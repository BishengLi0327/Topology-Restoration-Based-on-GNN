"""
This code is a extension version of KNNGraph + SEAL, which aims to incorporate metric learning to compute the
distance while constructing the KNN Graph.
"""
import torch
import torch_geometric.nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.nn import Node2Vec
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, add_self_loops, negative_sampling,\
    coalesce, from_networkx, to_scipy_sparse_matrix, k_hop_subgraph, to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse.csgraph import shortest_path
from torch_geometric.transforms import KNNGraph
from tensorboardX import SummaryWriter

from models import DGCNN

import os
import time
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from itertools import chain

import argparse
import os.path as osp
import warnings
warnings.filterwarnings('ignore')


# class KNNGraph(object):
#     def __init__(self, k, loop=False, force_undirected=False, flow='source_to_target'):
#         super(KNNGraph, self).__init__()
#         self.k = k
#         self.loop = loop
#         self.force_undirected = force_undirected
#         self.flow = flow
#
#     def __call__(self, data):
#         data.edge_attr = None
#         batch = data.batch if 'batch' in data else None
#         edge_index = torch_geometric.nn.knn_graph(data.pos, self.k, batch, loop=self.loop, flow=self.flow)
#
#         if self.force_undirected:
#             edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
#
#         data.edge_index = edge_index
#
#     def __repr__(self):
#         return '{}(k={})'.format(self.__class__.__name__, self.k)


def load_data(dataset):
    if dataset == 'cora':
        dataset = Planetoid(root='../data/Planetoid', name='Cora')[0]
        dataset.one_hot_y = F.one_hot(dataset.y).to(torch.float)
        dataset.train_mask = dataset.val_mask = dataset.test_mask = None
        return dataset
    elif dataset == 'pubmed':
        dataset = Planetoid('../data/Planetoid', 'PubMed')[0]
        dataset.one_hot_y = F.one_hot(dataset.y).to(torch.float)
        dataset.train_mask = dataset.val_mask = dataset.test_mask = None
        return dataset
    elif dataset == 'airport':
        data_path = '/root/libisheng/HUAWEI/code/hgcn/data/airport'
        dataset_str = 'airport'
        graph = pickle.load(open(osp.join(data_path, dataset_str + '.p'), 'rb'))
        dataset = from_networkx(graph)
        dataset.x = dataset.feat
        dataset.feat = None
        return dataset
    elif dataset == 'disease':
        path = '../data/disease_lp/'
        edges = pd.read_csv(path + 'disease_lp.edges.csv')
        labels = np.load(path + 'disease_lp.labels.npy')
        features = sp.load_npz(path + 'disease_lp.feats.npz').todense()
        dataset = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edges.values).t().contiguous(),
            one_hot_y=F.one_hot(torch.tensor(labels))
        )
        return dataset
    else:
        raise ValueError('Invalid dataset!')


def drnl_node_labeling(edge_index, src, dst, num_nodes=None):
    global max_z
    # Double-radius node labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                             indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                             indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    max_z = max(int(z.max()), max_z)

    return z.to(torch.long)


def extract_enclosing_subgraphs(data, link_index, edge_index, y):
    data_list = []
    for src, dst in tqdm(link_index.t().tolist(), desc='Extracting...'):
        # src: source   dst: destination
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], num_hops=2, edge_index=edge_index, relabel_nodes=True, num_nodes=data.num_nodes
        )
        src, dst = mapping.tolist()

        # remove target link from the subgraph
        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]

        # calculate node labeling
        z = drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))

        sub_data = Data(x=data.x[sub_nodes], z=z, edge_index=sub_edge_index, y=y, sub_nodes_index=sub_nodes)
        if 'one_hot_y' in data.keys:
            sub_data.one_hot_y = data.one_hot_y[sub_nodes]
        if 'pretrained_features' in data.keys:
            sub_data.pretrained_features = data.pretrained_features[sub_nodes]

        data_list.append(sub_data)

    return data_list


def extract_subgraphs(data, use_label: bool, use_feat: bool):
    print('=' * 50)
    print('Starting extracting subgraphs...')
    train_pos_list = extract_enclosing_subgraphs(
        data, data.train_pos_edge_index, data.edge_index, 1
    )
    train_neg_list = extract_enclosing_subgraphs(
        data, data.train_neg_edge_index, data.edge_index, 0
    )

    val_pos_list = extract_enclosing_subgraphs(
        data, data.val_pos_edge_index, data.edge_index, 1
    )
    val_neg_list = extract_enclosing_subgraphs(
        data, data.val_neg_edge_index, data.edge_index, 0
    )

    test_pos_list = extract_enclosing_subgraphs(
        data, data.test_pos_edge_index, data.edge_index, 1
    )
    test_neg_list = extract_enclosing_subgraphs(
        data, data.test_neg_edge_index, data.edge_index, 0
    )
    print('Finished extracting subgraphs.')
    print('=' * 50)

    for data in chain(train_pos_list, train_neg_list, val_pos_list, val_neg_list, test_pos_list, test_neg_list):
        # data.x = torch.cat((F.one_hot(data.z, max_z + 1).to(torch.float), data.knn_emb), dim=1)
        if use_feat and 'x' in data.keys:
            data.x = torch.cat((data.x, F.one_hot(data.z, max_z+1).to(torch.float)), dim=1)
        else:
            data.x = F.one_hot(data.z, max_z + 1).to(torch.float)
        data.z = None
        if use_label and 'one_hot_y' in data.keys:
            data.x = torch.cat((data.x, data.one_hot_y), dim=1)
            data.one_hot_y = None

    return train_pos_list + train_neg_list, val_pos_list + val_neg_list, test_pos_list + test_neg_list


def train_node2vec_emb(data):
    print('=' * 50)
    print('Start train node2vec model on the knn graph.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Node2Vec(data.edge_index, embedding_dim=32, walk_length=10, context_size=5, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=False, num_nodes=data.num_nodes).to(device)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    minimal_loss = 1e9
    patience = 0
    patience_threshold = 10
    for epoch in range(1, 201):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss / len(loader)
        if loss < minimal_loss:
            minimal_loss = loss
            patience = 0
        else:
            patience += 1
        if patience >= patience_threshold:
            print('Early Stop.')
            break
        print("Epoch: {:02d}, loss: {:.4f}".format(epoch, loss))
    print('Finished training.')
    print('=' * 50)
    return model()


def train(model, train_loader, device, optimizer, train_dataset):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.batch)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader, model, device):
    model.eval()

    y_pred, y_true = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))

    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred)), \
           average_precision_score(torch.cat(y_true), torch.cat(y_pred))


def construct_KNN_graph(dataset, weight):
    dataset.pos = torch.mm(dataset.x, weight)
    k = int(dataset.num_edges / dataset.num_nodes) + 1

    trans = KNNGraph(k, loop=False, force_undirected=True)
    knn_graph = trans(dataset.clone())
    return knn_graph


def run():
    parser = argparse.ArgumentParser('Configurations for SEAL with data augmentations')
    parser.add_argument('--dataset', default='cora', type=str)
    parser.add_argument('--use_label', action='store_true',
                        help='whether to use label information as additional features')
    parser.add_argument('--epochs', default=401, type=int, help='training epochs')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool)
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decaying')
    parser.add_argument('--val_ratio', default=0.05, type=float, help='validation links ratio')
    parser.add_argument('--test_ratio', default=0.10, type=float, help='test link ratio')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--use_feat', action='store_true', help='whether to use original feature')
    parser.add_argument('--knn_usage', default='add_feat', choices=['add_feat', 'concat_graph'])
    parser.add_argument('--patience', default=20, type=int, help='early stop steps')
    args = parser.parse_args()
    print(args)

    dataset = load_data(args.dataset)

    # train/val/test split
    data = train_test_split_edges(dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    data.edge_index = data.train_pos_edge_index

    train_graphs, val_graphs, test_graphs = extract_subgraphs(data, args.use_label, args.use_feat)

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    model = DGCNN(train_graphs, hidden_channels=32, num_layers=3).to(device)
    weight1 = torch.nn.Parameter(torch.randn(dataset.num_features, 32), requires_grad=True)
    optimizer = torch.optim.Adam([{'params': weight1}, {'params': model.parameters()}], lr=args.lr,
                                 weight_decay=args.wd)

    best_val_auc = test_auc = test_ap = 0
    patience = 0

    for epoch in range(1, args.epochs):

        knn_graph = construct_KNN_graph(dataset, weight1)
        knn_emb = train_node2vec_emb(knn_graph)

        train_loader = DataLoader(train_graphs, batch_size=args.bs, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=args.bs, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=args.bs, shuffle=False)

        loss = train(model, train_loader, device, optimizer, train_graphs)
        val_auc, val_ap = test(val_loader, model, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc, test_ap = test(test_loader, model, device)
            patience = 0
            # saving model parameters
            state = {'model': model.state_dict(), 'auc': test_auc, 'ap': test_ap, 'epoch': epoch}
            save_path = '../checkpoint/KNN-SEAL/'
            if not osp.exists(save_path):
                os.mkdir(save_path)
            torch.save(state, osp.join(save_path, args.dataset + '-' + 'ckpt.pth'))
        else:
            patience += 1
        if patience >= args.patience:
            print('Early Stop! Best Val AUC: {:.4f}, Test AUC: {:.4f}'.format(best_val_auc, test_auc))
            break

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val_AUC: {val_auc:.4f}, Val_AP: {val_ap:.4f}, '
              f'Test_AUC: {test_auc:.4f}, Test_AP: {test_ap:.4f}')


if __name__ == '__main__':
    max_z = 0
    seed_everything(11)
    run()
