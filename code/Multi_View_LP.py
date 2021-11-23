import math
import os.path as osp

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import ModuleList, BCEWithLogitsLoss
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_add_pool, global_sort_pool
from torch_geometric.utils import negative_sampling, k_hop_subgraph, add_self_loops,\
    to_scipy_sparse_matrix, train_test_split_edges
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid, CoraFull, Flickr

import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.sparse.csgraph import shortest_path
from itertools import chain
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')


class Feature_Net(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(Feature_Net, self).__init__()
        self.conv1 = GCNConv(in_channels=in_feats, out_channels=h_feats)
        self.conv2 = GCNConv(in_channels=h_feats, out_channels=out_feats)

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        return x


class DGCNN(torch.nn.Module):
    def __init__(self, train_dataset, hidden_channels, num_layers, GNN=GCNConv, k=0.6):
        super(DGCNN, self).__init__()

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.convs = ModuleList()
        self.convs.append(GNN(train_dataset[0].topo_feat.size(1), hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = nn.Linear(dense_dim, 32)

    def forward(self, x, edge_index, batch):
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = self.lin1(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)
        return x  # [num_graphs, 16]


class Topology_Net(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(Topology_Net, self).__init__()
        self.conv1 = SAGEConv(in_channels=in_feats, out_channels=h_feats)
        self.conv2 = SAGEConv(in_channels=h_feats, out_channels=out_feats)

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        return x


class GAT(torch.nn.Module):

    def __init__(self, in_channels, h_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, h_channels, heads=8)
        self.conv2 = GATConv(8 * h_channels, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index, batch):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        return x


class Net(nn.Module):
    def __init__(self, train_dataset, topology_h_feats, feature_in_feats,
                 feature_h_feats, out_feats, num_classes):
        super(Net, self).__init__()
        self.model_t = DGCNN(
            train_dataset=train_dataset,
            hidden_channels=topology_h_feats,
            num_layers=3
        )
        # self.model_t = Topology_Net(topology_in_feats, topology_h_feats, out_feats)
        # self.model_f = Feature_Net(feature_in_feats, feature_h_feats, out_feats)
        self.model_f = GAT(feature_in_feats, feature_h_feats, out_feats)
        self.att_t = nn.Linear(out_feats, 32)
        self.att_f = nn.Linear(out_feats, 32)
        self.query = nn.Linear(32, 1)
        self.lin1 = nn.Linear(out_feats, 32)
        self.lin2 = nn.Linear(32, num_classes)

    def forward(self, data):
        feat, topo, edge_index, batch = data.x, data.topo_feat, data.edge_index, data.batch
        x_t = self.model_t(topo, edge_index, batch)
        x_f = self.model_f(feat, edge_index, batch)
        att_t = self.query(F.tanh(self.att_t(x_t)))
        att_f = self.query(F.tanh(self.att_f(x_f)))
        alpha_t = torch.exp(att_t) / (torch.exp(att_t) + torch.exp(att_f))
        alpha_f = torch.exp(att_f) / (torch.exp(att_t) + torch.exp(att_f))
        x = alpha_t * x_t + alpha_f * x_f
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


def drnl_node_labeling(edge_index, src, dst, num_nodes=None):
    global topo_feat_length
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

    # max_z = max(int(z.max()), max_z)
    topo_feat_length = max((int(z.max())), topo_feat_length)

    return z.to(torch.long)


def extract_enclosing_subgraphs(data, link_index, edge_index, y):
    data_list = []
    for src, dst in tqdm(link_index.t().tolist(), desc='Extracting...'):
        # src: source   dst: destination
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], num_hops=2, edge_index=edge_index, relabel_nodes=True
        )
        src, dst = mapping.tolist()

        # remove target link from the subgraph
        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]

        # calculate node labeling
        z = drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))

        sub_data = Data(x=data.x[sub_nodes], z=z, edge_index=sub_edge_index, y=y)
        data_list.append(sub_data)

    return data_list


def train(model, train_loader, optimizer, device):
    model.train()

    total_loss, num_graphs = 0, 0
    for data in train_loader:
        num_graphs += data.num_graphs
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = BCEWithLogitsLoss()(logits[:, 0], data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss * data.num_graphs

    return total_loss / num_graphs


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    y_pred, y_true = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data)
        y_pred.append(logits[:, 0].cpu())
        y_true.append(data.y.cpu().to(torch.float))
    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred))


def run():
    parser = argparse.ArgumentParser('Configuration for Link Prediction')
    parser.add_argument('--dataset', default='Cora', type=str, help='dataset')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--weight_decaying', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool)
    parser.add_argument('--epochs', default=1001, type=int, help='epochs to run')
    parser.add_argument('--val_ratio', default=0.05, type=float, help='validation ratio')
    parser.add_argument('--test_ratio', default=0.1, type=float, help='test ratio')
    parser.add_argument('--patience', default=50, type=int, help='early stop steps')
    args = parser.parse_args()
    print(args)

    print('Start loading dataset.')
    if args.dataset == 'Cora':
        dataset = Planetoid(root='../data/Planetoid', name='Cora')
        data = dataset[0]
    elif args.dataset == 'CiteSeer':
        dataset = Planetoid(root='../data/Planetoid', name='CiteSeer')
        data = dataset[0]
    elif args.dataset == 'PubMed':
        dataset = Planetoid(root='../data/Planetoid', name='PubMed')
        data = dataset[0]
    elif args.dataset == 'CoraFull':
        dataset = CoraFull(root='../data/CoraFull')
        data = dataset[0]
    elif args.dataset == 'Flickr':
        dataset = Flickr(root='../data/Flickr')
        data = dataset[0]
    else:
        raise KeyError("Invalid dataset name!")
    print('Dataset loaded!')

    print('Start train test split edges')
    data = train_test_split_edges(data, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    print('Done with train test split edges!')

    print('='*50)
    print('Start extracting subgraphs~~~')
    train_pos_list = extract_enclosing_subgraphs(
        data, data.train_pos_edge_index, data.train_pos_edge_index, 1
    )
    train_neg_list = extract_enclosing_subgraphs(
        data, data.train_neg_edge_index, data.train_pos_edge_index, 0
    )

    val_pos_list = extract_enclosing_subgraphs(
        data, data.val_pos_edge_index, data.train_pos_edge_index, 1
    )
    val_neg_list = extract_enclosing_subgraphs(
        data, data.val_neg_edge_index, data.train_pos_edge_index, 0
    )

    test_pos_list = extract_enclosing_subgraphs(
        data, data.test_pos_edge_index, data.train_pos_edge_index, 1
    )
    test_neg_list = extract_enclosing_subgraphs(
        data, data.test_neg_edge_index, data.train_pos_edge_index, 0
    )

    for data in chain(train_pos_list, train_neg_list, val_pos_list, val_neg_list, test_pos_list, test_neg_list):
        data.topo_feat = F.one_hot(data.z, topo_feat_length+1).to(torch.float)
        data.z = None

    print('Done with extracting subgraphs.')
    print('='*50)

    train_graphs = train_pos_list + train_neg_list
    val_graphs = val_pos_list + val_neg_list
    test_graphs = test_pos_list + test_neg_list
    train_loader = DataLoader(train_graphs, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.bs)
    test_loader = DataLoader(test_graphs, batch_size=args.bs)

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    model = Net(
        train_dataset=train_graphs,
        # topology_in_feats=train_graphs[0].topo_feat.shape[1],
        topology_h_feats=64,
        feature_in_feats=train_graphs[0].x.shape[1],
        feature_h_feats=128,
        out_feats=32,
        num_classes=2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decaying)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decaying)

    df = pd.DataFrame(columns=['epoch', 'train loss', 'val auc', 'test auc'])
    path = osp.join('../results/MultiView_LP', args.dataset, 'result.csv')
    df.to_csv(path, index=False)

    best_val_auc, test_auc, patience = 0, 0, 0
    for epoch in range(1, args.epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_auc = test(model, val_loader, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = test(model, test_loader, device)
            state = {'model': model.state_dict(), 'auc': test_auc, 'epoch': epoch}
            path = osp.join('../checkpoint/MultiView_LP/', args.dataset+'ckpt.pt')
            torch.save(state, path)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
        print(f"Epoch: {epoch:04d}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
        record_list = [epoch, train_loss, val_auc, test_auc]
        data = pd.DataFrame([record_list])
        data.to_csv(path, mode='a', header=False, index=False)


if __name__ == '__main__':
    topo_feat_length = 0
    run()
