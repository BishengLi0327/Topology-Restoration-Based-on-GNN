from tqdm import tqdm
from itertools import chain

from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorboardX import SummaryWriter
import os
from models import DGCNN
import argparse
import os.path as osp

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import torch
import torch.nn.functional as F
# from gae_huawei import train, test
from models import GCN, SGC, GAT

from torch_geometric.nn import GAE
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, k_hop_subgraph,
                                   to_scipy_sparse_matrix)

import warnings
warnings.filterwarnings("ignore")

max_z = 0


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(path):
    alarm_graph = np.load(path, allow_pickle=True)
    node_list = list(alarm_graph.nodes)
    edge_tmp = []
    for edge in list(alarm_graph.edges):
        if edge[0] != edge[1]:
            edge_tmp.append(edge)  # 这一步是为了保证没有self-loops
    edge_list = []
    for i in range(len(edge_tmp)):
        a = node_list.index(edge_tmp[i][0])
        b = node_list.index(edge_tmp[i][1])
        edge_list.append([a, b])

    alarm_names = []
    for ne_name in list(alarm_graph.nodes):
        for alarm in alarm_graph.nodes[ne_name].keys():
            if alarm != 'NE_TYPE' and alarm not in alarm_names:
                alarm_names.append(alarm)

    labels = np.zeros([len(node_list), 3])
    for i in range(len(alarm_graph.nodes)):
        if alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'NODEB':
            labels[i][0] = 1
        elif alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'MICROWAVE':
            labels[i][1] = 1
        elif alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'ROUTER':
            labels[i][2] = 1

    label_list = []
    for i in range(len(alarm_graph.nodes)):
        if alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'NODEB':
            label_list.append(1)
        elif alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'MICROWAVE':
            label_list.append(2)
        elif alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'ROUTER':
            label_list.append(3)

    attribute_length = len(alarm_names)
    num_of_nodes = len(alarm_graph.nodes)
    attribute_one_hot = np.zeros([num_of_nodes, attribute_length])

    # one-hot
    for i in range(len(alarm_graph.nodes)):
        for alarm in alarm_graph.nodes[list(alarm_graph.nodes)[i]].keys():
            if alarm != 'NE_TYPE':
                attribute_one_hot[i][alarm_names.index(alarm)] = 1
    return node_list, edge_list, attribute_one_hot, labels, label_list


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
        if 'pretrained_features' in data.keys:
            sub_data.pretrained_features = data.pretrained_features[sub_nodes]
        if 'alarm_features' in data.keys:
            sub_data.alarm_features = data.alarm_features[sub_nodes]

        data_list.append(sub_data)

    return data_list


def load_huawei_dataset():
    path = '../data/alarm_project_hitsz/preprocessed/G'
    nodes, edge_list, attribute, node_labels, labels = load_data(path)
    dataset = Data(x=torch.tensor(attribute, dtype=torch.float),
                   edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
                   y=torch.tensor(node_labels, dtype=torch.float), labels=labels)
    return dataset


def load_disease_dataset():
    path = '../data/disease_lp/'
    edges = pd.read_csv(path + 'disease_lp.edges.csv')
    labels = np.load(path + 'disease_lp.labels.npy')
    features = sp.load_npz(path + 'disease_lp.feats.npz').todense()
    dataset = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=torch.tensor(edges.values).t().contiguous(),
        y=F.one_hot(torch.tensor(labels))
    )
    return dataset


def load_cora_dataset():
    path = osp.join(osp.dirname(osp.relpath('seal_dataset.py')), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, 'Cora')[0]
    dataset.y = F.one_hot(dataset.y).to(torch.float)
    # 这里做了一个处理，将标签转化维one-hot向量
    return dataset


def pre_train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss), z


def pre_test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def process(args):

    seed = 327
    set_seed(seed)

    print('Loading dataset~~~')
    if args.dataset == 'huawei':
        dataset = load_huawei_dataset()
        if args.use_alarm:
            alarm_feature_path = '../data/alarm_construct_graph/embedding_10.pt'
            dataset.alarm_features = torch.load(alarm_feature_path)
    elif args.dataset == 'disease':
        dataset = load_disease_dataset()
    elif args.dataset == 'cora':
        dataset = load_cora_dataset()
    else:
        raise ValueError("Invalid dataset type")

    data = train_test_split_edges(dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # =============================================================================================
    # the flowing step is just for huawei dataset
    if args.dataset == 'huawei' and args.hierarchical:
        # data.val_pos_edge_index
        edge_index = data.val_pos_edge_index.t().tolist()
        mask = [False] * len(edge_index)
        for i in range(len(edge_index)):
            if (data.labels[edge_index[i][0]] == 1 and data.labels[edge_index[i][1]] != 1) \
                    or (data.labels[edge_index[i][0]] != 1 and data.labels[edge_index[i][1]] == 1):
                mask[i] = True
        data.val_pos_edge_index = torch.tensor(edge_index, dtype=torch.long)[mask].t().contiguous()

        # data.val_neg_edge_index
        edge_index = data.val_neg_edge_index.t().tolist()
        mask = [False] * len(edge_index)
        for i in range(len(edge_index)):
            if (data.labels[edge_index[i][0]] == 1 and data.labels[edge_index[i][1]] != 1) \
                    or (data.labels[edge_index[i][0]] != 1 and data.labels[edge_index[i][1]] == 1):
                mask[i] = True
        data.val_neg_edge_index = torch.tensor(edge_index, dtype=torch.long)[mask].t().contiguous()

        # data.test_neg_edge_index
        edge_index = data.test_pos_edge_index.t().tolist()
        mask = [False] * len(edge_index)
        for i in range(len(edge_index)):
            if (data.labels[edge_index[i][0]] == 1 and data.labels[edge_index[i][1]] != 1) \
                    or (data.labels[edge_index[i][0]] != 1 and data.labels[edge_index[i][1]] == 1):
                mask[i] = True
        data.test_pos_edge_index = torch.tensor(edge_index, dtype=torch.long)[mask].t().contiguous()

        # data.test_neg_edge_index
        edge_index = data.test_neg_edge_index.t().tolist()
        mask = [False] * len(edge_index)
        for i in range(len(edge_index)):
            if (data.labels[edge_index[i][0]] == 1 and data.labels[edge_index[i][1]] != 1) \
                    or (data.labels[edge_index[i][0]] != 1 and data.labels[edge_index[i][1]] == 1):
                mask[i] = True
        data.test_neg_edge_index = torch.tensor(edge_index, dtype=torch.long)[mask].t().contiguous()
    # =================================================================================================

    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    print('The dataset and the split edges are done!!!')

    if args.pretrain:
        pretrained_data = data.clone()
        pretrained_data.train_pos_edge_index = torch.cat(
            (pretrained_data.train_pos_edge_index, pretrained_data.train_neg_edge_index), dim=1
        )
        pretrained_data.train_neg_edge_index = None
        # 这个地方如果选择pretrain，应该采取negative injection的方式，重新训练得到特征
        print('-' * 60)
        print('Pretraining')
        if args.pre_encoder == 'GCN':
            pre_model = GAE(GCN(dataset.num_features, 32))
        else:
            raise ValueError('Invalid model type!')

        optimizer = torch.optim.Adam(pre_model.parameters(), lr=0.001)

        best_auc = 0
        patience = 0
        for pretrained_epoch in range(1, args.pretrained_epochs):
            train_loss, node_embedding = pre_train(pre_model, pretrained_data, optimizer)
            val_auc, val_ap = pre_test(pre_model, data.x, data.train_pos_edge_index,
                                       data.val_pos_edge_index, data.val_neg_edge_index)
            print(f"Epoch: {pretrained_epoch:03d}, Loss: {train_loss:.4f}, Val_AUC: {val_auc:.4f}, Val_AP: {val_ap:.4f}")
            if val_auc > best_auc:
                best_auc = val_auc
                patience = 0
            else:
                patience += 1
            if patience > args.patience:
                break

        print('-' * 60)
        print('Finished pretraining')
        data.pretrained_features = node_embedding.detach()

    if args.embedding == 'DRNL':
        pass
    else:
        data.x = data.y

    print('Starting extracting subgraphs~~~')
    # collect a list of subgraphs of training, validation and test
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
    print('Finished extracting subgraphs.')

    if args.embedding == 'DRNL':
        # convert labels to one-hot features
        for data in chain(train_pos_list, train_neg_list,
                          val_pos_list, val_neg_list,
                          test_pos_list, test_neg_list):
            data.x = F.one_hot(data.z, max_z + 1).to(torch.float)
    elif args.embedding == 'DRNL_SelfFeat':
        for data in chain(train_pos_list, train_neg_list,
                          val_pos_list, val_neg_list,
                          test_pos_list, test_neg_list):
            data.x = torch.cat((F.one_hot(data.z, max_z + 1).to(torch.float), data.x), dim=1)
    elif args.embedding == 'SelfFeat':
        pass
    else:
        raise ValueError("Unsupported embedding type.")

    if args.pretrain:
        for data in chain(train_pos_list, train_neg_list,
                          val_pos_list, val_neg_list,
                          test_pos_list, test_neg_list):
            data.x = torch.cat((data.x, data.pretrained_features), dim=1)
            data.pretrained_features = None

    if args.use_alarm:
        for data in chain(train_pos_list, train_neg_list,
                          val_pos_list, val_neg_list,
                          test_pos_list, test_neg_list):
            data.x = torch.cat((data.x, data.alarm_features), dim=1)
            data.alarm_features = None

    return train_pos_list + train_neg_list, val_pos_list + val_neg_list, test_pos_list + test_neg_list


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


def run():
    parser = argparse.ArgumentParser("Configurations for seal")
    parser.add_argument('--dataset', default='huawei', type=str, help='dataset')
    parser.add_argument('--embedding', default='DRNL', type=str,
                        help='node encoding(["DRNL", "DRNL_SelfFeat", "SelfFeat"])')
    parser.add_argument('--epochs', default=101, type=int, help='training epochs')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='cuda')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--val_ratio', default=0.05, type=float)
    parser.add_argument('--test_ratio', default=0.10, type=float)
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrained_epochs', default=401, type=int)
    parser.add_argument('--pre_encoder', default='GCN', type=str, choices=['GCN'])
    parser.add_argument('--patience', default=50, type=int, help='early stop steps')
    parser.add_argument('--use_alarm', action='store_true')
    parser.add_argument('--hierarchical', action='store_true')
    args = parser.parse_args()
    print(args)

    args.split_ratio = str(int((1-args.val_ratio-args.test_ratio)*100)) \
                       + str(int(args.val_ratio*100)) + str(int(args.test_ratio*100))

    train_dataset, val_dataset, test_dataset = process(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda:1' if args.cuda else 'cpu')
    model = DGCNN(train_dataset, hidden_channels=32, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainWriter = SummaryWriter('../{}/{}/{}/{}/{}'.format(
        'runs', 'SEAL', args.dataset + '_' + args.split_ratio + '_pretrained_' + str(args.pretrain),
        args.embedding, 'Train'
    ))
    valWriter = SummaryWriter('../{}/{}/{}/{}/{}'.format(
        'runs', 'SEAL', args.dataset + '_' + args.split_ratio + '_pretrained_' + str(args.pretrain),
        args.embedding, 'Val'
    ))

    best_val_auc = test_auc = test_ap = 0

    for epoch in range(1, args.epochs):
        loss = train(model, train_loader, device, optimizer, train_dataset)
        trainWriter.add_scalar(tag='Train Loss', scalar_value=loss, global_step=epoch)
        val_auc, val_ap = test(val_loader, model, device)
        valWriter.add_scalar(tag='Val AUC', scalar_value=val_auc, global_step=epoch)
        valWriter.add_scalar(tag='Val AP', scalar_value=val_ap, global_step=epoch)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc, test_ap = test(test_loader, model, device)
            # saving model parameters
            state = {'model': model.state_dict(), 'auc': test_auc, 'ap': test_ap, 'epoch': epoch}
            save_path = '../checkpoint/SEAL/'
            if not osp.exists(save_path):
                os.mkdir(save_path)
            torch.save(state, osp.join(save_path, args.dataset+'-'+args.split_ratio+'-'+args.embedding+'-'+'ckpt.pth'))

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val_AUC: {val_auc:.4f}, Val_AP: {val_ap:.4f},'
              f'Test_AUC: {test_auc:.4f}, Test_AP: {test_ap:.4f}')


if __name__ == "__main__":
    run()




