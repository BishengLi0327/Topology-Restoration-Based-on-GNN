from tqdm import tqdm
from itertools import chain
import argparse
import os.path as osp
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import torch
import torch.nn.functional as F
from models import GCN

from torch_geometric.nn import GAE
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (negative_sampling, add_self_loops, train_test_split_edges, k_hop_subgraph,
                                   to_scipy_sparse_matrix, to_undirected, remove_self_loops, coalesce, from_networkx)

import warnings
warnings.filterwarnings("ignore")

max_z = 0


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class HUAWEI_Dataset(InMemoryDataset):
    def __init__(self):
        super().__init__()
        self.data = torch.load('../data/huawei_graph.pt')


# def load_data(path):
#     alarm_graph = np.load(path, allow_pickle=True)
#     node_list = list(alarm_graph.nodes)
#     edge_tmp = []
#     for edge in list(alarm_graph.edges):
#         if edge[0] != edge[1]:
#             edge_tmp.append(edge)  # 这一步是为了保证没有self-loops
#     edge_list = []
#     for i in range(len(edge_tmp)):
#         a = node_list.index(edge_tmp[i][0])
#         b = node_list.index(edge_tmp[i][1])
#         edge_list.append([a, b])
#
#     alarm_names = []
#     for ne_name in list(alarm_graph.nodes):
#         for alarm in alarm_graph.nodes[ne_name].keys():
#             if alarm != 'NE_TYPE' and alarm not in alarm_names:
#                 alarm_names.append(alarm)
#
#     labels = np.zeros([len(node_list), 3])
#     for i in range(len(alarm_graph.nodes)):
#         if alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'NODEB':
#             labels[i][0] = 1
#         elif alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'MICROWAVE':
#             labels[i][1] = 1
#         elif alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'ROUTER':
#             labels[i][2] = 1
#
#     attribute_length = len(alarm_names)
#     num_of_nodes = len(alarm_graph.nodes)
#     attribute_one_hot = np.zeros([num_of_nodes, attribute_length])
#
#     # one-hot
#     for i in range(len(alarm_graph.nodes)):
#         for alarm in alarm_graph.nodes[list(alarm_graph.nodes)[i]].keys():
#             if alarm != 'NE_TYPE':
#                 attribute_one_hot[i][alarm_names.index(alarm)] = 1
#     return node_list, edge_list, attribute_one_hot, labels


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
    # path = '../data/alarm_project_hitsz/preprocessed/G'
    # nodes, edge_list, attribute, node_labels = load_data(path)
    # dataset = Data(x=torch.tensor(attribute, dtype=torch.float),
    #                edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
    #                y=torch.tensor(node_labels, dtype=torch.float))
    dataset = HUAWEI_Dataset()[0]
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


def load_pubmed_dataset():
    path = '../data/Planetoid'
    dataset = Planetoid(path, 'PubMed')[0]
    dataset.train_mask = dataset.val_mask = dataset.test_mask = None
    dataset.y = F.one_hot(dataset.y).to(torch.float)
    return dataset


def load_airport_dataset():
    data_path = '/root/libisheng/HUAWEI/code/hgcn/data/airport'
    dataset_str = 'airport'
    graph = pickle.load(open(osp.join(data_path, dataset_str + '.p'), 'rb'))
    graph_pyg = from_networkx(graph)
    graph_pyg.x = graph_pyg.feat
    return graph_pyg


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

    seed = 925
    set_seed(seed)

    print('Loading dataset~~~')
    if args.dataset == 'huawei':
        dataset = load_huawei_dataset()
        if args.use_alarm:
            alarm_feature_path = '../data/alarm_construct_graph/embedding_80.pt'
            dataset.alarm_features = torch.load(alarm_feature_path)
    elif args.dataset == 'disease':
        dataset = load_disease_dataset()
    elif args.dataset == 'cora':
        dataset = load_cora_dataset()
    elif args.dataset == 'pubmed':
        dataset = load_pubmed_dataset()
    elif args.dataset == 'airport':
        dataset = load_airport_dataset()
    else:
        raise ValueError("Invalid dataset type")

    data = train_test_split_edges(dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    print('The dataset and the split edges are done!!!')

    # data.edge_index = data.train_pos_edge_index

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # load another graph, this step is for a small experiment
    # edge_path = '../data/alarm_construct_graph/alarm_graph_edge_index_10'
    # f = open(edge_path + '.pkl', 'rb')
    # edge_index = pickle.load(f)
    # f.close()
    #
    # edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).t().contiguous()
    # tmp_data = Data(edge_index=edge_index, num_nodes=41143)
    # edge_index = to_undirected(remove_self_loops(tmp_data.edge_index)[0])
    #
    # data.edge_index = coalesce(torch.cat((data.edge_index, edge_index), dim=1))
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
