import torch
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn import Node2Vec, GAE, GCN, GATConv
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix, k_hop_subgraph, from_networkx
from torch_geometric.transforms import RandomLinkSplit
import pickle
import os.path as osp
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import chain
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
max_z = 0


class SEAL_Dataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, dataset, num_hops=1, split='train'):
        self.d_name = str(dataset)[:-2]
        self.data = dataset[0]
        self.num_hops = num_hops
        super(SEAL_Dataset, self).__init__(dataset.root)
        index = ['train', 'val', 'test'].index(split)
        self.data, self.slices = torch.load(self.processed_paths[index])

    @property
    def processed_file_names(self):
        return ['SEAL_' + self.d_name + '_train_data.pt',
                'SEAL_' + self.d_name + '_val_data.pt',
                'SEAL_' + self.d_name + '_test_data.pt']

    def process(self):
        transform = RandomLinkSplit(num_val=0.05, num_test=0.1,
                                    is_undirected=True, split_labels=True)
        train_data, val_data, test_data = transform(self.data)

        self._max_z = 0

        # Collect a list of subgraphs for training, validation and testing:
        train_pos_data_list = self.extract_enclosing_subgraphs(
            train_data.edge_index, train_data.pos_edge_label_index, 1)
        train_neg_data_list = self.extract_enclosing_subgraphs(
            train_data.edge_index, train_data.neg_edge_label_index, 0)

        val_pos_data_list = self.extract_enclosing_subgraphs(
            val_data.edge_index, val_data.pos_edge_label_index, 1)
        val_neg_data_list = self.extract_enclosing_subgraphs(
            val_data.edge_index, val_data.neg_edge_label_index, 0)

        test_pos_data_list = self.extract_enclosing_subgraphs(
            test_data.edge_index, test_data.pos_edge_label_index, 1)
        test_neg_data_list = self.extract_enclosing_subgraphs(
            test_data.edge_index, test_data.neg_edge_label_index, 0)

        # Convert node labeling to one-hot features.
        for data in chain(train_pos_data_list, train_neg_data_list,
                          val_pos_data_list, val_neg_data_list,
                          test_pos_data_list, test_neg_data_list):
            # We solely learn links from structure, dropping any node features:
            data.x = F.one_hot(data.z, self._max_z + 1).to(torch.float)

        torch.save(self.collate(train_pos_data_list + train_neg_data_list),
                   self.processed_paths[0])
        torch.save(self.collate(val_pos_data_list + val_neg_data_list),
                   self.processed_paths[1])
        torch.save(self.collate(test_pos_data_list + test_neg_data_list),
                   self.processed_paths[2])

    def extract_enclosing_subgraphs(self, edge_index, edge_label_index, y):
        data_list = []
        for src, dst in edge_label_index.t().tolist():
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src, dst], self.num_hops, edge_index, relabel_nodes=True)
            src, dst = mapping.tolist()

            # Remove target link from the subgraph.
            mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
            sub_edge_index = sub_edge_index[:, mask1 & mask2]

            # Calculate node labeling.
            z = self.drnl_node_labeling(sub_edge_index, src, dst,
                                        num_nodes=sub_nodes.size(0))

            data = Data(x=self.data.x[sub_nodes], z=z,
                        edge_index=sub_edge_index, y=y)
            data_list.append(data)

        return data_list

    def drnl_node_labeling(self, edge_index, src, dst, num_nodes=None):
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

        self._max_z = max(int(z.max()), self._max_z)

        return z.to(torch.long)


class GAT(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


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

        sub_data = Data(x=data.x[sub_nodes], z=z, edge_index=sub_edge_index, y=y)
        if 'one_hot_y' in data.keys:
            sub_data.one_hot_y = data.one_hot_y[sub_nodes]
        if 'pretrained_features' in data.keys:
            sub_data.pretrained_features = data.pretrained_features[sub_nodes]
        if 'knn_emb' in data.keys:
            sub_data.knn_emb = data.knn_emb[sub_nodes]

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
        if 'knn_emb' in data.keys:
            data.x = torch.cat((data.x, data.knn_emb), dim=1)
            data.knn_emb = None
        if use_label and 'one_hot_y' in data.keys:
            data.x = torch.cat((data.x, data.one_hot_y), dim=1)
            data.one_hot_y = None
        if 'pretrained_features' in data.keys:
            data.x = torch.cat((data.x, data.pretrained_features), dim=1)
            data.pretrained_features = None

    return train_pos_list + train_neg_list, val_pos_list + val_neg_list, test_pos_list + test_neg_list


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
    return model().detach().cpu()


class Discriminator(torch.nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        # self.f_k = torch.nn.Bilinear(n_h, n_h, 1)
        self.lin = torch.nn.Linear(n_h, n_h)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, h_pl, h_mi):
        assert h_pl.shape == h_mi.shape

        logits = torch.mm(self.lin(h_pl), torch.transpose(h_mi, dim0=0, dim1=1))

        return logits


class Contrastive_Net(torch.nn.Module):
    def __init__(self, topo_in_channels, feat_in_channels, out_channels):
        super(Contrastive_Net, self).__init__()
        self.model_topo = GAE(GCN(topo_in_channels, out_channels, 2))
        self.model_feat = GAT(feat_in_channels, out_channels)
        self.disc = Discriminator(out_channels)

    def reset_parameters(self):
        self.model_feat.reset_parameters()
        self.model_topo.reset_parameters()

    def forward(self, topo_data, feat_data):
        topo_z = self.model_topo.encode(topo_data.x, topo_data.train_pos_edge_index)
        feat_z = self.model_feat(feat_data.x, feat_data.edge_index)

        res = self.disc(topo_z, feat_z)
        # return topo_z, feat_z
        return res


def CL(args, knn_graph, data):

    pretrained_data = data.clone()
    # 这个地方如果选择pretrain，应该采取negative injection的方式，重新训练得到特征
    pretrained_data.train_pos_edge_index = torch.cat(
        (pretrained_data.train_pos_edge_index, pretrained_data.train_neg_edge_index), dim=1
    )
    pretrained_data.train_neg_edge_index = None
    # knn_graph.x = torch.randn([knn_graph.num_nodes, 128])
    knn_graph.x = data.x

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    model = Contrastive_Net(data.num_features, knn_graph.x.shape[1], 32).to(device)
    disc = Discriminator(32).to(device)
    pretrained_data = pretrained_data.to(device)
    knn_graph = knn_graph.to(device)
    # pre_model_topo = GAE(GCN(data.num_features, 32))
    # pre_model_feat = GAT(knn_graph.x.shape[1], 32)
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': disc.parameters()}],
                                 lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    min_loss = 1e9
    patience = 0
    for epoch in range(1, args.pretrained_epochs):
        scheduler.step()
        optimizer.zero_grad()
        res = model(pretrained_data, knn_graph).view(-1)
        lbls = torch.eye(knn_graph.num_nodes).to(device).view(-1)
        loss = torch.nn.BCEWithLogitsLoss()(res, lbls)
        loss.backward()
        optimizer.step()
        if loss < min_loss:
            min_loss = loss
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            print('Early Stop...')
            break
        print(f'Epoch: {epoch:3d}, Loss: {loss:.4f}')
    # torch.save(model.state_dict(), '../checkpoint/CL_model.pt')
    # model = Contrastive_Net(data.num_features, knn_graph.x.shape[1], 32)
    # model.load_state_dict('../checkpoint/CL_model.pt')
    # return topo_z.detach().cpu(), feat_z.detach().cpu()
    return model.model_topo.encode(pretrained_data.x, pretrained_data.train_pos_edge_index).detach().cpu(),\
           model.model_feat(knn_graph.x, knn_graph.edge_index).detach().cpu()
