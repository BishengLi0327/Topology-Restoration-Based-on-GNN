import random
import os.path as osp

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_add_pool
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid, CoraFull, Flickr

import argparse
from tqdm import tqdm
import numpy as np
from scipy.sparse.csgraph import shortest_path
from itertools import chain
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')


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
    def __init__(self, topology_in_feats, topology_h_feats, feature_in_feats,
                 feature_h_feats, out_feats, num_classes):
        super(Net, self).__init__()
        self.model_t = Topology_Net(topology_in_feats, topology_h_feats, out_feats)
        self.model_f = Feature_Net(feature_in_feats, feature_h_feats, out_feats)
        # self.model_f = GAT(feature_in_feats, feature_h_feats, out_feats)
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


def node_labeling(edge_index, node, num_nodes=None):
    # 可自由发挥
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    dist = shortest_path(adj, directed=False, unweighted=True, indices=node)
    dist = torch.from_numpy(dist).squeeze(dim=0)
    return dist.to(torch.long)


def extract_subgraphs(data):
    subgraphs = []
    for i in tqdm(range(data.num_nodes)):
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            i, num_hops=2, edge_index=data.edge_index, relabel_nodes=True
        )
        # node = mapping
        topo_feat = node_labeling(
            edge_index=sub_edge_index,
            node=mapping,
            num_nodes=sub_nodes.size(0)
        )
        sub_graph = Data(x=data.x[sub_nodes], topo_feat=topo_feat, edge_index=sub_edge_index, y=data.y[i])
        subgraphs.append(sub_graph)
    return subgraphs


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    num_graphs = 0
    for data in train_loader:
        num_graphs += data.num_graphs
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits, data.y)
        loss.backward()
        optimizer.step()
        correct = (pred == data.y).float().sum()
        total_correct += correct
        total_loss += loss.item() * data.num_graphs

    return total_loss / num_graphs, (total_correct / num_graphs).item()


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    total_correct = 0
    num_graphs = 0
    for data in loader:
        num_graphs += data.num_graphs
        data = data.to(device)
        logits = model(data)
        pred = logits.argmax(1)
        correct = (pred == data.y).float().sum()
        total_correct += correct
    return (total_correct / num_graphs).item()


def run():
    parser = argparse.ArgumentParser('Configurations for Node Classification')
    parser.add_argument('--dataset', default='Cora', type=str, help='dataset')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--weight_decaying', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='GPU')
    parser.add_argument('--epochs', default=201, type=int, help='epoch')
    parser.add_argument('--val_ratio', default=0.4, type=float, help='validation ratio')
    parser.add_argument('--test_ratio', default=0.4, type=float, help='test ratio')
    parser.add_argument('--patience', default=50, type=int, help='early stop patience')
    args = parser.parse_args()
    print(args)

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
        raise KeyError('Invalid dataset name!')

    print('=' * 50)
    print('Start extracting subgraphs~~~')
    subgraphs = extract_subgraphs(data)
    print('Done with extracting subgraphs.')
    print('=' * 50)
    if 'train_mask' in data.keys:
        train_graphs = [graph for graph, b in zip(subgraphs, data.train_mask) if b == True]
        val_graphs = [graph for graph, b in zip(subgraphs, data.val_mask) if b == True]
        test_graphs = [graph for graph, b in zip(subgraphs, data.test_mask) if b == True]
    else:
        num_val = int(np.ceil(data.num_nodes * args.val_ratio))
        num_test = int(np.ceil(data.num_nodes * args.test_ratio))
        random.shuffle(subgraphs)
        train_graphs = subgraphs[: (data.num_nodes - num_val - num_test)]
        val_graphs = subgraphs[(data.num_nodes - num_val - num_test): (data.num_nodes - num_test)]
        test_graphs = subgraphs[(data.num_nodes - num_test):]

    for data in chain(train_graphs, val_graphs, test_graphs):
        data.topo_feat = F.one_hot(data.topo_feat, 3).to(torch.float)

    train_loader = DataLoader(train_graphs, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.bs)
    test_loader = DataLoader(test_graphs, batch_size=args.bs)

    device = torch.device('cuda:0' if args.cuda else 'cpu')

    model = Net(
        topology_in_feats=train_graphs[0].topo_feat.shape[1],
        topology_h_feats=64,
        feature_in_feats=train_graphs[0].x.shape[1],
        feature_h_feats=128,
        out_feats=64,
        num_classes=dataset.num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    df = pd.DataFrame(columns=['epoch', 'train loss', 'train acc', 'val acc', 'test acc'])
    path = osp.join('../results/MultiView_NC', args.dataset, 'result.csv')
    df.to_csv(path, index=False)
    best_val_acc, test_acc = 0, 0
    patience = 0
    trainWriter = SummaryWriter('../{}/{}/{}/{}'.format('runs', 'MultiView_NC', args.dataset, 'Train'))
    valWriter = SummaryWriter('../{}/{}/{}/{}'.format('runs', 'MultiView_NC', args.dataset, 'Val'))
    for epoch in range(1, args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        val_acc = test(model, val_loader, device)
        trainWriter.add_scalar(tag='Train Acc', scalar_value=train_acc, global_step=epoch)
        valWriter.add_scalar(tag='Val Acc', scalar_value=val_acc, global_step=epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = test(model, test_loader, device)
            state = {'model': model.state_dict(), 'acc': test_acc, 'epoch': epoch}
            checkpoint_path = osp.join('../checkpoint/MultiView_NC', args.dataset + "_ckpt.pt")
            torch.save(state, checkpoint_path)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
        print(f"Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},"
              f"Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
        record_list = [epoch, train_loss, train_acc, val_acc, test_acc]
        data = pd.DataFrame([record_list])
        data.to_csv(path, mode='a', header=False, index=False)


if __name__ == '__main__':
    run()
