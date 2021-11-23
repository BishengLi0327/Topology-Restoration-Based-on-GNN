import math
import torch
from torch.nn import ModuleList, Conv1d, MaxPool1d, Linear
import torch.nn.functional as F
from torch_geometric.nn import global_sort_pool
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SGConv
from torch_geometric.nn import GATConv

import manifolds
from layers.hyp_layers import HNNLayer


class GCN(torch.nn.Module):

    def __init__(self, input_feat, output_feat):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_feat, 2 * output_feat)
        self.conv2 = GCNConv(2 * output_feat, output_feat)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GAT(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class SGC(torch.nn.Module):

    def __init__(self, input_feat, output_feat):
        super(SGC, self).__init__()
        self.conv = SGConv(input_feat, output_feat, K=2)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class DGCNN(torch.nn.Module):
    def __init__(self, train_dataset, hidden_channels, num_layers, GNN=GCNConv, k=0.6):
        super(DGCNN, self).__init__()

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.convs = ModuleList()
        self.convs.append(GNN(train_dataset[0].num_features, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

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
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class H_DGCNN(torch.nn.Module):
    def __init__(self, train_dataset, hidden_channels, num_layers, GNN=GCNConv, k=0.6):
        super(H_DGCNN, self).__init__()

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.convs = ModuleList()
        self.convs.append(GNN(train_dataset[0].num_features, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)
        self.manifold = manifolds.PoincareBall()
        self.hlin2 = HNNLayer(self.manifold, 128, 1, 1.0, 0.0, lambda x:x, use_bias=True)

    def forward(self, x, edge_index, batch):
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index))]
        x = torch.cat(xs[1:], dim=-1)
        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden] (32, 1, 970)
        x = F.relu(self.conv1(x)) # (32 16 10)
        x = self.maxpool1d(x) # (32 16 5)
        x = F.relu(self.conv2(x)) # (32, 32, 1)
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim] # (32 32)
        # MLP.
        x = F.relu(self.lin1(x)) # (32, 128)
        x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x) # (32, 1)
        x = self.manifold.expmap0(x, 1.0)
        x = self.manifold.proj(x, 1.0)
        x = self.hlin2(x)
        x = self.manifold.logmap0(x, 1.0)
        return x
