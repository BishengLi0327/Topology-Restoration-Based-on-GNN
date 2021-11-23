import torch.nn
import torch.nn.functional as F # noqa
import torch_geometric.loader

from torch_geometric.nn import GCNConv, GATConv, global_add_pool # noqa
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T # noqa
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data, DataLoader

import argparse
import os.path as osp
from tqdm import tqdm


class GCN(torch.nn.Module):
    def __init__(self, in_channels, h_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=in_channels, out_channels=h_channels, cached=True)
        self.conv2 = GCNConv(in_channels=h_channels, out_channels=out_channels, cached=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x=x, edge_index=edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels=in_channels, out_channels=8, heads=8, dropout=0.6)
        self.conv2 = GATConv(in_channels=8 * 8, out_channels=out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x=x, edge_index=edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index)
        return F.log_softmax(x, dim=1)


def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


parser = argparse.ArgumentParser('Experiment Configuration')
parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--model', default='GAT', type=str)
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decaying')
parser.add_argument('-bs', default=32, type=int)
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool)
parser.add_argument('--epochs', default=401, type=int)
parser.add_argument('--patience', default=100, type=int, help='early stop epochs')
args = parser.parse_args()
print(args)

if args.dataset == 'Cora':
    dataset = Planetoid(root='../data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
else:
    raise ValueError('Unsupported dataset')

device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
data = data.to(device)
if args.model == 'GCN':
    model = GCN(dataset.num_features, 128, dataset.num_classes).to(device)
elif args.model == 'GAT':
    model = GAT(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

best_val_acc = test_acc = 0
patience = 0
for epoch in range(1, args.epochs):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        state = model.state_dict()
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        print('Early Stop!!! Best Val Acc: {:.4f}, Test Acc: {:.4f}'.format(best_val_acc, test_acc))
        break
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

torch.save(state, osp.join('../checkpoint', args.model + '_ckpt.pt'))
print('Finished Training')


def extract_subgraphs(data):
    subgraphs = []
    for i in tqdm(range(data.num_nodes)):
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            i, num_hops=2, edge_index=data.edge_index, relabel_nodes=True
        )
        sub_graph = Data(x=data.x[sub_nodes], edge_index=sub_edge_index, y=data.y[i])
        subgraphs.append(sub_graph)
    return subgraphs


print('=' * 50)
print('Start extracting subgraphs~~~')
subgraphs = extract_subgraphs(data)
print('Done with extracting subgraphs.')
print('=' * 50)

train_graphs = [graph for graph, b in zip(subgraphs, data.train_mask) if b == True]
val_graphs = [graph for graph, b in zip(subgraphs, data.val_mask) if b == True]
test_graphs = [graph for graph, b in zip(subgraphs, data.test_mask) if b == True]

train_loader = DataLoader(train_graphs, batch_size=args.bs, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=args.bs)
test_loader = DataLoader(test_graphs, batch_size=args.bs)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GATConv(in_channels=in_channels, out_channels=8, heads=8, dropout=0.6)
        self.conv2 = GATConv(in_channels=8 * 8, out_channels=out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x=x, edge_index=edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index)
        x = global_add_pool(x, batch)
        return F.log_softmax(x, dim=1)


model = Net(dataset.num_features, dataset.num_classes).to(device)
model.load_state_dict(torch.load(osp.join('../checkpoint', args.model + '_ckpt.pt')))
# All keys matched successfully


@torch.no_grad()
def evaluate(model, loader):
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


val_acc = evaluate(model, val_loader)
test_acc = evaluate(model, test_loader)
print('Val Acc: {:.4f}, Test Acc: {:.4f}'.format(val_acc, test_acc))
