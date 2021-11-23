"""
This code is an advanced version of link prediction. We first utilize the features to construct KNNGraph,
the two different GNN models are used to extract embedding from original graph and KNNGraph,
we use the contrastive learning methods to enhance the learning process and use the learnt embedding as
the additional features for link prediction.
"""
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from torch_geometric import seed_everything
from torch_geometric.nn import GAE, GCN
from torch_geometric.loader import DataLoader
from torch_geometric.utils import train_test_split_edges, add_self_loops, negative_sampling, coalesce
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.transforms import KNNGraph
from tensorboardX import SummaryWriter
import os
import time
from models import DGCNN
import argparse
import os.path as osp
import warnings
from utils import *
warnings.filterwarnings('ignore')


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
    # parser.add_argument('--knn_usage', default='add_feat', choices=['add_feat', 'concat_graph'])
    # parser.add_argument('--pretrain', action='store_true', help='whether to use pretrained features')
    parser.add_argument('--pretrained_epochs', default=201, type=int)
    parser.add_argument('--patience', default=50, type=int, help='early stop steps')
    args = parser.parse_args()
    print(args)

    # load dataset
    dataset = load_data(args.dataset)

    # construct KNN graph
    dataset.pos = dataset.x
    k = int(dataset.num_edges / dataset.num_nodes) + 1
    trans = KNNGraph(k, loop=False, force_undirected=True)
    knn_graph = trans(dataset.clone())
    dataset.pos, knn_graph.pos, knn_graph.x = None, None, None

    # train/val/test split
    data = train_test_split_edges(dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    data.edge_index = data.train_pos_edge_index

    pretrained_features, knn_emb = CL(args, knn_graph, data)
    data.pretrained_features, data.knn_emb = pretrained_features, knn_emb
    # data.knn_emb = knn_emb

    # extract subgraphs for each link
    train_graphs, val_graphs, test_graphs = extract_subgraphs(data, args.use_label, args.use_feat)
    # split_data = {'train_dataset': train_graphs, 'val_dataset': val_graphs, 'test_dataset': test_graphs}
    # torch.save(split_data, osp.join('../data/', args.dataset.title() + '_split_data.pt'))
    # split_data = torch.load(osp.join('../data/', args.dataset.title() + '_split_data.pt'))
    # train_graphs, val_graphs, test_graphs = \
    #     split_data['train_dataset'], split_data['val_dataset'], split_data['test_dataset']

    train_loader = DataLoader(train_graphs, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.bs, shuffle=False)

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    model = DGCNN(train_graphs, hidden_channels=32, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    best_val_auc = test_auc = test_ap = 0
    patience = 0

    for epoch in range(1, args.epochs):
        scheduler.step()
        loss = train(model, train_loader, device, optimizer, train_graphs)
        val_auc, val_ap = test(val_loader, model, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc, test_ap = test(test_loader, model, device)
            patience = 0
            # saving model parameters
            state = {'model': model.state_dict(), 'auc': test_auc, 'ap': test_ap, 'epoch': epoch}
            save_path = '../checkpoint/CL-KNN-SEAL/'
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

    return test_auc


if __name__ == '__main__':
    test_auc = []
    for _ in range(10):
        """
        Random Seed:
            Cora: 1
            Disease: 2
        """
        seed_everything(3)
        auc = run()
        test_auc.append(auc)
    print('Test AUC:', test_auc)
    print('the mean test auc is {:.4f}'.format(sum(test_auc) / 10))
