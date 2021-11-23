import os
import os.path as osp

import argparse

import torch
from torch_geometric.nn import GAE
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges, negative_sampling, add_self_loops
from tensorboardX import SummaryWriter

from models import *
from data_process import load_huawei_data, load_cora_dataset


def load_data(args):
    if args.dataset == 'huawei':
        node_list, edge_list, alarm_names, attribute_one_hot, attribute_count, labels = load_huawei_data()
        if args.embedding == 'one-hot':
            dataset = Data(x=torch.tensor(attribute_one_hot, dtype=torch.float),
                           edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous())
        elif args.embedding == 'count':
            dataset = Data(x=torch.tensor(attribute_count, dtype=torch.float),
                           edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous())
        elif args.embedding == 'random':
            dataset = Data(x=torch.randn([len(node_list), len(alarm_names)]),
                           edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous())
        elif args.embedding == 'node2vec':
            alarm_feature_path = '../data/alarm_construct_graph/embedding.pt'
            alarm_features = torch.load(alarm_feature_path)
            dataset = Data(x=alarm_features, edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous())
        else:
            raise ValueError('Invalid embedding type!')
    elif args.dataset == 'cora':
        dataset = load_cora_dataset()
    else:
        raise ValueError('Invalid dataset type!')

    return dataset


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index, data.train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss), z


def test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def run():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--variational', action='store_true')
    # parser.add_argument('--linear', action='store_true')
    parser.add_argument('--dataset', default='huawei', type=str, choices=['huawei', 'disease', 'cora'])
    parser.add_argument('--embedding', default='node2vec', type=str, choices=['one-hot', 'count', 'random', 'node2vec'])
    parser.add_argument('--encoder', default='GAT', type=str, choices=['GCN', 'SGC', 'GAT'])
    parser.add_argument('--epochs', default=401, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool)
    parser.add_argument('--val_ratio', default=0.05, type=float)
    parser.add_argument('--test_ratio', default=0.1, type=float)
    args = parser.parse_args()
    args.split_ratio = str(int((1-args.val_ratio-args.test_ratio)*100)) \
                       + str(int(args.val_ratio*100)) + str(int(args.test_ratio*100))
    print(args)

    dataset = load_data(args)
    data = train_test_split_edges(dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    if args.encoder == 'GCN':
        model = GAE(GCN(dataset.num_features, 32))
    elif args.encoder == 'SGC':
        model = GAE(SGC(dataset.num_features, 32))
    elif args.encoder == 'GAT':
        model = GAE(GAT(dataset.num_features, 32))
    else:
        raise ValueError('Invalid model type!')

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = test_auc = test_ap = 0
    trainWriter = SummaryWriter('../{}/{}/{}/{}/{}'.
                                format('runs', 'GAE', args.dataset.capitalize()+'_'+args.encoder+args.split_ratio,
                                       args.embedding, 'Train'))
    valWriter = SummaryWriter('../{}/{}/{}/{}/{}'.
                              format('runs', 'GAE', args.dataset.capitalize()+'_'+args.encoder+args.split_ratio,
                                     args.embedding, 'Val'))
    for epoch in range(1, args.epochs):
        train_loss, _ = train(model, data, optimizer)
        trainWriter.add_scalar(tag='Train Loss', scalar_value=train_loss, global_step=epoch)
        val_auc, val_ap = test(model, data.x, data.train_pos_edge_index,
                               data.val_pos_edge_index, data.val_neg_edge_index)
        valWriter.add_scalar(tag='Val AUC', scalar_value=val_auc, global_step=epoch)
        valWriter.add_scalar(tag='Val AP', scalar_value=val_ap, global_step=epoch)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc, test_ap = test(model, data.x, data.train_pos_edge_index,
                                     data.test_pos_edge_index, data.test_neg_edge_index)
            state = {'model': model.state_dict(), 'auc': test_auc, 'epoch': epoch}
            if not osp.isdir(osp.join('../checkpoint/GAE/')):
                os.mkdir(osp.join('../checkpoint/GAE/'))
            torch.save(state, '../checkpoint/GAE/' + args.encoder + '_' + args.embedding + '_'+args.split_ratio+'_' + 'ckpt.pth')

        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val_AUC: {val_auc:.4f}, '
              f'Val_AP: {val_ap:.4f}, Test_AUC: {test_auc:.4f}, Test_AP: {test_ap:.4f}')


if __name__ == '__main__':
    run()
