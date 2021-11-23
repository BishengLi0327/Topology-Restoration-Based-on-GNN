import numpy as np
import random
import os

from sklearn.metrics import roc_auc_score

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, train_test_split_edges, negative_sampling
from tensorboardX import SummaryWriter
import argparse

from data_process import load_huawei_data
from models import *

node_list, edge_list, new_edge_list, alarm_names, attribute_one_hot, attribute_count = load_huawei_data()


def compute_prob(edge, embedding):
    in_edge, out_edge = [], []
    for edge in edge:
        in_edge.append(edge[0])
        out_edge.append(edge[1])

    pred_targets = torch.sigmoid(torch.sum(embedding[in_edge] * embedding[out_edge], axis=1))
    return pred_targets


# train
def train(data, model, optimizer, train_edges, train_y):
    model.train()
    optimizer.zero_grad()
    embedding = model(data.x, data.edge_index)
    pred_targets = compute_prob(train_edges, embedding)
    loss = F.binary_cross_entropy(
        pred_targets, torch.tensor(train_y, dtype=torch.float)
    )
    loss.backward()
    optimizer.step()
    auc = roc_auc_score(np.array(train_y), pred_targets.detach().numpy())
    return loss, auc


# test
def test(data, model, test_edges, test_y):
    model.eval()
    embedding = model(data.x, data.edge_index)
    pred_targets = compute_prob(test_edges, embedding)
    loss = F.binary_cross_entropy(
        pred_targets, torch.tensor(test_y, dtype=torch.float)
    )
    auc = roc_auc_score(np.array(test_y), pred_targets.detach().numpy())
    return auc


def run():
    parser = argparse.ArgumentParser(description='Link Prediction Baselines')
    parser.add_argument('--model', default='GCN', type=str, help='which model to use')
    parser.add_argument('--original_embedding', default='one-hot', type=str,
                        help='what kinds of embedding to use')
    parser.add_argument('--epochs', default=401, type=int, help='number of epochs to run')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    args = parser.parse_args()
    print(args)

    # 模型及优化器
    if args.model == 'GCN':
        model = GCN(len(alarm_names), 32)
    elif args.model == 'GAT':
        model = GAT(len(alarm_names), 32)
    elif args.model == 'SGC':
        model = SGC(len(alarm_names), 32)
    else:
        raise ValueError('Invalid model type~~~')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainWriter = SummaryWriter('../{}/{}/{}/{}'.
                                format('runs', args.model, args.original_embedding, 'Train'))
    valWriter = SummaryWriter('../{}/{}/{}/{}'.
                               format('runs', args.model, args.original_embedding, 'Val'))

    dataset = Data(edge_index=torch.tensor(new_edge_list, dtype=torch.long).t().contiguous())
    dataset.num_nodes = len(node_list)
    data = train_test_split_edges(dataset)
    # data里含有train_pos_edge_index, train_neg_adj_mask, val_pos_edge_index
    # val_neg_edge_index, test_pos_edge_index, test_neg_edge_index
    edge_index, _ = add_self_loops(data.train_pos_edge_index)  # 这个是用于训练构图的edge_index，已经去除了test和val

    # 负采样作为训练时的负样本
    data.train_neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )

    # 构建数据
    if args.original_embedding == 'one-hot':
        x = torch.tensor(attribute_one_hot, dtype=torch.float)  # One-hot encoding
    elif args.original_embedding == 'count':
        x = torch.tensor(attribute_count, dtype=torch.float)  # count encoding
    elif args.original_embedding == 'random':
        x = torch.randn([len(node_list), len(alarm_names)])  # random encoding
    # y = torch.tensor(labels, dtype=torch.long)
    # data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    train_graph = Data(x=x, edge_index=edge_index)

    train_edges = data.train_pos_edge_index.t().contiguous().tolist() + \
                  data.train_neg_edge_index.t().contiguous().tolist()
    train_y = [1] * data.train_pos_edge_index.size(1) + [0] * data.train_neg_edge_index.size(1)
    train_data = list(zip(train_edges, train_y))
    random.shuffle(train_data)
    train_edges[:], train_y[:] = zip(*train_data)

    val_edges = data.val_pos_edge_index.t().contiguous().tolist() + \
                 data.val_neg_edge_index.t().contiguous().tolist()
    val_y = [1] * data.val_pos_edge_index.size(1) + [0] * data.val_neg_edge_index.size(1)
    val_data = list(zip(val_edges, val_y))
    random.shuffle(val_data)
    val_edges[:], val_y[:] = zip(*val_data)

    test_edges = data.test_pos_edge_index.t().contiguous().tolist() + \
                 data.test_neg_edge_index.t().contiguous().tolist()
    test_y = [1] * data.test_pos_edge_index.size(1) + [0] * data.test_neg_edge_index.size(1)
    test_data = list(zip(test_edges, test_y))
    random.shuffle(test_data)
    test_data[:], test_y[:] = zip(*test_data)

    best_val_auc = test_auc = 0
    for epoch in range(1, args.epochs):
        training_loss, training_auc = train(train_graph, model, optimizer, train_edges, train_y)
        trainWriter.add_scalar(tag='Training Loss', scalar_value=training_loss, global_step=epoch)

        val_auc = test(train_graph, model, val_edges, val_y)
        valWriter.add_scalar(tag='Val AUC', scalar_value=val_auc, global_step=epoch)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = test(train_graph, model, test_edges, test_y)
            state = {'model': model.state_dict(), 'auc': test_auc, 'epoch': epoch}
            if not os.path.isdir('../checkpoint'):
                os.mkdir('../checkpoint')
            torch.save(state, '../checkpoint/'+str(args.model)+'_'+str(args.original_embedding)+'_'+'ckpt.pth')

        print(f'Epoch: {epoch}, Training Loss: {training_loss:4f}, Val AUC: {val_auc:4f}, Test AUC: {test_auc:4f}')


if __name__ == '__main__':
    run()
