import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, ndcg_score
from tensorboardX import SummaryWriter
import os
from models import DGCNN, H_DGCNN
import argparse
import os.path as osp
from seal_dataset import process
import warnings
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
           average_precision_score(torch.cat(y_true), torch.cat(y_pred)), \
           recall_score(torch.cat(y_true), torch.where(torch.cat(y_pred) > torch.cat(y_pred).median(), 1, 0)), \
           ndcg_score(torch.cat(y_true).unsqueeze(dim=0), torch.cat(y_pred).unsqueeze(dim=0))


def run():
    parser = argparse.ArgumentParser("Configurations for seal")
    parser.add_argument('--dataset', default='huawei', type=str, help='dataset')
    parser.add_argument('--embedding', default='DRNL', type=str,
                        help='node encoding(["DRNL", "DRNL_SelfFeat", "SelfFeat"])')
    # parser.add_argument('--runs', default=10, type=int, help='training runs')
    parser.add_argument('--epochs', default=101, type=int, help='training epochs')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='cuda')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--val_ratio', default=0.05, type=float)
    parser.add_argument('--test_ratio', default=0.10, type=float)
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrained_epochs', default=1001, type=int)
    parser.add_argument('--pre_encoder', default='GCN', type=str, choices=['GCN'])
    parser.add_argument('--patience', default=50, type=int, help='early stop steps')
    parser.add_argument('--use_alarm', action='store_true')
    args = parser.parse_args()
    print(args)

    args.split_ratio = str(int((1-args.val_ratio-args.test_ratio)*100)) \
                       + str(int(args.val_ratio*100)) + str(int(args.test_ratio*100))

    train_dataset, val_dataset, test_dataset = process(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    model = H_DGCNN(train_dataset, hidden_channels=32, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # trainWriter = SummaryWriter('../{}/{}/{}/{}/{}'.format(
    #     'runs', 'SEAL', args.dataset + '_' + args.split_ratio + '_pretrained_' + str(args.pretrain) + '_use_alarm_' + str(args.use_alarm),
    #     args.embedding, 'Train'
    # ))
    # valWriter = SummaryWriter('../{}/{}/{}/{}/{}'.format(
    #     'runs', 'SEAL', args.dataset + '_' + args.split_ratio + '_pretrained_' + str(args.pretrain) + '_use_alarm_' + str(args.use_alarm),
    #     args.embedding, 'Val'
    # ))

    best_val_auc = test_auc = test_ap = test_recall = test_ndcg = 0

    for epoch in range(1, args.epochs):
        loss = train(model, train_loader, device, optimizer, train_dataset)
        # trainWriter.add_scalar(tag='Train Loss', scalar_value=loss, global_step=epoch)
        val_auc, val_ap, val_recall, val_ndcg = test(val_loader, model, device)
        # valWriter.add_scalar(tag='Val AUC', scalar_value=val_auc, global_step=epoch)
        # valWriter.add_scalar(tag='Val AP', scalar_value=val_ap, global_step=epoch)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc, test_ap, test_recall, test_ndcg = test(test_loader, model, device)
            # saving model parameters
            state = {'model': model.state_dict(), 'auc': test_auc, 'ap': test_ap, 'epoch': epoch}
            save_path = '../checkpoint/SEAL/'
            if not osp.exists(save_path):
                os.mkdir(save_path)
            torch.save(state, osp.join(save_path, args.dataset+'-'+args.split_ratio+'-'+args.embedding+'-'+'ckpt.pth'))

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val_AUC: {val_auc:.4f}, Val_AP: {val_ap:.4f},'
              f'Test_AUC: {test_auc:.4f}, Test_AP: {test_ap:.4f}, Test_Recall: {test_recall:.4f}, Test_NDCG: {test_ndcg:.4f}')


if __name__ == "__main__":
    run()
