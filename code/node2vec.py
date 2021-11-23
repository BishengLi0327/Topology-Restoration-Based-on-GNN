import numpy as np
import pickle

import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected

count_threshold = 10
edge_path = '../data/alarm_construct_graph/alarm_graph_edge_index_{}'.format(count_threshold)
f = open(edge_path + '.pkl', 'rb')
edge_index = pickle.load(f)
f.close()

edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).t().contiguous()
data = Data(edge_index=edge_index, num_nodes=41143)
data.edge_index = add_self_loops(to_undirected(remove_self_loops(data.edge_index)[0]))[0]

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = Node2Vec(data.edge_index, embedding_dim=32, walk_length=10, context_size=5, walks_per_node=10,
                 num_negative_samples=1, p=1, q=1, sparse=True, num_nodes=data.num_nodes).to(device)
loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


for epoch in range(1, 101):
    loss = train()
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")

print('Finished training.')

embedding = model().detach().cpu()
path = '../data/alarm_construct_graph/embedding_{}.pt'.format(count_threshold)
torch.save(embedding, f=path)
print(embedding.shape)
