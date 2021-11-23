import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
from torch_geometric.datasets import Planetoid


def load_cora_dataset():
    path = osp.join(osp.dirname(osp.relpath('data_process.py')), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, 'Cora')[0]
    # dataset.x = F.one_hot(dataset.y).to(torch.float)
    # 这里做了一个处理，不使用原本的特征，而使用节点标签的one-hot编码。
    return dataset


def load_huawei_data():
    # 读取数据并统计节点和边的数目。
    alarm_graph = np.load('../data/alarm_project_hitsz/preprocessed/G', allow_pickle=True)
    node_list = list(alarm_graph.nodes)
    edge_ori = list(alarm_graph.edges)
    edge_list = []
    for edge in edge_ori:
        if edge[0] != edge[1]:
            edge_list.append(edge)  # 这一步是为了保证没有self-loops

    print('There are', len(node_list), 'nodes in the Graph!')
    print('There are', len(edge_list), 'edges in the Graph!')

    # 查看图中所有告警的数量。
    alarm_names = []
    for ne_name in list(alarm_graph.nodes):
        for alarm in alarm_graph.nodes[ne_name].keys():
            if alarm != 'NE_TYPE' and alarm not in alarm_names:
                alarm_names.append(alarm)

    print('Total different alarms:', len(alarm_names))

    # 查看不同类型的节点数目。
    site_type = {}
    for node in list(alarm_graph.nodes):
        site_type[alarm_graph.nodes[node]['NE_TYPE']] = site_type.get(alarm_graph.nodes[node]['NE_TYPE'], 0) + 1
    # 可以看到的是：一共有三种节点，分别为'Router':507, 'Microwave':24515, 'NodeB':16121
    # 根据节点的不同类型构建labels。维度为41143 * 3。
    labels = np.zeros([len(node_list), 3])
    for i in range(len(alarm_graph.nodes)):
        if alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'NODEB':
            labels[i][0] = 1
        elif alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'MICROWAVE':
            labels[i][1] = 1
        elif alarm_graph.nodes[list(alarm_graph.nodes)[i]]['NE_TYPE'] == 'ROUTER':
            labels[i][2] = 1

    # 根据告警构建节点初始属性，这里构建的是baseline，简单的0、1向量表示。
    attribute_length = len(alarm_names)
    num_of_nodes = len(alarm_graph.nodes)
    attribute_one_hot = np.zeros([num_of_nodes, attribute_length])
    attribute_count = np.zeros([num_of_nodes, attribute_length])

    # # one-hot
    for i in range(len(alarm_graph.nodes)):
        for alarm in alarm_graph.nodes[list(alarm_graph.nodes)[i]].keys():
            if alarm != 'NE_TYPE':
                attribute_one_hot[i][alarm_names.index(alarm)] = 1

    # one-hot + count
    for i in range(len(alarm_graph.nodes)):
        for alarm, alarm_time in alarm_graph.nodes[list(alarm_graph.nodes)[i]].items():
            if alarm != 'NE_TYPE':
                attribute_count[i][alarm_names.index(alarm)] = len(alarm_time)

    print('The construction of original attribute is Done~~~')
    # print('The ratio of non-zero element in attribute matrix is',
    # attribute.shape[0] * attribute.shape[1] / attribute.sum(), '%')

    # 将边的信息编码成torch.long类型数据。
    new_edge_list = []
    for i in range(len(edge_list)):
        a = node_list.index(edge_list[i][0])
        b = node_list.index(edge_list[i][1])
        new_edge_list.append([a, b])
    # 其中边以列表的形式存在，数字化代表第i个节点，节点列表已构建。

    return node_list, new_edge_list, alarm_names, attribute_one_hot, attribute_count, labels
