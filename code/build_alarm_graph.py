import numpy as np
import pickle
from tqdm import tqdm
from typing import List

# 读取数据并统计节点和边的数目
alarm_graph = np.load('../data/alarm_project_hitsz/preprocessed/G', allow_pickle=True)

# 首先将每一个设备（节点）上的告警时间转化为一个list的形式
for node in tqdm(list(alarm_graph.nodes), desc='Converting alarm time to list: '):
    alarm_time = []
    del alarm_graph.nodes[node]['NE_TYPE']
    for key in alarm_graph.nodes[node].keys():
        alarm_time += alarm_graph.nodes[node][key]
    alarm_graph.nodes[node]['alarm_time'] = alarm_time
    alarm_key = []
    for key in alarm_graph.nodes[node].keys():
        if key != 'alarm_time':
            alarm_key.append(key)
    for key in alarm_key:
        del alarm_graph.nodes[node][key]


def count_close_time(t1: List[int], t2: List[int], t_interval: int) -> int:
    count = 0
    if not t1 and not t2:
        return 0
    for t_a in t1:
        for t_b in t2:
            if abs(t_a - t_b) <= t_interval:
                count += 1
    return count


# edge_dict = {}  # 用来保存可能合理的边以及他们连接的次数
# t_interval = 3600  # 表示时间间隔

# num_processed = 0
# for node_A in tqdm(list(alarm_graph.nodes)):
#     for node_B in list(alarm_graph.nodes):
#         t1 = alarm_graph.nodes[node_A]['alarm_time']
#         t2 = alarm_graph.nodes[node_B]['alarm_time']
#         count = count_close_time(t1, t2, t_interval)
#         if count != 0:
#             num_processed += 1
#             edge_dict[(node_A, node_B)] = count
# print(num_processed)


key = 0
id_to_key = {}
num_nodes = len(alarm_graph.nodes)
edge_dict = [0] * int(num_nodes ** 2)
time_intervals = []
for node in tqdm(list(alarm_graph.nodes), desc='Mapping string to integer:'):
    id_to_key[node] = key
    t1 = alarm_graph.nodes[node]['alarm_time']
    for t in t1:
        time_intervals.append([t, key])
    key += 1

# with open("id_to_key.pkl", 'wb') as f:
#     pickle.dump(id_to_key, f)
#
# with open('time_intervals.pkl', 'wb') as f:
#     pickle.dump(time_intervals, f)

# f = open('time_intervals.pkl', 'rb')
# time_intervals = pickle.load(f)
# f.close()
time_intervals.sort(key=lambda x: x[0])
print(time_intervals[0])
n = len(time_intervals)
t_interval = 300
print(n, t_interval)


def binary_search(l, r, s):
    while l < r:
        mid = (l + r + 1) >> 1
        if time_intervals[mid][0] <= s:
            l = mid
        else:
            r = mid - 1
    return l


def divide_conquer(l, r):
    n = r - l + 1
    num_processed = 0
    if n == 2:
        if time_intervals[r][0] - time_intervals[l][0] <= t_interval:
            node_A = time_intervals[r][1]
            node_B = time_intervals[l][1]
            edge_dict[node_A * num_nodes + node_B] += 1
            return 1
    elif n <= 1:
        return 0
    else:
        mid = (l + r) >> 1
        left_processed = divide_conquer(l, mid)
        right_processed = divide_conquer(mid + 1, r)
        ll = binary_search(l, mid, time_intervals[mid + 1][0] - t_interval)
        rr = binary_search(mid + 1, r, time_intervals[mid][0] + t_interval)
        j = mid + 1
        for i in range(ll, mid + 1):
            j = binary_search(j, rr, time_intervals[i][0] + t_interval)
            node_B = time_intervals[i][1]
            for k in (mid, j):
                node_A = time_intervals[k][1]
                edge_dict[node_A * num_nodes + node_B] += 1
                num_processed += 1
        return num_processed + left_processed + right_processed


# ans = 0
# ans2 = 0
# for i in range(1, len(time_intervals)):
#     if time_intervals[i][0] == time_intervals[i - 1][0]:
#         ans += 1
#     else:
#         ans2 += ans * ans
#         ans = 0
# print(ans2)

# print(divide_conquer(0, n - 1))

num_processed = 0
j = 1
for i in tqdm(range(n)):
    j = binary_search(j, n - 1, time_intervals[i][0] + t_interval)
    # while j < n and time_intervals[j][0] - time_intervals[i][0] <= t_interval:
    #     j += 1
    node_B = time_intervals[i][1]
    for k in range(i, j):
        node_A = time_intervals[k][1]
        edge_dict[node_A * num_nodes + node_B] += 1
        num_processed += 1
    # print(num_processed)
print(num_processed)

# with open('edge.pkl', 'wb') as f:
#     pickle.dump(edge_dict, f)
