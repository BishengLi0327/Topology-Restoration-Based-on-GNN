import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
#
# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot auc curve for parallel model

# auc_data_1 = pd.read_csv('../results/parallel_results_disease_drnl.csv', delimiter=' ')
# auc_data_2 = pd.read_csv('../results/parallel_results_disease_drnl_nl.csv', delimiter=' ')
# auc_data_3 = pd.read_csv('../results/parallel_results_disease_nl.csv', delimiter=' ')
#
# plt.figure()
# plt.plot(auc_data_1['val_auc'], label='DRNL')
# plt.plot(auc_data_2['val_auc'], label='DRNL+NL')
# plt.plot(auc_data_3['val_auc'], label='NL')
# plt.xlabel('epoch')
# plt.ylabel('AUC')
# # plt.title('AUC curve for parallel model')
# plt.grid(True, linestyle='--')
# plt.legend()
# plt.show()
#
# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HGCN for HUAWEI
# ea = event_accumulator.EventAccumulator(
#     '/root/libisheng/HUAWEI/code/hgcn_modify/runs/HGCN/huawei/count/Val/events.out.tfevents.1628823242.0d09caa0b086'
# )
# ea.Reload()
#
# val_auc = ea.scalars.Items('Val_AUC')
# # val_ap = ea.scalars.Items('Val_AP')
# auc = [i.value for i in val_auc]
# # ap = [i.value for i in val_ap]
#
# plt.figure()
# plt.plot(auc)
# plt.xlabel('step')
# plt.ylabel('Val AUC')
# plt.grid(True, linestyle='--')
# plt.show()

# plt.figure()
# plt.plot(ap)
# plt.xlabel('step')
# plt.ylabel('Val AP')
# plt.grid(True, linestyle='--')
# plt.show()

# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # 加载tensorboard中的数据
# ea = event_accumulator.EventAccumulator(
#     '/root/libisheng/HUAWEI/runs/PARALLEL/disease/DRNL/Val/events.out.tfevents.1628652047.0d09caa0b086'
# )
# ea.Reload()
# val_auc = ea.scalars.Items('Val_AUC')
# auc = [i.value for i in val_auc]
# plt.figure()
# plt.plot(auc)
# plt.xlabel('epoch')
# plt.ylabel('Val AUC')
# plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# GAE(GCN/SGC/GAT)画图
# ea_one_hot = event_accumulator.EventAccumulator(
#     '/root/libisheng/HUAWEI/runs/GAE/Huawei_GAT/one-hot/Val/events.out.tfevents.1627720157.0d09caa0b086'
# )
# ea_count = event_accumulator.EventAccumulator(
#     '/root/libisheng/HUAWEI/runs/GAE/Huawei_GAT/count/Val/events.out.tfevents.1627720150.0d09caa0b086'
# )
# ea_random = event_accumulator.EventAccumulator(
#     '/root/libisheng/HUAWEI/runs/GAE/Huawei_GAT/random/Val/events.out.tfevents.1627718874.0d09caa0b086'
# )
# ea_one_hot.Reload()
# ea_count.Reload()
# ea_random.Reload()
#
# val_auc_one_hot = ea_one_hot.scalars.Items('Val_AUC')
# val_ap_one_hot = ea_one_hot.scalars.Items('Val_AP')
# val_auc_count = ea_count.scalars.Items('Val_AUC')
# val_ap_count = ea_count.scalars.Items('Val_AP')
# val_auc_random = ea_random.scalars.Items('Val_AUC')
# val_ap_random = ea_random.scalars.Items('Val_AP')
#
# auc_one_hot = [i.value for i in val_auc_one_hot]
# ap_one_hot = [i.value for i in val_ap_one_hot]
# auc_count = [i.value for i in val_auc_count]
# ap_count = [i.value for i in val_ap_count]
# auc_random = [i.value for i in val_auc_random]
# ap_random = [i.value for i in val_ap_random]
#
# plt.figure()
# plt.plot(auc_one_hot, label='one-hot')
# plt.plot(auc_count, label='count')
# plt.plot(auc_random, label='random')
# plt.xlabel('epoch')
# plt.ylabel('Val AUC')
# plt.legend()
# plt.grid(True, linestyle='--')
# plt.show()
#
# plt.figure()
# plt.plot(ap_one_hot, label='one-hot')
# plt.plot(ap_count, label='count')
# plt.plot(ap_random, label='random')
# plt.xlabel('epoch')
# plt.ylabel('Val AP')
# plt.legend()
# plt.grid(True, linestyle='--')
# plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # SEAL画图
ea_drnl = event_accumulator.EventAccumulator(
    '/root/libisheng/HUAWEI/runs/SEAL/huawei_85510_pretrained_True_use_alarm_True/DRNL/Val/events.out.tfevents.1633434090.0d09caa0b086'
)
ea_drnl_nl = event_accumulator.EventAccumulator(
    '/root/libisheng/HUAWEI/runs/SEAL/huawei_85510_pretrained_True_use_alarm_True/DRNL_SelfFeat/Val/events.out.tfevents.1633443274.0d09caa0b086'
)
ea_nl = event_accumulator.EventAccumulator(
    '/root/libisheng/HUAWEI/runs/SEAL/huawei_85510_pretrained_True_use_alarm_True/SelfFeat/Val/events.out.tfevents.1633443406.0d09caa0b086'
)
ea_drnl.Reload()
ea_drnl_nl.Reload()
ea_nl.Reload()

val_auc_drnl = ea_drnl.scalars.Items('Val_AUC')
val_auc_drnl_nl = ea_drnl_nl.scalars.Items('Val_AUC')
val_auc_nl = ea_nl.scalars.Items('Val_AUC')

val_ap_drnl = ea_drnl.scalars.Items('Val_AP')
val_ap_drnl_nl = ea_drnl_nl.scalars.Items('Val_AP')
val_ap_nl = ea_nl.scalars.Items('Val_AP')

auc_drnl = [i.value for i in val_auc_drnl]
auc_drnl_nl = [i.value for i in val_auc_drnl_nl]
auc_nl = [i.value for i in val_auc_nl]

ap_drnl = [i.value for i in val_ap_drnl]
ap_drnl_nl = [i.value for i in val_ap_drnl_nl]
ap_nl = [i.value for i in val_ap_nl]

plt.figure()
plt.plot(auc_drnl, label='DRNL')
plt.plot(auc_drnl_nl, label='DRNL+NL')
plt.plot(auc_nl, label='NL')
plt.xlabel('epoch')
plt.ylabel('Val AUC')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()

plt.figure()
plt.plot(ap_drnl, label='DRNL')
plt.plot(ap_drnl_nl, label='DRNL+NL')
plt.plot(ap_nl, label='NL')
plt.xlabel('epoch')
plt.ylabel('Val AP')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()
