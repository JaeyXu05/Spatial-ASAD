import numpy as np
import os
import h5py
import torch
import config as cfg
from train_valid_and_test import train_valid_model,test_model
from AADdataset import sliding_window,sliding_window2


def from_mat_to_tensor(raw_data):
    #transpose, the dimention of mat and numpy is contrary
    Transpose = np.transpose(raw_data)
    Nparray = np.array(Transpose)
    return Nparray

# all the number of sbjects in the experiment
# train one model for every subject

# read the data
eegname = cfg.process_data_dir + '/' +  cfg.dataset_name
eegdata = h5py.File(eegname, 'r')# h5py 读取出来的数组维度是 转置 的
data = from_mat_to_tensor(eegdata['EEG'])  
label = from_mat_to_tensor(eegdata['ENV'])  # 0 or 1, representing the attended direction  

# random seed
torch.manual_seed(cfg.torch_seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(cfg.torch_seed)

res = torch.zeros((cfg.sbnum, 1))
for test_sb in range(cfg.sbnum):
    train_eeg_list = []
    train_label_list = []
    for train_sb in range(cfg.sbnum):
        if train_sb == test_sb:
            continue  # 跳过测试的subject
        eegdata_sb = data[train_sb]  # shape: (num_trials, time, channels)
        eeglabel_sb = label[train_sb]  # shape: (num_trials, time)
        eegdata_sb_windowed = sliding_window(eegdata_sb, cfg.decision_window,cfg.decision_window//2)
        eeglabel_sb_windowed = sliding_window(eeglabel_sb, cfg.decision_window, cfg.decision_window//2)
        train_eeg_list.append(eegdata_sb_windowed)
        train_label_list.append(eeglabel_sb_windowed)
    train_eegdata = np.concatenate(train_eeg_list, axis=0)
    train_eeglabel = np.concatenate(train_label_list, axis=0)
    
    # 测试数据
    test_eegdata_list = []
    test_eeglabel_list = []
    for tr in range(cfg.trnum):
        eegdata_sb = data[test_sb][tr]
        eeglabel_sb = label[test_sb][tr]
        test_eegdata = sliding_window2(eegdata_sb, cfg.decision_window, cfg.decision_window//2)
        test_eeglabel = sliding_window2(eeglabel_sb, cfg.decision_window, cfg.decision_window//2)
        test_eegdata_list.append(test_eegdata)
        test_eeglabel_list.append(test_eeglabel)
    test_eegdata = np.concatenate(test_eegdata_list, axis=0)
    test_eeglabel = np.concatenate(test_eeglabel_list, axis=0)
    train_valid_model(train_eegdata, train_eeglabel, test_id=test_sb)
    res[test_sb, 0] = test_model(test_eegdata, test_eeglabel,  test_id=test_sb)

for sb in range(cfg.sbnum):
    print(f"Subject {sb}: {res[sb, 0].item():.3f}%")

print(f"\nAverage Accuracy: {torch.mean(res).item():.3f}%")
os.makedirs(cfg.result_dir, exist_ok=True)
save_path = f"{cfg.result_dir}/result_loso.csv"
np.savetxt(save_path, res.numpy(), delimiter=',')