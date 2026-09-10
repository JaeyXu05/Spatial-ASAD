import numpy as np
import os
import h5py
import torch
import config as cfg
from AADdataset import sliding_window,sliding_window2
from sklearn.model_selection import train_test_split
from train_valid_and_test import pretrain_model, finetune_model, test_model
from csp import CSP
import numpy as np

def from_mat_to_tensor(raw_data):
    #transpose, the dimention of mat and numpy is contrary
    Transpose = np.transpose(raw_data)
    Nparray = np.array(Transpose)
    return Nparray

def normalize_channels(eeg_data):
    mean = np.mean(eeg_data, axis=(0, 1), keepdims=True)
    std = np.std(eeg_data, axis=(0, 1), keepdims=True) + 1e-8
    normalized = (eeg_data - mean) / std
    return normalized, mean, std

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

res = torch.zeros((cfg.sbnum,1))



for fine_and_test_sb in range(cfg.sbnum):

    fine_and_test_eeg=data[fine_and_test_sb][:]
    fine_and_test_label=label[fine_and_test_sb][:]
    
    fine_eeg, test_eeg, fine_label, test_label = train_test_split(
        fine_and_test_eeg,
        fine_and_test_label,
        test_size=0.75,
        stratify=fine_and_test_label, 
        random_state=cfg.torch_seed
    )

    csp = CSP()

    pretrain_eeg=np.concatenate([data[:fine_and_test_sb],data[fine_and_test_sb+1:]],axis=0)
    pretrain_label=np.concatenate([label[:fine_and_test_sb],label[fine_and_test_sb+1:]],axis=0)
    pretrain_eeg=pretrain_eeg.reshape(-1,pretrain_eeg.shape[2],pretrain_eeg.shape[3])
    pretrain_label=pretrain_label.reshape(-1,pretrain_label.shape[2])
    pretrain_eeg=sliding_window(pretrain_eeg,cfg.decision_window,cfg.decision_window//2)
    pretrain_label=sliding_window(pretrain_label,cfg.decision_window,cfg.decision_window//2)

    pretrain_label_trial = pretrain_label[:, 0]
    csp.fit(pretrain_eeg, pretrain_label_trial)
    #pretrain_csp = csp.transform(pretrain_eeg)
    #pretrain_eeg = np.concatenate([pretrain_eeg, pretrain_csp], axis=2)
    #pretrain_eeg, pretrain_mean, pretrain_std = normalize_channels(pretrain_eeg)
    pretrain_csp = csp.transform(pretrain_eeg)  
    pretrain_eeg, pretrain_mean, pretrain_std = normalize_channels(pretrain_csp)

    fine_eeg=sliding_window(fine_eeg,cfg.decision_window,0)
    fine_label=sliding_window(fine_label,cfg.decision_window,0)
    test_eeg=sliding_window(test_eeg,cfg.decision_window,0)
    test_label=sliding_window(test_label,cfg.decision_window,0)
    fine_label = fine_label[:, 0]
    fine_csp = csp.transform(fine_eeg)
    test_csp = csp.transform(test_eeg)

    # fine_eeg = np.concatenate([fine_eeg, fine_csp], axis=2)
    # test_eeg = np.concatenate([test_eeg, test_csp], axis=2)
    fine_eeg = (fine_csp - pretrain_mean) / pretrain_std
    test_eeg = (test_csp - pretrain_mean) / pretrain_std

    pretrain_model(pretrain_eeg, pretrain_label, fine_and_test_sb)
    finetune_model(fine_eeg, fine_label, fine_and_test_sb)
    res[fine_and_test_sb] = test_model(test_eeg, test_label, fine_and_test_sb)


for sb in range(cfg.sbnum):
    mean_acc = torch.mean(res[sb][:])
    print(f"Subject {sb}: mean accuracy = {mean_acc:.3f}%")

os.makedirs(cfg.result_dir, exist_ok=True)
save_path = f"{cfg.result_dir}/result_subject_adaptive.csv"
np.savetxt(save_path, res.numpy(), delimiter=',')


