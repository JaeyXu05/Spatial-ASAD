import numpy as np
import os
import h5py
import torch
import config as cfg
from train_valid_and_test import train_valid_model,test_model
from sklearn.model_selection import train_test_split
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

res = torch.zeros((cfg.sbnum,cfg.trnum))


from sklearn.model_selection import KFold,train_test_split
kfold = KFold(n_splits=cfg.kfold_num, shuffle=True, random_state=cfg.torch_seed)#kfold 对象包含的是如何把数据分成5份的规则

for sb in range(cfg.sbnum):
    # get the data of specific subject
    eegdata = data[sb]
    eeglabel = label[sb]

    for tr in range(cfg.trnum):
        test_eegdata = eegdata[tr]
        test_eegdata = sliding_window2(test_eegdata, cfg.decision_window, cfg.decision_window//2)
        test_eeglabel = eeglabel[tr]
        test_eeglabel = sliding_window2(test_eeglabel, cfg.decision_window, cfg.decision_window//2)
        
        #train
        train_eegdata = np.concatenate([eegdata[:tr], eegdata[tr+1:]], axis=0)##拼接
        train_eeglabel = np.concatenate([eeglabel[:tr], eeglabel[tr+1:]], axis=0)
        train_eegdata = sliding_window(train_eegdata,cfg.decision_window,cfg.decision_window//2)
        train_eeglabel = sliding_window(train_eeglabel,cfg.decision_window,cfg.decision_window//2)

        train_valid_model(train_eegdata, train_eeglabel, sb, test_id=tr)#训练
        res[sb,tr] = test_model(test_eegdata, test_eeglabel, sb,test_id=tr)

for sb in range(cfg.sbnum):
    mean_acc = torch.mean(res[sb][:])
    print(f"Subject {sb}: mean accuracy = {mean_acc:.3f}%")

os.makedirs(cfg.result_dir, exist_ok=True)
save_path = f"{cfg.result_dir}/result_loto.csv"
np.savetxt(save_path, res.numpy(), delimiter=',')

