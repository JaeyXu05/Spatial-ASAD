import numpy as np
import os
import h5py
import torch
import config as cfg
from train_valid_and_test import train_valid_model, test_model
from AADdataset import sliding_window2
from sklearn.model_selection import KFold,train_test_split

def from_mat_to_tensor(raw_data):
    #transpose, the dimention of mat and numpy is contrary
    Transpose = np.transpose(raw_data)
    Nparray = np.array(Transpose)
    return Nparray

# read the data
eegname = cfg.process_data_dir + '/' + cfg.dataset_name
eegdata = h5py.File(eegname, 'r')
data = from_mat_to_tensor(eegdata['EEG'])  
label = from_mat_to_tensor(eegdata['ENV'])

# random seed
torch.manual_seed(cfg.torch_seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(cfg.torch_seed)

# 存储结果: (subject, fold)
res = torch.zeros((cfg.sbnum, cfg.kfold_num))

for sb in range(cfg.sbnum):
    print(f"\n{'='*60}")
    print(f"Processing Subject {sb}")
    print(f"{'='*60}")
    
    # 收集这个subject的所有trial数据
    all_trials_eeg = []
    all_trials_label = []
    for tr in range(cfg.trnum):
        eeg_trial = data[sb][tr]  # shape: (time, channels)
        label_trial = label[sb][tr]  # shape: (time,)
        all_trials_eeg.append(eeg_trial)
        all_trials_label.append(label_trial)
    
    combined_eeg = np.concatenate(all_trials_eeg, axis=0)  # (total_time, channels)
    combined_label = np.concatenate(all_trials_label, axis=0)  # (total_time,)
    windows_eeg = sliding_window2(combined_eeg, cfg.decision_window, cfg.decision_window // 2)
    windows_label = sliding_window2(combined_label, cfg.decision_window, cfg.decision_window // 2)

    kfold = KFold(n_splits=cfg.kfold_num, shuffle=True, random_state=cfg.torch_seed)
    fold_accuracies = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(windows_eeg)):
        train_eeg_w = windows_eeg[train_idx]
        train_label_w = windows_label[train_idx]
        test_eeg_w = windows_eeg[test_idx]
        test_label_w = windows_label[test_idx]

        if len(train_eeg_w) == 0 or len(test_eeg_w) == 0:
            print(f"    Warning: Fold {fold_idx} has insufficient windows")
            fold_accuracies.append(float('nan'))
            continue

        best_model = train_valid_model(
            train_eeg_w, train_label_w, sb, 
            test_id=fold_idx, 
        )

        acc = test_model(
            test_eeg_w, test_label_w, sb,
            test_id=fold_idx, 
        )
        fold_accuracies.append(acc)
        res[sb, fold_idx] = acc

    valid_acc = [acc for acc in fold_accuracies if not np.isnan(acc)]
    if len(valid_acc) > 0:
        mean_acc = np.mean(valid_acc)
        std_acc = np.std(valid_acc)
        print(f"\nSubject {sb} average: {mean_acc:.3f}% ± {std_acc:.3f}")

# 全局平均准确率
global_test = res[~torch.isnan(res)]
if len(global_test) > 0:
    global_mean = torch.mean(global_test)
    global_std = torch.std(global_test)
    print(f"\n{'='*60}")
    print(f"Global Average: {global_mean:.3f}% ± {global_std:.3f}")
    print(f"Total samples: {len(global_test)}")

os.makedirs(cfg.result_dir, exist_ok=True)
save_path = f"{cfg.result_dir}/result_within_trial_subject_level_{cfg.kfold_num}folds.csv"
np.savetxt(save_path, res.numpy(), delimiter=',', fmt='%.3f')
print(f"\nResults saved to {save_path}")