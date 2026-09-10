import torch
# paths: put KUL3_1D.mat into ./data (or override these two lines)
process_data_dir='./data'
result_dir='./results'

# models
model_names = ['CNN_CSP_Riemannian', 'CNN_CovarianceEnhanced']
model_name = model_names[0]
if model_name == 'CNN_CSP_Riemannian':
    result_dir='./CNN_CSP_Riemannian'
    pretrain_model_dir='./CNN_CSP_Riemannian/pretrain_model'
    finetune_model_dir='./CNN_CSP_Riemannian/finetune_model'
elif model_name == 'CNN_CovarianceEnhanced':
    result_dir='./CNN_CovarianceEnhanced'
    pretrain_model_dir='./CNN_CovarianceEnhanced/pretrain_model'
    finetune_model_dir='./CNN_CovarianceEnhanced/finetune_model'
dataset_name = 'KUL3_1D.mat'

# GPU selection: use the CUDA_VISIBLE_DEVICES env var to pick a GPU; defaults to cuda:0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch_num = 100
finetune_epoch_num=50
batch_size = 64
sample_rate = 128
categorie_num = 2
sbnum = 16
trnum=8
kfold_num = 5
fine_ratio=0.2
lr=1e-3
finetune_lr=5e-5
weight_decay=0.01
torch_seed=2025
# the length of decision window
decision_window = 128*10 


csp_n_filters = 6  # CSP滤波器数量
