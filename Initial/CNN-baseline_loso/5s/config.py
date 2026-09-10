import torch
# paths: put KUL3_1D.mat into ./data (or override these two lines)
process_data_dir='./data'
result_dir='./results'

# models
model_names = ['CNN_baseline','CNN_2D','DenseNet_37']
model_name = model_names[0] # you could change the code to other models by only changing the number
model_dir='./model'
if model_name == 'CNN_baseline':
    dataset_name = 'KUL3_1D.mat'
else:
    dataset_name = 'KUL3_2D.mat'

# GPU selection: use the CUDA_VISIBLE_DEVICES env var to pick a GPU; defaults to cuda:0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch_num = 100
batch_size = 128
sample_rate = 128
categorie_num = 2
sbnum = 16
trnum=8
kfold_num = 5

lr=1e-3
weight_decay=0.01
torch_seed=2024
# the length of decision window
decision_window = 128*5



