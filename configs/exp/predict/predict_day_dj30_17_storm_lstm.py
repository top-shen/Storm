workdir = "workdir"
tag = "predict_day_dj30_17_storm_lstm"
exp_path = f"{workdir}/{tag}"
log_file = "storm.log"
tensorboard_path = "tensorboard"
checkpoint_path = "checkpoint"
wandb_path = "wandb"
project = "storm"
model_file = "best.pth"
seed = 1337
if_remove = False

storm_data_config = "configs/exp/pretrain/pretrain_day_dj30_17_dynamic_single_vqvae_time_series.py"
# Reuses the main STORM data config, which now uses StandardScaler(train fit -> valid/test transform).

label_column = "ret1"
history_timestamps = 64
num_assets = 17
feature_dim = 152
batch_size = 32
num_workers = 1

hidden_size = 128
num_layers = 2
dropout = 0.0

learning_rate = 1e-3
weight_decay = 0.0
grad_clip = 1.0
num_epochs = 200
early_stop = 10
