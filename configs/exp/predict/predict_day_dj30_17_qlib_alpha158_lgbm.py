workdir = "workdir"
tag = "predict_day_dj30_17_qlib_alpha158_lgbm"
exp_path = f"{workdir}/{tag}"
log_file = "storm.log"
tensorboard_path = "tensorboard"
checkpoint_path = "checkpoint"
wandb_path = "wandb"
model_file = "best.pkl"
project = "storm"
seed = 1337
if_remove = False

qlib_init = dict(
    provider_uri=None,
    region="us",
)

data = dict(
    start_time="2008-04-01",
    end_time="2024-04-01",
    fit_start_time="2008-04-01",
    fit_end_time="2021-04-01",
    instruments=None,
    instruments_path="configs/_asset_list_/dj30_17.json",
)

segments = dict(
    train=("2008-04-01", "2021-04-01"),
    valid=("2021-04-01", "2024-04-01"),
    test=("2021-04-01", "2024-04-01"),
)

label_column = "LABEL0"
label = ["Ref($close, -1) / $close - 1"]

model = dict(
    loss="mse",
    colsample_bytree=0.8879,
    learning_rate=0.2,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=20,
    seed=seed,
)
