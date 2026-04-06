workdir = "workdir"
tag = "predict_day_dj30_17_qlib_lgbm"
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
    region="cn",
)

history_timestamps = 64
include_asset_identity = True
feature_columns = [
    "open","high","low","close","adj_close","kmid","kmid2","klen","kup","kup2","klow","klow2","ksft","ksft2",
    "roc_5","roc_10","roc_20","roc_30","roc_60","ma_5","ma_10","ma_20","ma_30","ma_60",
    "std_5","std_10","std_20","std_30","std_60","beta_5","beta_10","beta_20","beta_30","beta_60",
    "max_5","max_10","max_20","max_30","max_60","min_5","min_10","min_20","min_30","min_60",
    "qtlu_5","qtlu_10","qtlu_20","qtlu_30","qtlu_60","qtld_5","qtld_10","qtld_20","qtld_30","qtld_60",
    "rank_5","rank_10","rank_20","rank_30","rank_60","imax_5","imax_10","imax_20","imax_30","imax_60",
    "imin_5","imin_10","imin_20","imin_30","imin_60","imxd_5","imxd_10","imxd_20","imxd_30","imxd_60",
    "rsv_5","rsv_10","rsv_20","rsv_30","rsv_60","cntp_5","cntp_10","cntp_20","cntp_30","cntp_60",
    "cntn_5","cntn_10","cntn_20","cntn_30","cntn_60","cntd_5","cntd_10","cntd_20","cntd_30","cntd_60",
    "corr_5","corr_10","corr_20","corr_30","corr_60","cord_5","cord_10","cord_20","cord_30","cord_60",
    "sump_5","sump_10","sump_20","sump_30","sump_60","sumn_5","sumn_10","sumn_20","sumn_30","sumn_60",
    "sumd_5","sumd_10","sumd_20","sumd_30","sumd_60","vma_5","vma_10","vma_20","vma_30","vma_60",
    "vstd_5","vstd_10","vstd_20","vstd_30","vstd_60","wvma_5","wvma_10","wvma_20","wvma_30","wvma_60",
    "vsump_5","vsump_10","vsump_20","vsump_30","vsump_60","vsumn_5","vsumn_10","vsumn_20","vsumn_30","vsumn_60",
    "vsumd_5","vsumd_10","vsumd_20","vsumd_30","vsumd_60",
    "day","weekday","month",
]
label_column = "ret1"

data = dict(
    data_path="workdir/processd_day_dj30_17/features",
    assets_path="configs/_asset_list_/dj30_17.json",
    start_time="2008-04-01",
    end_time="2024-04-01",
)

segments = dict(
    train=("2008-04-01", "2021-04-01"),
    valid=("2021-04-01", "2024-04-01"),
    test=("2021-04-01", "2024-04-01"),
)

model = dict(
    loss="mse",
    colsample_bytree=0.8879,
    learning_rate=0.0421,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=20,
    seed=seed,
)




