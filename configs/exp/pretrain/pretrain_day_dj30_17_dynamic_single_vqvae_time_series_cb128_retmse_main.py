_base_ = [
    "./pretrain_day_dj30_17_dynamic_single_vqvae_time_series_cb128.py"
]

tag = "single_vqvae_retmse_128"
use_ema_for_eval = False
best_metric = "ret_mse"
best_metric_mode = "min"
