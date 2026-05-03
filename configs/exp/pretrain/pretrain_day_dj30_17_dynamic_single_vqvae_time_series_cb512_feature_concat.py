_base_ = [
    "./pretrain_day_dj30_17_dynamic_single_vqvae_time_series_retmse_main.py"
]

tag = "single_vqvae_feature_concat_512"

# Keep the existing quantized-only configs untouched. This line explicitly
# opens a feature-concat experiment: factors = concat(enc, quantized).
use_quantized_only_for_factors = False
vae = dict(use_quantized_only_for_factors=False)

use_ema_for_eval = False
best_metric = "ret_mse"
best_metric_mode = "min"
