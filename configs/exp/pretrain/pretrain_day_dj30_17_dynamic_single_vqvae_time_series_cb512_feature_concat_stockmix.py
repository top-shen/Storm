_base_ = [
    "./pretrain_day_dj30_17_dynamic_single_vqvae_time_series_cb512_feature_concat.py"
]

tag = "single_vqvae_feature_concat_stockmix_512"

vae = dict(
    use_quantized_only_for_factors=False,
    use_stock_mixing=True,
    stock_mixing_market_dim=8,
    stock_mixing_dropout=0.1,
    stock_mixing_residual_scale=0.1,
)

use_ema_for_eval = False
best_metric = "ret_mse"
best_metric_mode = "min"
