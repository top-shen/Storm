_base_ = [
    "./pretrain_day_dj30_17_dynamic_single_vqvae_time_series_cb128_feature_concat.py"
]

tag = "single_vqvae_stockmix_clean_128"

# Clean StockMixing ablation: feature_concat + cross-sectional mixer only.
# No IC/ranking loss is enabled here, so the effect of StockMixing is isolated.
vae = dict(
    use_quantized_only_for_factors=False,
    use_stock_mixing=True,
    stock_mixing_market_dim=4,
    stock_mixing_dropout=0.05,
    stock_mixing_residual_scale=0.3,
)

use_ema_for_eval = False
best_metric = "ret_mse"
best_metric_mode = "min"
