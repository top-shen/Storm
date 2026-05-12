_base_ = [
    "./pretrain_day_dj30_17_dynamic_single_vqvae_time_series_cb256_feature_concat.py"
]

tag = "single_vqvae_feature_concat_stockmix_256"

# Stock-representation-level cross-sectional mixing:
# aggregate patch tokens to one representation per stock, mix 17 stocks through
# a small market bottleneck, then broadcast the market residual back to tokens.
vae = dict(
    use_quantized_only_for_factors=False,
    use_stock_mixing=True,
    stock_mixing_market_dim=4,
    stock_mixing_dropout=0.05,
    stock_mixing_residual_scale=0.2,
    stock_mixing_aggregation="mean",
)

# Keep this line clean: return MSE + StockMixer-style pairwise ranking loss.
# Do not add IC loss here; use ret_mse for best checkpoint selection.
ranking_loss_weight = 0.1
ic_loss_weight = 0.0
loss_funcs_config = dict(
    vae_loss=dict(
        ranking_loss_weight=ranking_loss_weight,
        ic_loss_weight=ic_loss_weight,
        ranking_loss_type="stockmixer_pairwise",
    )
)

use_ema_for_eval = False
best_metric = "return_rank_loss"
best_metric_mode = "min"
