_base_ = [
    "./pretrain_day_dj30_17_dynamic_single_vqvae_time_series_cb256_feature_concat.py"
]

tag = "single_vqvae_feature_concat_ranking_ic_256"

# Ranking-aware return prediction: keep return regression, then add direct
# cross-sectional ranking and Pearson-IC optimization signals.
ranking_loss_weight = 0.1
ic_loss_weight = 0.05
rank_temperature = 0.01

loss_funcs_config = dict(
    vae_loss=dict(
        ranking_loss_weight=ranking_loss_weight,
        ic_loss_weight=ic_loss_weight,
        rank_temperature=rank_temperature,
    )
)

best_metric = "ic"
best_metric_mode = "max"
