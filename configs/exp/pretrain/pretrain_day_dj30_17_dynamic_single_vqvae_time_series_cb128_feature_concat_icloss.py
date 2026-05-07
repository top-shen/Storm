_base_ = [
    "./pretrain_day_dj30_17_dynamic_single_vqvae_time_series_cb128_feature_concat.py"
]

tag = "single_vqvae_feature_concat_icloss_128"

# First stabilize the IC objective before re-introducing pairwise ranking loss.
ranking_loss_weight = 0.0
ic_loss_weight = 0.02
rank_temperature = 0.05

loss_funcs_config = dict(
    vae_loss=dict(
        ranking_loss_weight=ranking_loss_weight,
        ic_loss_weight=ic_loss_weight,
        rank_temperature=rank_temperature,
    )
)

best_metric = "ret_mse"
best_metric_mode = "min"
