_base_ = [
    "./pretrain_day_dj30_17_dynamic_single_vqvae_time_series_cb256_feature_concat_stockmix.py"
]

tag = "cb256_e_sm8_r01"

# E: mean aggregation, wider market bottleneck, residual_scale=0.1.
vae = dict(
    use_quantized_only_for_factors=False,
    use_stock_mixing=True,
    stock_mixing_market_dim=8,
    stock_mixing_dropout=0.05,
    stock_mixing_residual_scale=0.1,
    stock_mixing_aggregation="mean",
)

trainer = dict(type="DynamicSingleVQVAETwoStageTrainer")
two_stage_resume_model_only = True
two_stage_vqvae_epochs = 120
stage1_min_epochs = 70
stage1_patience = 30
stage1_monitor = "loss"
stage1_monitor_mode = "min"
stage1_min_delta = 1e-4
stage2_min_epochs = 45
stage2_patience = 35
stage2_monitor = "return_rank_loss"
stage2_monitor_mode = "min"
stage2_min_delta = 1e-8

reset_optimizer_on_stage_switch = True
stage1_lr = 1e-4
stage1_weight_decay = 0.05
stage1_betas = (0.9, 0.95)
stage1_scheduler_type = "CosineWithWarmupScheduler"
stage1_warmup_epochs = 12
stage2_lr = 3e-5
stage2_weight_decay = 0.01
stage2_betas = (0.9, 0.95)
stage2_scheduler_type = "CosineWithWarmupScheduler"
stage2_warmup_epochs = 8

use_ema_for_eval = False
best_metric = "return_rank_loss"
best_metric_mode = "min"
