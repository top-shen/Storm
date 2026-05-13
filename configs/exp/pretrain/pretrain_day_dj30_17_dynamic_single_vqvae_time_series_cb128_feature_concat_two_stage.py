_base_ = [
    "./pretrain_day_dj30_17_dynamic_single_vqvae_time_series_cb128.py"
]

tag = "single_vqvae_feature_concat_two_stage_128"

# factors = concat(enc, quantized).
use_quantized_only_for_factors = False
vae = dict(use_quantized_only_for_factors=False)

# Stage 1: train VQ-VAE representation only.
# Stage 2: freeze VQ-VAE and train prior/posterior return prediction.
trainer = dict(type="DynamicSingleVQVAETwoStageTrainer")
two_stage_resume_model_only = True
two_stage_vqvae_epochs = 120
stage1_min_epochs = 70
stage1_patience = 30
stage1_monitor = "price_mse"
stage1_monitor_mode = "min"
stage1_min_delta = 1e-3
stage2_min_epochs = 45
stage2_patience = 35
stage2_monitor = "ret_mse"
stage2_monitor_mode = "min"
stage2_min_delta = 1e-8

# Rebuild optimizer/scheduler per stage. Stage 1 keeps the representation
# learning rate; Stage 2 uses a gentler fresh LR for the noisy return head.
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
best_metric = "ret_mse"
best_metric_mode = "min"
