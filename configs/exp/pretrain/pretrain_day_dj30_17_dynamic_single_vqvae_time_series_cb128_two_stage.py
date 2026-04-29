_base_ = [
    "./pretrain_day_dj30_17_dynamic_single_vqvae_time_series_cb128_retmse_main.py"
]

tag = "single_vqvae_two_stage_128"

trainer.update(dict(type="DynamicSingleVQVAETwoStageTrainer"))

# Stage 1: train VQ-VAE representation with reconstruction/codebook losses.
# Stage 2: freeze VQ-VAE modules and train prior/posterior return prediction.
two_stage_training = True
vqvae_pretrain_epochs = 100
freeze_vqvae_in_predictor_stage = True
predictor_stage_lr = vae_lr

# Do not save stage-1 checkpoints as best.pth; stage 1 is representation pretraining.
best_after_epoch = vqvae_pretrain_epochs + 1
best_metric = "ret_mse"
best_metric_mode = "min"
use_ema_for_eval = False
