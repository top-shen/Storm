_base_ = [
    "./pretrain_cb256_a_prior_ubal_two_stage.py"
]

tag = "cb256_a_pubal05_lcb"

# With EMA disabled, usage-balance gradients act more directly on the learned
# codebook. Use a moderate weight rather than the stronger 0.1 setting.
config = dict(
    quantizer_config=dict(
        ema_update=False,
        learnable_codebook=True,
        threshold_ema_dead_code=0,
        codebook_usage_balance_loss_weight=0.05,
    )
)
