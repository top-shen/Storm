_base_ = [
    "./pretrain_cb256_a_prior_two_stage.py"
]

tag = "cb256_a_plcb"

# Disable hard-assignment EMA codebook updates and learn the codebook through
# the training objective. This isolates the effect on the original a_prior line.
config = dict(
    quantizer_config=dict(
        ema_update=False,
        learnable_codebook=True,
        threshold_ema_dead_code=0,
    )
)
