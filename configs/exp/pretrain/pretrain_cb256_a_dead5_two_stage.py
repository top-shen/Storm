_base_ = [
    "./pretrain_cb256_a_nosmix_two_stage.py"
]

tag = "cb256_a_dead5"

# A3: cb=256 no-stockmix with more aggressive dead-code replacement.
config = dict(
    quantizer_config=dict(
        threshold_ema_dead_code=5,
    )
)
