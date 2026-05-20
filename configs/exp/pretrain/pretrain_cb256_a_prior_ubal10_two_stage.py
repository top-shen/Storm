_base_ = [
    "./pretrain_cb256_a_prior_ubal_two_stage.py"
]

tag = "cb256_a_pubal10"

# Stronger global usage-balance pressure than cb256_a_pubal (0.02).
# Keep the same temperature to isolate the effect of the loss weight.
config = dict(
    quantizer_config=dict(
        codebook_usage_balance_loss_weight=0.1,
    )
)
