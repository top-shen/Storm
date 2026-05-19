_base_ = [
    "./pretrain_cb256_a_prior_div02_two_stage.py"
]

tag = "cb256_a_pubal"

# Directly penalize global code-usage imbalance on top of the existing
# diversity loss. Start small so reconstruction is not over-regularized.
config = dict(
    quantizer_config=dict(
        codebook_usage_balance_loss_weight=0.02,
        codebook_usage_balance_temperature=1.0,
    )
)
