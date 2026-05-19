_base_ = [
    "./pretrain_cb256_a_prior_two_stage.py"
]

tag = "cb256_a_pdiv05"

# Increase diversity pressure from the current best prior+div0.2 baseline.
config = dict(
    quantizer_config=dict(
        codebook_diversity_loss_weight=0.5,
    )
)
