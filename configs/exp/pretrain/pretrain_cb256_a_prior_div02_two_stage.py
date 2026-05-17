_base_ = [
    "./pretrain_cb256_a_prior_two_stage.py"
]

tag = "cb256_a_pdiv02"

# Combine posterior-guided prior learning with stronger codebook diversity.
config = dict(
    quantizer_config=dict(
        codebook_diversity_loss_weight=0.2,
    )
)
