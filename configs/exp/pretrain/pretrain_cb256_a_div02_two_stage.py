_base_ = [
    "./pretrain_cb256_a_nosmix_two_stage.py"
]

tag = "cb256_a_div02"

# A2: cb=256 no-stockmix with stronger codebook diversity regularization.
config = dict(
    quantizer_config=dict(
        codebook_diversity_loss_weight=0.2,
    )
)
