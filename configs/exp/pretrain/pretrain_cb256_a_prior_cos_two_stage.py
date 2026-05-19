_base_ = [
    "./pretrain_cb256_a_prior_div02_two_stage.py"
]

tag = "cb256_a_pcos"

# Cosine quantization reduces norm-driven nearest-code attraction. A sharper
# diversity temperature keeps the soft assignment close to cosine argmax.
config = dict(
    quantizer_config=dict(
        use_cosine_sim=True,
        codebook_diversity_temperature=10.0,
    )
)
