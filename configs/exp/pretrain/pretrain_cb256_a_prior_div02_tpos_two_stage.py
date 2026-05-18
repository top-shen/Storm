_base_ = [
    "./pretrain_cb256_a_prior_div02_two_stage.py"
]

tag = "cb256_a_pdiv02_tpos"

# Keep temporal position information in the encoder, but remove stock-index
# spatial position before quantization to reduce asset-specific code collapse.
config = dict(
    encoder_config=dict(
        sep_pos_embed_mode="temporal_only",
    )
)
