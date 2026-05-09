_base_ = [
    "./pretrain_day_dj30_17_dynamic_single_vqvae_time_series_retmse_main.py"
]

tag = "single_vqvae_recon152_512"
reconstruction_target = "features"

config = dict(
    decoder_config=dict(
        output_dim=152,
    ),
)
