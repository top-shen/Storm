_base_ = [
    "./pretrain_cb256_a_nosmix_two_stage.py"
]

tag = "cb256_a_prior"

# Posterior remains a label-conditioned teacher, while the prior prediction
# receives direct return/ranking supervision to match the inference path.
loss_funcs_config = dict(
    vae_loss=dict(
        posterior_loss_weight=0.5,
        prior_loss_weight=1.0,
    )
)

stage2_metric_prediction = "prior"
