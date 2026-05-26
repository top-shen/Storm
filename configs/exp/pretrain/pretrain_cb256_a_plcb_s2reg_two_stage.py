_base_ = [
    "./pretrain_cb256_a_prior_lcb_two_stage.py"
]

tag = "cb256_a_plcb_s2reg"

# Keep the return-rank monitor and posterior-guided prior objective unchanged,
# but make stage2 harder to overfit.
stage2_lr = 1e-5
stage2_weight_decay = 0.05
stage2_min_epochs = 25
stage2_patience = 15
stage2_min_delta = 1e-5
