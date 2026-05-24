_base_ = [
    "./pretrain_cb256_a_prior_lcb_two_stage.py"
]

tag = "cb256_a_plcb_prank"

stage2_monitor = "prior_return_rank_loss"
stage2_monitor_mode = "min"
