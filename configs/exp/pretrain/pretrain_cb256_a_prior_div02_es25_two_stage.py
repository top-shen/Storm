_base_ = [
    "./pretrain_cb256_a_prior_div02_two_stage.py"
]

tag = "cb256_a_pdiv02_es25"

# Early-stop comparison line for posterior-guided prior + diversity=0.2.
stage2_patience = 25
