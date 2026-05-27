_base_ = [
    "./pretrain_cb256_a_prior_lcb_two_stage.py"
]

tag = "cb256_a_plcb_reg_mid"

# Medium stage2 regularization: keep return-rank selection and the
# posterior-guided prior objective, while avoiding overly strong shrinkage.
stage2_lr = 1.5e-5
stage2_weight_decay = 0.03
stage2_min_epochs = 25
stage2_patience = 20
stage2_min_delta = 5e-6
