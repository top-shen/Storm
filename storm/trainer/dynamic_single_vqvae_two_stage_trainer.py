import os
import json
import random
import torch
from glob import glob
from typing import Any, Dict, Tuple

import numpy as np

from storm.registry import TRAINER, OPTIMIZER, SCHEDULER
from storm.utils import check_data
from storm.utils import SmoothedValue
from storm.utils import MetricLogger
from storm.utils import Records
from storm.metrics import MSE
from storm.models import patchify

from .dynamic_single_vqvae_trainer import DynamicSingleVQVAETrainer


@TRAINER.register_module(force=True)
class DynamicSingleVQVAETwoStageTrainer(DynamicSingleVQVAETrainer):
    """Two-stage variant for single VQ-VAE experiments.

    Stage 1 trains the representation path only:
        reconstruction + vector-quantization objectives.

    Stage 2 freezes the VQ-VAE path and trains the return-prediction path:
        ret/KL/ranking/IC objectives, depending on the configured loss weights.
    """

    VQ_MODULE_NAMES = ("embed_layer", "encoder", "quantizer", "decoder")
    PRED_MODULE_NAMES = (
        "stock_mixer",
        "to_pd_encode_latent",
        "post_distribution_layer",
        "to_pd_decode_latent",
        "alpha_distribution_layer",
        "beta_layer",
        "multi_head_attention_layer",
        "prior_distribution_layer",
    )

    def __init__(self, *args, **kwargs):
        self.two_stage_vqvae_epochs = None
        self._active_stage = None
        super().__init__(*args, **kwargs)

    def _init_params(self):
        """Same initialization as the base trainer, but without static graph.

        Static graph is not safe here because the trainable parameter set changes
        between representation learning and prediction learning.
        """

        self.logger.info("| Init parameters for two-stage single VQ-VAE trainer...")

        torch.set_default_dtype(self.dtype)

        self.is_main_process = self.accelerator.is_local_main_process

        self.model = self.accelerator.prepare(self.vae)
        self.model_ema = self.accelerator.prepare(self.vae_ema)

        if self.vae_loss_fn:
            self.vae_loss_fn = self.accelerator.prepare(self.vae_loss_fn)
        if self.price_cont_loss_fn:
            self.price_cont_loss_fn = self.accelerator.prepare(self.price_cont_loss_fn)

        self.optimizer = self.accelerator.prepare(self.vae_optimizer)
        self.scheduler = self.accelerator.prepare(self.vae_scheduler)
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        self.valid_dataloader = self.accelerator.prepare(self.valid_dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        torch.set_default_dtype(self.dtype)

        self.checkpoint_path = os.path.join(self.exp_path, "checkpoint")
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.plot_path = os.path.join(self.exp_path, "plot")
        os.makedirs(self.plot_path, exist_ok=True)

        self.two_stage_vqvae_epochs = int(getattr(self.config, "two_stage_vqvae_epochs", 120))
        self.stage1_min_epochs = int(getattr(self.config, "stage1_min_epochs", max(1, self.two_stage_vqvae_epochs // 2)))
        self.stage1_patience = int(getattr(self.config, "stage1_patience", 30))
        self.stage1_monitor = getattr(self.config, "stage1_monitor", "price_mse")
        self.stage1_monitor_mode = getattr(self.config, "stage1_monitor_mode", "min")
        self.stage1_min_delta = float(getattr(self.config, "stage1_min_delta", 0.0))
        self.stage2_min_epochs = int(getattr(self.config, "stage2_min_epochs", 45))
        self.stage2_patience = int(getattr(self.config, "stage2_patience", 35))
        self.stage2_monitor = getattr(self.config, "stage2_monitor", self.best_metric)
        self.stage2_monitor_mode = getattr(self.config, "stage2_monitor_mode", self.best_metric_mode)
        self.stage2_min_delta = float(getattr(self.config, "stage2_min_delta", 0.0))
        self.stage2_metric_prediction = getattr(self.config, "stage2_metric_prediction", "default")
        self.reset_optimizer_on_stage_switch = bool(getattr(self.config, "reset_optimizer_on_stage_switch", True))
        self.two_stage_resume_model_only = bool(getattr(self.config, "two_stage_resume_model_only", True))
        self.stage1_lr = float(getattr(self.config, "stage1_lr", getattr(self.config, "vae_lr", 1e-4)))
        self.stage1_weight_decay = float(getattr(
            self.config,
            "stage1_weight_decay",
            getattr(self.config, "vae_weight_decay", 0.05),
        ))
        self.stage1_betas = tuple(getattr(self.config, "stage1_betas", getattr(self.config, "vae_betas", (0.9, 0.95))))
        self.stage1_scheduler_type = getattr(self.config, "stage1_scheduler_type", "CosineWithWarmupScheduler")
        self.stage1_warmup_epochs = int(getattr(self.config, "stage1_warmup_epochs", 12))
        self.stage2_lr = float(getattr(self.config, "stage2_lr", 3e-5))
        self.stage2_weight_decay = float(getattr(self.config, "stage2_weight_decay", 0.01))
        self.stage2_betas = tuple(getattr(self.config, "stage2_betas", getattr(self.config, "vae_betas", (0.9, 0.95))))
        self.stage2_scheduler_type = getattr(self.config, "stage2_scheduler_type", "CosineWithWarmupScheduler")
        self.stage2_warmup_epochs = int(getattr(self.config, "stage2_warmup_epochs", 8))

        loaded_full_optimizer_state = False
        if self.resume:
            if self.two_stage_resume_model_only:
                self.start_epoch = self._load_latest_model_state_only() + 1
            else:
                try:
                    self.start_epoch = self.load_checkpoint() + 1
                    loaded_full_optimizer_state = True
                except ValueError as exc:
                    if "parameter group" not in str(exc):
                        raise
                    self.logger.warning(
                        "| Full optimizer resume failed because the optimizer parameter groups changed. "
                        "Fallback to model-only resume for two-stage training."
                    )
                    self.start_epoch = self._load_latest_model_state_only() + 1
        else:
            self.start_epoch = 1
            self.logger.info("| Resume disabled. Start training from epoch 1 without loading the latest checkpoint.")

        self.check_batch_info_flag = True

        self.global_train_step = 0
        self.global_valid_step = 0
        self.global_test_step = 0

        self.logger.info(
            "| Two-stage schedule: "
            f"stage1(VQ-VAE representation)<= {self.two_stage_vqvae_epochs} epochs, "
            f"monitor={self.stage1_monitor}/{self.stage1_monitor_mode}, "
            f"min_epochs={self.stage1_min_epochs}, patience={self.stage1_patience}, "
            f"min_delta={self.stage1_min_delta}; "
            f"stage2(return prediction) monitor={self.stage2_monitor}/{self.stage2_monitor_mode}, "
            f"min_epochs={self.stage2_min_epochs}, patience={self.stage2_patience}, "
            f"min_delta={self.stage2_min_delta}; "
            f"stage1_lr={self.stage1_lr}, stage2_lr={self.stage2_lr}, "
            f"reset_optimizer_on_stage_switch={self.reset_optimizer_on_stage_switch}, "
            f"resume_model_only={self.two_stage_resume_model_only}"
        )
        if self.reset_optimizer_on_stage_switch and not loaded_full_optimizer_state:
            initial_stage = self._stage_for_epoch(self.start_epoch)
            total_epochs = (
                max(1, self.two_stage_vqvae_epochs - self.start_epoch + 1)
                if initial_stage == "vqvae"
                else max(1, self.num_training_epochs - self.start_epoch + 1)
            )
            warmup_epochs = self.stage1_warmup_epochs if initial_stage == "vqvae" else self.stage2_warmup_epochs
            self._reset_stage_optimizer(
                stage=initial_stage,
                total_epochs=total_epochs,
                warmup_epochs=warmup_epochs,
            )

    def _stage_for_epoch(self, epoch: int) -> str:
        if self._active_stage is not None:
            return self._active_stage
        return "vqvae" if int(epoch) <= self.two_stage_vqvae_epochs else "predict"

    @staticmethod
    def _initial_best_value(mode: str) -> float:
        if mode == "min":
            return float("inf")
        if mode == "max":
            return float("-inf")
        raise ValueError(f"Unsupported monitor mode: {mode}")

    @staticmethod
    def _is_better(metric: float, best_metric: float, mode: str, min_delta: float = 0.0) -> bool:
        if mode == "min":
            return metric < best_metric - min_delta
        if mode == "max":
            return metric > best_metric + min_delta
        raise ValueError(f"Unsupported monitor mode: {mode}")

    def _save_named_checkpoint(self, epoch: int, filename: str):
        if not self.accelerator.is_local_main_process:
            return
        checkpoint_file = os.path.join(self.checkpoint_path, filename)
        state = {
            "epoch": epoch,
            "model_state": self.accelerator.unwrap_model(self.model).state_dict(),
            "model_ema_state": self.accelerator.unwrap_model(self.model_ema).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        self.accelerator.save(state, checkpoint_file)
        self.logger.info(f"| Checkpoint saved: {checkpoint_file}")

    def _load_model_state_only(self, filename: str) -> int:
        checkpoint_file = os.path.join(self.checkpoint_path, filename)
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Two-stage checkpoint not found: {checkpoint_file}")
        return self._load_model_state_only_from_file(checkpoint_file)

    def _latest_checkpoint_file(self):
        checkpoint_files = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
        if not checkpoint_files:
            return None
        return sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]

    def _load_latest_model_state_only(self) -> int:
        checkpoint_file = self._latest_checkpoint_file()
        if checkpoint_file is None:
            self.logger.info(f"| No checkpoint found in {self.checkpoint_path}.")
            return 0
        return self._load_model_state_only_from_file(checkpoint_file)

    def _load_model_state_only_from_file(self, checkpoint_file: str) -> int:
        self.logger.info(f"| Load model weights only: {checkpoint_file}")
        state = torch.load(checkpoint_file, map_location=self.device)
        self.accelerator.unwrap_model(self.model).load_state_dict(state["model_state"])
        if "model_ema_state" in state:
            self.accelerator.unwrap_model(self.model_ema).load_state_dict(state["model_ema_state"])
        return int(state["epoch"])

    def _unwrap_model(self, model=None):
        if model is None:
            model = self.model
        return self.accelerator.unwrap_model(model)

    @staticmethod
    def _set_modules_trainable(model, module_names, trainable: bool, train_mode: bool):
        for name in module_names:
            module = getattr(model, name, None)
            if module is None:
                continue
            module.train(train_mode)
            for param in module.parameters():
                param.requires_grad_(trainable)

    def _configure_stage_modules(self, stage: str, active_model, if_train: bool):
        model = self._unwrap_model(active_model)

        if not if_train:
            model.eval()
            return

        if stage == "vqvae":
            self._set_modules_trainable(model, self.VQ_MODULE_NAMES, trainable=True, train_mode=True)
            self._set_modules_trainable(model, self.PRED_MODULE_NAMES, trainable=False, train_mode=False)
        elif stage == "predict":
            self._set_modules_trainable(model, self.VQ_MODULE_NAMES, trainable=False, train_mode=False)
            self._set_modules_trainable(model, self.PRED_MODULE_NAMES, trainable=True, train_mode=True)
        else:
            raise ValueError(f"Unsupported two-stage stage: {stage}")

    def _stage_optimizer_settings(self, stage: str) -> Dict[str, Any]:
        if stage == "vqvae":
            return {
                "module_names": self.VQ_MODULE_NAMES,
                "lr": self.stage1_lr,
                "weight_decay": self.stage1_weight_decay,
                "betas": self.stage1_betas,
                "scheduler_type": self.stage1_scheduler_type,
            }
        if stage == "predict":
            return {
                "module_names": self.PRED_MODULE_NAMES,
                "lr": self.stage2_lr,
                "weight_decay": self.stage2_weight_decay,
                "betas": self.stage2_betas,
                "scheduler_type": self.stage2_scheduler_type,
            }
        raise ValueError(f"Unsupported two-stage stage: {stage}")

    def _collect_trainable_stage_params(self, module_names) -> list:
        model = self._unwrap_model(self.model)
        params = []
        for name in module_names:
            module = getattr(model, name, None)
            if module is None:
                continue
            params.extend(param for param in module.parameters() if param.requires_grad)
        return params

    def _reset_stage_optimizer(self, stage: str, total_epochs: int, warmup_epochs: int):
        """Rebuild optimizer/scheduler for the active stage.

        The two stages optimize different parameter subsets and different
        objectives, so sharing one decayed scheduler across them makes Stage 2
        start with an almost exhausted learning rate. Rebuilding here gives the
        prediction head a clean, controlled optimization window.
        """

        self._configure_stage_modules(stage, self.model, if_train=True)
        settings = self._stage_optimizer_settings(stage)
        params = self._collect_trainable_stage_params(settings["module_names"])
        if not params:
            raise RuntimeError(f"No trainable parameters found for two-stage stage={stage}.")

        optimizer_cfg = {
            "type": "AdamW",
            "params": params,
            "lr": settings["lr"],
            "weight_decay": settings["weight_decay"],
            "betas": settings["betas"],
        }
        optimizer = OPTIMIZER.build(optimizer_cfg)

        steps_per_epoch = max(1, len(self.train_dataloader))
        total_steps = max(1, int(total_epochs) * steps_per_epoch)
        warmup_steps = max(0, min(int(warmup_epochs) * steps_per_epoch, total_steps - 1))
        scheduler_cfg = {
            "type": settings["scheduler_type"],
            "optimizer": optimizer,
            "num_warmup_steps": warmup_steps,
            "num_training_steps": total_steps,
        }
        scheduler = SCHEDULER.build(scheduler_cfg)

        self.optimizer = self.accelerator.prepare(optimizer)
        self.scheduler = self.accelerator.prepare(scheduler)
        self.vae_optimizer = self.optimizer
        self.vae_scheduler = self.scheduler
        self.logger.info(
            "| Reset two-stage optimizer: "
            f"stage={stage}, lr={settings['lr']}, weight_decay={settings['weight_decay']}, "
            f"scheduler={settings['scheduler_type']}, total_steps={total_steps}, "
            f"warmup_steps={warmup_steps}, trainable_params={sum(p.numel() for p in params)}"
        )

    def run_step(self,
                 epoch,
                 if_use_writer=True,
                 if_use_wandb=True,
                 if_plot=False,
                 mode="train",
                 if_update=True,
                 model_override=None):
        self.check_batch_info_flag = True if epoch == self.start_epoch else False

        stage = self._stage_for_epoch(epoch)
        if_train = mode == "train" and if_update

        records = Records(accelerator=self.accelerator)
        metric_logger = MetricLogger(delimiter="  ")

        active_model = model_override if model_override is not None else self.model

        if if_train:
            active_model.train(True)
        else:
            active_model.eval()
        self._configure_stage_modules(stage, active_model, if_train)

        if self.accelerator.use_distributed:
            patch_size = active_model.module.patch_size
            if_mask = active_model.module.if_mask
        else:
            patch_size = active_model.patch_size
            if_mask = active_model.if_mask

        if mode == "train":
            if if_train:
                metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
            header = f"| Train Epoch: [{epoch}/{self.num_training_epochs}] [{stage}]"
            global_step = self.global_train_step
            dataloader = self.train_dataloader
        elif mode == "valid":
            header = f"| Valid Epoch: [{epoch}/{self.num_valid_epochs}] [{stage}]"
            global_step = self.global_valid_step
            dataloader = self.valid_dataloader
        else:
            header = f"| Test Epoch: [{epoch}/{self.num_testing_epochs}] [{stage}]"
            global_step = self.global_test_step
            dataloader = self.test_dataloader

        sample_batchs = []
        if if_plot:
            num_plot_sample_batch = int(self.num_plot_samples // self.plot.sample_num)
            sample_batchs = random.sample(range(len(dataloader)), num_plot_sample_batch)

        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader,
                                                                       self.logger,
                                                                       self.print_freq,
                                                                       header)):
            loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

            self.check_batch_info(batch)

            asset = batch["asset"]
            history = batch["history"]

            start_timestamp = history["start_timestamp"]
            end_timestamp = history["end_timestamp"]
            start_index = history["start_index"]
            end_index = history["end_index"]
            features = history["features"]
            labels = history["labels"]
            prices = history["prices"]
            timestamps = history["timestamps"]
            prices_mean = history["prices_mean"]
            prices_std = history["prices_std"]
            text = history["text"]

            features = features.to(self.device, self.dtype)
            prices = prices.to(self.device, self.dtype)
            prices_mean = prices_mean.to(self.device, self.dtype)
            prices_std = prices_std.to(self.device, self.dtype)

            if len(features.shape) == 4:
                features = features.unsqueeze(1)
            if len(prices.shape) == 4:
                prices = prices.unsqueeze(1)
            if len(prices_mean.shape) == 4:
                prices_mean = prices_mean.unsqueeze(1)
            if len(prices_std.shape) == 4:
                prices_std = prices_std.unsqueeze(1)
            if len(labels.shape) == 4:
                labels = labels.unsqueeze(1)

            labels = labels[:, :, -1:, :, 0]

            return_prediction = stage != "vqvae"
            if if_train:
                output = active_model(features, labels, training=True, return_prediction=return_prediction)
            else:
                with torch.no_grad():
                    output = active_model(features, labels, training=False, return_prediction=return_prediction)

            pred_recon = output["recon"]
            (
                recon_target,
                restored_target_recon,
                restored_pred_recon,
                patched_target_recon,
                patched_pred_recon,
                patch_info,
            ) = self._build_reconstruction_tensors(
                prices=prices,
                prices_mean=prices_mean,
                prices_std=prices_std,
                pred_recon=pred_recon,
                patch_size=patch_size,
            )

            if if_plot and data_iter_step in sample_batchs:
                if self.is_main_process:
                    try:
                        save_dir = os.path.join(self.plot_path, "comparison_kline")
                        save_prefix = "{}_epoch_{:06d}_batch_{:06d}".format(mode, epoch, data_iter_step)
                        self.plot.plot_comparison_kline(
                            asset,
                            start_timestamp.detach().cpu().numpy(),
                            end_timestamp.detach().cpu().numpy(),
                            timestamps.detach().cpu().numpy(),
                            restored_target_recon.squeeze(1).detach().cpu().numpy(),
                            restored_pred_recon.squeeze(1).detach().cpu().numpy(),
                            save_dir=save_dir,
                            save_prefix=save_prefix
                        )
                    except Exception as e:
                        self.logger.error(f"Plot error: {e}")

            weighted_quantized_loss = output["weighted_quantized_loss"]
            weighted_commit_loss = output["weighted_commit_loss"]
            weighted_codebook_diversity_loss = output["weighted_codebook_diversity_loss"]
            weighted_codebook_usage_balance_loss = output.get(
                "weighted_codebook_usage_balance_loss",
                torch.tensor(0.0, device=self.device, dtype=self.dtype),
            )
            weighted_orthogonal_reg_loss = output["weighted_orthogonal_reg_loss"]

            records.update({
                "stage_id": torch.tensor(1.0 if stage == "vqvae" else 2.0, device=self.device, dtype=self.dtype),
                "weighted_quantized_loss": weighted_quantized_loss,
                "weighted_commit_loss": weighted_commit_loss,
                "weighted_codebook_diversity_loss": weighted_codebook_diversity_loss,
                "weighted_codebook_usage_balance_loss": weighted_codebook_usage_balance_loss,
                "weighted_orthogonal_reg_loss": weighted_orthogonal_reg_loss,
            })

            mask = output["mask"]
            pred_label = output["pred_label"]
            pred_label_prior = output.get("pred_label_prior", None)
            posterior = output["posterior"]
            prior = output["prior"]
            labels = labels.squeeze(1).squeeze(1)

            weighted_ranking_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            weighted_ic_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            ranking_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            ic_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            weighted_prior_ret_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            weighted_prior_ranking_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            weighted_prior_ic_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

            if self.vae_loss_fn:
                loss_dict = self.vae_loss_fn(
                    sample=patched_pred_recon,
                    target_sample=patched_target_recon,
                    pred_label=pred_label,
                    label=labels,
                    posterior=posterior,
                    prior=prior,
                    mask=mask,
                    if_mask=if_mask,
                    compute_prediction_losses=stage != "vqvae",
                    pred_label_prior=pred_label_prior,
                )

                weighted_nll_loss = loss_dict["weighted_nll_loss"]
                weighted_kl_loss = loss_dict["weighted_kl_loss"]
                weighted_ret_loss = loss_dict["weighted_ret_loss"]
                ranking_loss = loss_dict.get(
                    "ranking_loss",
                    torch.tensor(0.0, device=self.device, dtype=self.dtype),
                )
                weighted_ranking_loss = loss_dict.get(
                    "weighted_ranking_loss",
                    torch.tensor(0.0, device=self.device, dtype=self.dtype),
                )
                ic_loss = loss_dict.get(
                    "ic_loss",
                    torch.tensor(0.0, device=self.device, dtype=self.dtype),
                )
                weighted_ic_loss = loss_dict.get(
                    "weighted_ic_loss",
                    torch.tensor(0.0, device=self.device, dtype=self.dtype),
                )
                weighted_prior_ret_loss = loss_dict.get(
                    "weighted_prior_ret_loss",
                    torch.tensor(0.0, device=self.device, dtype=self.dtype),
                )
                prior_ranking_loss = loss_dict.get(
                    "prior_ranking_loss",
                    torch.tensor(0.0, device=self.device, dtype=self.dtype),
                )
                weighted_prior_ranking_loss = loss_dict.get(
                    "weighted_prior_ranking_loss",
                    torch.tensor(0.0, device=self.device, dtype=self.dtype),
                )
                prior_ic_loss = loss_dict.get(
                    "prior_ic_loss",
                    torch.tensor(0.0, device=self.device, dtype=self.dtype),
                )
                weighted_prior_ic_loss = loss_dict.get(
                    "weighted_prior_ic_loss",
                    torch.tensor(0.0, device=self.device, dtype=self.dtype),
                )

                records.update({
                    "weighted_nll_loss": weighted_nll_loss,
                    "weighted_kl_loss": weighted_kl_loss,
                    "weighted_ret_loss": weighted_ret_loss,
                    "ranking_loss": ranking_loss,
                    "weighted_ranking_loss": weighted_ranking_loss,
                    "ic_loss": ic_loss,
                    "weighted_ic_loss": weighted_ic_loss,
                    "weighted_prior_ret_loss": weighted_prior_ret_loss,
                    "prior_ranking_loss": prior_ranking_loss,
                    "weighted_prior_ranking_loss": weighted_prior_ranking_loss,
                    "prior_ic_loss": prior_ic_loss,
                    "weighted_prior_ic_loss": weighted_prior_ic_loss,
                })

                if stage == "vqvae":
                    loss = loss + weighted_quantized_loss + weighted_nll_loss
                else:
                    loss = (
                        loss
                        + weighted_kl_loss
                        + weighted_ret_loss
                        + weighted_ranking_loss
                        + weighted_ic_loss
                        + weighted_prior_ret_loss
                        + weighted_prior_ranking_loss
                        + weighted_prior_ic_loss
                    )
            else:
                if stage == "vqvae":
                    loss = loss + weighted_quantized_loss

            if self.price_cont_loss_fn:
                loss_dict = self.price_cont_loss_fn(prices=restored_pred_recon.squeeze(1))
                weighted_cont_loss = loss_dict["weighted_cont_loss"]
                records.update({"weighted_cont_loss": weighted_cont_loss})
                if stage == "vqvae":
                    loss = loss + weighted_cont_loss

            records.update({"loss": loss})

            restored_pred_recon = patchify(restored_pred_recon, patch_info=patch_info)
            restored_target_recon = patchify(restored_target_recon, patch_info=patch_info)
            restored_pred_recon = restored_pred_recon.detach()
            restored_target_recon = restored_target_recon.detach()

            if if_mask and if_mask:
                mask = mask.detach()
                mask = mask.repeat(1, 1, recon_target.shape[-1])
                mask_target_recon = restored_target_recon * mask
                mask_pred_recon = restored_pred_recon * mask
                nomask_target_recon = restored_target_recon * (1.0 - mask)
                nomask_pred_recon = restored_pred_recon * (1.0 - mask)

                mask_mse = MSE(mask_target_recon, mask_pred_recon)
                nomask_mse = MSE(nomask_target_recon, nomask_pred_recon)
                mse = MSE(restored_target_recon, restored_pred_recon)
                records.update({"mask_mse": mask_mse, "nomask_mse": nomask_mse, "mse": mse})
            else:
                mse = MSE(restored_target_recon, restored_pred_recon)
                records.update({"mse": mse})

            with torch.no_grad():
                metric_pred_label = pred_label
                metric_weighted_ranking_loss = weighted_ranking_loss
                if (
                    stage == "predict"
                    and self.stage2_metric_prediction == "prior"
                    and pred_label_prior is not None
                ):
                    metric_pred_label = pred_label_prior
                    metric_weighted_ranking_loss = weighted_prior_ranking_loss

                if metric_pred_label is None:
                    ret_mse = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    ic = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    direction_counts = {
                        key: torch.tensor(0.0, device=self.device, dtype=self.dtype)
                        for key in ("direction_tp", "direction_tn", "direction_fp", "direction_fn")
                    }
                    acc, mcc = 0.0, 0.0
                else:
                    ret_mse = MSE(labels.detach(), metric_pred_label.detach())
                    ic = self._pearson_ic_series(metric_pred_label.detach(), labels.detach()).mean()
                    direction_counts = self._direction_counts(metric_pred_label.detach(), labels.detach())
                    acc, mcc = self._direction_metrics_from_counts(
                        *(direction_counts[key].item() for key in ("direction_tp", "direction_tn", "direction_fp", "direction_fn"))
                    )

                records.update({
                    "recon_mse": mse,
                    "ret_mse": ret_mse,
                    "return_rank_loss": ret_mse + metric_weighted_ranking_loss.detach(),
                    "ic": ic,
                    **direction_counts,
                    "acc": torch.tensor(acc, device=self.device, dtype=self.dtype),
                    "mcc": torch.tensor(mcc, device=self.device, dtype=self.dtype),
                    "price_mse": mse,
                })

            if if_train:
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                lr = self.optimizer.param_groups[0]["lr"]
                records.update_value({"lr": lr})
                self._update_model_ema()

            records.gather(self.train_gather_multi_gpu)
            gathered_item = records.gathered_item

            global_step += 1

            prefix = mode
            if global_step % self.print_freq == 0:
                wandb_dict = {}
                for key, value in gathered_item.items():
                    if if_use_writer and self.writer:
                        self.writer.log_scalar(f"{prefix}/{key}", value, global_step)
                    if if_use_wandb and self.wandb:
                        wandb_dict[f"{prefix}/{key}"] = value

                self.wandb.log(wandb_dict)

            metric_logger.update(**gathered_item)
            metric_logger.synchronize_between_processes()

        if if_use_writer and self.is_main_process:
            self.writer.flush()

        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if all(key in stats for key in ("direction_tp", "direction_tn", "direction_fp", "direction_fn")):
            stats["acc"], stats["mcc"] = self._direction_metrics_from_counts(
                stats["direction_tp"],
                stats["direction_tn"],
                stats["direction_fp"],
                stats["direction_fn"],
            )

        if mode == "train":
            self.global_train_step = global_step
            log_str = "| Train averaged stats: "
        elif mode == "valid":
            self.global_valid_step = global_step
            log_str = "| Valid averaged stats: "
        else:
            self.global_test_step = global_step
            log_str = "| Test averaged stats: "

        for name, value in stats.items():
            log_str += f"- {name}: {value:.4f}"
        self.logger.info(log_str)

        return stats

    def train(self):
        self.logger.info("| Start two-stage training with train/valid only...")

        stage = self._stage_for_epoch(self.start_epoch)
        self._active_stage = stage

        stage1_best_value = self._initial_best_value(self.stage1_monitor_mode)
        stage1_bad_epochs = 0
        stage1_epochs_seen = 0
        stage1_best_epoch = None

        stage2_best_value = self._initial_best_value(self.stage2_monitor_mode)
        stage2_bad_epochs = 0
        stage2_epochs_seen = 0

        for epoch in range(self.start_epoch, self.num_training_epochs + 1):
            self._active_stage = stage
            train_stats = self.run_step(epoch, mode="train")
            valid_stats = self.run_step(epoch, mode="valid")

            log_stats = {"epoch": epoch, "stage": stage}
            log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
            log_stats.update({f"valid_{k}": v for k, v in valid_stats.items()})

            if self.is_main_process:
                with open(os.path.join(self.exp_path, "train_log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if epoch % self.checkpoint_period == 0:
                self.save_checkpoint(epoch)

            if stage == "vqvae":
                if self.stage1_monitor not in valid_stats:
                    raise KeyError(
                        f"stage1_monitor={self.stage1_monitor} not found in valid_stats. "
                        f"Available metrics: {sorted(valid_stats.keys())}"
                    )
                stage1_epochs_seen += 1
                metric = valid_stats[self.stage1_monitor]
                if self._is_better(
                    metric,
                    stage1_best_value,
                    self.stage1_monitor_mode,
                    min_delta=self.stage1_min_delta,
                ):
                    stage1_best_value = metric
                    stage1_bad_epochs = 0
                    stage1_best_epoch = epoch
                    self._save_named_checkpoint(epoch, "best_stage1.pth")
                else:
                    stage1_bad_epochs += 1

                hit_stage1_cap = stage1_epochs_seen >= self.two_stage_vqvae_epochs
                hit_stage1_patience = (
                    stage1_epochs_seen >= self.stage1_min_epochs
                    and stage1_bad_epochs >= self.stage1_patience
                )
                if hit_stage1_cap or hit_stage1_patience:
                    self.accelerator.wait_for_everyone()
                    if stage1_best_epoch is not None:
                        self._load_model_state_only("best_stage1.pth")
                    self.logger.info(
                        "| Stage 1 finished. "
                        f"best_epoch={stage1_best_epoch}, best_{self.stage1_monitor}={stage1_best_value:.6f}. "
                        "Stage 2 will freeze VQ-VAE and optimize prediction losses."
                    )
                    stage = "predict"
                    self._active_stage = stage
                    if self.reset_optimizer_on_stage_switch:
                        remaining_epochs = max(1, self.num_training_epochs - epoch)
                        self._reset_stage_optimizer(
                            stage="predict",
                            total_epochs=remaining_epochs,
                            warmup_epochs=self.stage2_warmup_epochs,
                        )
                continue

            if self.stage2_monitor not in valid_stats:
                raise KeyError(
                    f"stage2_monitor={self.stage2_monitor} not found in valid_stats. "
                    f"Available metrics: {sorted(valid_stats.keys())}"
                )
            stage2_epochs_seen += 1
            metric = valid_stats[self.stage2_monitor]
            is_best = self._is_better(
                metric,
                stage2_best_value,
                self.stage2_monitor_mode,
                min_delta=self.stage2_min_delta,
            )

            if is_best:
                stage2_best_value = metric
                stage2_bad_epochs = 0
                self.save_checkpoint(epoch, if_best=True)
            else:
                stage2_bad_epochs += 1

            if stage2_epochs_seen >= self.stage2_min_epochs and stage2_bad_epochs >= self.stage2_patience:
                self.logger.info(
                    "| Stage 2 early stop. "
                    f"best_{self.stage2_monitor}={stage2_best_value:.6f}, "
                    f"patience={self.stage2_patience}"
                )
                break
