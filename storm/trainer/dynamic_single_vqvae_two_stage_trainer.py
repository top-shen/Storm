import os
import json
import random
from typing import Dict

import torch

from storm.metrics import MSE
from storm.models import get_patch_info, patchify
from storm.registry import TRAINER
from storm.trainer.dynamic_single_vqvae_trainer import DynamicSingleVQVAETrainer
from storm.utils import MetricLogger, Records, SmoothedValue


@TRAINER.register_module(force=True)
class DynamicSingleVQVAETwoStageTrainer(DynamicSingleVQVAETrainer):
    """Experimental two-stage trainer for single VQ-VAE.

    Stage 1 optimizes representation/reconstruction losses. Stage 2 can freeze
    VQ-VAE modules and optimize return-prediction losses only. Keeping this in a
    separate trainer avoids changing the default single_vqvae training line.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vqvae_pretrain_epochs = getattr(self.config, "vqvae_pretrain_epochs", 0)
        self.freeze_vqvae_in_predictor_stage = getattr(self.config, "freeze_vqvae_in_predictor_stage", True)
        self.predictor_stage_lr = getattr(self.config, "predictor_stage_lr", None)
        self.best_after_epoch = getattr(self.config, "best_after_epoch", self.vqvae_pretrain_epochs + 1)

    def _get_train_stage(self, epoch: int) -> str:
        if epoch <= self.vqvae_pretrain_epochs:
            return "vqvae_recon"
        return "predictor"

    @staticmethod
    def _set_module_trainable(module: torch.nn.Module, trainable: bool):
        for param in module.parameters():
            param.requires_grad = trainable

    def _configure_train_stage(self, active_model: torch.nn.Module, stage: str):
        model = self.accelerator.unwrap_model(active_model)

        for param in model.parameters():
            param.requires_grad = True

        if stage != "predictor" or not self.freeze_vqvae_in_predictor_stage:
            return

        frozen_module_names = ["embed_layer", "encoder", "quantizer", "decoder"]
        for name in frozen_module_names:
            module = getattr(model, name, None)
            if module is not None:
                self._set_module_trainable(module, False)
                module.eval()

    def _use_fixed_predictor_lr(self, stage: str) -> bool:
        return stage == "predictor" and self.predictor_stage_lr is not None

    def _apply_stage_lr(self, stage: str):
        if not self._use_fixed_predictor_lr(stage):
            return
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.predictor_stage_lr

    def run_step(self,
                 epoch,
                 if_use_writer=True,
                 if_use_wandb=True,
                 if_plot=False,
                 mode="train",
                 if_update=True,
                 model_override=None):
        self.check_batch_info_flag = True if epoch == self.start_epoch else False

        if_train = mode == "train" and if_update

        records = Records(accelerator=self.accelerator)
        metric_logger = MetricLogger(delimiter="  ")

        active_model = model_override if model_override is not None else self.model
        train_stage = self._get_train_stage(epoch) if mode in ["train", "valid"] else "test"

        if if_train:
            active_model.train(True)
            self._configure_train_stage(active_model, train_stage)
        else:
            active_model.eval()

        if self.accelerator.use_distributed:
            patch_size = active_model.module.patch_size
            if_mask = active_model.module.if_mask
        else:
            patch_size = active_model.patch_size
            if_mask = active_model.if_mask

        if mode == "train":
            if if_train:
                metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

            header = f"| Train Epoch: [{epoch}/{self.num_training_epochs}]"
            global_step = self.global_train_step
            dataloader = self.train_dataloader
        elif mode == "valid":
            header = f"| Valid Epoch: [{epoch}/{self.num_valid_epochs}]"
            global_step = self.global_valid_step
            dataloader = self.valid_dataloader
        else:
            header = f"| Test Epoch: [{epoch}/{self.num_testing_epochs}]"
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
            features = history["features"]
            labels = history["labels"]
            prices = history["prices"]
            timestamps = history["timestamps"]
            prices_mean = history["prices_mean"]
            prices_std = history["prices_std"]

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

            if if_train:
                output = active_model(features, labels, training=True)
            else:
                with torch.no_grad():
                    output = active_model(features, labels, training=False)

            input_size = prices.shape

            pred_prices = output["recon"]
            price_patch_size = (patch_size[0], patch_size[1], prices.shape[-1])
            patch_info = get_patch_info(input_size, price_patch_size)

            restored_target_prices = prices * prices_std + prices_mean
            restored_pred_prices = pred_prices * prices_std + prices_mean

            patched_target_prices = patchify(prices, patch_info=patch_info)
            patched_pred_prices = patchify(pred_prices, patch_info=patch_info)

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
                            restored_target_prices.squeeze(1).detach().cpu().numpy(),
                            restored_pred_prices.squeeze(1).detach().cpu().numpy(),
                            save_dir=save_dir,
                            save_prefix=save_prefix
                        )
                    except Exception as e:
                        self.logger.error(f"Plot error: {e}")

            weighted_quantized_loss = output["weighted_quantized_loss"]
            weighted_commit_loss = output["weighted_commit_loss"]
            weighted_codebook_diversity_loss = output["weighted_codebook_diversity_loss"]
            weighted_orthogonal_reg_loss = output["weighted_orthogonal_reg_loss"]

            if train_stage == "vqvae_recon":
                loss += weighted_quantized_loss

            records.update({
                "weighted_quantized_loss": weighted_quantized_loss,
                "weighted_commit_loss": weighted_commit_loss,
                "weighted_codebook_diversity_loss": weighted_codebook_diversity_loss,
                "weighted_orthogonal_reg_loss": weighted_orthogonal_reg_loss
            })

            mask = output["mask"]
            pred_label = output["pred_label"]
            posterior = output["posterior"]
            prior = output["prior"]
            labels = labels.squeeze(1).squeeze(1)

            if self.vae_loss_fn:
                loss_dict = self.vae_loss_fn(sample=patched_pred_prices,
                                             target_sample=patched_target_prices,
                                             pred_label=pred_label,
                                             label=labels,
                                             posterior=posterior,
                                             prior=prior,
                                             mask=mask,
                                             if_mask=if_mask)

                weighted_nll_loss = loss_dict["weighted_nll_loss"]
                weighted_kl_loss = loss_dict["weighted_kl_loss"]
                weighted_ret_loss = loss_dict["weighted_ret_loss"]

                records.update({
                    "weighted_nll_loss": weighted_nll_loss,
                    "weighted_kl_loss": weighted_kl_loss,
                    "weighted_ret_loss": weighted_ret_loss
                })

                if train_stage == "vqvae_recon":
                    loss += weighted_nll_loss
                elif train_stage == "predictor":
                    loss += weighted_kl_loss + weighted_ret_loss

            if self.price_cont_loss_fn:
                loss_dict = self.price_cont_loss_fn(prices=restored_pred_prices.squeeze(1))
                weighted_cont_loss = loss_dict["weighted_cont_loss"]

                records.update({
                    "weighted_cont_loss": weighted_cont_loss
                })

                if train_stage == "vqvae_recon":
                    loss += weighted_cont_loss

            records.update({
                "loss": loss
            })

            restored_pred_prices = patchify(restored_pred_prices, patch_info=patch_info)
            restored_target_prices = patchify(restored_target_prices, patch_info=patch_info)
            restored_pred_prices = restored_pred_prices.detach()
            restored_target_prices = restored_target_prices.detach()

            if if_mask and if_mask:
                mask = mask.detach()
                mask = mask.repeat(1, 1, prices.shape[-1])
                mask_target_prices = restored_target_prices * mask
                mask_pred_prices = restored_pred_prices * mask
                nomask_target_prices = restored_target_prices * (1.0 - mask)
                nomask_pred_prices = restored_pred_prices * (1.0 - mask)

                mask_mse = MSE(mask_target_prices, mask_pred_prices)
                nomask_mse = MSE(nomask_target_prices, nomask_pred_prices)
                mse = MSE(restored_target_prices, restored_pred_prices)

                records.update({
                    "mask_mse": mask_mse,
                    "nomask_mse": nomask_mse,
                    "mse": mse
                })

            else:
                mse = MSE(restored_target_prices, restored_pred_prices)

                records.update({
                    "mse": mse
                })

            with torch.no_grad():
                ret_mse = MSE(labels.detach(), pred_label.detach())
                direction_counts = self._direction_counts(pred_label.detach(), labels.detach())
                acc, mcc = self._direction_metrics_from_counts(
                    *(direction_counts[key].item() for key in ("direction_tp", "direction_tn", "direction_fp", "direction_fn"))
                )

                records.update({
                    "price_mse": mse,
                    "ret_mse": ret_mse,
                    **direction_counts,
                    "acc": torch.tensor(acc, device=self.device, dtype=self.dtype),
                    "mcc": torch.tensor(mcc, device=self.device, dtype=self.dtype),
                })

            if if_train:
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self._apply_stage_lr(train_stage)
                self.optimizer.step()

                if self.scheduler and not self._use_fixed_predictor_lr(train_stage):
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
        self.logger.info("| Start two-stage training and evaluating VAE...")

        if self.best_metric_mode == "min":
            best_metric_value = float("inf")
        elif self.best_metric_mode == "max":
            best_metric_value = float("-inf")
        else:
            raise ValueError(f"Unsupported best_metric_mode: {self.best_metric_mode}")

        for epoch in range(self.start_epoch, self.num_training_epochs + 1):
            train_stage = self._get_train_stage(epoch)
            self.logger.info(f"| Training stage: {train_stage}")
            train_stats = self.run_step(epoch, mode="train")
            valid_stats = self.run_step(epoch, mode="valid")
            test_stats = self.run_test(epoch, mode="test")

            if self.best_metric not in valid_stats:
                raise KeyError(
                    f"best_metric={self.best_metric} not found in valid_stats. "
                    f"Available metrics: {sorted(valid_stats.keys())}"
                )
            metric = valid_stats[self.best_metric]

            log_stats = {"epoch": epoch, "stage": train_stage}
            log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
            log_stats.update({f"valid_{k}": v for k, v in valid_stats.items()})
            log_stats.update({f"test_{k}": v for k, v in test_stats.items()})

            if self.is_main_process:
                with open(os.path.join(self.exp_path, "train_log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if epoch % self.checkpoint_period == 0:
                self.save_checkpoint(epoch)

            can_update_best = epoch >= self.best_after_epoch
            if self.best_metric_mode == "min":
                is_best = can_update_best and metric < best_metric_value
            else:
                is_best = can_update_best and metric > best_metric_value

            if is_best:
                best_metric_value = metric
                self.save_checkpoint(epoch, if_best=True)
