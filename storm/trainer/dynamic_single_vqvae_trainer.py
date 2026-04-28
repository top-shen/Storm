import os
import torch
from typing import Any, Dict, List, Tuple, Union
from glob import glob
import json
import random
import numpy as np
from collections import Counter

from storm.registry import TRAINER
from storm.registry import DOWNSTREAM
from storm.utils import check_data
from storm.utils import SmoothedValue
from storm.utils import MetricLogger
from storm.metrics import MSE, RankICIR, RankIC, RankICSeries
from storm.models import get_patch_info, patchify
from storm.utils import convert_int_to_timestamp
from storm.utils import save_joblib
from storm.utils import save_json
from storm.utils import Records
from storm.qlib_adapter import build_prediction_payload

@TRAINER.register_module(force=True)
class DynamicSingleVQVAETrainer():
    def __init__(self,
                 *args,
                 config,
                 vae,
                 vae_ema,
                 dit,
                 dit_ema,
                 diffusion,
                 train_dataloader,
                 valid_dataloader,
                 test_dataloader,
                 loss_funcs,
                 vae_optimizer,
                 dit_optimizer,
                 vae_scheduler,
                 dit_scheduler,
                 logger,
                 device,
                 dtype,
                 writer: Any = None,
                 wandb: Any = None,
                 num_plot_samples: int = 50,
                 plot: Any = None,
                 accelerator: Any = None,
                 print_freq: int = 20,
                 train_gather_multi_gpu: bool = False,
                 **kwargs):

        self.config = config
        self.batch_size = self.config.batch_size
        self.start_epoch = self.config.start_epoch
        self.resume = getattr(self.config, "resume", True)
        self.ema_decay = getattr(self.config, "ema_decay", None)
        self.ema_update_after_step = getattr(self.config, "ema_update_after_step", 0)
        self.use_ema_for_eval = getattr(self.config, "use_ema_for_eval", False)
        self.best_metric = getattr(self.config, "best_metric", "ret_mse")
        self.best_metric_mode = getattr(self.config, "best_metric_mode", "min")
        self.num_training_epochs = self.config.num_training_epochs
        self.num_valid_epochs = self.config.num_training_epochs
        self.num_testing_epochs = 1
        self.num_checkpoint_del = self.config.num_checkpoint_del
        self.checkpoint_period = self.config.checkpoint_period
        self.vae = vae
        self.vae_ema = vae_ema
        self.dit = dit
        self.dit_ema = dit_ema
        self.diffusion = diffusion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.vae_loss_fn = loss_funcs.get("vae_loss", None)
        self.price_cont_loss_fn = loss_funcs.get("price_cont_loss", None)
        self.vae_optimizer = vae_optimizer
        self.dit_optimizer = dit_optimizer
        self.vae_scheduler = vae_scheduler
        self.dit_scheduler = dit_scheduler
        self.logger = logger
        self.device = device
        self.dtype = dtype
        self.exp_path = config.exp_path
        self.writer = writer
        self.wandb = wandb
        self.num_plot_samples = num_plot_samples
        self.plot = plot
        self.accelerator = accelerator
        self.print_freq = print_freq
        self.train_gather_multi_gpu = train_gather_multi_gpu

        self.downstream = DOWNSTREAM.build(self.config.downstream)

        self._init_params()

    def _init_params(self):

        self.logger.info("| Init parameters for VAE trainer...")

        torch.set_default_dtype(self.dtype)

        self.is_main_process = self.accelerator.is_local_main_process

        self.model = self.accelerator.prepare(self.vae)
        self.model_ema = self.accelerator.prepare(self.vae_ema)

        if hasattr(self.model, "_set_static_graph"):
            self.model._set_static_graph()

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

        if self.resume:
            self.start_epoch = self.load_checkpoint() + 1
        else:
            self.start_epoch = 1
            self.logger.info("| Resume disabled. Start training from epoch 1 without loading the latest checkpoint.")

        self.check_batch_info_flag = True

        self.global_train_step = 0
        self.global_valid_step = 0
        self.global_test_step = 0

    def _update_model_ema(self):
        if self.ema_decay is None or self.ema_decay <= 0.0 or self.ema_decay >= 1.0:
            return
        if self.global_train_step < self.ema_update_after_step:
            return

        model = self.accelerator.unwrap_model(self.model)
        model_ema = self.accelerator.unwrap_model(self.model_ema)

        with torch.no_grad():
            model_params = dict(model.named_parameters())
            ema_params = dict(model_ema.named_parameters())
            for name, param in model_params.items():
                if name in ema_params:
                    ema_params[name].data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

            model_buffers = dict(model.named_buffers())
            ema_buffers = dict(model_ema.named_buffers())
            for name, buffer in model_buffers.items():
                if name in ema_buffers:
                    ema_buffers[name].data.copy_(buffer.data)

    def _get_eval_model(self):
        if self.use_ema_for_eval:
            return self.model_ema, "ema"
        return self.model, "main"

    @staticmethod
    def _direction_counts(pred_label: torch.Tensor, true_label: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_up = pred_label > 0
        true_up = true_label > 0
        return {
            "direction_tp": (pred_up & true_up).sum().to(torch.float32),
            "direction_tn": (~pred_up & ~true_up).sum().to(torch.float32),
            "direction_fp": (pred_up & ~true_up).sum().to(torch.float32),
            "direction_fn": (~pred_up & true_up).sum().to(torch.float32),
        }

    @staticmethod
    def _direction_metrics_from_counts(tp: float, tn: float, fp: float, fn: float) -> Tuple[float, float]:
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0.0
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0
        return float(acc), float(mcc)

    @staticmethod
    def _pearson_ic_series(pred_label: torch.Tensor, true_label: torch.Tensor) -> torch.Tensor:
        if pred_label.dim() == 1:
            pred_label = pred_label.unsqueeze(0)
        else:
            pred_label = pred_label.reshape(pred_label.shape[0], -1)

        if true_label.dim() == 1:
            true_label = true_label.unsqueeze(0)
        else:
            true_label = true_label.reshape(true_label.shape[0], -1)

        pred_label = pred_label.float()
        true_label = true_label.float()
        pred_centered = pred_label - pred_label.mean(dim=1, keepdim=True)
        true_centered = true_label - true_label.mean(dim=1, keepdim=True)

        covariance = torch.mean(pred_centered * true_centered, dim=1)
        pred_std = torch.sqrt(torch.mean(pred_centered.square(), dim=1))
        true_std = torch.sqrt(torch.mean(true_centered.square(), dim=1))
        denominator = pred_std * true_std

        ic_values = torch.zeros_like(covariance)
        valid = denominator > 0
        ic_values[valid] = covariance[valid] / denominator[valid]
        return torch.nan_to_num(ic_values, nan=0.0, posinf=0.0, neginf=0.0)

    def save_checkpoint(self, epoch: int, if_best: bool = False):
        if not self.accelerator.is_local_main_process:
            return  # Only save checkpoint on the main process

        if if_best:
            checkpoint_file = os.path.join(self.checkpoint_path, f"best.pth")
        else:
            checkpoint_file = os.path.join(self.checkpoint_path, "checkpoint_{:06d}.pth".format(epoch))

        # Save model, optimizer, and scheduler states
        state = {
            'epoch': epoch,
            'model_state': self.accelerator.unwrap_model(self.model).state_dict(),
            'model_ema_state': self.accelerator.unwrap_model(self.model_ema).state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }

        self.accelerator.save(state, checkpoint_file)

        # Manage saved checkpoints
        checkpoint_files = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
        checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if len(checkpoint_files) > self.num_checkpoint_del:
            for checkpoint_file in checkpoint_files[:-self.num_checkpoint_del]:
                os.remove(checkpoint_file)
                self.logger.info(f"| Checkpoint deleted: {checkpoint_file}")
        self.logger.info(f"| Checkpoint saved: {checkpoint_file}")

    def load_checkpoint(self, epoch: int = -1, checkpoint_file: str = None, if_best: bool = False):

        if checkpoint_file is None:
            best_checkpoint_file = os.path.join(self.checkpoint_path, "best.pth")
            epoch_checkpoint_file = os.path.join(self.checkpoint_path, f"checkpoint_{epoch:06d}.pth")

            latest_checkpoint_file = None
            checkpoint_files = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
            if not checkpoint_files:
                self.logger.info(f"| No checkpoint found in {self.checkpoint_path}.")
            else:
                checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                checkpoint_file = checkpoint_files[-1]
                latest_checkpoint_file = checkpoint_file

            if if_best and os.path.exists(best_checkpoint_file):
                checkpoint_file = best_checkpoint_file
                self.logger.info(f"| Load best checkpoint: {checkpoint_file}")
            elif epoch >= 0 and os.path.exists(epoch_checkpoint_file):
                checkpoint_file = epoch_checkpoint_file
                self.logger.info(f"| Load epoch checkpoint: {checkpoint_file}")
            else:
                if latest_checkpoint_file:
                    checkpoint_file = latest_checkpoint_file
                    self.logger.info(f"| Load latest checkpoint: {checkpoint_file}")
                else:
                    checkpoint_file = None
                    self.logger.info(f"| Checkpoint not found.")
        else:
            self.logger.info(f"| Load checkpoint: {checkpoint_file}")

        if checkpoint_file is not None:

            state = torch.load(checkpoint_file, map_location=self.device)

            # Unwrap the model to load state dict into the underlying model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(state['model_state'])

            unwrapped_model_ema = self.accelerator.unwrap_model(self.model_ema)
            unwrapped_model_ema.load_state_dict(state['model_ema_state'])

            self.optimizer.load_state_dict(state['optimizer_state'])
            self.scheduler.load_state_dict(state['scheduler_state'])

            return state['epoch']
        else:
            return 0

    def __str__(self):
        return f"DynamicSingleVQVAETrainer(num_training_epochs={self.num_training_epochs}, start_epoch={self.start_epoch}, batch_size={self.batch_size})"

    def check_batch_info(self, batch: Dict):

        if self.check_batch_info_flag:
            asset = batch["asset"]
            self.logger.info(f"| Asset: {check_data(asset)}")

            for key in batch:
                if key not in ["asset"]:
                    data = batch[key]
                    log_str = f"| {key}: "
                    for key, value in data.items():
                        log_str += f"\n {key}: {check_data(value)}"
                    self.logger.info(log_str)

            self.check_batch_info_flag = False

    def run_step(self,
                 epoch,
                 if_use_writer = True,
                 if_use_wandb = True,
                 if_plot = False,
                 mode = "train",
                 if_update = True,
                 model_override = None
                  ):
        self.check_batch_info_flag = True if epoch == self.start_epoch else False

        if_train = mode == "train" and if_update

        records = Records(accelerator=self.accelerator)
        metric_logger = MetricLogger(delimiter="  ")

        active_model = model_override if model_override is not None else self.model

        if if_train:
            active_model.train(True)
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

        rankics = []
        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader,
                                                                       self.logger,
                                                                       self.print_freq,
                                                                       header)):

            loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

            self.check_batch_info(batch)

            asset = batch["asset"]  # (N, T)
            history = batch["history"]

            start_timestamp = history["start_timestamp"]  # (N,)
            end_timestamp = history["end_timestamp"]  # (N,)
            start_index = history["start_index"]  # (N,)
            end_index = history["end_index"]  # (N,)
            features = history["features"]  # (N, T, S, F)
            labels = history["labels"]  # (N, T, S, 2)
            prices = history["prices"]  # (N, T, S, 5)
            timestamps = history["timestamps"]  # (N, T, S)
            prices_mean = history["prices_mean"]  # (N, T, S, 5)
            prices_std = history["prices_std"]  # (N, T, S, 5)
            text = history["text"]  # (N, T)

            features = features.to(self.device, self.dtype)
            prices = prices.to(self.device, self.dtype)
            prices_mean = prices_mean.to(self.device, self.dtype)
            prices_std = prices_std.to(self.device, self.dtype)

            if len(features.shape) == 4:
                features = features.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices.shape) == 4:
                prices = prices.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_mean.shape) == 4:
                prices_mean = prices_mean.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_std.shape) == 4:
                prices_std = prices_std.unsqueeze(1)  # (N, C, T, S, F)
            if len(labels.shape) == 4:
                labels = labels.unsqueeze(1) # (N, C, T, S, 2)

            # The next day returns of the last day
            labels = labels[:, :, -1:, :, 0] # (N, C, T, S)

            # Forward
            if if_train:
                output = active_model(features, labels, training = True)
            else:
                with torch.no_grad():
                    output = active_model(features, labels, training = False)

            # Restore prices
            input_size = prices.shape

            pred_prices = output["recon"]
            patch_size = (patch_size[0], patch_size[1], prices.shape[-1])
            patch_info = get_patch_info(input_size, patch_size)

            restored_target_prices = prices * prices_std + prices_mean
            restored_pred_prices = pred_prices
            restored_pred_prices = restored_pred_prices * prices_std + prices_mean # (N, C, T, S, 5)

            patched_target_prices = patchify(prices, patch_info=patch_info)  # (N, L, D)
            patched_pred_prices = patchify(pred_prices, patch_info=patch_info)  # (N, L, D)

            # plot
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

            # Compute loss
            weighted_quantized_loss = output["weighted_quantized_loss"]
            weighted_commit_loss = output["weighted_commit_loss"]
            weighted_codebook_diversity_loss = output["weighted_codebook_diversity_loss"]
            weighted_orthogonal_reg_loss = output["weighted_orthogonal_reg_loss"]

            # `weighted_quantized_loss` already contains the full quantizer
            # objective. The breakdown terms are logged separately and should
            # not be added again here.
            loss += weighted_quantized_loss

            records.update({
                "weighted_quantized_loss": weighted_quantized_loss,
                "weighted_commit_loss": weighted_commit_loss,
                "weighted_codebook_diversity_loss": weighted_codebook_diversity_loss,
                "weighted_orthogonal_reg_loss": weighted_orthogonal_reg_loss
            })

            mask = output["mask"]
            pred_label = output["pred_label"] # (N, S)
            posterior = output["posterior"]
            prior = output["prior"]
            labels = labels.squeeze(1).squeeze(1)  # (N, S)

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

                loss += weighted_nll_loss + weighted_kl_loss + weighted_ret_loss

            if self.price_cont_loss_fn:
                loss_dict = self.price_cont_loss_fn(prices=restored_pred_prices.squeeze(1))
                weighted_cont_loss = loss_dict["weighted_cont_loss"]

                records.update({
                    "weighted_cont_loss": weighted_cont_loss
                })

                loss += weighted_cont_loss

            records.update({
                "loss": loss
            })

            # Compute metrics
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

            # pred_label = pred_label.detach()
            # labels = labels.detach()
            #
            # rankic = RankIC(pred_label, labels)
            # rankics.append(rankic)
            #
            # rankicir = RankICIR(rankics)
            #
            # records.update(data = {"RANKIC":rankic, "RANKICIR": rankicir },
            #                extra_info={"pred_label": pred_label,"true_label": labels})

            if if_train:
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                lr = self.optimizer.param_groups[0]["lr"]
                records.update_value({"lr": lr})
                self._update_model_ema()

            # gather data from multi gpu
            records.gather(self.train_gather_multi_gpu)
            gathered_item = records.gathered_item

            global_step += 1

            prefix = mode
            if global_step % self.print_freq == 0:

                wandb_dict = {}
                # For records
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

    def run_test(self,
                 epoch,
                 if_use_writer=True,
                 if_use_wandb=True,
                 if_plot=False,
                 mode="test",
                 save_predictions: bool = False,
                 model_override = None,
                 eval_model_name: str = "main"
                  ):

        self.check_batch_info_flag = True if epoch == self.start_epoch else False

        records = Records(accelerator=self.accelerator)

        metric_logger = MetricLogger(delimiter="  ")
        active_model = model_override if model_override is not None else self.model
        active_model.eval()

        if self.accelerator.use_distributed:
            patch_size = active_model.module.patch_size
            if_mask = active_model.module.if_mask
        else:
            patch_size = active_model.patch_size
            if_mask = active_model.if_mask

        header = f"| Test Epoch: [{epoch}/{self.num_testing_epochs}]"
        dataloader = self.test_dataloader

        sample_batchs = []
        if if_plot:
            num_plot_sample_batch = int(self.num_plot_samples // self.plot.sample_num)
            sample_batchs = random.sample(range(len(dataloader)), num_plot_sample_batch)

        ics = []
        rankics = []
        price_mses = []
        ret_mses = []
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        pred_labels_all = []
        true_labels_all = []
        end_timestamps_all = []
        assets_all = []
        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader,
                                                                       self.logger,
                                                                       self.print_freq,
                                                                       header)):

            self.check_batch_info(batch)

            asset = batch["asset"]  # (N, T)
            history = batch["history"]

            start_timestamp = history["start_timestamp"]  # (N,)
            end_timestamp = history["end_timestamp"]  # (N,)
            start_index = history["start_index"]  # (N,)
            end_index = history["end_index"]  # (N,)
            features = history["features"]  # (N, T, S, F)
            labels = history["labels"]  # (N, T, S, 2)
            prices = history["prices"]  # (N, T, S, 5)
            timestamps = history["timestamps"]  # (N, T, S)
            prices_mean = history["prices_mean"]  # (N, T, S, 5)
            prices_std = history["prices_std"]  # (N, T, S, 5)
            text = history["text"]  # (N, T)

            features = features.to(self.device, self.dtype)
            prices = prices.to(self.device, self.dtype)
            prices_mean = prices_mean.to(self.device, self.dtype)
            prices_std = prices_std.to(self.device, self.dtype)

            if len(features.shape) == 4:
                features = features.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices.shape) == 4:
                prices = prices.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_mean.shape) == 4:
                prices_mean = prices_mean.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_std.shape) == 4:
                prices_std = prices_std.unsqueeze(1)  # (N, C, T, S, F)
            if len(labels.shape) == 4:
                labels = labels.unsqueeze(1)  # (N, C, T, S, 2)

            # The next day returns of the last day
            labels = labels[:, :, -1:, :, 0]  # (N, C, T, S)

            with torch.no_grad():
                output = active_model(features, labels, training=False)

            # Restore prices
            input_size = prices.shape

            pred_prices = output["recon"]
            patch_size = (patch_size[0], patch_size[1], prices.shape[-1])
            patch_info = get_patch_info(input_size, patch_size)

            restored_target_prices = prices * prices_std + prices_mean
            restored_pred_prices = pred_prices
            restored_pred_prices = restored_pred_prices * prices_std + prices_mean  # (N, C, T, S, 5)

            patched_target_prices = patchify(prices, patch_info=patch_info)  # (N, L, D)
            patched_pred_prices = patchify(pred_prices, patch_info=patch_info)  # (N, L, D)

            # plot
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

            mask = output["mask"]
            pred_label = output["pred_label"]  # (N, S)
            posterior = output["posterior"]
            prior = output["prior"]
            labels = labels.squeeze(1).squeeze(1)  # (N, S)

            # Compute metrics
            restored_pred_prices = patchify(restored_pred_prices, patch_info=patch_info)
            restored_target_prices = patchify(restored_target_prices, patch_info=patch_info)
            restored_pred_prices = restored_pred_prices.detach()
            restored_target_prices = restored_target_prices.detach()

            pred_label = pred_label.detach()
            labels = labels.detach()
            end_timestamp = end_timestamp.detach()

            price_mse = MSE(restored_target_prices, restored_pred_prices)
            ret_mse = MSE(labels, pred_label)
            mse_tensor = torch.stack([price_mse.detach().float(), ret_mse.detach().float()])
            if self.accelerator is not None:
                mse_tensor = self.accelerator.gather_for_metrics(mse_tensor)
            mse_tensor = mse_tensor.reshape(-1, 2).detach().cpu().numpy()
            price_mses.extend(mse_tensor[:, 0].tolist())
            ret_mses.extend(mse_tensor[:, 1].tolist())

            direction_counts = self._direction_counts(pred_label, labels)
            tp += int(direction_counts["direction_tp"].detach().cpu().item())
            tn += int(direction_counts["direction_tn"].detach().cpu().item())
            fp += int(direction_counts["direction_fp"].detach().cpu().item())
            fn += int(direction_counts["direction_fn"].detach().cpu().item())

            batch_ics = self._pearson_ic_series(pred_label, labels)
            batch_rankics = RankICSeries(pred_label, labels)
            if self.accelerator is not None:
                batch_ics = self.accelerator.gather_for_metrics(batch_ics)
                batch_rankics = self.accelerator.gather_for_metrics(batch_rankics)
            ics.extend(batch_ics.detach().cpu().tolist())
            rankics.extend(batch_rankics.detach().cpu().tolist())

            end_timestamps_all.append(end_timestamp.cpu().numpy())
            pred_labels_all.append(pred_label.cpu().numpy())
            true_labels_all.append(labels.cpu().numpy())
            assets_all.extend(batch["asset"])

        metrics = dict()
        metrics["MSE"] = float(np.mean(ret_mses)) if len(ret_mses) > 0 else 0.0
        metrics["RET_MSE"] = metrics["MSE"]
        metrics["PRICE_MSE"] = float(np.mean(price_mses)) if len(price_mses) > 0 else 0.0

        count_tensor = torch.tensor([tp, tn, fp, fn], dtype=torch.float64, device=self.device)
        if self.accelerator is not None:
            count_tensor = self.accelerator.gather(count_tensor).view(-1, 4).sum(dim=0)
        tp, tn, fp, fn = [float(x) for x in count_tensor.detach().cpu().tolist()]
        metrics["ACC"], metrics["MCC"] = self._direction_metrics_from_counts(tp, tn, fp, fn)

        ic_values = np.asarray(ics, dtype=np.float64)
        metrics["IC"] = float(np.mean(ic_values)) if ic_values.size > 0 else 0.0

        rankic_values = np.asarray(rankics, dtype=np.float64)
        if rankic_values.size > 0:
            metrics["RANKIC"] = float(np.mean(rankic_values))
        else:
            metrics["RANKIC"] = 0.0

        if rankic_values.size > 1:
            rankic_std = float(np.std(rankic_values))
            metrics["RANKICIR"] = float(np.mean(rankic_values) / rankic_std) if rankic_std > 0 else 0.0
        else:
            metrics["RANKICIR"] = 0.0

        # Process extra records
        if self.is_main_process and len(end_timestamps_all) > 0:
            end_timestamps = np.concatenate(end_timestamps_all, axis=0)
            pred_labels = np.concatenate(pred_labels_all, axis=0)
            true_labels = np.concatenate(true_labels_all, axis=0)

            # sort according to end_timestamp
            indices = np.argsort(end_timestamps)
            end_timestamps = end_timestamps[indices]
            pred_labels = pred_labels[indices]
            true_labels = true_labels[indices]
            assets_all = [assets_all[idx] for idx in indices.tolist()]

            downstream_metrics = self.downstream(pred_labels=pred_labels,
                                                 true_labels=true_labels)

            metrics.update({
                "CW": downstream_metrics["CW"],
                "ARR%": downstream_metrics["ARR%"],
                "SR": downstream_metrics["SR"],
                "CR": downstream_metrics["CR"],
                "SOR": downstream_metrics["SOR"],
                "DD": downstream_metrics["DD"],
                "MDD%": downstream_metrics["MDD%"],
                "VOL": downstream_metrics["VOL"],
            })

            if save_predictions:
                row_timestamps = []
                row_assets = []
                row_preds = []
                row_trues = []
                for end_timestamp, batch_assets, pred_row, true_row in zip(end_timestamps, assets_all, pred_labels, true_labels):
                    date_str = convert_int_to_timestamp(int(end_timestamp)).strftime("%Y-%m-%d")
                    for asset, pred_value, true_value in zip(batch_assets, pred_row, true_row):
                        row_timestamps.append(date_str)
                        row_assets.append(asset)
                        row_preds.append(float(pred_value))
                        row_trues.append(float(true_value))

                payload = build_prediction_payload(row_timestamps, row_assets, row_preds, row_trues)
                prediction_file = f"{mode}_predictions.joblib" if eval_model_name == "main" else f"{mode}_predictions_{eval_model_name}.joblib"
                save_joblib(payload, os.path.join(self.exp_path, prediction_file))

        # Round
        for key, value in metrics.items():
            metrics[key] = float(np.round(value, 4))

        prefix = mode
        wandb_dict = {}
        # For records
        for key, value in metrics.items():
            if if_use_writer and self.writer:
                self.writer.log_scalar(f"{prefix}/{key}", value, epoch)
            if if_use_wandb and self.wandb:
                wandb_dict[f"{prefix}/{key}"] = value
        self.wandb.log(wandb_dict)

        if if_use_writer and self.is_main_process:
            self.writer.flush()

        log_str = "| Test averaged stats: "

        for name, meter in metrics.items():
            log_str += f"- {name}: {meter}"
        self.logger.info(log_str)

        return metrics

    def run_state(self,
                 epoch,
                 mode="test"
                 ):

        self.check_batch_info_flag = True if epoch == self.start_epoch else False

        records = Records(accelerator=self.accelerator)

        metric_logger = MetricLogger(delimiter="  ")
        self.model.eval()

        if self.accelerator.use_distributed:
            patch_size = self.model.module.patch_size
            if_mask = self.model.module.if_mask
            n_size = self.model.module.embed_layer.n_size
            n_num = self.model.module.embed_layer.n_num
        else:
            patch_size = self.model.patch_size
            if_mask = self.model.if_mask
            n_size = self.model.embed_layer.n_size
            n_num = self.model.embed_layer.n_num

        header = f"| Test Epoch: [{epoch}/{self.num_testing_epochs}]"
        dataloader = self.test_dataloader

        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader,
                                                                       self.logger,
                                                                       self.print_freq,
                                                                       header)):

            self.check_batch_info(batch)

            asset = batch["asset"]  # (N, T)
            history = batch["history"]

            start_timestamp = history["start_timestamp"]  # (N,)
            end_timestamp = history["end_timestamp"]  # (N,)
            start_index = history["start_index"]  # (N,)
            end_index = history["end_index"]  # (N,)
            features = history["features"]  # (N, T, S, F)
            labels = history["labels"]  # (N, T, S, 2)
            prices = history["prices"]  # (N, T, S, 5)
            timestamps = history["timestamps"]  # (N, T, S)
            prices_mean = history["prices_mean"]  # (N, T, S, 5)
            prices_std = history["prices_std"]  # (N, T, S, 5)
            text = history["text"]  # (N, T)

            features = features.to(self.device, self.dtype)
            prices = prices.to(self.device, self.dtype)
            prices_mean = prices_mean.to(self.device, self.dtype)
            prices_std = prices_std.to(self.device, self.dtype)

            if len(features.shape) == 4:
                features = features.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices.shape) == 4:
                prices = prices.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_mean.shape) == 4:
                prices_mean = prices_mean.unsqueeze(1)  # (N, C, T, S, F)
            if len(prices_std.shape) == 4:
                prices_std = prices_std.unsqueeze(1)  # (N, C, T, S, F)
            if len(labels.shape) == 4:
                labels = labels.unsqueeze(1)  # (N, C, T, S, 2)

            # The next day returns of the last day
            labels = labels[:, :, -1:, :, 0]  # (N, C, T, S)

            with torch.no_grad():
                output = self.model(features, labels, training=False)

            factors = output["factors"]
            embed_ind = output["embed_ind"]

            factors = factors.detach()
            embed_ind = embed_ind.detach()

            end_timestamp = end_timestamp.detach()

            records.update(extra_info = {
                "factors": factors,
                "embed_ind": embed_ind,
                "end_timestamps": end_timestamp,
            })

            # gather data from multi gpu
            records.gather(train_gather_multi_gpu = True)

        extra_combiner = records.extra_combiner

        # Process extra records
        if self.is_main_process:
            end_timestamps = extra_combiner["end_timestamps"]
            factors = extra_combiner["factors"]
            embed_ind = extra_combiner["embed_ind"]

            end_timestamps = np.concatenate(end_timestamps, axis=0)
            factors = np.concatenate(factors, axis=0)
            embed_ind = np.concatenate(embed_ind, axis=0)

            # count embedding index
            embed_ind = embed_ind.flatten().tolist()
            count_embed_ind = Counter(embed_ind)

            # sort according to end_timestamp
            indices = np.argsort(end_timestamps)
            end_timestamps = end_timestamps[indices]
            factors = factors[indices]

            end_timestamps = end_timestamps.tolist()
            end_timestamps = [convert_int_to_timestamp(end_timestamps).strftime("%Y-%m-%d") for end_timestamps in end_timestamps]

            meta = {
                "n_size": n_size,
                "n_num": n_num,
                "factors_shape": factors.shape,
                "timestamps": end_timestamps,
                "nums": len(end_timestamps),
            }

            print(meta)

            items = {}
            for end_timestamp, factor in zip(end_timestamps, factors):
                items[end_timestamp] = {
                    "factor": factor
                }

            count = {
                "count": count_embed_ind,
            }

            info = {
                "meta": meta,
                "items": items,
                "count": count
            }

            self.logger.info(f"| State info")
            if self.is_main_process:
                save_joblib(info, os.path.join(self.exp_path, f"state.joblib"))
                save_json(count, os.path.join(self.exp_path, f"count.json"))

    def train(self):
        self.logger.info("| Start training and evaluating VAE...")

        if self.best_metric_mode == "min":
            best_metric_value = float("inf")
        elif self.best_metric_mode == "max":
            best_metric_value = float("-inf")
        else:
            raise ValueError(f"Unsupported best_metric_mode: {self.best_metric_mode}")

        for epoch in range(self.start_epoch, self.num_training_epochs + 1):
            train_stats = self.run_step(epoch, mode="train")
            valid_stats = self.run_step(epoch, mode="valid")
            test_stats = self.run_test(epoch, mode="test")

            if self.best_metric not in valid_stats:
                raise KeyError(
                    f"best_metric={self.best_metric} not found in valid_stats. "
                    f"Available metrics: {sorted(valid_stats.keys())}"
                )
            metric = valid_stats[self.best_metric]

            log_stats = {"epoch": epoch}
            log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
            log_stats.update({f"valid_{k}": v for k, v in valid_stats.items()})
            log_stats.update({f"test_{k}": v for k, v in test_stats.items()})

            if self.is_main_process:
                with open(os.path.join(self.exp_path, "train_log.txt"), "a", ) as f:
                    f.write(json.dumps(log_stats) + "\n")

            if epoch % self.checkpoint_period == 0:
                self.save_checkpoint(epoch)

            if self.best_metric_mode == "min":
                is_best = metric < best_metric_value
            else:
                is_best = metric > best_metric_value

            if is_best:
                best_metric_value = metric
                self.save_checkpoint(epoch, if_best=True)

    def test(self, checkpoint_path: str = None):
        self.logger.info("| Start testing VAE...")

        if checkpoint_path is None:
            best_checkpoint_path = os.path.join(self.checkpoint_path, "best.pth")
            if os.path.exists(best_checkpoint_path):
                epoch = self.load_checkpoint(if_best=True)
            else:
                self.logger.info("| Best checkpoint not found. Load the last checkpoint.")
                epoch = self.load_checkpoint()
        else:
            epoch = self.load_checkpoint(checkpoint_file=checkpoint_path)

        log_stats = {"epoch": epoch}
        eval_model, eval_model_name = self._get_eval_model()
        log_stats["eval_model"] = eval_model_name

        train_stats = self.run_step(epoch,
                                    mode="train",
                                    if_update=False,
                                    if_use_writer=False,
                                    if_use_wandb=False,
                                    if_plot=False,
                                    model_override=eval_model)
        valid_stats = self.run_step(epoch,
                                    mode="valid",
                                    if_use_writer=False,
                                    if_use_wandb=False,
                                    if_plot=False,
                                    model_override=eval_model)

        test_stats = self.run_test(epoch,
                                   mode="test",
                                   if_use_writer=False,
                                   if_use_wandb=False,
                                   if_plot=False,
                                   save_predictions=True,
                                   model_override=eval_model,
                                   eval_model_name=eval_model_name)

        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
        log_stats.update({f"valid_{k}": v for k, v in valid_stats.items()})
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})

        if self.is_main_process:
            with open(os.path.join(self.exp_path, "test_log.txt"), "w", ) as f:
                f.write(json.dumps(log_stats) + "\n")
            with open(os.path.join(self.exp_path, f"test_log_{eval_model_name}.txt"), "w", ) as f:
                f.write(json.dumps(log_stats) + "\n")

        self.logger.info("| Test finished.")

    def state(self, checkpoint_path: str = None):
        self.logger.info("| Start state VAE...")

        if checkpoint_path is None:
            best_checkpoint_path = os.path.join(self.checkpoint_path, "best.pth")
            if os.path.exists(best_checkpoint_path):
                epoch = self.load_checkpoint(if_best=True)
            else:
                self.logger.info("| Best checkpoint not found. Load the last checkpoint.")
                epoch = self.load_checkpoint()
        else:
            epoch = self.load_checkpoint(checkpoint_file=checkpoint_path)

        log_stats = {"epoch": epoch}

        self.run_state(epoch)

        if self.is_main_process:
            with open(os.path.join(self.exp_path, "state_log.txt"), "w", ) as f:
                f.write(json.dumps(log_stats) + "\n")

        self.logger.info("| State finished.")
