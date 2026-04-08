import argparse
import copy
import json
import os
import pathlib
import sys
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from mmengine import Config, DictAction
from torch.utils.data import DataLoader

root = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(root)

from storm.config import build_config
from storm.log import logger
from storm.models.storm_lstm import StormLSTM
from storm.qlib_adapter import calc_prediction_metrics
from storm.registry import COLLATE_FN, DATASET
from storm.utils import assemble_project_path, convert_int_to_timestamp, get_model_numel, save_joblib


def get_args_parser():
    parser = argparse.ArgumentParser(description="Train/test STORM-style LSTM baseline for dj30_17")
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "exp", "predict", "predict_day_dj30_17_storm_lstm.py"),
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--no_train", action="store_false", dest="train")
    parser.set_defaults(train=True)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no_test", action="store_false", dest="test")
    parser.set_defaults(test=True)
    parser.add_argument("--checkpoint_path_override", type=str, default=None)
    return parser


def _prepare_data_configs(config) -> Tuple[Dict, Dict, Dict, Dict]:
    base_config = Config.fromfile(assemble_project_path(config.storm_data_config))

    train_dataset_cfg = copy.deepcopy(base_config.train_dataset)
    valid_dataset_cfg = copy.deepcopy(base_config.valid_dataset)
    test_dataset_cfg = copy.deepcopy(base_config.test_dataset)
    collate_fn_cfg = copy.deepcopy(base_config.collate_fn)

    for dataset_cfg in [train_dataset_cfg, valid_dataset_cfg, test_dataset_cfg]:
        dataset_cfg["exp_path"] = config.exp_path

    return train_dataset_cfg, valid_dataset_cfg, test_dataset_cfg, collate_fn_cfg


def _build_dataloaders(config):
    train_dataset_cfg, valid_dataset_cfg, test_dataset_cfg, collate_fn_cfg = _prepare_data_configs(config)

    collate_fn = COLLATE_FN.build(collate_fn_cfg)

    train_dataset = DATASET.build(train_dataset_cfg)
    valid_dataset = DATASET.build(valid_dataset_cfg)
    test_dataset = DATASET.build(test_dataset_cfg)

    logger.info(f"| Train dataset: \n{train_dataset}")
    logger.info(f"| Valid dataset: \n{valid_dataset}")
    logger.info(f"| Test dataset: \n{test_dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, train_eval_loader, valid_loader, test_loader


def _extract_inputs_and_targets(batch, device):
    history = batch["history"]
    features = history["features"]
    labels = history["labels"]

    if features.dim() == 5:
        features = features.squeeze(1)
    if labels.dim() == 5:
        labels = labels.squeeze(1)

    features = features.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.float32)
    target = labels[:, -1, :, 0]

    return features, target


def _build_metric_inputs(
    end_timestamps: torch.Tensor,
    assets: List[List[str]],
    pred: torch.Tensor,
    target: torch.Tensor,
    label_column: str,
):
    end_timestamp_values: List[str] = []
    asset_values: List[str] = []
    pred_values: List[float] = []
    true_values: List[float] = []

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    end_timestamp_np = end_timestamps.detach().cpu().numpy()

    for batch_idx, raw_timestamp in enumerate(end_timestamp_np):
        date = convert_int_to_timestamp(int(raw_timestamp))
        batch_assets = assets[batch_idx]
        for asset_idx, asset in enumerate(batch_assets):
            end_timestamp_values.append(date.strftime("%Y-%m-%d"))
            asset_values.append(asset)
            pred_values.append(float(pred_np[batch_idx, asset_idx]))
            true_values.append(float(target_np[batch_idx, asset_idx]))

    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(end_timestamp_values), asset_values],
        names=["datetime", "instrument"],
    )
    label_df = pd.DataFrame({("label", label_column): true_values}, index=index)
    pred_series = pd.Series(pred_values, index=index, name="score")

    payload = {
        "end_timestamp": end_timestamp_values,
        "asset": asset_values,
        "pred_label": pred_values,
        "true_label": true_values,
    }

    return label_df, pred_series, payload


def _run_epoch(model, dataloader, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        features, target = _extract_inputs_and_targets(batch, device)
        pred = model(features)
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


def _evaluate(model, dataloader, device, label_column, save_path=None):
    model.eval()

    total_squared_error = 0.0
    total_count = 0

    all_label_frames = []
    all_pred_series = []
    payload = {
        "end_timestamp": [],
        "asset": [],
        "pred_label": [],
        "true_label": [],
    }

    with torch.no_grad():
        for batch in dataloader:
            features, target = _extract_inputs_and_targets(batch, device)
            pred = model(features)

            total_squared_error += float(torch.sum((pred - target) ** 2).item())
            total_count += int(target.numel())

            label_df, pred_series, batch_payload = _build_metric_inputs(
                end_timestamps=batch["history"]["end_timestamp"],
                assets=batch["asset"],
                pred=pred,
                target=target,
                label_column=label_column,
            )
            all_label_frames.append(label_df)
            all_pred_series.append(pred_series)

            for key in payload:
                payload[key].extend(batch_payload[key])

    merged_label_df = pd.concat(all_label_frames, axis=0).sort_index()
    merged_pred_series = pd.concat(all_pred_series, axis=0).sort_index()
    metrics = calc_prediction_metrics(merged_label_df, merged_pred_series, label_column=label_column)
    metrics["MSE"] = round(total_squared_error / max(total_count, 1), 6)

    if save_path is not None:
        save_joblib(payload, save_path)

    return metrics


def _save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def _load_checkpoint(model, optimizer, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    return int(state.get("epoch", 0))


def main(args):
    config = build_config(assemble_project_path(args.config), args)
    logger.init_logger(config.log_path)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    logger.info(f"| Device: {device}")

    train_loader, train_eval_loader, valid_loader, test_loader = _build_dataloaders(config)

    model = StormLSTM(
        num_assets=config.num_assets,
        feature_dim=config.feature_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(f"| Model: \n{model}")
    logger.info(f"| Model numel: {model_numel}, trainable: {model_numel_trainable}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model_path = os.path.join(config.checkpoint_path, config.model_file)

    if args.train:
        best_valid_mse = float("inf")
        patience = 0

        train_log_path = os.path.join(config.exp_path, "train_log.txt")
        with open(train_log_path, "w", encoding="utf-8") as _:
            pass

        for epoch in range(1, config.num_epochs + 1):
            train_loss = _run_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_clip=config.grad_clip,
            )
            valid_metrics = _evaluate(
                model=model,
                dataloader=valid_loader,
                device=device,
                label_column=config.label_column,
            )

            log_item = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "valid_MSE": valid_metrics["MSE"],
                "valid_RANKIC": valid_metrics["RANKIC"],
                "valid_RANKICIR": valid_metrics["RANKICIR"],
            }
            with open(train_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_item) + "\n")
            logger.info(f"| Epoch {epoch}: {log_item}")

            if valid_metrics["MSE"] < best_valid_mse:
                best_valid_mse = valid_metrics["MSE"]
                patience = 0
                _save_checkpoint(model, optimizer, epoch, model_path)
                logger.info(f"| Saved best STORM LSTM model: {model_path}")
            else:
                patience += 1
                if patience >= config.early_stop:
                    logger.info(f"| Early stop at epoch {epoch}")
                    break

        _load_checkpoint(model, optimizer=None, checkpoint_path=model_path, device=device)
        final_stats = {}
        for split, loader in [("train", train_eval_loader), ("valid", valid_loader), ("test", test_loader)]:
            metrics = _evaluate(
                model=model,
                dataloader=loader,
                device=device,
                label_column=config.label_column,
                save_path=os.path.join(config.exp_path, f"{split}_predictions.joblib"),
            )
            final_stats.update({f"{split}_{key}": value for key, value in metrics.items()})

        with open(os.path.join(config.exp_path, "best_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(final_stats) + "\n")
        logger.info(f"| STORM LSTM train/best stats: {final_stats}")

    if args.test:
        ckpt = args.checkpoint_path_override or model_path
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"STORM LSTM checkpoint not found: {ckpt}")
        _load_checkpoint(model, optimizer=None, checkpoint_path=ckpt, device=device)
        logger.info(f"| Load STORM LSTM model: {ckpt}")

        test_stats = {}
        for split, loader in [("train", train_eval_loader), ("valid", valid_loader), ("test", test_loader)]:
            metrics = _evaluate(
                model=model,
                dataloader=loader,
                device=device,
                label_column=config.label_column,
                save_path=os.path.join(config.exp_path, f"{split}_predictions.joblib"),
            )
            test_stats.update({f"{split}_{key}": value for key, value in metrics.items()})

        with open(os.path.join(config.exp_path, "test_log.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(test_stats) + "\n")
        logger.info(f"| STORM LSTM test stats: {test_stats}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
