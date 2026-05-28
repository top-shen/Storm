import argparse
import json
import os
import pathlib
import sys

import pandas as pd

root = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(root)

from mmengine import DictAction

from storm.config import build_config
from storm.log import logger
from storm.qlib_adapter import calc_prediction_metrics, build_prediction_payload_from_frame
from storm.utils import assemble_project_path, load_json, save_joblib, load_joblib


def get_args_parser():
    parser = argparse.ArgumentParser(description="Train/test Qlib-native Alpha158 LGBM baseline")
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "exp", "predict", "predict_day_dj30_17_qlib_alpha158_lgbm.py"),
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--no_train", action="store_false", dest="train")
    parser.set_defaults(train=True)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no_test", action="store_false", dest="test")
    parser.set_defaults(test=True)
    parser.add_argument("--checkpoint_path_override", type=str, default=None)
    return parser


def _init_qlib(config):
    try:
        import qlib
        from qlib.constant import REG_CN
    except ImportError as exc:
        raise ImportError("qlib is required for train_qlib_alpha158_lgbm.py.") from exc

    provider_uri = config.qlib_init.get("provider_uri", None)
    region = config.qlib_init.get("region", "cn")
    region_value = REG_CN if str(region).lower() == "cn" else region
    if provider_uri:
        qlib.init(provider_uri=provider_uri, region=region_value)
    else:
        qlib.init(region=region_value)


def _load_instruments(config):
    instruments = config.data.get("instruments", None)
    if instruments:
        return instruments

    instruments_path = config.data.get("instruments_path", None)
    if not instruments_path:
        raise ValueError("Set either data.instruments or data.instruments_path in the config.")

    payload = load_json(assemble_project_path(instruments_path))
    if isinstance(payload, dict):
        return list(payload.keys())
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return [item["symbol"] for item in payload]
        return payload
    raise ValueError(f"Unsupported instruments payload from {instruments_path}: {type(payload)}")


def _build_dataset(config):
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    handler = Alpha158(
        instruments=_load_instruments(config),
        start_time=config.data.start_time,
        end_time=config.data.end_time,
        fit_start_time=config.data.fit_start_time,
        fit_end_time=config.data.fit_end_time,
        label=config.label,
    )
    dataset = DatasetH(handler=handler, segments=config.segments)
    return dataset


def _normalize_label_frame(label_df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    if isinstance(label_df.columns, pd.MultiIndex):
        target = ("label", label_column)
        if target in label_df.columns:
            return label_df[[target]]

        label_cols = [col for col in label_df.columns if col[0] == "label"]
        if len(label_cols) == 1:
            normalized = label_df[[label_cols[0]]].copy()
            normalized.columns = pd.MultiIndex.from_tuples([target])
            return normalized
    else:
        if label_column in label_df.columns:
            series = label_df[label_column]
        elif label_df.shape[1] == 1:
            series = label_df.iloc[:, 0]
        else:
            raise KeyError(f"Could not find label column {label_column!r} in {list(label_df.columns)}")

        return pd.DataFrame(
            {("label", label_column): series},
            index=label_df.index,
        )

    raise KeyError(f"Could not find label column {label_column!r} in {list(label_df.columns)}")


def _segment_label_frame(dataset, segment, label_column: str):
    from qlib.data.dataset.handler import DataHandlerLP

    label_df = dataset.prepare(segment, col_set=["label"], data_key=DataHandlerLP.DK_L)
    return _normalize_label_frame(label_df, label_column=label_column)


def _save_predictions(exp_path, split, pred_series, label_df, label_column="LABEL0"):
    _, payload = build_prediction_payload_from_frame(label_df, pred_series, label_column=label_column)
    save_joblib(payload, os.path.join(exp_path, f"{split}_predictions.joblib"))


def _evaluate(model, dataset, exp_path, label_column="LABEL0", splits=("train", "valid", "test")):
    stats = {}
    for split in splits:
        pred = model.predict(dataset, segment=split)
        label_df = _segment_label_frame(dataset, split, label_column=label_column)
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        pred.name = "score"
        metrics = calc_prediction_metrics(label_df, pred, label_column=label_column)
        _save_predictions(exp_path, split, pred, label_df, label_column=label_column)
        stats.update({f"{split}_{key}": value for key, value in metrics.items()})
    return stats


def main(args):
    config = build_config(assemble_project_path(args.config), args)
    logger.init_logger(config.log_path)

    os.makedirs(config.exp_path, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)

    _init_qlib(config)
    dataset = _build_dataset(config)

    model_path = os.path.join(config.checkpoint_path, config.model_file)

    try:
        from qlib.contrib.model.gbdt import LGBModel
    except ImportError as exc:
        raise ImportError("qlib LGBModel is unavailable. Please ensure pyqlib and lightgbm are installed.") from exc

    if args.train:
        model = LGBModel(**config.model)
        model.fit(dataset)
        save_joblib(model, model_path)
        logger.info(f"| Saved Qlib Alpha158 LGBM model: {model_path}")

        stats = _evaluate(model, dataset, config.exp_path, label_column=config.label_column, splits=("train", "valid"))
        with open(os.path.join(config.exp_path, "train_log.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        logger.info(f"| Qlib Alpha158 LGBM train/valid stats: {stats}")

    if args.test:
        ckpt = args.checkpoint_path_override or model_path
        model = load_joblib(ckpt)
        if model is None:
            raise FileNotFoundError(f"Qlib Alpha158 LGBM checkpoint not found: {ckpt}")
        logger.info(f"| Load Qlib Alpha158 LGBM model: {ckpt}")

        stats = _evaluate(model, dataset, config.exp_path, label_column=config.label_column)
        with open(os.path.join(config.exp_path, "test_log.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        logger.info(f"| Qlib Alpha158 LGBM test stats: {stats}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
