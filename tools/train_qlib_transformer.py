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
from storm.qlib_adapter import apply_qlib_like_processors, build_qlib_dataframe, calc_prediction_metrics
from storm.utils import assemble_project_path, load_joblib, save_joblib


def get_args_parser():
    parser = argparse.ArgumentParser(description="Train/test Qlib Transformer baseline for Storm dj30_17")
    parser.add_argument("--config", default=os.path.join("configs", "exp", "predict", "predict_day_dj30_17_qlib_transformer.py"))
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
        raise ImportError("qlib is required for train_qlib_transformer.py. Please install pyqlib in the runtime environment.") from exc

    provider_uri = config.qlib_init.get("provider_uri", None)
    region = config.qlib_init.get("region", "cn")
    if provider_uri:
        qlib.init(provider_uri=provider_uri, region=REG_CN if region.lower() == "cn" else region)
    else:
        qlib.init(region=REG_CN if region.lower() == "cn" else region)


def _build_dataset(config):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    raw_df = build_qlib_dataframe(
        data_path=config.data.data_path,
        assets_path=config.data.assets_path,
        feature_columns=config.feature_columns,
        label_column=config.label_column,
        start_time=config.data.start_time,
        end_time=config.data.end_time,
    )

    fit_start_time, fit_end_time = config.segments["train"]
    if getattr(config, "processor", None) and config.processor.get("use_qlib_processors", False):
        model_df = apply_qlib_like_processors(
            raw_df,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            label_column=config.label_column,
            clip_outlier=config.processor.get("clip_outlier", True),
        )
    else:
        model_df = raw_df

    handler = DataHandlerLP.from_df(model_df)
    dataset = DatasetH(handler=handler, segments=config.segments)
    return dataset, raw_df, model_df


def _segment_label_frame(qlib_df, config, segment):
    start_time, end_time = config.segments[segment]
    datetimes = qlib_df.index.get_level_values(0)
    mask = (datetimes >= pd.Timestamp(start_time)) & (datetimes <= pd.Timestamp(end_time))
    return qlib_df.loc[mask, [("label", config.label_column)]].copy()


def _save_predictions(exp_path, split, pred_series, label_df):
    pred_frame = pred_series.to_frame(name="score")
    pred_frame.columns = pd.MultiIndex.from_tuples([("prediction", "score")])
    merged = pd.concat([label_df, pred_frame], axis=1, join="inner")
    payload = {
        "end_timestamp": [idx[0].strftime("%Y-%m-%d") for idx in merged.index],
        "asset": [idx[1] for idx in merged.index],
        "pred_label": merged[("prediction", "score")].to_numpy(),
        "true_label": merged[("label", "ret1")].to_numpy(),
    }
    save_joblib(payload, os.path.join(exp_path, f"{split}_predictions.joblib"))


def _evaluate(model, dataset, raw_qlib_df, config, exp_path):
    stats = {}
    for split in ["train", "valid", "test"]:
        pred = model.predict(dataset, segment=split)
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        pred.name = "score"
        label_df = _segment_label_frame(raw_qlib_df, config, split)
        metrics = calc_prediction_metrics(label_df, pred, label_column=config.label_column)
        _save_predictions(exp_path, split, pred, label_df)
        stats.update({f"{split}_{k}": v for k, v in metrics.items()})
    return stats


def main(args):
    config = build_config(assemble_project_path(args.config), args)
    logger.init_logger(config.log_path)

    os.makedirs(config.exp_path, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)

    _init_qlib(config)
    dataset, raw_qlib_df, model_qlib_df = _build_dataset(config)

    logger.info(f"| Raw Qlib dataframe shape: {raw_qlib_df.shape}")
    logger.info(f"| Model Qlib dataframe shape: {model_qlib_df.shape}")
    logger.info(
        f"| Qlib dataframe index range: {raw_qlib_df.index.get_level_values(0).min()} -> {raw_qlib_df.index.get_level_values(0).max()}"
    )

    model_path = os.path.join(config.checkpoint_path, config.model_file)

    try:
        from qlib.contrib.model.pytorch_transformer import TransformerModel
    except ImportError as exc:
        raise ImportError(
            "qlib pytorch_transformer is unavailable. Please ensure pyqlib and torch are installed in the runtime environment."
        ) from exc

    if args.train:
        model = TransformerModel(**config.model)
        model.fit(dataset, save_path=model_path)
        save_joblib(model, model_path + '.joblib')
        logger.info(f"| Saved Qlib Transformer model: {model_path}")

        stats = _evaluate(model, dataset, raw_qlib_df, config, config.exp_path)
        with open(os.path.join(config.exp_path, "train_log.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        logger.info(f"| Qlib Transformer train/test stats: {stats}")

    if args.test:
        ckpt = args.checkpoint_path_override or model_path + '.joblib'
        model = load_joblib(ckpt)
        if model is None:
            raise FileNotFoundError(f"Qlib Transformer checkpoint not found: {ckpt}")
        logger.info(f"| Load Qlib Transformer model: {ckpt}")

        stats = _evaluate(model, dataset, raw_qlib_df, config, config.exp_path)
        with open(os.path.join(config.exp_path, "test_log.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        logger.info(f"| Qlib Transformer test stats: {stats}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
