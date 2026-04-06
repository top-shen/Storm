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
from storm.qlib_adapter import build_windowed_qlib_dataframe, calc_prediction_metrics
from storm.utils import assemble_project_path, save_joblib, load_joblib


def get_args_parser():
    parser = argparse.ArgumentParser(description="Train/test Qlib LGBM baseline for Storm dj30_17")
    parser.add_argument("--config", default=os.path.join("configs", "exp", "predict", "predict_day_dj30_17_qlib_lgbm.py"))
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
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
        raise ImportError("qlib is required for train_qlib_lgbm.py. Please install pyqlib in the runtime environment.") from exc

    provider_uri = config.qlib_init.get("provider_uri", None)
    region = config.qlib_init.get("region", "cn")
    if provider_uri:
        qlib.init(provider_uri=provider_uri, region=REG_CN if region.lower() == "cn" else region)
    else:
        qlib.init(region=REG_CN if region.lower() == "cn" else region)


def _build_dataset(config):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    df = build_windowed_qlib_dataframe(
        data_path=config.data.data_path,
        assets_path=config.data.assets_path,
        feature_columns=config.feature_columns,
        history_timestamps=config.history_timestamps,
        label_column=config.label_column,
        start_time=config.data.start_time,
        end_time=config.data.end_time,
        include_asset_identity=getattr(config, "include_asset_identity", True),
    )

    handler = DataHandlerLP.from_df(df)
    dataset = DatasetH(handler=handler, segments=config.segments)
    return dataset, df


def _segment_label_frame(dataset, segment):
    from qlib.data.dataset.handler import DataHandlerLP
    return dataset.prepare(segment, col_set=["label"], data_key=DataHandlerLP.DK_L)


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


def _evaluate(model, dataset, exp_path):
    stats = {}
    for split in ["train", "valid", "test"]:
        pred = model.predict(dataset, segment=split)
        label_df = _segment_label_frame(dataset, split)
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        pred.name = "score"
        metrics = calc_prediction_metrics(label_df, pred, label_column="ret1")
        _save_predictions(exp_path, split, pred, label_df)
        stats.update({f"{split}_{k}": v for k, v in metrics.items()})
    return stats


def main(args):
    config = build_config(assemble_project_path(args.config), args)
    logger.init_logger(config.log_path)

    os.makedirs(config.exp_path, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)

    _init_qlib(config)
    dataset, qlib_df = _build_dataset(config)

    logger.info(f"| Qlib windowed dataframe shape: {qlib_df.shape}")
    logger.info(f"| Qlib dataframe index range: {qlib_df.index.get_level_values(0).min()} -> {qlib_df.index.get_level_values(0).max()}")

    model_path = os.path.join(config.checkpoint_path, config.model_file)

    try:
        from qlib.contrib.model.gbdt import LGBModel
    except ImportError as exc:
        raise ImportError("qlib LGBModel is unavailable. Please ensure pyqlib and lightgbm are installed in the runtime environment.") from exc

    if args.train:
        model = LGBModel(**config.model)
        model.fit(dataset)
        save_joblib(model, model_path)
        logger.info(f"| Saved Qlib LGBM model: {model_path}")

        stats = _evaluate(model, dataset, config.exp_path)
        with open(os.path.join(config.exp_path, "train_log.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        logger.info(f"| Qlib LGBM train/test stats: {stats}")

    if args.test:
        ckpt = args.checkpoint_path_override or model_path
        model = load_joblib(ckpt)
        if model is None:
            raise FileNotFoundError(f"Qlib LGBM checkpoint not found: {ckpt}")
        logger.info(f"| Load Qlib LGBM model: {ckpt}")

        stats = _evaluate(model, dataset, config.exp_path)
        with open(os.path.join(config.exp_path, "test_log.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        logger.info(f"| Qlib LGBM test stats: {stats}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
