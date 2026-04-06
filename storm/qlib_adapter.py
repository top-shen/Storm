import os
from typing import Dict, Iterable, List

import pandas as pd

from storm.utils import assemble_project_path
from storm.utils import load_json


def _load_assets(assets_path: str) -> List[str]:
    assets = load_json(assemble_project_path(assets_path))
    if isinstance(assets, dict):
        return list(assets.keys())
    if isinstance(assets, list):
        return [asset["symbol"] for asset in assets]
    raise ValueError("Unsupported assets format. Expected a dict or a list of dicts.")


def build_qlib_dataframe(data_path: str,
                         assets_path: str,
                         feature_columns: Iterable[str],
                         label_column: str = "ret1",
                         start_time: str = None,
                         end_time: str = None) -> pd.DataFrame:
    data_path = assemble_project_path(data_path)
    assets = _load_assets(assets_path)
    feature_columns = list(feature_columns)

    frames = []
    for asset in assets:
        csv_path = os.path.join(data_path, f"{asset}.csv")
        df = pd.read_csv(csv_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "datetime"

        if start_time is not None:
            df = df.loc[pd.Timestamp(start_time):]
        if end_time is not None:
            df = df.loc[:pd.Timestamp(end_time)]

        use_columns = feature_columns + [label_column]
        df = df[use_columns].copy()
        df["instrument"] = asset
        df = df.reset_index().set_index(["datetime", "instrument"]).sort_index()

        feature_df = df[feature_columns].copy()
        feature_df.columns = pd.MultiIndex.from_product([["feature"], feature_df.columns])

        label_df = df[[label_column]].copy()
        label_df.columns = pd.MultiIndex.from_product([["label"], label_df.columns])

        frames.append(pd.concat([feature_df, label_df], axis=1))

    qlib_df = pd.concat(frames, axis=0).sort_index()
    qlib_df = qlib_df.replace([float("inf"), float("-inf")], pd.NA)
    qlib_df = qlib_df.dropna(subset=[("label", label_column)])
    return qlib_df


def calc_prediction_metrics(label_df: pd.DataFrame, pred_series: pd.Series, label_column: str = "ret1") -> Dict[str, float]:
    pred_name = pred_series.name if pred_series.name is not None else "score"
    pred_frame = pred_series.to_frame(name=pred_name)
    pred_frame.columns = pd.MultiIndex.from_tuples([("prediction", pred_name)])
    merged = pd.concat([label_df, pred_frame], axis=1, join="inner")

    y_true = merged[("label", label_column)].astype(float)
    y_pred = merged[("prediction", pred_name)].astype(float)
    mse = float(((y_true - y_pred) ** 2).mean())

    rankics = []
    for _, group in merged.groupby(level=0):
        if len(group) < 2:
            continue
        rankic = group[("label", label_column)].corr(group[("prediction", pred_name)], method="spearman")
        if pd.notna(rankic):
            rankics.append(float(rankic))

    if rankics:
        rankic = float(pd.Series(rankics).mean())
        rankic_std = float(pd.Series(rankics).std(ddof=0))
        rankicir = float(rankic / rankic_std) if len(rankics) > 1 and rankic_std > 0 else 0.0
    else:
        rankic = 0.0
        rankicir = 0.0

    return {
        "MSE": round(mse, 6),
        "RANKIC": round(rankic, 6),
        "RANKICIR": round(rankicir, 6),
    }
