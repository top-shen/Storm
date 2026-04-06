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


def _load_asset_frames(data_path: str,
                       assets_path: str,
                       feature_columns: Iterable[str],
                       label_column: str,
                       start_time: str = None,
                       end_time: str = None):
    data_path = assemble_project_path(data_path)
    assets = _load_assets(assets_path)
    feature_columns = list(feature_columns)

    frames = {}
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
        frames[asset] = df[use_columns].copy()
    return assets, frames, feature_columns


def build_windowed_qlib_dataframe(data_path: str,
                                  assets_path: str,
                                  feature_columns: Iterable[str],
                                  history_timestamps: int,
                                  label_column: str = "ret1",
                                  start_time: str = None,
                                  end_time: str = None) -> pd.DataFrame:
    assets, frames, feature_columns = _load_asset_frames(
        data_path=data_path,
        assets_path=assets_path,
        feature_columns=feature_columns,
        label_column=label_column,
        start_time=start_time,
        end_time=end_time,
    )

    first_asset = assets[0]
    first_df = frames[first_asset]

    flattened_columns = []
    for lag in range(history_timestamps):
        offset = history_timestamps - 1 - lag
        for feature in feature_columns:
            flattened_columns.append(f"{feature}_t-{offset}")

    rows = []
    index = []

    for i in range(history_timestamps, len(first_df)):
        end_index = i - 1
        end_timestamp = first_df.index[end_index]

        for asset in assets:
            asset_df = frames[asset]
            window = asset_df.iloc[i - history_timestamps:i][feature_columns]
            if len(window) != history_timestamps:
                continue

            label_value = asset_df.iloc[end_index][label_column]
            if pd.isna(label_value):
                continue

            rows.append(list(window.to_numpy(dtype="float32").reshape(-1)) + [float(label_value)])
            index.append((end_timestamp, asset))

    columns = pd.MultiIndex.from_tuples(
        [("feature", col) for col in flattened_columns] + [("label", label_column)]
    )
    qlib_df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=["datetime", "instrument"]), columns=columns)
    qlib_df = qlib_df.replace([float("inf"), float("-inf")], pd.NA)
    qlib_df = qlib_df.dropna(subset=[("label", label_column)])
    return qlib_df.sort_index()


def calc_prediction_metrics(label_df: pd.DataFrame, pred_series: pd.Series, label_column: str = "ret1") -> Dict[str, float]:
    pred_name = pred_series.name if pred_series.name is not None else "score"
    pred_frame = pred_series.to_frame(name=pred_name)
    merged = label_df.join(pred_frame, how="inner")

    y_true = merged[("label", label_column)].astype(float)
    y_pred = merged[pred_name].astype(float)
    mse = float(((y_true - y_pred) ** 2).mean())

    rankics = []
    for _, group in merged.groupby(level=0):
        if len(group) < 2:
            continue
        rankic = group[("label", label_column)].corr(group[pred_name], method="spearman")
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
