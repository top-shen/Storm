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


def _load_asset_frames(
    data_path: str,
    assets_path: str,
    feature_columns: Iterable[str],
    label_column: str,
    start_time: str = None,
    end_time: str = None,
):
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


def build_qlib_dataframe(
    data_path: str,
    assets_path: str,
    feature_columns: Iterable[str],
    label_column: str = "ret1",
    start_time: str = None,
    end_time: str = None,
) -> pd.DataFrame:
    assets, frames, feature_columns = _load_asset_frames(
        data_path=data_path,
        assets_path=assets_path,
        feature_columns=feature_columns,
        label_column=label_column,
        start_time=start_time,
        end_time=end_time,
    )

    rows = []
    for asset in assets:
        frame = frames[asset].copy()
        frame["instrument"] = asset
        frame = frame.reset_index().set_index(["datetime", "instrument"])
        frame.columns = pd.MultiIndex.from_tuples(
            [("feature", col) for col in feature_columns] + [("label", label_column)]
        )
        rows.append(frame)

    qlib_df = pd.concat(rows, axis=0).sort_index()
    qlib_df = qlib_df.replace([float("inf"), float("-inf")], pd.NA)
    qlib_df = qlib_df.dropna(subset=[("label", label_column)])
    return qlib_df


def build_windowed_qlib_dataframe(
    data_path: str,
    assets_path: str,
    feature_columns: Iterable[str],
    history_timestamps: int,
    label_column: str = "ret1",
    start_time: str = None,
    end_time: str = None,
    include_asset_identity: bool = True,
) -> pd.DataFrame:
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

    identity_columns = []
    if include_asset_identity:
        identity_columns = [f"asset_id_{asset}" for asset in assets]

    rows = []
    index = []

    for i in range(history_timestamps, len(first_df)):
        end_index = i - 1
        end_timestamp = first_df.index[end_index]

        for asset_idx, asset in enumerate(assets):
            asset_df = frames[asset]
            window = asset_df.iloc[i - history_timestamps:i][feature_columns]
            if len(window) != history_timestamps:
                continue

            label_value = asset_df.iloc[end_index][label_column]
            if pd.isna(label_value):
                continue

            row = list(window.to_numpy(dtype="float32").reshape(-1))
            if include_asset_identity:
                identity = [0.0] * len(assets)
                identity[asset_idx] = 1.0
                row.extend(identity)

            row.append(float(label_value))
            rows.append(row)
            index.append((end_timestamp, asset))

    columns = pd.MultiIndex.from_tuples(
        [("feature", col) for col in flattened_columns + identity_columns] + [("label", label_column)]
    )
    qlib_df = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["datetime", "instrument"]),
        columns=columns,
    )
    qlib_df = qlib_df.replace([float("inf"), float("-inf")], pd.NA)
    qlib_df = qlib_df.dropna(subset=[("label", label_column)])
    return qlib_df.sort_index()


def apply_qlib_like_processors(
    qlib_df: pd.DataFrame,
    fit_start_time: str,
    fit_end_time: str,
    label_column: str = "ret1",
    clip_outlier: bool = True,
) -> pd.DataFrame:
    processed = qlib_df.copy()

    feature_df = processed["feature"].astype(float)
    feature_df = feature_df.replace([float("inf"), float("-inf")], pd.NA)

    datetimes = processed.index.get_level_values(0)
    fit_mask = (datetimes >= pd.Timestamp(fit_start_time)) & (datetimes <= pd.Timestamp(fit_end_time))
    fit_feature_df = feature_df.loc[fit_mask]

    median = fit_feature_df.median(axis=0)
    mad = (fit_feature_df - median).abs().median(axis=0)
    scale = (mad * 1.4826).replace(0, pd.NA)

    feature_df = (feature_df - median) / scale
    if clip_outlier:
        feature_df = feature_df.clip(-3, 3)
    feature_df = feature_df.fillna(0.0)

    label_df = processed["label"].astype(float).replace([float("inf"), float("-inf")], pd.NA)
    label_df = label_df.dropna(subset=[label_column])
    label_series = label_df[label_column].groupby(level=0).rank(pct=True) - 0.5
    label_df[label_column] = label_series

    processed = pd.concat(
        [
            pd.concat({"feature": feature_df}, axis=1),
            pd.concat({"label": label_df}, axis=1),
        ],
        axis=1,
        join="inner",
    ).sort_index()

    return processed


def calc_prediction_metrics(
    label_df: pd.DataFrame,
    pred_series: pd.Series,
    label_column: str = "ret1",
) -> Dict[str, float]:
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
