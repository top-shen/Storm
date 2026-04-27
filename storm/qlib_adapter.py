import os
from typing import Dict, Iterable, List

import numpy as np
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
    timestamp_alignment: str = "intersection",
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

    if timestamp_alignment == "intersection":
        common_index = None
        for asset in assets:
            asset_index = frames[asset].index
            common_index = asset_index if common_index is None else common_index.intersection(asset_index)
        common_index = common_index.sort_values()
        for asset in assets:
            frames[asset] = frames[asset].loc[common_index].copy()
    elif timestamp_alignment == "check":
        base_asset = assets[0]
        base_index = frames[base_asset].index
        for asset in assets[1:]:
            asset_index = frames[asset].index
            if not asset_index.equals(base_index):
                missing = base_index.difference(asset_index)
                extra = asset_index.difference(base_index)
                raise ValueError(
                    f"Timestamp misalignment detected between {base_asset} and {asset}. "
                    f"missing={len(missing)}, extra={len(extra)}"
                )
    elif timestamp_alignment == "none":
        pass
    else:
        raise ValueError(
            f"Unsupported timestamp_alignment={timestamp_alignment}. "
            "Expected one of: intersection, check, none"
        )

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


def _safe_div_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)


def _rank_ic_single_np(pred_row: np.ndarray, futr_row: np.ndarray) -> float:
    pred_row = np.asarray(pred_row, dtype=np.float64)
    futr_row = np.asarray(futr_row, dtype=np.float64)

    if pred_row.size <= 1 or futr_row.size <= 1:
        return 0.0
    if np.unique(pred_row).size < 2 or np.unique(futr_row).size < 2:
        return 0.0

    pred_rank = pd.Series(pred_row).rank(method="average").to_numpy(dtype=np.float64)
    futr_rank = pd.Series(futr_row).rank(method="average").to_numpy(dtype=np.float64)

    pred_centered = pred_rank - pred_rank.mean()
    futr_centered = futr_rank - futr_rank.mean()

    pred_std = pred_centered.std(ddof=0)
    futr_std = futr_centered.std(ddof=0)
    if pred_std <= 0 or futr_std <= 0:
        return 0.0

    covariance = np.mean(pred_centered * futr_centered)
    return float(covariance / (pred_std * futr_std))


def _calc_rankic_np(preds: List[np.ndarray], futrs: List[np.ndarray]) -> np.ndarray:
    values = [_rank_ic_single_np(pred_row.reshape(-1), futr_row.reshape(-1)) for pred_row, futr_row in zip(preds, futrs)]
    return np.asarray(values, dtype=np.float64)


def _direction_metrics_np(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    pred_up = np.asarray(y_pred, dtype=np.float64) > 0
    true_up = np.asarray(y_true, dtype=np.float64) > 0

    tp = float(np.logical_and(pred_up, true_up).sum())
    tn = float(np.logical_and(~pred_up, ~true_up).sum())
    fp = float(np.logical_and(pred_up, ~true_up).sum())
    fn = float(np.logical_and(~pred_up, true_up).sum())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0

    return {
        "ACC": round(float(acc), 6),
        "MCC": round(float(mcc), 6),
    }


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
    direction_metrics = _direction_metrics_np(y_pred.to_numpy(), y_true.to_numpy())

    pred_groups = []
    true_groups = []
    for _, group in merged.groupby(level=0):
        if len(group) < 2:
            continue
        pred_groups.append(group[("prediction", pred_name)].to_numpy(dtype=np.float64)[None, :])
        true_groups.append(group[("label", label_column)].to_numpy(dtype=np.float64)[None, :])

    if pred_groups:
        rankic_values = _calc_rankic_np(pred_groups, true_groups)
        rankic = float(rankic_values.mean())
        rankic_std = float(rankic_values.std())
        rankicir = float(rankic / rankic_std) if rankic_values.size > 1 and rankic_std > 0 else 0.0
    else:
        rankic = 0.0
        rankicir = 0.0

    return {
        "MSE": round(mse, 6),
        "ACC": direction_metrics["ACC"],
        "MCC": direction_metrics["MCC"],
        "RANKIC": round(rankic, 6),
        "RANKICIR": round(rankicir, 6),
    }


def _direction_label(value: float) -> str:
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def build_prediction_payload(
    end_timestamps: Iterable,
    assets: Iterable[str],
    pred_values: Iterable[float],
    true_values: Iterable[float],
):
    end_timestamps = [str(ts) for ts in end_timestamps]
    assets = [str(asset) for asset in assets]
    pred_values = [float(v) for v in pred_values]
    true_values = [float(v) for v in true_values]

    pred_direction = [_direction_label(v) for v in pred_values]
    true_direction = [_direction_label(v) for v in true_values]

    payload = {
        "end_timestamp": end_timestamps,
        "asset": assets,
        "pred_label": pred_values,
        "true_label": true_values,
        "pred_direction": pred_direction,
        "true_direction": true_direction,
    }

    rows = pd.DataFrame(
        {
            "end_timestamp": end_timestamps,
            "asset": assets,
            "pred_label": pred_values,
            "true_label": true_values,
            "pred_direction": pred_direction,
            "true_direction": true_direction,
        }
    )

    rankings_by_date = []
    for date, group in rows.groupby("end_timestamp", sort=True):
        ordered = group.sort_values(["pred_label", "asset"], ascending=[False, True]).reset_index(drop=True)
        ranking = []
        for rank, item in enumerate(ordered.itertuples(index=False), start=1):
            ranking.append(
                {
                    "rank": rank,
                    "asset": item.asset,
                    "pred_label": float(item.pred_label),
                    "true_label": float(item.true_label),
                    "pred_direction": item.pred_direction,
                    "true_direction": item.true_direction,
                }
            )
        rankings_by_date.append({"end_timestamp": date, "ranking": ranking})

    payload["rankings_by_date"] = rankings_by_date
    return payload


def build_prediction_payload_from_frame(
    label_df: pd.DataFrame,
    pred_series: pd.Series,
    label_column: str = "ret1",
):
    pred_name = pred_series.name if pred_series.name is not None else "score"
    pred_frame = pred_series.to_frame(name=pred_name)
    pred_frame.columns = pd.MultiIndex.from_tuples([("prediction", pred_name)])
    merged = pd.concat([label_df, pred_frame], axis=1, join="inner")

    payload = build_prediction_payload(
        [idx[0].strftime("%Y-%m-%d") for idx in merged.index],
        [idx[1] for idx in merged.index],
        merged[("prediction", pred_name)].to_numpy(),
        merged[("label", label_column)].to_numpy(),
    )
    return merged, payload
