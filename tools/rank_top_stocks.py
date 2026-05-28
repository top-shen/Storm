import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storm.utils import load_joblib


DEFAULT_EXPERIMENTS = [
    "workdir/predict_day_dj30_17_qlib_lgbm",
    "workdir/predict_day_dj30_17_qlib_lstm",
    "workdir/predict_day_dj30_17_storm_lstm",
    "workdir/predict_day_dj30_17_storm_transformer_s3",
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rank stocks by saved prediction scores. This is a model signal report, "
            "not financial advice."
        )
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment directories containing <split>_predictions.joblib.",
    )
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--date", default="latest", help="Date to rank, or 'latest'.")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Also report an average-rank ensemble across all available experiments.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _load_prediction_frame(exp_dir: Path, split: str) -> pd.DataFrame:
    path = exp_dir / f"{split}_predictions.joblib"
    payload = load_joblib(str(path))
    if payload is None:
        raise FileNotFoundError(f"prediction file not found: {path}")

    required = ("end_timestamp", "asset", "pred_label", "true_label")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")

    frame = pd.DataFrame(
        {
            "end_timestamp": pd.to_datetime(payload["end_timestamp"]).strftime("%Y-%m-%d"),
            "asset": [str(asset) for asset in payload["asset"]],
            "pred_label": np.asarray(payload["pred_label"], dtype=np.float64),
            "true_label": np.asarray(payload["true_label"], dtype=np.float64),
        }
    )
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred_label"])


def _select_date(frame: pd.DataFrame, date_arg: str) -> str:
    dates = sorted(frame["end_timestamp"].unique())
    if not dates:
        raise ValueError("prediction frame has no dates")
    if date_arg == "latest":
        return dates[-1]
    date = pd.Timestamp(date_arg).strftime("%Y-%m-%d")
    if date not in set(dates):
        raise ValueError(f"date {date} not found; available range: {dates[0]} -> {dates[-1]}")
    return date


def _rank_for_date(frame: pd.DataFrame, date: str, topk: int):
    day = frame[frame["end_timestamp"] == date].copy()
    day = day.sort_values(["pred_label", "asset"], ascending=[False, True]).reset_index(drop=True)
    day["rank"] = np.arange(1, len(day) + 1)
    top = day.head(topk)
    return [
        {
            "rank": int(row.rank),
            "asset": row.asset,
            "pred_label": float(row.pred_label),
            "true_label": float(row.true_label),
        }
        for row in top.itertuples(index=False)
    ], day


def _experiment_name(exp_dir: Path) -> str:
    return exp_dir.name


def _ensemble_rank(day_frames: list[tuple[str, pd.DataFrame]], topk: int):
    pieces = []
    for name, frame in day_frames:
        item = frame[["asset", "rank", "pred_label"]].copy()
        item = item.rename(columns={"rank": f"{name}_rank", "pred_label": f"{name}_score"})
        pieces.append(item.set_index("asset"))

    merged = pd.concat(pieces, axis=1, join="inner")
    rank_cols = [col for col in merged.columns if col.endswith("_rank")]
    score_cols = [col for col in merged.columns if col.endswith("_score")]
    merged["avg_rank"] = merged[rank_cols].mean(axis=1)
    merged["avg_score_z"] = merged[score_cols].apply(
        lambda row: np.mean(
            [
                (value - merged[col].mean()) / (merged[col].std(ddof=0) or 1.0)
                for value, col in zip(row, score_cols)
            ]
        ),
        axis=1,
    )
    merged = merged.sort_values(["avg_rank", "avg_score_z"], ascending=[True, False]).reset_index()
    return [
        {
            "rank": idx + 1,
            "asset": row.asset,
            "avg_rank": float(row.avg_rank),
            "avg_score_z": float(row.avg_score_z),
        }
        for idx, row in enumerate(merged.head(topk).itertuples(index=False))
    ]


def main():
    args = _parse_args()
    results = {
        "split": args.split,
        "topk": args.topk,
        "note": "Model-score ranking only; not financial advice.",
        "experiments": [],
    }
    day_frames = []
    selected_dates = []

    for exp_arg in args.experiments:
        exp_dir = _resolve_path(exp_arg)
        frame = _load_prediction_frame(exp_dir, args.split)
        date = _select_date(frame, args.date)
        top, day_frame = _rank_for_date(frame, date, args.topk)
        name = _experiment_name(exp_dir)
        selected_dates.append(date)
        day_frames.append((name, day_frame))
        results["experiments"].append(
            {
                "experiment": name,
                "date": date,
                "top": top,
            }
        )

    if args.ensemble:
        if len(set(selected_dates)) != 1:
            raise ValueError(f"Ensemble requires the same selected date, got {sorted(set(selected_dates))}")
        results["ensemble"] = {
            "date": selected_dates[0],
            "top": _ensemble_rank(day_frames, args.topk),
        }

    text = json.dumps(results, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        out_path = _resolve_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
