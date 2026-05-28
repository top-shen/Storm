import argparse
import json
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(root)

from storm.utils import assemble_project_path, load_json


def _parse_args():
    parser = argparse.ArgumentParser(description="Dump DJ30_17 CSV prices into Qlib binary provider format.")
    parser.add_argument(
        "--source-price-dir",
        default="workdir/processd_day_dj30_17/price",
        help="Directory containing one CSV per asset with timestamp/open/high/low/close/volume columns.",
    )
    parser.add_argument(
        "--assets-path",
        default="configs/_asset_list_/dj30_17.json",
        help="Asset list used to select CSV files.",
    )
    parser.add_argument(
        "--qlib-dir",
        default="workdir/qlib_data/dj30_17",
        help="Output Qlib provider directory.",
    )
    parser.add_argument(
        "--use-adjusted-price",
        action="store_true",
        help="Adjust OHLC by adj_close / close when adj_close is available.",
    )
    parser.add_argument(
        "--vwap-method",
        choices=("typical", "close"),
        default="typical",
        help="How to synthesize vwap when intraday VWAP is unavailable.",
    )
    return parser.parse_args()


def _load_assets(path: str):
    payload = load_json(assemble_project_path(path))
    if isinstance(payload, dict):
        return list(payload.keys())
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return [item["symbol"] for item in payload]
        return [str(item) for item in payload]
    raise ValueError(f"Unsupported asset list format: {type(payload)}")


def _read_price_csv(path: Path, symbol: str, use_adjusted_price: bool, vwap_method: str):
    df = pd.read_csv(path)
    date_col = "timestamp" if "timestamp" in df.columns else "date"
    if date_col not in df.columns:
        raise KeyError(f"{path} must contain timestamp or date column.")

    rename = {"adjClose": "adj_close", "Adj Close": "adj_close"}
    df = df.rename(columns=rename)
    required = [date_col, "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    out = df[required + (["adj_close"] if "adj_close" in df.columns else [])].copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.strftime("%Y-%m-%d")
    out = out.rename(columns={date_col: "date"})
    out = out.sort_values("date").drop_duplicates("date", keep="first")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if use_adjusted_price and "adj_close" in out.columns:
        out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
        factor = out["adj_close"] / out["close"].replace(0, np.nan)
        for col in ["open", "high", "low", "close"]:
            out[col] = out[col] * factor
        out["close"] = out["adj_close"]

    if "vwap" not in out.columns:
        if vwap_method == "close":
            out["vwap"] = out["close"]
        else:
            out["vwap"] = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0

    out["factor"] = 1.0
    out["symbol"] = symbol.upper()
    return out[["date", "symbol", "open", "high", "low", "close", "volume", "vwap", "factor"]]


def _write_feature_bin(values: pd.Series, calendar: list[str], out_path: Path):
    values = values.reindex(calendar)
    first_valid = values.first_valid_index()
    if first_valid is None:
        return

    start_idx = calendar.index(first_valid)
    payload = values.loc[first_valid:].to_numpy(dtype=np.float32)
    out = np.hstack([np.asarray([start_idx], dtype=np.float32), payload]).astype("<f4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(out_path)


def main():
    args = _parse_args()
    source_dir = Path(assemble_project_path(args.source_price_dir))
    qlib_dir = Path(assemble_project_path(args.qlib_dir))
    assets = _load_assets(args.assets_path)

    frames = {}
    for symbol in assets:
        csv_path = source_dir / f"{symbol}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing price CSV for {symbol}: {csv_path}")
        frames[symbol.upper()] = _read_price_csv(
            csv_path,
            symbol=symbol,
            use_adjusted_price=args.use_adjusted_price,
            vwap_method=args.vwap_method,
        )

    calendar = sorted(set().union(*[set(df["date"]) for df in frames.values()]))
    calendars_dir = qlib_dir / "calendars"
    instruments_dir = qlib_dir / "instruments"
    features_dir = qlib_dir / "features"
    calendars_dir.mkdir(parents=True, exist_ok=True)
    instruments_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    (calendars_dir / "day.txt").write_text("\n".join(calendar) + "\n", encoding="utf-8")

    instrument_lines = []
    feature_fields = ["open", "high", "low", "close", "volume", "vwap", "factor"]
    for symbol, df in frames.items():
        start = str(df["date"].min())
        end = str(df["date"].max())
        instrument_lines.append(f"{symbol}\t{start}\t{end}")

        indexed = df.set_index("date")
        symbol_dir = features_dir / symbol.lower()
        for field in feature_fields:
            _write_feature_bin(indexed[field].astype(float), calendar, symbol_dir / f"{field}.day.bin")

    (instruments_dir / "all.txt").write_text("\n".join(instrument_lines) + "\n", encoding="utf-8")

    print(f"Saved Qlib provider to: {qlib_dir}")
    print(f"assets: {len(frames)}")
    print(f"calendar: {calendar[0]} -> {calendar[-1]} ({len(calendar)} rows)")


if __name__ == "__main__":
    main()
