import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"No JSON rows found in {path}")
    return rows


def _numeric_keys(rows: Iterable[Dict]) -> List[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                keys.add(key)
    keys.discard("epoch")
    return sorted(keys)


def _filter_existing_keys(rows: List[Dict], keys: List[str]) -> List[str]:
    existing = []
    for key in keys:
        if any(key in row and isinstance(row[key], (int, float)) for row in rows):
            existing.append(key)
    return existing


def _plot_group(rows: List[Dict], x_key: str, keys: List[str], title: str, output_path: Path) -> None:
    keys = _filter_existing_keys(rows, keys)
    if not keys:
        return

    x_values = [row.get(x_key, idx + 1) for idx, row in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(12, 7))

    for key in keys:
        y_values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                y_values.append(float(value))
            else:
                y_values.append(math.nan)
        ax.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=3.5, label=key)

    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot JSONL training logs into PNG charts.")
    parser.add_argument("--log", required=True, help="Path to a JSONL log file such as train_log.txt")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to write figures into. Defaults to <log_dir>/plots",
    )
    parser.add_argument(
        "--x-key",
        default="epoch",
        help="Field to use on the x-axis. Defaults to epoch.",
    )
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    rows = _load_jsonl(log_path)

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else log_path.parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    log_prefix = log_path.stem.replace("_log", "")
    if len(rows) == 1:
        print(f"Warning: only 1 JSON row found in {log_path.name}; this is usually expected for test/state logs, not full training curves.")

    all_numeric = _numeric_keys(rows)

    default_groups = {
        f"{log_prefix}_loss_overview.png": ["train_loss", "valid_loss", "loss", "valid_MSE", "valid_mse", "train_mse"],
        f"{log_prefix}_rank_metrics.png": ["valid_RANKIC", "valid_RANKICIR", "train_RANKIC", "train_RANKICIR"],
        f"{log_prefix}_vq_losses_train.png": [
            "train_weighted_quantized_loss",
            "train_weighted_commit_loss",
            "train_weighted_codebook_diversity_loss",
            "train_weighted_orthogonal_reg_loss",
            "train_weighted_nll_loss",
            "train_weighted_ret_loss",
            "train_weighted_kl_loss",
            "train_weighted_cont_loss",
        ],
        f"{log_prefix}_vq_losses_valid.png": [
            "valid_weighted_quantized_loss",
            "valid_weighted_commit_loss",
            "valid_weighted_codebook_diversity_loss",
            "valid_weighted_orthogonal_reg_loss",
            "valid_weighted_nll_loss",
            "valid_weighted_ret_loss",
            "valid_weighted_kl_loss",
            "valid_weighted_cont_loss",
        ],
    }

    for filename, keys in default_groups.items():
        _plot_group(rows, args.x_key, keys, filename.replace("_", " ").replace(".png", ""), outdir / filename)

    default_keys = {key for keys in default_groups.values() for key in keys}
    extra_keys = [key for key in all_numeric if key not in default_keys]
    _plot_group(rows, args.x_key, extra_keys, "extra metrics", outdir / f"{log_prefix}_extra_metrics.png")

    print(f"Saved plots to: {outdir}")
    for path in sorted(outdir.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
