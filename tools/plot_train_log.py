import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


def _value(row: Dict, key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _series(rows: List[Dict], key: str) -> List[float]:
    return [_value(row, key) if _value(row, key) is not None else float("nan") for row in rows]


def _x_values(rows: List[Dict], x_key: str) -> List[float]:
    return [row.get(x_key, idx + 1) for idx, row in enumerate(rows)]


def _stage_name(row: Dict) -> str | None:
    stage = row.get("stage")
    if isinstance(stage, str):
        return stage

    stage_ids = []
    for key in ("train_stage_id", "valid_stage_id", "stage_id"):
        value = _value(row, key)
        if value is not None:
            stage_ids.append(round(value))

    if 2 in stage_ids:
        return "predict"
    if 1 in stage_ids:
        return "vqvae"
    return None


def _stage_rows(rows: List[Dict], stage: str) -> List[Dict]:
    return [row for row in rows if _stage_name(row) == stage]


def _has_any(rows: List[Dict], keys: Sequence[str]) -> bool:
    return any(
        any(_value(row, key) is not None for row in rows)
        for key in keys
    )


def _plot_group(
        rows: List[Dict],
        x_key: str,
        keys: List[str],
        title: str,
        output_path: Path,
        yscale: str = "linear",
) -> None:
    keys = _filter_existing_keys(rows, keys)
    if not keys:
        return

    x_values = _x_values(rows, x_key)
    fig, ax = plt.subplots(figsize=(12, 7))

    for key in keys:
        y_values = _series(rows, key)
        ax.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=3.5, label=key)

    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel("value")
    ax.set_yscale(yscale)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _metric_keys(metric: str, prefixes: Sequence[str] = ("train", "valid")) -> List[str]:
    return [f"{prefix}_{metric}" for prefix in prefixes]


def _plot_metric_panels(
        rows: List[Dict],
        x_key: str,
        metrics: Sequence[Tuple[str, str]],
        title: str,
        output_path: Path,
        prefixes: Sequence[str] = ("train", "valid"),
        yscale: str = "linear",
) -> None:
    available = [
        (metric, label)
        for metric, label in metrics
        if _has_any(rows, _metric_keys(metric, prefixes))
    ]
    if not available:
        return

    ncols = 2
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), squeeze=False)
    x_values = _x_values(rows, x_key)

    for ax, (metric, label) in zip(axes.flatten(), available):
        for prefix in prefixes:
            key = f"{prefix}_{metric}"
            if not _has_any(rows, [key]):
                continue
            ax.plot(
                x_values,
                _series(rows, key),
                marker="o",
                linewidth=1.6,
                markersize=3.0,
                label=prefix,
            )
        ax.set_title(label)
        ax.set_xlabel(x_key)
        ax.set_ylabel(metric)
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    for ax in axes.flatten()[len(available):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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

    if len(rows) == 1:
        print(f"Warning: only 1 JSON row found in {log_path.name}; this is usually expected for test/state logs, not full training curves.")

    all_numeric = _numeric_keys(rows)

    stage1_rows = _stage_rows(rows, "vqvae") or rows
    stage2_rows = _stage_rows(rows, "predict")

    stage1_metrics = [
        ("loss", "stage1 total loss"),
        ("mse", "stage1 normalized reconstruction MSE"),
        ("price_mse", "stage1 restored price MSE"),
        ("recon_mse", "stage1 reconstruction MSE"),
        ("weighted_nll_loss", "stage1 weighted reconstruction loss"),
        ("weighted_quantized_loss", "stage1 weighted VQ loss"),
        ("weighted_commit_loss", "stage1 weighted commit loss"),
        ("weighted_codebook_diversity_loss", "stage1 codebook diversity loss"),
        ("weighted_orthogonal_reg_loss", "stage1 orthogonal regularization"),
        ("weighted_cont_loss", "stage1 OHLC constraint loss"),
    ]
    _plot_metric_panels(
        stage1_rows,
        args.x_key,
        stage1_metrics,
        "stage1 representation and reconstruction",
        outdir / "stage1_representation.png",
    )

    stage2_loss_metrics = [
        ("loss", "stage2 total loss"),
        ("return_rank_loss", "stage2 return + ranking monitor"),
        ("ret_mse", "stage2 return MSE"),
        ("weighted_ret_loss", "stage2 weighted return loss"),
        ("weighted_prior_ret_loss", "stage2 weighted prior return loss"),
        ("ranking_loss", "stage2 raw ranking loss"),
        ("weighted_ranking_loss", "stage2 weighted ranking loss"),
        ("prior_ranking_loss", "stage2 raw prior ranking loss"),
        ("weighted_prior_ranking_loss", "stage2 weighted prior ranking loss"),
        ("weighted_kl_loss", "stage2 weighted KL loss"),
    ]
    stage2_signal_metrics = [
        ("ic", "stage2 Pearson IC"),
        ("acc", "stage2 direction accuracy"),
        ("mcc", "stage2 direction MCC"),
        ("direction_tp", "stage2 direction TP"),
        ("direction_tn", "stage2 direction TN"),
        ("direction_fp", "stage2 direction FP"),
        ("direction_fn", "stage2 direction FN"),
    ]

    if stage2_rows:
        _plot_metric_panels(
            stage2_rows,
            args.x_key,
            stage2_loss_metrics,
            "stage2 prediction losses",
            outdir / "stage2_prediction_losses.png",
        )

        _plot_metric_panels(
            stage2_rows,
            args.x_key,
            stage2_signal_metrics,
            "stage2 prediction signals",
            outdir / "stage2_prediction_signals.png",
        )

    _plot_group(
        rows,
        args.x_key,
        ["train_stage_id", "valid_stage_id", "train_lr"],
        "training schedule",
        outdir / "training_schedule.png",
    )

    paper_metric_keys = [
        "valid_IC",
        "valid_RIC",
        "valid_PRECISION@10",
        "valid_SR",
        "train_IC",
        "train_RIC",
        "train_PRECISION@10",
        "train_SR",
    ]
    _plot_group(rows, args.x_key, paper_metric_keys, "paper metrics", outdir / "paper_metrics.png")

    plotted_metrics = list(stage1_metrics)
    if stage2_rows:
        plotted_metrics.extend(stage2_loss_metrics)
        plotted_metrics.extend(stage2_signal_metrics)

    default_keys = {
        f"{prefix}_{metric}"
        for prefix in ("train", "valid")
        for metric, _ in plotted_metrics
    }
    default_keys.update({"train_stage_id", "valid_stage_id", "train_lr", *paper_metric_keys})
    extra_keys = [key for key in all_numeric if key not in default_keys]
    _plot_group(rows, args.x_key, extra_keys, "extra metrics", outdir / "extra_metrics.png")

    print(f"Saved plots to: {outdir}")
    for path in sorted(outdir.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
