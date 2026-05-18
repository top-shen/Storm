import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storm.utils import load_joblib


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose dominant VQ code usage from state.joblib and checkpoint codebook."
    )
    parser.add_argument("--state", required=True, help="Path to state.joblib.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth checkpoint.")
    parser.add_argument("--outdir", default=None, help="Output directory.")
    parser.add_argument(
        "--factor-part",
        choices=["auto", "quantized", "encoder", "full"],
        default="auto",
        help="Factor slice used for nearest-code assignment.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Number of top codes to summarize.")
    parser.add_argument("--chunk-size", type=int, default=32768, help="Assignment chunk size.")
    return parser.parse_args()


def _load_codebook(checkpoint_path: Path):
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    model_state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    matches = [
        key
        for key in model_state.keys()
        if key.endswith("quantizer._codebook.embed") or key.endswith("_codebook.embed")
    ]
    if not matches:
        raise KeyError("Cannot find quantizer._codebook.embed in checkpoint.")
    key = sorted(matches, key=len)[0]
    codebook = model_state[key].detach().cpu().float().numpy()
    if codebook.ndim == 3 and codebook.shape[0] == 1:
        codebook = codebook[0]
    elif codebook.ndim == 3:
        codebook = codebook.reshape(-1, codebook.shape[-1])
    if codebook.ndim != 2:
        raise ValueError(f"Unsupported codebook shape: {codebook.shape}")
    return codebook, key


def _extract_factor_tensor(state_obj, codebook_dim: int, factor_part: str):
    items = state_obj.get("items", {})
    if not items:
        raise ValueError("No factor items found in state.joblib.")

    timestamps = list(items.keys())
    factors = []
    for timestamp in timestamps:
        factor = np.asarray(items[timestamp]["factor"], dtype=np.float32)
        factors.append(factor.reshape(-1, factor.shape[-1]))
    tensor = np.stack(factors, axis=0)
    factor_dim = tensor.shape[-1]

    resolved_part = factor_part
    if factor_part == "auto":
        resolved_part = "full" if factor_dim == codebook_dim else "quantized"

    if resolved_part == "full":
        if factor_dim != codebook_dim:
            raise ValueError(
                f"Full factor dim {factor_dim} does not match codebook dim {codebook_dim}."
            )
        selected = tensor
    elif resolved_part == "quantized":
        selected = tensor if factor_dim == codebook_dim else tensor[..., codebook_dim:]
    elif resolved_part == "encoder":
        selected = tensor if factor_dim == codebook_dim else tensor[..., :codebook_dim]
    else:
        raise ValueError(f"Unsupported factor part: {factor_part}")

    return timestamps, selected.astype(np.float32, copy=False), resolved_part


def _assign_nearest_codes(tokens: np.ndarray, codebook: np.ndarray, chunk_size: int):
    flat = tokens.reshape(-1, tokens.shape[-1])
    codebook_t = codebook.astype(np.float32, copy=False)
    code_norm = np.sum(codebook_t * codebook_t, axis=1)[None, :]

    assignments = np.empty(flat.shape[0], dtype=np.int64)
    margins = np.empty(flat.shape[0], dtype=np.float32)
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        chunk = flat[start:end]
        dist = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + code_norm
            - 2.0 * chunk @ codebook_t.T
        )
        order = np.argpartition(dist, kth=1, axis=1)[:, :2]
        best_dist = dist[np.arange(end - start), order[:, 0]]
        second_dist = dist[np.arange(end - start), order[:, 1]]
        swap = second_dist < best_dist
        best_idx = order[:, 0].copy()
        second_idx = order[:, 1].copy()
        best_idx[swap], second_idx[swap] = second_idx[swap], best_idx[swap]
        best_dist, second_dist = (
            dist[np.arange(end - start), best_idx],
            dist[np.arange(end - start), second_idx],
        )
        assignments[start:end] = best_idx
        margins[start:end] = second_dist - best_dist

    return assignments.reshape(tokens.shape[:-1]), margins.reshape(tokens.shape[:-1])


def _infer_grid_shape(state_obj, token_num: int):
    meta = state_obj.get("meta", {})
    n_size = meta.get("n_size")
    if n_size is not None:
        dims = tuple(int(v) for v in n_size)
        if np.prod(dims) == token_num:
            return dims
    return (token_num,)


def _plot_position_heatmap(assignments, code_id: int, grid_shape, out_path: Path):
    usage_by_position = np.mean(assignments == code_id, axis=0)
    if len(grid_shape) >= 2:
        data = usage_by_position.reshape(grid_shape)
        if len(grid_shape) == 3 and grid_shape[-1] == 1:
            data = data[..., 0]
    else:
        data = usage_by_position.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto", cmap="magma")
    ax.set_title(f"Dominant Code {code_id} Usage by Token Position")
    ax.set_xlabel("Stock index" if data.ndim == 2 else "Token index")
    ax.set_ylabel("Time patch index" if data.ndim == 2 else "")
    fig.colorbar(im, ax=ax, label="Usage ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_usage_over_time(timestamps, assignments, top_codes, out_path: Path):
    x = np.arange(len(timestamps))
    fig, ax = plt.subplots(figsize=(14, 5))
    for code_id in top_codes:
        daily_usage = np.mean(assignments == code_id, axis=1) * 100.0
        ax.plot(x, daily_usage, linewidth=1.4, label=f"code {code_id}")
    ax.set_title("Top Code Usage Over Time")
    ax.set_ylabel("Daily token share (%)")
    ax.set_xlabel("Timestamp")
    tick_step = max(1, len(timestamps) // 12)
    tick_positions = list(range(0, len(timestamps), tick_step))
    if tick_positions[-1] != len(timestamps) - 1:
        tick_positions.append(len(timestamps) - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([timestamps[i] for i in tick_positions], rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", ncol=min(4, len(top_codes)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_position_entropy(assignments, codebook_size: int, grid_shape, out_path: Path):
    flat_pos = assignments.reshape(assignments.shape[0], -1)
    entropy_values = []
    for pos in range(flat_pos.shape[1]):
        counts = np.bincount(flat_pos[:, pos], minlength=codebook_size).astype(np.float64)
        prob = counts / max(counts.sum(), 1.0)
        nonzero = prob > 0
        entropy = -np.sum(prob[nonzero] * np.log(prob[nonzero]))
        entropy_values.append(entropy / np.log(codebook_size))
    entropy_values = np.asarray(entropy_values)

    if len(grid_shape) >= 2:
        data = entropy_values.reshape(grid_shape)
        if len(grid_shape) == 3 and grid_shape[-1] == 1:
            data = data[..., 0]
    else:
        data = entropy_values.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Code Assignment Entropy by Token Position")
    ax.set_xlabel("Stock index" if data.ndim == 2 else "Token index")
    ax.set_ylabel("Time patch index" if data.ndim == 2 else "")
    fig.colorbar(im, ax=ax, label="Normalized entropy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = _parse_args()
    state_path = Path(args.state).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else state_path.parent / "state_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    state_obj = load_joblib(str(state_path))
    codebook, codebook_key = _load_codebook(checkpoint_path)
    timestamps, tokens, factor_part = _extract_factor_tensor(
        state_obj,
        codebook_dim=codebook.shape[-1],
        factor_part=args.factor_part,
    )
    assignments, margins = _assign_nearest_codes(tokens, codebook, args.chunk_size)

    counts = np.bincount(assignments.reshape(-1), minlength=codebook.shape[0])
    usage_pct = counts / counts.sum() * 100.0
    top_codes = np.argsort(usage_pct)[::-1][: args.top_k]
    grid_shape = _infer_grid_shape(state_obj, assignments.shape[1])

    _plot_position_heatmap(
        assignments,
        int(top_codes[0]),
        grid_shape,
        outdir / "dominant_code_position_heatmap.png",
    )
    _plot_usage_over_time(
        timestamps,
        assignments,
        [int(v) for v in top_codes[: min(args.top_k, 5)]],
        outdir / "top_code_usage_over_time.png",
    )
    _plot_position_entropy(
        assignments,
        codebook.shape[0],
        grid_shape,
        outdir / "code_assignment_entropy_by_position.png",
    )

    top_margin = margins[assignments == int(top_codes[0])]
    other_margin = margins[assignments != int(top_codes[0])]
    summary = {
        "state": str(state_path),
        "checkpoint": str(checkpoint_path),
        "codebook_key": codebook_key,
        "factor_part": factor_part,
        "tokens_shape": tuple(tokens.shape),
        "grid_shape": grid_shape,
        "top_codes": [
            {"code": int(idx), "usage_pct": float(usage_pct[idx]), "count": int(counts[idx])}
            for idx in top_codes
        ],
        "dead_codes": int(np.sum(counts == 0)),
        "used_codes": int(np.sum(counts > 0)),
        "dominant_code_margin_mean": float(np.mean(top_margin)) if top_margin.size else 0.0,
        "other_codes_margin_mean": float(np.mean(other_margin)) if other_margin.size else 0.0,
    }
    (outdir / "codebook_spike_diagnostics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostics to: {outdir}")


if __name__ == "__main__":
    main()
