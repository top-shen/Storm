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
        description=(
            "Plot latent factor and codebook embedding distributions without "
            "using code assignment counts."
        )
    )
    parser.add_argument(
        "--state",
        required=True,
        help="Path to state.joblib, or the workdir containing state.joblib.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. If omitted, try <state_dir>/checkpoint/best.pth.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to <state_dir>/latent_codebook_distribution.",
    )
    parser.add_argument(
        "--factor-part",
        choices=("auto", "quantized", "encoder", "full"),
        default="auto",
        help=(
            "Which saved factor part to plot. For concatenated [encoder, quantized] "
            "factors, auto uses quantized."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("tsne", "umap", "pca"),
        default="tsne",
        help="2D projection method for the joint latent-codebook scatter plot.",
    )
    parser.add_argument(
        "--num-factor-samples",
        type=int,
        default=8000,
        help="Number of latent tokens sampled for projection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for sampling and projection.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=50.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--tsne-iters",
        type=int,
        default=1200,
        help="Number of t-SNE optimization iterations.",
    )
    parser.add_argument(
        "--pre-pca-dim",
        type=int,
        default=50,
        help="Pre-reduce vectors before t-SNE/UMAP. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.15,
        help="UMAP min_dist.",
    )
    parser.add_argument(
        "--annotate-codebook",
        choices=("none", "all", "every"),
        default="every",
        help="How to annotate codebook embedding indices on the scatter plot.",
    )
    parser.add_argument(
        "--annotate-every",
        type=int,
        default=16,
        help="Annotate every N-th codebook index when --annotate-codebook=every.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=60,
        help="Number of bins for norm and distance histograms.",
    )
    return parser.parse_args()


def _resolve_state_path(state_arg: str):
    path = Path(state_arg).resolve()
    if path.is_dir():
        path = path / "state.joblib"
    if not path.exists():
        raise FileNotFoundError(f"state.joblib not found: {path}")
    return path


def _find_checkpoint(state_path: Path, checkpoint_arg: str | None):
    if checkpoint_arg:
        checkpoint_path = Path(checkpoint_arg).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    checkpoint_dir = state_path.parent / "checkpoint"
    candidates = [
        checkpoint_dir / "best.pth",
        checkpoint_dir / "best_stage1.pth",
        checkpoint_dir / "latest.pth",
        checkpoint_dir / "last.pth",
        state_path.parent / "best.pth",
    ]
    if checkpoint_dir.exists():
        candidates.extend(sorted(checkpoint_dir.glob("*.pth")))

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find checkpoint automatically. Pass --checkpoint explicitly."
    )


def _load_checkpoint_state_dict(checkpoint_path: Path):
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_ema_state", "state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise ValueError(f"Unsupported checkpoint payload type: {type(checkpoint)}")


def _load_codebook(checkpoint_path: Path):
    state_dict = _load_checkpoint_state_dict(checkpoint_path)
    candidates = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key.replace("module.", "")
        if (
            clean_key.endswith("quantizer._codebook.embed")
            or clean_key.endswith("_codebook.embed")
            or clean_key.endswith("codebook.embed")
        ):
            candidates.append((clean_key, value.detach().cpu()))

    if not candidates:
        raise KeyError(
            "Could not find codebook embedding in checkpoint. Expected a key like "
            "'quantizer._codebook.embed'."
        )

    key, tensor = sorted(candidates, key=lambda item: len(item[0]))[0]
    codebook = np.squeeze(tensor.float().numpy())
    if codebook.ndim != 2:
        raise ValueError(f"Unsupported codebook shape from {key}: {codebook.shape}")
    return codebook.astype(np.float32, copy=False), key


def _extract_factor_tokens(state_obj, codebook_dim: int, factor_part: str):
    items = state_obj.get("items", {})
    if not items:
        raise ValueError("No factor items found in state.joblib.")

    token_chunks = []
    for timestamp in items:
        factor = np.asarray(items[timestamp]["factor"], dtype=np.float32)
        token_chunks.append(factor.reshape(-1, factor.shape[-1]))

    tokens = np.concatenate(token_chunks, axis=0)
    factor_dim = tokens.shape[-1]

    resolved_part = factor_part
    if factor_part == "auto":
        resolved_part = "full" if factor_dim == codebook_dim else "quantized"

    if resolved_part == "full":
        if factor_dim != codebook_dim:
            raise ValueError(
                f"Full factor dim {factor_dim} cannot be compared with codebook dim {codebook_dim}. "
                "Use --factor-part quantized or --factor-part encoder."
            )
        selected = tokens
    elif resolved_part == "quantized":
        if factor_dim == codebook_dim:
            selected = tokens
        elif factor_dim == codebook_dim * 2:
            selected = tokens[:, codebook_dim:]
        else:
            raise ValueError(
                f"Cannot infer quantized half from factor dim {factor_dim} and codebook dim {codebook_dim}."
            )
    elif resolved_part == "encoder":
        if factor_dim == codebook_dim:
            selected = tokens
        elif factor_dim == codebook_dim * 2:
            selected = tokens[:, :codebook_dim]
        else:
            raise ValueError(
                f"Cannot infer encoder half from factor dim {factor_dim} and codebook dim {codebook_dim}."
            )
    else:
        raise ValueError(f"Unsupported factor part: {factor_part}")

    return tokens, selected.astype(np.float32, copy=False), resolved_part


def _sample_rows(x: np.ndarray, num_samples: int, seed: int):
    if num_samples <= 0 or num_samples >= x.shape[0]:
        return x
    rng = np.random.default_rng(seed)
    indices = rng.choice(x.shape[0], size=num_samples, replace=False)
    return x[indices]


def _standardize(x: np.ndarray):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x - mean) / std


def _pre_pca(x: np.ndarray, n_dims: int):
    if n_dims <= 0 or n_dims >= x.shape[1]:
        return x
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:n_dims].T


def _project_2d(
    x: np.ndarray,
    method: str,
    seed: int,
    perplexity: float,
    tsne_iters: int,
    pre_pca_dim: int,
    umap_neighbors: int,
    umap_min_dist: float,
):
    x = _standardize(x).astype(np.float32, copy=False)

    if method == "pca":
        _u, s, vt = np.linalg.svd(x, full_matrices=False)
        total_var = float(np.sum(s ** 2))
        explained = (s[:2] ** 2) / total_var if total_var > 0 else np.zeros(2, dtype=np.float64)
        return x @ vt[:2].T, {"pc1_var": float(explained[0]), "pc2_var": float(explained[1])}

    x = _pre_pca(x, pre_pca_dim).astype(np.float32, copy=False)

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError(
                "t-SNE requires scikit-learn. Install scikit-learn or run with --method pca."
            ) from exc

        max_perplexity = max(5.0, (x.shape[0] - 1) / 3.0)
        perplexity = min(float(perplexity), max_perplexity)
        kwargs = dict(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        try:
            reducer = TSNE(max_iter=tsne_iters, **kwargs)
        except TypeError:
            reducer = TSNE(n_iter=tsne_iters, **kwargs)
        coords = reducer.fit_transform(x)
        return coords, {
            "perplexity": float(perplexity),
            "tsne_iters": int(tsne_iters),
            "pre_pca_dim": int(pre_pca_dim),
        }

    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP requires umap-learn. Install umap-learn or run with --method tsne/pca."
            ) from exc

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=seed,
        )
        coords = reducer.fit_transform(x)
        return coords, {
            "umap_neighbors": int(umap_neighbors),
            "umap_min_dist": float(umap_min_dist),
            "pre_pca_dim": int(pre_pca_dim),
        }

    raise ValueError(f"Unsupported projection method: {method}")


def _annotate_codebook(ax, codebook_xy: np.ndarray, mode: str, every: int):
    if mode == "none":
        return
    if mode == "all":
        indices = range(codebook_xy.shape[0])
    else:
        every = max(1, int(every))
        indices = range(0, codebook_xy.shape[0], every)

    for idx in indices:
        ax.annotate(
            str(idx),
            (codebook_xy[idx, 0], codebook_xy[idx, 1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#475569",
            alpha=0.85,
        )


def _plot_joint_distribution(
    factor_points: np.ndarray,
    codebook: np.ndarray,
    out_path: Path,
    factor_part: str,
    method: str,
    args,
):
    combined = np.concatenate([factor_points, codebook], axis=0)
    combined_xy, projection_stats = _project_2d(
        combined,
        method=method,
        seed=args.seed,
        perplexity=args.perplexity,
        tsne_iters=args.tsne_iters,
        pre_pca_dim=args.pre_pca_dim,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    factor_xy = combined_xy[: factor_points.shape[0]]
    codebook_xy = combined_xy[factor_points.shape[0] :]

    title_method = {"tsne": "t-SNE", "umap": "UMAP", "pca": "PCA"}[method]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(
        factor_xy[:, 0],
        factor_xy[:, 1],
        s=8,
        color="#98d47f",
        alpha=0.28,
        linewidths=0,
        label=f"Latent factors ({factor_part}, sampled)",
    )
    ax.scatter(
        codebook_xy[:, 0],
        codebook_xy[:, 1],
        s=95,
        color="#22d3ee",
        marker="^",
        edgecolors="#111827",
        linewidths=0.55,
        alpha=0.92,
        label="Codebook embeddings",
        zorder=3,
    )
    _annotate_codebook(ax, codebook_xy, args.annotate_codebook, args.annotate_every)

    ax.set_title(f"Latent Factors and Codebook Embeddings Distribution ({title_method})")
    if method == "pca":
        ax.set_xlabel(f"PC1 ({projection_stats['pc1_var'] * 100:.2f}% var)")
        ax.set_ylabel(f"PC2 ({projection_stats['pc2_var'] * 100:.2f}% var)")
    else:
        ax.set_xlabel(f"{title_method} 1")
        ax.set_ylabel(f"{title_method} 2")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.22)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

    return {
        "projection_method": method,
        **projection_stats,
    }


def _pairwise_distances(x: np.ndarray):
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist[np.triu_indices_from(dist, k=1)]


def _nearest_codebook_distances(latent: np.ndarray, codebook: np.ndarray, chunk_size: int = 4096):
    distances = []
    for start in range(0, latent.shape[0], chunk_size):
        chunk = latent[start : start + chunk_size]
        diff = chunk[:, None, :] - codebook[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        distances.append(np.min(dist, axis=1))
    return np.concatenate(distances, axis=0)


def _plot_distribution_diagnostics(
    factor_points: np.ndarray,
    codebook: np.ndarray,
    out_path: Path,
    bins: int,
):
    latent_norm = np.linalg.norm(factor_points, axis=1)
    codebook_norm = np.linalg.norm(codebook, axis=1)
    codebook_pairwise = _pairwise_distances(codebook)
    nearest_dist = _nearest_codebook_distances(factor_points, codebook)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].hist(latent_norm, bins=bins, color="#8bcf74", alpha=0.82)
    axes[0, 0].set_title("Latent Factor Norms")
    axes[0, 0].set_xlabel("L2 norm")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(codebook_norm, bins=min(bins, max(12, codebook.shape[0] // 4)), color="#22d3ee", alpha=0.88)
    axes[0, 1].set_title("Codebook Embedding Norms")
    axes[0, 1].set_xlabel("L2 norm")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].hist(codebook_pairwise, bins=bins, color="#60a5fa", alpha=0.86)
    axes[1, 0].set_title("Codebook Pairwise Distances")
    axes[1, 0].set_xlabel("Euclidean distance")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(nearest_dist, bins=bins, color="#f59e0b", alpha=0.86)
    axes[1, 1].set_title("Latent-to-Codebook Nearest Distances")
    axes[1, 1].set_xlabel("Euclidean distance")
    axes[1, 1].set_ylabel("Count")

    for ax in axes.ravel():
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return {
        "latent_norm_mean": float(np.mean(latent_norm)),
        "latent_norm_std": float(np.std(latent_norm)),
        "codebook_norm_mean": float(np.mean(codebook_norm)),
        "codebook_norm_std": float(np.std(codebook_norm)),
        "codebook_pairwise_distance_mean": float(np.mean(codebook_pairwise)),
        "codebook_pairwise_distance_std": float(np.std(codebook_pairwise)),
        "latent_to_codebook_nearest_distance_mean": float(np.mean(nearest_dist)),
        "latent_to_codebook_nearest_distance_std": float(np.std(nearest_dist)),
    }


def _save_summary(
    out_path: Path,
    state_path: Path,
    checkpoint_path: Path,
    codebook_key: str,
    raw_tokens: np.ndarray,
    selected_tokens: np.ndarray,
    sampled_tokens: np.ndarray,
    codebook: np.ndarray,
    factor_part: str,
    plot_stats: dict,
    distribution_stats: dict,
):
    payload = {
        "state": str(state_path),
        "checkpoint": str(checkpoint_path),
        "codebook_key": codebook_key,
        "factor_part": factor_part,
        "raw_factor_tokens_shape": list(raw_tokens.shape),
        "selected_factor_tokens_shape": list(selected_tokens.shape),
        "sampled_factor_tokens_shape": list(sampled_tokens.shape),
        "codebook_shape": list(codebook.shape),
        "projection": plot_stats,
        "distribution": distribution_stats,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = _parse_args()
    state_path = _resolve_state_path(args.state)
    checkpoint_path = _find_checkpoint(state_path, args.checkpoint)
    outdir = Path(args.outdir).resolve() if args.outdir else state_path.parent / "latent_codebook_distribution"
    outdir.mkdir(parents=True, exist_ok=True)

    state_obj = load_joblib(str(state_path))
    codebook, codebook_key = _load_codebook(checkpoint_path)
    raw_tokens, selected_tokens, factor_part = _extract_factor_tokens(
        state_obj,
        codebook_dim=codebook.shape[1],
        factor_part=args.factor_part,
    )
    sampled_tokens = _sample_rows(selected_tokens, args.num_factor_samples, args.seed)

    scatter_path = outdir / f"latent_codebook_distribution_{args.method}.png"
    diagnostics_path = outdir / "latent_codebook_distribution_diagnostics.png"
    summary_path = outdir / "latent_codebook_distribution_summary.json"

    plot_stats = _plot_joint_distribution(
        sampled_tokens,
        codebook,
        scatter_path,
        factor_part=factor_part,
        method=args.method,
        args=args,
    )
    distribution_stats = _plot_distribution_diagnostics(
        sampled_tokens,
        codebook,
        diagnostics_path,
        bins=args.hist_bins,
    )
    _save_summary(
        summary_path,
        state_path=state_path,
        checkpoint_path=checkpoint_path,
        codebook_key=codebook_key,
        raw_tokens=raw_tokens,
        selected_tokens=selected_tokens,
        sampled_tokens=sampled_tokens,
        codebook=codebook,
        factor_part=factor_part,
        plot_stats=plot_stats,
        distribution_stats=distribution_stats,
    )

    print(f"Saved scatter plot to: {scatter_path}")
    print(f"Saved diagnostics plot to: {diagnostics_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"factor_part: {factor_part}")
    print(f"raw_factor_tokens_shape: {raw_tokens.shape}")
    print(f"selected_factor_tokens_shape: {selected_tokens.shape}")
    print(f"sampled_factor_tokens_shape: {sampled_tokens.shape}")
    print(f"codebook_shape: {codebook.shape}")


if __name__ == "__main__":
    main()
