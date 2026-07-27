#!/usr/bin/env python3
"""
Patch Embedding Distance Report Generator

For each model configuration:
- Load the frozen encoder (same loading code as generate_patch_embedding_report.py)
- For each mini dataset (a small manifest of clips), sample a fixed set of clips
  and extract the FULL patch-token grid embedding for every clip (not just one
  random patch)
- Videos within a mini dataset may have different lengths (different numbers
  of temporal tokens), so each clip's patch grid is collapsed into a single
  video embedding before comparison: average over spatial positions within
  each frame, then average across frames, then L2-normalize
- Within each mini dataset, compare every pair of clips' video embeddings
  using several magnitude-invariant distance metrics (cosine, angular,
  Pearson correlation, Spearman rank correlation)
- Report the mean and standard deviation of each distance metric per model per
  mini dataset, and render a polished Markdown report
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.generate_patch_embedding_report import (
    apply_pretrained_frame_skip,
    configure_logging,
    load_config,
    load_data_for_model,
    load_manifest,
    prepare_encoder,
    rebuild_attn_mask_after_frame_skip,
    sample_rows,
    slugify,
    MODELS_CONFIG,
)

logger = logging.getLogger(__name__)

# Each entry is a small manifest ("mini dataset") whose within-dataset
# embedding-distance statistics are reported and compared across models.
# Replace the placeholder CSV paths with real manifests once they exist -
# each should be a whitespace-separated file with a video path per row,
# exactly like the manifests consumed by generate_patch_embedding_report.py.
DATASETS_CONFIG = [
    {"name": "Random videos", "csv": "data/minidataset_random.csv"},
    {"name": "Wash action", "csv": "data/minidataset_wash_action.csv"},
    {"name": "Plate object", "csv": "data/minidataset_plate_object.csv"},
    {"name": "Same person same day", "csv": "data/minidataset_same_person_same_day.csv"},
    {"name": "Consecutive clips", "csv": "data/minidataset_same_person_same_day_consecutive.csv"},
]

# All metrics are expressed as *distances*: lower means more similar. Cosine,
# Pearson and Spearman are reported as (1 - similarity); angular distance is
# already a distance. All four are invariant to the magnitude of the
# embedding vectors.
METRICS = ("cosine_distance", "angular_distance", "pearson_distance", "spearman_distance")

METRIC_LABELS = {
    "cosine_distance": "Cosine distance",
    "angular_distance": "Angular distance",
    "pearson_distance": "Pearson correlation distance",
    "spearman_distance": "Spearman correlation distance",
}


@dataclass
class GridEmbedding:
    sample_index: int
    video_path: str
    grid: np.ndarray  # (num_temporal_tokens, num_spatial_tokens, embed_dim)
    valid_mask: np.ndarray  # (num_temporal_tokens, num_spatial_tokens) bool; False = padding, not a real frame


def validate_datasets_config(datasets_config: List[dict]) -> None:
    names = [entry["name"] for entry in datasets_config]
    duplicates = {name for name, count in Counter(names).items() if count > 1}
    if duplicates:
        raise ValueError(
            f"DATASETS_CONFIG has duplicate dataset name(s): {sorted(duplicates)}. "
            "Each mini dataset must have a unique 'name' - results are grouped by it, "
            "so a duplicate silently merges two different manifests into one row."
        )


def sample_dataset_manifest(dataset_cfg: dict, num_samples: int, seed: int, output_dir: Path) -> dict:
    manifest = load_manifest(dataset_cfg["csv"])
    sampled_manifest = sample_rows(manifest, num_samples, seed)

    dataset_slug = slugify(dataset_cfg["name"])
    dataset_dir = output_dir / "datasets" / dataset_slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sampled_csv = dataset_dir / "sampled_manifest.csv"
    sampled_manifest.to_csv(sampled_csv, sep=" ", header=False, index=False)

    return {
        "name": dataset_cfg["name"],
        "slug": dataset_slug,
        "csv": str(sampled_csv),
        "num_candidates": int(len(sampled_manifest)),
    }


def collect_grid_embeddings(
    encoder,
    config: dict,
    dataset_csv: str,
    batch_size: int,
    device: str,
    model_name: str,
) -> List[GridEmbedding]:
    """Extract the full patch-token grid embedding for every clip in `dataset_csv`,
    along with a mask marking which tokens are real frames vs. batch padding
    (shorter clips are zero-padded to the batch's max length).

    Unlike generate_patch_embedding_report.py (which keeps one random token per
    clip), every real token is kept here so they can later be averaged into a
    single per-clip embedding.
    """
    sampled_manifest = load_manifest(dataset_csv)

    args_data = config["data"]
    patch_size = args_data["patch_size"]
    tubelet_size = args_data["tubelet_size"]

    data_loader, frame_skip_info = load_data_for_model(config, dataset_csv, batch_size=batch_size)

    embeddings: List[GridEmbedding] = []
    skipped_empty = 0

    for batch_idx, data in enumerate(tqdm(data_loader, total=len(data_loader), desc=f"{model_name}")):
        try:
            clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
            clip_indices = data[2]
            if clip_indices is not None:
                clip_indices = [d.to(device, non_blocking=True) for d in clip_indices]
            attn_mask = [[dij.to(device, non_blocking=True) for dij in di] for di in data[3]]

            if frame_skip_info is not None:
                frames_to_skip, previous_tubelet_size = frame_skip_info
                pre_skip_validity = torch.diagonal(attn_mask[0][0][:, 0], dim1=-2, dim2=-1)
                clips = [
                    [apply_pretrained_frame_skip(view, frames_to_skip, previous_tubelet_size) for view in clip]
                    for clip in clips
                ]
                # See generate_patch_embedding_report.py: the mask was sized for the
                # pre-skip clip length, so rebuild it at the post-skip token count.
                _, _, t, h, w = clips[0][0].shape
                num_spatial_tokens = (h // patch_size) * (w // patch_size)
                num_tokens = (t // tubelet_size) * num_spatial_tokens
                attn_mask = [
                    [
                        rebuild_attn_mask_after_frame_skip(
                            pre_skip_validity,
                            num_spatial_tokens=num_spatial_tokens,
                            tubelet_size=tubelet_size,
                            frames_to_skip=frames_to_skip,
                            previous_tubelet_size=previous_tubelet_size,
                            num_tokens_post=num_tokens,
                        )
                    ]
                ]

            with torch.no_grad():
                outputs, attn_mask = encoder(clips, clip_indices, attn_mask)

            view_output = outputs[0]
            view_attn_mask = attn_mask[0]
            videos = clips[0][0]
            batch_size_actual = videos.shape[0]
            token_validity = torch.diagonal(view_attn_mask[:, 0], dim1=-2, dim2=-1)

            _, _, height, width = videos[0].shape
            grid_h = height // patch_size
            grid_w = width // patch_size
            grid_size = grid_h * grid_w

            for batch_index in range(batch_size_actual):
                # Order-preserving map back into sampled_manifest: shuffle=False and
                # drop_last=False mean every batch but the last is exactly
                # batch_size long, so this recovers the flat sample position.
                global_idx = batch_idx * batch_size + batch_index
                valid = token_validity[batch_index]
                if not bool(valid.any()):
                    skipped_empty += 1
                    continue

                token_embeddings = view_output[batch_index].detach().cpu().numpy().astype(np.float32, copy=False)
                num_tokens, embed_dim = token_embeddings.shape
                num_temporal = num_tokens // grid_size
                grid = token_embeddings.reshape(num_temporal, grid_size, embed_dim)
                # Batches mix clips of different real lengths: shorter clips are
                # zero-padded to the batch's max length, and this mask marks
                # which (temporal, spatial) tokens are real frames vs. padding.
                valid_mask = valid.detach().cpu().numpy().reshape(num_temporal, grid_size)

                video_path = ""
                if global_idx < len(sampled_manifest):
                    video_path = str(sampled_manifest.iloc[global_idx]["video_path"])

                embeddings.append(
                    GridEmbedding(sample_index=global_idx, video_path=video_path, grid=grid, valid_mask=valid_mask)
                )
        except Exception as exc:
            logger.error("Failed processing batch %d for %s: %s", batch_idx, model_name, exc)

    if skipped_empty:
        logger.warning(
            "%s: skipped %d clip(s) with no valid tokens at all",
            model_name,
            skipped_empty,
        )

    return embeddings


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def center(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x - x.mean(axis=axis, keepdims=True)


def rank_along_axis(x: np.ndarray, axis: int = -1) -> np.ndarray:
    order = np.argsort(x, axis=axis)
    ranks = np.argsort(order, axis=axis).astype(np.float64)
    return ranks


def video_embedding(grid: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """(T, S, D) patch grid -> single L2-normalized video embedding: average
    over real (non-padding) tokens only, then L2-normalize. Padding tokens
    from batch-padded shorter clips (valid_mask == False) are excluded so
    they don't pull the average toward zero."""
    weights = valid_mask.astype(np.float64)
    num_valid = weights.sum()
    weighted_sum = np.tensordot(weights, grid.astype(np.float64), axes=([0, 1], [0, 1]))
    return l2_normalize(weighted_sum / num_valid, axis=-1)


def analyze_dataset(embeddings: List[GridEmbedding]) -> Optional[Dict[str, Dict[str, float]]]:
    if len(embeddings) < 2:
        return None

    embed_dims = Counter(e.grid.shape[-1] for e in embeddings)
    common_dim, _ = embed_dims.most_common(1)[0]
    if len(embed_dims) > 1:
        logger.warning(
            "Dropping %d clip(s) with a mismatched embedding dim (kept dim %d)",
            len(embeddings) - embed_dims[common_dim],
            common_dim,
        )
        embeddings = [e for e in embeddings if e.grid.shape[-1] == common_dim]

    if len(embeddings) < 2:
        return None

    # Videos may have different numbers of temporal tokens (different clip
    # lengths), so each clip is first collapsed to a single embedding rather
    # than compared patch-by-patch.
    vectors = np.stack([video_embedding(e.grid, e.valid_mask) for e in embeddings], axis=0)  # (N, D)
    n, d = vectors.shape

    cosine_sim = vectors @ vectors.T
    angular = np.arccos(np.clip(cosine_sim, -1.0, 1.0)) / np.pi
    pearson_vecs = l2_normalize(center(vectors, axis=-1), axis=-1)
    pearson_sim = pearson_vecs @ pearson_vecs.T
    ranks = rank_along_axis(vectors, axis=-1)
    spearman_vecs = l2_normalize(center(ranks, axis=-1), axis=-1)
    spearman_sim = spearman_vecs @ spearman_vecs.T

    distances = {
        "cosine_distance": 1.0 - cosine_sim,
        "angular_distance": angular,
        "pearson_distance": 1.0 - pearson_sim,
        "spearman_distance": 1.0 - spearman_sim,
    }

    triu_i, triu_j = np.triu_indices(n, k=1)
    results: Dict[str, Dict[str, float]] = {}
    for metric_name, matrix in distances.items():
        pair_values = matrix[triu_i, triu_j]
        results[metric_name] = {
            "mean": float(pair_values.mean()),
            "std": float(pair_values.std()),
            "num_pairs": int(pair_values.shape[0]),
        }

    results["_grid_shape"] = {"embed_dim": d, "num_samples_used": n}
    return results


def save_distance_bar_chart(model_name: str, model_df: pd.DataFrame, output_path: Path) -> None:
    dataset_names = list(dict.fromkeys(model_df["dataset"]))
    if not dataset_names:
        return

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(dataset_names)), 6), dpi=150)
    x = np.arange(len(dataset_names))
    width = 0.8 / len(METRICS)

    for i, metric in enumerate(METRICS):
        metric_df = model_df[model_df["metric"] == metric].set_index("dataset")
        means = [metric_df.loc[name, "mean"] if name in metric_df.index else np.nan for name in dataset_names]
        stds = [metric_df.loc[name, "std"] if name in metric_df.index else 0.0 for name in dataset_names]
        offset = (i - (len(METRICS) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=3, label=METRIC_LABELS[metric])

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=20, ha="right")
    ax.set_ylabel("Distance (mean ± std across clip pairs)")
    ax.set_title(f"{model_name}: within-dataset patch embedding distance")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_chart(results_df: pd.DataFrame, output_path: Path) -> None:
    """One figure, one subplot per metric, comparing every model's mean ± std
    distance across all mini datasets side by side."""
    if results_df.empty:
        return

    models = list(dict.fromkeys(results_df["model"]))
    datasets = list(dict.fromkeys(results_df["dataset"]))
    if not models or not datasets:
        return

    fig, axes = plt.subplots(2, 2, figsize=(max(10, 2.6 * len(datasets)), 10), dpi=150)
    axes = axes.flatten()
    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    for ax, metric in zip(axes, METRICS):
        metric_df = results_df[results_df["metric"] == metric]
        for i, model in enumerate(models):
            model_df = metric_df[metric_df["model"] == model].set_index("dataset")
            means = [model_df.loc[name, "mean"] if name in model_df.index else np.nan for name in datasets]
            stds = [model_df.loc[name, "std"] if name in model_df.index else 0.0 for name in datasets]
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, means, width=width, yerr=stds, capsize=2, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.grid(axis="y", alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(models), 4), bbox_to_anchor=(0.5, 1.04), fontsize=9)
    fig.suptitle("Cross-model comparison: within-dataset patch embedding distance", y=1.08)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def append_model_comparison_section(
    lines: List[str],
    results_df: pd.DataFrame,
    model_meta: Dict[str, dict],
    output_dir: Path,
    comparison_chart_path: Optional[Path],
) -> None:
    if results_df.empty:
        return

    models = [name for name in model_meta if name in set(results_df["model"])]
    datasets = list(dict.fromkeys(results_df["dataset"]))
    if not models or not datasets:
        return

    lines.append("## Model Comparison")
    lines.append("")
    lines.append("Direct comparison of every tested model across all mini datasets and metrics.")
    lines.append("")

    if comparison_chart_path is not None:
        rel_chart = Path(comparison_chart_path).relative_to(output_dir)
        lines.append(f"![Model comparison]({rel_chart.as_posix()})")
        lines.append("")

    for metric in METRICS:
        metric_df = results_df[results_df["metric"] == metric]
        lines.append(f"### {METRIC_LABELS[metric]}")
        lines.append("")
        lines.append("| Dataset | " + " | ".join(models) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(models)) + " |")
        for dataset_name in datasets:
            row_cells = []
            for model in models:
                row = metric_df[(metric_df["dataset"] == dataset_name) & (metric_df["model"] == model)]
                if row.empty:
                    row_cells.append("n/a")
                else:
                    mean = row.iloc[0]["mean"]
                    std = row.iloc[0]["std"]
                    row_cells.append(f"{mean:.4f} ± {std:.4f}")
            lines.append(f"| {dataset_name} | " + " | ".join(row_cells) + " |")
        lines.append("")


def generate_markdown_report(
    results_df: pd.DataFrame,
    model_meta: Dict[str, dict],
    dataset_infos: List[dict],
    output_dir: Path,
    report_path: Path,
    comparison_chart_path: Optional[Path] = None,
) -> None:
    lines: List[str] = []
    lines.append("# Patch Embedding Distance Report")
    lines.append("")
    lines.append(
        "This report measures how consistent patch-token embeddings are *within* each "
        "mini dataset, for every model. Clips in a mini dataset may have different "
        "lengths, so each clip's patch-token grid is first collapsed into a single "
        "video embedding - averaging over spatial positions within a frame, then "
        "across frames, then L2-normalizing - before every pair of clips is compared. "
        "All metrics are magnitude-invariant, so a uniform rescaling of an embedding "
        "vector does not change its reported distance to another embedding."
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("All metrics are reported as *distances* (lower = more similar):")
    lines.append("")
    lines.append("- **Cosine distance** = 1 − cosine similarity. Range [0, 2].")
    lines.append(
        "- **Angular distance** = arccos(cosine similarity) / π. A true metric "
        "(satisfies the triangle inequality) derived from cosine similarity. Range [0, 1]."
    )
    lines.append(
        "- **Pearson correlation distance** = 1 − Pearson correlation between the two "
        "embedding vectors. Invariant to magnitude *and* additive shift. Range [0, 2]."
    )
    lines.append(
        "- **Spearman correlation distance** = 1 − Spearman rank correlation. "
        "Invariant to any monotonic transform of the embedding values. Range [0, 2]."
    )
    lines.append("")
    lines.append("## Mini datasets")
    lines.append("")
    lines.append("| Dataset | Manifest | Candidate clips sampled |")
    lines.append("| --- | --- | --- |")
    for info in dataset_infos:
        lines.append(f"| {info['name']} | `{info['csv']}` | {info['num_candidates']} |")
    lines.append("")

    if results_df.empty:
        lines.append("No results were produced - check the logs for errors.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    for model_name in model_meta:
        model_df = results_df[results_df["model"] == model_name]
        if model_df.empty:
            continue

        meta = model_meta[model_name]
        lines.append(f"## {model_name}")
        lines.append("")
        lines.append(f"- Checkpoint config: `{meta['config']}`")
        lines.append(f"- Embedding dim: {meta.get('embed_dim', 'n/a')}")
        lines.append("")

        chart_path = meta.get("chart_path")
        if chart_path is not None:
            rel_chart = Path(chart_path).relative_to(output_dir)
            lines.append(f"![{model_name} distance chart]({rel_chart.as_posix()})")
            lines.append("")

        lines.append(
            "| Dataset | Clips used | Pairs | "
            + " | ".join(METRIC_LABELS[m] for m in METRICS)
            + " |"
        )
        lines.append("| --- | --- | --- | " + " | ".join(["---"] * len(METRICS)) + " |")

        for dataset_name in dict.fromkeys(model_df["dataset"]):
            row_df = model_df[model_df["dataset"] == dataset_name]
            if row_df.empty:
                continue
            first = row_df.iloc[0]
            cells = []
            for metric in METRICS:
                metric_row = row_df[row_df["metric"] == metric]
                if metric_row.empty:
                    cells.append("n/a")
                    continue
                mean = metric_row.iloc[0]["mean"]
                std = metric_row.iloc[0]["std"]
                cells.append(f"{mean:.4f} ± {std:.4f}")
            lines.append(
                f"| {dataset_name} | {int(first['num_samples_used'])} | "
                f"{int(first['num_pairs'])} | " + " | ".join(cells) + " |"
            )

        lines.append("")

    append_model_comparison_section(lines, results_df, model_meta, output_dir, comparison_chart_path)

    lines.append("## Raw data")
    lines.append("")
    lines.append(f"Full per-model/per-dataset/per-metric statistics: `{(output_dir / 'distance_stats.csv').name}`")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a patch embedding distance report")
    parser.add_argument(
        "--output-dir",
        default="preliminary_experiments/evals/vitl/vjepa_ek100_patch_distance",
        help="Directory where the report artifacts will be saved.",
    )
    parser.add_argument("--num-samples", type=int, default=24, help="Number of clips to sample per mini dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of clips to run through the encoder per forward pass.",
    )
    return parser.parse_args()


def main() -> None:
    validate_datasets_config(DATASETS_CONFIG)

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(str(output_dir / "patch_distance_report.log"))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Sample every mini dataset once so every model is evaluated on the exact
    # same clips, keeping cross-model comparisons meaningful.
    dataset_infos = []
    for dataset_cfg in DATASETS_CONFIG:
        try:
            dataset_infos.append(sample_dataset_manifest(dataset_cfg, args.num_samples, args.seed, output_dir))
        except Exception as exc:
            logger.error("Failed to sample dataset %s: %s", dataset_cfg["name"], exc)
    logger.info("Prepared %d mini dataset(s)", len(dataset_infos))

    results_rows: List[dict] = []
    model_meta: Dict[str, dict] = {}

    for model_cfg in MODELS_CONFIG:
        model_name = model_cfg["name"]
        model_slug = slugify(model_name)
        logger.info("%s", "=" * 80)
        logger.info("Processing model: %s", model_name)
        logger.info("%s", "=" * 80)

        try:
            config = load_config(model_cfg["config"])
            encoder, _ = prepare_encoder(config, device)
        except Exception as exc:
            logger.error("Failed to load %s: %s", model_name, exc)
            continue

        model_meta[model_name] = {"config": model_cfg["config"]}
        model_dir = output_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        for dataset_info in dataset_infos:
            embeddings = collect_grid_embeddings(
                encoder,
                config,
                dataset_info["csv"],
                batch_size=args.batch_size,
                device=device,
                model_name=model_name,
            )
            if not embeddings:
                logger.warning("No embeddings collected for %s / %s", model_name, dataset_info["name"])
                continue

            stats = analyze_dataset(embeddings)
            if stats is None:
                logger.warning(
                    "Not enough usable clips to form pairs for %s / %s", model_name, dataset_info["name"]
                )
                continue

            grid_meta = stats.pop("_grid_shape")
            model_meta[model_name]["embed_dim"] = grid_meta["embed_dim"]

            for metric_name, values in stats.items():
                results_rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset_info["name"],
                        "metric": metric_name,
                        "mean": values["mean"],
                        "std": values["std"],
                        "num_pairs": values["num_pairs"],
                        "num_samples_used": grid_meta["num_samples_used"],
                        "embed_dim": grid_meta["embed_dim"],
                    }
                )

        del encoder
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results_rows)
    stats_csv_path = output_dir / "distance_stats.csv"
    results_df.to_csv(stats_csv_path, index=False)
    logger.info("Saved raw distance statistics to %s", stats_csv_path)

    for model_name in model_meta:
        model_df = results_df[results_df["model"] == model_name] if not results_df.empty else results_df
        if model_df.empty:
            continue
        model_slug = slugify(model_name)
        chart_path = output_dir / model_slug / "distance_comparison.png"
        save_distance_bar_chart(model_name, model_df, chart_path)
        model_meta[model_name]["chart_path"] = str(chart_path)

    comparison_chart_path = None
    if not results_df.empty and len(model_meta) > 1:
        comparison_chart_path = output_dir / "model_comparison.png"
        save_model_comparison_chart(results_df, comparison_chart_path)

    report_path = output_dir / "patch_distance_report.md"
    generate_markdown_report(
        results_df, model_meta, dataset_infos, output_dir, report_path, comparison_chart_path
    )
    logger.info("Markdown report saved to %s", report_path)


if __name__ == "__main__":
    main()
