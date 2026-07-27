#!/usr/bin/env python3
"""
Verb kNN Probing Report Generator

Evaluates every model in MODELS_CONFIG side by side on EPIC-KITCHENS-100 action
recognition, restricted to the *verb* task, with a non-parametric probe instead
of the attentive classifier used by evals/video_classification_frozen/eval.py:

- Load the frozen encoder (same loading code as the other report generators)
- Run the action-recognition *train* and *validation* manifests through the same
  dataloader stack as the supervised eval (VideoDataset + AttnMaskCollator, so
  variable-length action segments are padded and masked)
- Mean-average pool every valid patch token of a clip (all spatial positions of
  all temporal tokens) into a single clip embedding, then L2-normalize it
- Fit a k-nearest-neighbour probe on the train embeddings and score the
  validation embeddings with it (cosine similarity, uniform and
  softmax-weighted votes, with and without train-mean centering)
- Render a Markdown report comparing all models, all k values, and the
  per-verb behaviour of each model's best configuration

The encoder is the only learned component involved: no gradient step is taken,
so the numbers measure how linearly/locally separable each frozen
representation is for verbs, not how well a probe can be trained on it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.generate_patch_embedding_report import (
    apply_pretrained_frame_skip,
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

# Vote weighting schemes for the kNN probe.
#   uniform     - every one of the k neighbours contributes 1 vote
#   similarity  - neighbour i contributes max(cos_sim_i, 0): a mild
#                 closer-neighbours-count-more weighting
#   softmax     - neighbour i contributes exp(cos_sim_i / temperature), the
#                 weighting used by the standard DINO/MoCo kNN evaluation. At the
#                 usual temperature this is very peaked, so it behaves close to
#                 1-NN unless the temperature is raised.
WEIGHTINGS = ("uniform", "similarity", "softmax")

# Feature post-processing applied before the cosine kNN.
#   l2         - pooled clip embedding, L2-normalized
#   center_l2  - subtract the *train* feature mean, then re-L2-normalize. Removes
#                the large component shared by every clip of a dataset, which
#                usually dominates raw ViT features.
PREPROCS = ("l2", "center_l2")


def metric_labels(secondary_topk: int = 5, class_noun: str = "verb") -> Dict[str, str]:
    """Chart/table labels for the three headline metrics. The class noun is a
    parameter so the sibling action-group report can reuse these helpers."""
    return {
        "top1": "Top-1 accuracy (%)",
        "topk": f"Top-{secondary_topk} accuracy (%)",
        "mean_class_acc": f"Mean per-{class_noun} accuracy (%)",
    }


@dataclass
class FeatureSet:
    features: np.ndarray  # (N, D) float32, L2-normalized pooled clip embeddings
    labels: np.ndarray  # (N,) int64 verb class indices
    num_clips_seen: int = 0
    num_clips_skipped: int = 0
    elapsed_s: float = 0.0
    token_counts: List[int] = field(default_factory=list)

    @property
    def embed_dim(self) -> int:
        return int(self.features.shape[1]) if self.features.size else 0


def configure_logging(log_path: str) -> None:
    """Same shape as generate_patch_embedding_report.configure_logging, but bound
    to *this* module's logger - the sibling helper configures its own logger, so
    reusing it leaves this report's log file empty."""
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    logging.captureWarnings(True)


def load_label_map(json_path: Optional[str]) -> Dict[int, str]:
    if not json_path:
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            return {int(k): v for k, v in json.load(handle).items()}
    except Exception as exc:
        logger.warning("Could not load label map %s: %s", json_path, exc)
        return {}


def class_name(idx: int, mapping: Dict[int, str], fallback_prefix: str = "verb") -> str:
    return mapping.get(int(idx), f"{fallback_prefix}_{int(idx)}")


def prepare_split_manifest(
    csv_path: str,
    num_samples: int,
    seed: int,
    output_dir: Path,
    split: str,
) -> dict:
    """Subsample a split's manifest once, up front, so every model is evaluated
    on the exact same clips. `num_samples <= 0` keeps the full manifest."""
    manifest = load_manifest(csv_path)
    if num_samples and num_samples > 0:
        sampled = sample_rows(manifest, num_samples, seed)
    else:
        sampled = manifest.reset_index(drop=True)

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    sampled_csv = split_dir / f"{split}_manifest.csv"
    sampled.to_csv(sampled_csv, sep=" ", header=False, index=False)

    labels = sampled["label"].astype(int).to_numpy() if "label" in sampled.columns else np.zeros(len(sampled), int)
    return {
        "split": split,
        "source_csv": csv_path,
        "csv": str(sampled_csv),
        "num_clips": int(len(sampled)),
        "num_clips_available": int(len(manifest)),
        "num_verbs_present": int(len(np.unique(labels))),
        "labels": labels,
    }


def pool_clip_embeddings(view_output: torch.Tensor, token_validity: torch.Tensor) -> torch.Tensor:
    """(B, N, D) token embeddings -> (B, D) clip embeddings.

    Mean-average pools every *valid* token of the clip (all spatial positions of
    all temporal tokens) and L2-normalizes the result. Padding tokens introduced
    by AttnMaskCollator for shorter action segments are excluded so they cannot
    pull the average toward zero.
    """
    weights = token_validity.unsqueeze(-1).to(torch.float32)
    summed = (view_output.to(torch.float32) * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1.0)
    return F.normalize(summed / counts, dim=-1)


def extract_features(
    encoder,
    config: dict,
    dataset_csv: str,
    batch_size: int,
    device: str,
    model_name: str,
    split: str,
    use_amp: bool,
) -> FeatureSet:
    """Encode every clip of `dataset_csv` into one pooled embedding + its verb label."""
    args_data = config["data"]
    patch_size = args_data["patch_size"]
    tubelet_size = args_data["tubelet_size"]

    data_loader, frame_skip_info = load_data_for_model(config, dataset_csv, batch_size=batch_size)

    feature_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    token_counts: List[int] = []
    num_seen = 0
    num_skipped = 0
    start = time.time()

    desc = f"{model_name} [{split}]"
    for batch_idx, data in enumerate(tqdm(data_loader, total=len(data_loader), desc=desc)):
        try:
            clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
            labels = data[1]
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
                _, _, t_post, height, width = clips[0][0].shape
                if t_post < tubelet_size:
                    # Every clip of this batch was shorter than one
                    # frames_to_skip-sized chunk, so nothing survives the skip.
                    num_skipped += int(clips[0][0].shape[0])
                    num_seen += int(clips[0][0].shape[0])
                    continue
                num_spatial_tokens = (height // patch_size) * (width // patch_size)
                num_tokens_post = (t_post // tubelet_size) * num_spatial_tokens
                attn_mask = [
                    [
                        rebuild_attn_mask_after_frame_skip(
                            pre_skip_validity,
                            num_spatial_tokens=num_spatial_tokens,
                            tubelet_size=tubelet_size,
                            frames_to_skip=frames_to_skip,
                            previous_tubelet_size=previous_tubelet_size,
                            num_tokens_post=num_tokens_post,
                        )
                    ]
                ]

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp and device.startswith("cuda")):
                    outputs, out_attn_mask = encoder(clips, clip_indices, attn_mask)

                view_output = outputs[0]
                token_validity = torch.diagonal(out_attn_mask[0][:, 0], dim1=-2, dim2=-1)
                pooled = pool_clip_embeddings(view_output, token_validity)

            valid_counts = token_validity.sum(dim=1)
            keep = valid_counts > 0
            num_seen += int(keep.shape[0])
            num_skipped += int((~keep).sum().item())
            if not bool(keep.any()):
                continue

            feature_chunks.append(pooled[keep].detach().cpu().numpy().astype(np.float32, copy=False))
            label_chunks.append(np.asarray(labels, dtype=np.int64)[keep.detach().cpu().numpy()])
            token_counts.extend(int(c) for c in valid_counts[keep].detach().cpu().numpy())
        except Exception as exc:
            logger.error("Failed processing %s batch %d for %s: %s", split, batch_idx, model_name, exc)

    elapsed = time.time() - start
    if not feature_chunks:
        return FeatureSet(
            features=np.zeros((0, 0), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            num_clips_seen=num_seen,
            num_clips_skipped=num_skipped,
            elapsed_s=elapsed,
        )

    return FeatureSet(
        features=np.concatenate(feature_chunks, axis=0),
        labels=np.concatenate(label_chunks, axis=0),
        num_clips_seen=num_seen,
        num_clips_skipped=num_skipped,
        elapsed_s=elapsed,
        token_counts=token_counts,
    )


def postprocess_features(
    train_features: np.ndarray,
    val_features: np.ndarray,
    preproc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if preproc == "l2":
        return train_features, val_features
    if preproc == "center_l2":
        # Centering uses the *train* mean only - the validation split must never
        # inform its own representation.
        mean = train_features.mean(axis=0, keepdims=True)
        train_centered = train_features - mean
        val_centered = val_features - mean
        train_centered /= np.maximum(np.linalg.norm(train_centered, axis=1, keepdims=True), 1e-12)
        val_centered /= np.maximum(np.linalg.norm(val_centered, axis=1, keepdims=True), 1e-12)
        return train_centered, val_centered
    raise ValueError(f"Unknown feature post-processing: {preproc}")


def knn_neighbours(
    train_features: np.ndarray,
    val_features: np.ndarray,
    max_k: int,
    device: str,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine top-`max_k` neighbours of every validation clip among the train
    clips. Returns (similarities, train indices), both (num_val, max_k)."""
    train = torch.from_numpy(train_features).to(device)
    max_k = min(max_k, train.shape[0])

    sims_out, idx_out = [], []
    for start in range(0, val_features.shape[0], chunk_size):
        query = torch.from_numpy(val_features[start : start + chunk_size]).to(device)
        sims = query @ train.T
        top_sims, top_idx = sims.topk(max_k, dim=1)
        sims_out.append(top_sims.cpu())
        idx_out.append(top_idx.cpu())

    del train
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(sims_out, dim=0), torch.cat(idx_out, dim=0)


def knn_scores(
    neighbour_sims: torch.Tensor,
    neighbour_idx: torch.Tensor,
    train_labels: np.ndarray,
    num_classes: int,
    k: int,
    weighting: str,
    temperature: float,
) -> np.ndarray:
    """Class vote scores (num_val, num_classes) from the cached neighbour lists."""
    sims = neighbour_sims[:, :k].to(torch.float32)
    labels = torch.from_numpy(train_labels)[neighbour_idx[:, :k]]

    if weighting == "uniform":
        weights = torch.ones_like(sims)
    elif weighting == "similarity":
        weights = sims.clamp(min=0.0)
    elif weighting == "softmax":
        weights = (sims / temperature).exp()
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    scores = torch.zeros(sims.shape[0], num_classes, dtype=torch.float32)
    scores.scatter_add_(1, labels, weights)
    return scores.numpy()


def accuracy_metrics(
    scores: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
    secondary_topk: int = 5,
) -> Tuple[dict, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="stable")
    top1 = order[:, 0]
    topk = order[:, : min(secondary_topk, scores.shape[1])]

    correct1 = top1 == val_labels
    # A class that received no vote at all is not a prediction: with small k most
    # classes score exactly 0, and counting them as top-k candidates would credit
    # whichever class indices happen to sort first.
    topk_scores = np.take_along_axis(scores, topk, axis=1)
    correct_topk = ((topk == val_labels[:, None]) & (topk_scores > 0)).any(axis=1)

    per_class = np.full(num_classes, np.nan, dtype=np.float64)
    for cls in np.unique(val_labels):
        cls_mask = val_labels == cls
        per_class[cls] = 100.0 * float(correct1[cls_mask].mean())

    metrics = {
        "top1": 100.0 * float(correct1.mean()),
        "topk": 100.0 * float(correct_topk.mean()),
        "mean_class_acc": float(np.nanmean(per_class)),
        # How many distinct classes the probe is willing to predict at all. A large
        # k buys overall accuracy by collapsing onto the frequent classes, and this
        # column is what makes that visible.
        "classes_predicted": int(np.unique(top1).size),
    }
    return metrics, top1


def evaluate_model_knn(
    train_set: FeatureSet,
    val_set: FeatureSet,
    k_values: Sequence[int],
    temperature: float,
    num_classes: int,
    device: str,
    secondary_topk: int = 5,
) -> Tuple[List[dict], Dict[str, dict]]:
    """Every (preproc, k, weighting) combination for one model. The expensive
    part - the neighbour search - is done once per preproc and reused.

    Two winners are returned, because on a long-tailed label set they are rarely
    the same configuration: `top1` maximizes overall accuracy, `balanced`
    maximizes mean per-verb accuracy.
    """
    rows: List[dict] = []
    winners: Dict[str, dict] = {}
    max_k = min(max(k_values), train_set.features.shape[0])

    for preproc in PREPROCS:
        train_features, val_features = postprocess_features(train_set.features, val_set.features, preproc)
        neighbour_sims, neighbour_idx = knn_neighbours(train_features, val_features, max_k, device)

        for k in k_values:
            if k > max_k:
                logger.warning("Skipping k=%d: only %d train clips available", k, max_k)
                continue
            for weighting in WEIGHTINGS:
                scores = knn_scores(
                    neighbour_sims, neighbour_idx, train_set.labels, num_classes, k, weighting, temperature
                )
                metrics, predictions = accuracy_metrics(
                    scores, val_set.labels, num_classes, secondary_topk=secondary_topk
                )
                row = {"preproc": preproc, "k": k, "weighting": weighting, **metrics}
                rows.append(row)
                candidate = {
                    "preproc": preproc,
                    "k": k,
                    "weighting": weighting,
                    "metrics": metrics,
                    "predictions": predictions,
                }
                for criterion, metric_key in (("top1", "top1"), ("balanced", "mean_class_acc")):
                    incumbent = winners.get(criterion)
                    if incumbent is None or metrics[metric_key] > incumbent["metrics"][metric_key]:
                        winners[criterion] = candidate

    return rows, winners


def winner_row(model_name: str, winner: dict) -> dict:
    return {
        "model": model_name,
        "preproc": winner["preproc"],
        "k": winner["k"],
        "weighting": winner["weighting"],
        **winner["metrics"],
    }


def save_features(model_dir: Path, split_infos: Dict[str, dict], train_set: FeatureSet, val_set: FeatureSet) -> None:
    np.save(model_dir / "train_features.npy", train_set.features)
    np.save(model_dir / "train_labels.npy", train_set.labels)
    np.save(model_dir / "val_features.npy", val_set.features)
    np.save(model_dir / "val_labels.npy", val_set.labels)
    meta = {
        "train_source_csv": split_infos["train"]["source_csv"],
        "val_source_csv": split_infos["val"]["source_csv"],
        "train_clips_requested": split_infos["train"]["num_clips"],
        "val_clips_requested": split_infos["val"]["num_clips"],
        "train_tokens_per_clip_mean": float(np.mean(train_set.token_counts)) if train_set.token_counts else None,
        "val_tokens_per_clip_mean": float(np.mean(val_set.token_counts)) if val_set.token_counts else None,
        "encode_seconds": train_set.elapsed_s + val_set.elapsed_s,
    }
    (model_dir / "feature_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_cached_features(model_dir: Path, split_infos: Dict[str, dict]) -> Optional[Dict[str, FeatureSet]]:
    """Reload the pooled embeddings of a previous run so the probe can be re-swept
    without paying for the encoder again. Returns None unless the cache exists and
    was produced from the same manifests and clip counts."""
    meta_path = model_dir / "feature_meta.json"
    paths = {
        split: (model_dir / f"{split}_features.npy", model_dir / f"{split}_labels.npy") for split in ("train", "val")
    }
    if not meta_path.exists() or not all(p.exists() for pair in paths.values() for p in pair):
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Ignoring unreadable feature cache in %s: %s", model_dir, exc)
        return None

    for split in ("train", "val"):
        if meta.get(f"{split}_source_csv") != split_infos[split]["source_csv"] or meta.get(
            f"{split}_clips_requested"
        ) != split_infos[split]["num_clips"]:
            logger.warning(
                "Feature cache in %s was built from different %s clips - re-encoding",
                model_dir,
                split,
            )
            return None

    feature_sets = {}
    for split, (features_path, labels_path) in paths.items():
        features = np.load(features_path)
        labels = np.load(labels_path)
        feature_sets[split] = FeatureSet(
            features=features,
            labels=labels,
            num_clips_seen=int(features.shape[0]),
            elapsed_s=float(meta.get("encode_seconds", 0.0)) / 2.0,
            token_counts=[int(round(meta.get(f"{split}_tokens_per_clip_mean") or 0))] * int(features.shape[0]),
        )
    return feature_sets


def per_class_table(
    val_labels: np.ndarray,
    predictions: np.ndarray,
    train_labels: np.ndarray,
    label_map: Dict[int, str],
    fallback_prefix: str = "verb",
) -> pd.DataFrame:
    rows = []
    train_counts = Counter(int(v) for v in train_labels)
    for cls in np.unique(val_labels):
        cls_mask = val_labels == cls
        rows.append(
            {
                "class_idx": int(cls),
                "class_name": class_name(int(cls), label_map, fallback_prefix),
                "val_clips": int(cls_mask.sum()),
                "train_clips": int(train_counts.get(int(cls), 0)),
                "accuracy": 100.0 * float((predictions[cls_mask] == cls).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("val_clips", ascending=False).reset_index(drop=True)


def confusion_pairs(
    val_labels: np.ndarray,
    predictions: np.ndarray,
    label_map: Dict[int, str],
    top_n: int = 10,
    fallback_prefix: str = "verb",
) -> pd.DataFrame:
    wrong = predictions != val_labels
    counts = Counter(zip(val_labels[wrong].tolist(), predictions[wrong].tolist()))
    rows = [
        {
            "true_class": class_name(true_cls, label_map, fallback_prefix),
            "predicted_class": class_name(pred_cls, label_map, fallback_prefix),
            "count": count,
            "share_of_errors": 100.0 * count / max(1, int(wrong.sum())),
        }
        for (true_cls, pred_cls), count in counts.most_common(top_n)
    ]
    return pd.DataFrame(rows)


def baseline_metrics(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
    secondary_topk: int = 5,
) -> dict:
    """Two reference points the probe has to beat: always predicting the most
    frequent train class, and uniform random guessing."""
    if train_labels.size == 0 or val_labels.size == 0:
        return {}
    counts = Counter(int(v) for v in train_labels)
    majority_cls = counts.most_common(1)[0][0]
    topk_classes = [cls for cls, _ in counts.most_common(secondary_topk)]
    return {
        "majority_class_idx": majority_cls,
        "majority_top1": 100.0 * float((val_labels == majority_cls).mean()),
        "majority_topk": 100.0 * float(np.isin(val_labels, topk_classes).mean()),
        "chance_top1": 100.0 / num_classes,
        "secondary_topk": secondary_topk,
    }


def save_accuracy_vs_k_chart(results_df: pd.DataFrame, output_path: Path, class_noun: str = "verb") -> None:
    if results_df.empty:
        return

    models = list(dict.fromkeys(results_df["model"]))
    fig, axes = plt.subplots(1, len(PREPROCS), figsize=(6.2 * len(PREPROCS), 5), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, preproc in zip(axes, PREPROCS):
        preproc_df = results_df[results_df["preproc"] == preproc]
        for model in models:
            for weighting, linestyle in zip(WEIGHTINGS, ("-", "--", ":")):
                series = preproc_df[(preproc_df["model"] == model) & (preproc_df["weighting"] == weighting)]
                series = series.sort_values("k")
                if series.empty:
                    continue
                ax.plot(
                    series["k"],
                    series["top1"],
                    marker="o",
                    linestyle=linestyle,
                    label=f"{model} ({weighting})",
                )
        ax.set_xscale("log")
        ax.set_xticks(sorted(preproc_df["k"].unique()))
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("k (number of neighbours)")
        ax.set_title(f"Features: {preproc}")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(f"{class_noun.capitalize()} top-1 accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, max(1, len(labels))), bbox_to_anchor=(0.5, 1.12), fontsize=9)
    fig.suptitle(f"kNN {class_noun} probing: sensitivity to k", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_headline_chart(
    best_rows: List[dict],
    baselines: dict,
    output_path: Path,
    class_noun: str = "verb",
    secondary_topk: int = 5,
) -> None:
    if not best_rows:
        return

    labels_by_metric = metric_labels(secondary_topk, class_noun)
    models = [row["model"] for row in best_rows]
    metrics = list(labels_by_metric)
    fig, ax = plt.subplots(figsize=(max(7, 2.4 * len(models) + 3), 5.4), dpi=150)
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(models))

    for i, row in enumerate(best_rows):
        values = [row[m] for m in metrics]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=f"{row['model']} (k={row['k']}, {row['weighting']})")
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

    ax.set_ylim(0, max(1.0, max(row["topk"] for row in best_rows)) * 1.2)
    if baselines:
        ax.axhline(baselines["majority_top1"], color="#64748b", linestyle="--", linewidth=1.2)
        ax.text(
            -0.48,
            baselines["majority_top1"],
            f"majority {class_noun} ({baselines['majority_top1']:.1f}%)",
            va="bottom",
            ha="left",
            fontsize=8,
            color="#475569",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([labels_by_metric[m] for m in metrics], fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Best kNN {class_noun} probe per model")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_per_class_heatmap(
    per_class_frames: Dict[str, pd.DataFrame],
    output_path: Path,
    top_n: int = 20,
    class_noun: str = "verb",
) -> None:
    if not per_class_frames:
        return

    reference = next(iter(per_class_frames.values()))
    classes = reference.head(top_n)
    if classes.empty:
        return

    models = list(per_class_frames)
    matrix = np.full((len(models), len(classes)), np.nan)
    for row_idx, model in enumerate(models):
        frame = per_class_frames[model].set_index("class_idx")
        for col_idx, class_idx in enumerate(classes["class_idx"]):
            if class_idx in frame.index:
                matrix[row_idx, col_idx] = frame.loc[class_idx, "accuracy"]

    fig_height = max(2.6, 0.7 * len(models) + 1.8)
    fig, ax = plt.subplots(figsize=(max(9, 0.62 * len(classes) + 3), fig_height), dpi=150)
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(
        [f"{row.class_name}\n(n={row.val_clips})" for row in classes.itertuples()],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            if not np.isnan(matrix[row_idx, col_idx]):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{matrix[row_idx, col_idx]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[row_idx, col_idx] < 60 else "black",
                )
    fig.colorbar(image, ax=ax, label=f"Per-{class_noun} top-1 accuracy (%)", fraction=0.025, pad=0.02)
    ax.set_title(
        f"Per-{class_noun} accuracy on the {len(classes)} most frequent validation "
        f"{class_noun}s (top-1-best config per model)"
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def fmt_pct(value: float) -> str:
    return f"{value:.2f}"


def generate_markdown_report(
    report_path: Path,
    output_dir: Path,
    results_df: pd.DataFrame,
    model_meta: Dict[str, dict],
    split_infos: Dict[str, dict],
    baselines: dict,
    per_class_frames: Dict[str, pd.DataFrame],
    confusions: Dict[str, pd.DataFrame],
    charts: Dict[str, Optional[Path]],
    skipped_models: List[Tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    lines: List[str] = []
    best_rows = [meta["best_row"] for meta in model_meta.values() if meta.get("best_row")]
    best_rows.sort(key=lambda row: row["top1"], reverse=True)

    lines.append("# Frozen-Encoder Verb kNN Probing Report")
    lines.append("")
    lines.append(
        "How much *verb* structure does each frozen V-JEPA encoder already expose, with no "
        "trained probe on top? Each model turns every EPIC-KITCHENS-100 action clip into a "
        "single embedding by mean-average pooling all of its patch tokens; a k-nearest-neighbour "
        "vote over the training clips then predicts the verb of every validation clip. Nothing "
        "is fine-tuned, so the score is a direct read-out of how well verbs are already grouped "
        "in the representation."
    )
    lines.append("")

    # -- TL;DR
    lines.append("## TL;DR")
    lines.append("")
    if not best_rows:
        lines.append("No model produced usable features - check the log for errors.")
        lines.append("")
    else:
        leader = best_rows[0]
        margin = ""
        if len(best_rows) > 1:
            margin = (
                f" - {leader['top1'] - best_rows[1]['top1']:+.2f} points ahead of "
                f"**{best_rows[1]['model']}** ({fmt_pct(best_rows[1]['top1'])}%)"
            )
        lines.append(
            f"- **{leader['model']}** leads with **{fmt_pct(leader['top1'])}% verb top-1** "
            f"(k={leader['k']}, {leader['weighting']} votes, {leader['preproc']} features){margin}."
        )
        if baselines:
            uplift = leader["top1"] / max(baselines["majority_top1"], 1e-9)
            lines.append(
                f"- That is **{uplift:.1f}x** the majority-verb baseline "
                f"({fmt_pct(baselines['majority_top1'])}%) and "
                f"**{leader['top1'] / max(baselines['chance_top1'], 1e-9):.0f}x** random chance "
                f"({fmt_pct(baselines['chance_top1'])}%)."
            )
        lines.append(
            f"- Mean per-verb accuracy is only **{fmt_pct(leader['mean_class_acc'])}%** against "
            f"**{fmt_pct(leader['top1'])}%** overall: the probe rides EPIC-KITCHENS' heavy verb "
            f"imbalance, predicting just {leader['classes_predicted']} distinct verbs out of "
            f"{split_infos['val']['num_verbs_present']} present in the validation split."
        )
        leader_balanced = model_meta[leader["model"]].get("balanced_row")
        if leader_balanced:
            lines.append(
                f"- The trade-off is explicit: at k={leader_balanced['k']} "
                f"({leader_balanced['weighting']}, {leader_balanced['preproc']}) the same features "
                f"reach **{fmt_pct(leader_balanced['mean_class_acc'])}% mean per-verb** across "
                f"{leader_balanced['classes_predicted']} predicted verbs, at the cost of "
                f"{leader_balanced['top1'] - leader['top1']:+.2f} points of top-1. Small k spreads "
                "predictions over the tail; large k collapses onto `take`/`put`/`wash`."
            )
        lines.append("")

    # -- Headline table
    if best_rows:
        if charts.get("headline") is not None:
            lines.append(f"![Best kNN probe per model]({Path(charts['headline']).relative_to(output_dir).as_posix()})")
            lines.append("")
        lines.append("### Best configuration per model (selected on top-1)")
        lines.append("")
        lines.append(
            "| Model | Top-1 (%) | Top-5 (%) | Mean per-verb (%) | Verbs predicted | Best k | Votes | "
            "Features | Embed dim |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in best_rows:
            meta = model_meta[row["model"]]
            lines.append(
                f"| {row['model']} | **{fmt_pct(row['top1'])}** | {fmt_pct(row['topk'])} | "
                f"{fmt_pct(row['mean_class_acc'])} | {row['classes_predicted']} | {row['k']} | "
                f"{row['weighting']} | {row['preproc']} | {meta.get('embed_dim', 'n/a')} |"
            )
        if baselines:
            lines.append(
                f"| _majority verb baseline_ | {fmt_pct(baselines['majority_top1'])} | "
                f"{fmt_pct(baselines['majority_topk'])} | "
                f"{fmt_pct(100.0 / max(1, split_infos['val']['num_verbs_present']))} | 1 | - | - | - | - |"
            )
            lines.append(
                f"| _random chance_ | {fmt_pct(baselines['chance_top1'])} | "
                f"{fmt_pct(5 * baselines['chance_top1'])} | {fmt_pct(baselines['chance_top1'])} | - | - | - | - | - |"
            )
        lines.append("")

        balanced_rows = [meta["balanced_row"] for meta in model_meta.values() if meta.get("balanced_row")]
        if balanced_rows:
            lines.append("### Same features, selected on mean per-verb accuracy instead")
            lines.append("")
            lines.append(
                "One number cannot summarise a 97-way, long-tailed problem. Selecting the probe on "
                "overall top-1 rewards a configuration that answers `take`, `put` or `wash` and is "
                "right often enough; selecting on mean per-verb accuracy rewards one that actually "
                "attempts the tail. Both are reported so neither can flatter a model on its own."
            )
            lines.append("")
            lines.append(
                "| Model | Mean per-verb (%) | Top-1 (%) | Top-5 (%) | Verbs predicted | k | Votes | Features |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for row in sorted(balanced_rows, key=lambda r: r["mean_class_acc"], reverse=True):
                lines.append(
                    f"| {row['model']} | **{fmt_pct(row['mean_class_acc'])}** | {fmt_pct(row['top1'])} | "
                    f"{fmt_pct(row['topk'])} | {row['classes_predicted']} | {row['k']} | "
                    f"{row['weighting']} | {row['preproc']} |"
                )
            lines.append("")

    # -- Setup
    lines.append("## Setup")
    lines.append("")
    lines.append("### Protocol")
    lines.append("")
    lines.append(
        "1. **Frozen encoder.** Each model is loaded exactly as in the supervised eval "
        "(`vit_encoder_multiclip` wrapper around the pretrained `target_encoder`), in eval mode, "
        "with gradients disabled."
    )
    lines.append(
        "2. **Same dataloader as the supervised eval.** `VideoDataset` + `AttnMaskCollator`, so "
        "variable-length action segments are zero-padded within a batch and the attention mask "
        "marks which patch tokens are real. Per-model clip sampling (`fps`, `dataset_fpcs`, "
        "`crop_size`, `patch_size`, `tubelet_size`) is taken from that model's own pretraining "
        "config, so every encoder sees clips shaped the way it was trained on."
    )
    lines.append(
        "3. **Clip embedding.** All valid patch tokens of a clip - every spatial position of every "
        "temporal token - are mean-average pooled into one vector, which is then L2-normalized. "
        "Padding tokens are excluded from the mean."
    )
    lines.append(
        f"4. **kNN probe.** Cosine similarity against the pooled train embeddings; "
        f"k ∈ {{{', '.join(str(k) for k in args.k_values)}}}; three vote weightings (`uniform`, "
        f"`similarity` = max(cos, 0), `softmax` = exp(cos / {args.temperature}) as in the standard "
        "DINO kNN eval); features used raw (`l2`) and train-mean-centered (`center_l2`). Two "
        "configurations are reported per model: the one maximizing top-1 accuracy and the one "
        "maximizing mean per-verb accuracy."
    )
    lines.append("")
    lines.append(
        "Only the *verb* head of the action-recognition task is evaluated, on the "
        "`*_verbs.csv` manifests (97 verb classes). Models are encoded one after another over "
        "the identical, pre-sampled clip lists, so all numbers in this report are directly "
        "comparable."
    )
    lines.append("")

    lines.append("### Data")
    lines.append("")
    lines.append("| Split | Manifest | Clips used | Clips available | Verbs present |")
    lines.append("| --- | --- | --- | --- | --- |")
    for split in ("train", "val"):
        info = split_infos[split]
        lines.append(
            f"| {split} | `{info['source_csv']}` | {info['num_clips']} | "
            f"{info['num_clips_available']} | {info['num_verbs_present']} |"
        )
    lines.append("")
    unseen = split_infos.get("val_unseen_verbs")
    if unseen:
        lines.append(
            f"> {unseen['clips']} validation clips ({unseen['share']:.2f}%) belong to verbs that never "
            f"appear in the sampled train split, so the probe cannot possibly get them right."
        )
        lines.append("")

    lines.append("### Models")
    lines.append("")
    lines.append(
        "| Model | Pretraining config | Checkpoint | Embed dim | Train clips encoded | Val clips encoded | "
        "Mean tokens/clip | Encode time (min) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for model_name, meta in model_meta.items():
        lines.append(
            f"| {model_name} | `{meta['config']}` | `{meta['checkpoint']}` | {meta.get('embed_dim', 'n/a')} | "
            f"{meta.get('num_train', 0)} | {meta.get('num_val', 0)} | "
            f"{meta.get('mean_tokens', float('nan')):.0f} | {meta.get('encode_minutes', float('nan')):.1f} |"
        )
    lines.append("")
    if skipped_models:
        lines.append("Models declared in `MODELS_CONFIG` but skipped:")
        lines.append("")
        for name, reason in skipped_models:
            lines.append(f"- **{name}** - {reason}")
        lines.append("")

    # -- Full grid
    if not results_df.empty:
        lines.append("## Sensitivity to the probe's hyper-parameters")
        lines.append("")
        if charts.get("accuracy_vs_k") is not None:
            rel = Path(charts["accuracy_vs_k"]).relative_to(output_dir).as_posix()
            lines.append(f"![Accuracy vs k]({rel})")
            lines.append("")
        lines.append(
            "A kNN probe has no learned parameters, but it does have knobs. The table below is the "
            "full grid; it is worth checking that a model's ranking is stable across it rather than "
            "an artefact of one lucky k."
        )
        lines.append("")
        lines.append(
            "| Model | Features | Votes | k | Top-1 (%) | Top-5 (%) | Mean per-verb (%) | Verbs predicted |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        grid = results_df.sort_values(["model", "preproc", "weighting", "k"])
        for row in grid.itertuples():
            lines.append(
                f"| {row.model} | {row.preproc} | {row.weighting} | {row.k} | {fmt_pct(row.top1)} | "
                f"{fmt_pct(row.topk)} | {fmt_pct(row.mean_class_acc)} | {row.classes_predicted} |"
            )
        lines.append("")
        spread = (
            results_df.groupby("model")["top1"].agg(["min", "max"]).assign(spread=lambda d: d["max"] - d["min"])
        )
        for model_name, row in spread.iterrows():
            lines.append(
                f"- **{model_name}**: top-1 ranges {fmt_pct(row['min'])}% - {fmt_pct(row['max'])}% "
                f"across the grid (spread {fmt_pct(row['spread'])} points)."
            )
        lines.append("")

    # -- Per-verb behaviour
    if per_class_frames:
        lines.append("## Per-verb behaviour")
        lines.append("")
        lines.append(
            "Everything in this section uses each model's **top-1-best** configuration, so it "
            "describes the same predictions as the headline table."
        )
        lines.append("")
        if charts.get("per_class") is not None:
            rel = Path(charts["per_class"]).relative_to(output_dir).as_posix()
            lines.append(f"![Per-verb accuracy heatmap]({rel})")
            lines.append("")
        for model_name, frame in per_class_frames.items():
            best_row = model_meta[model_name]["best_row"]
            lines.append(
                f"### {model_name} (k={best_row['k']}, {best_row['weighting']}, {best_row['preproc']})"
            )
            lines.append("")
            scored = frame[frame["val_clips"] >= args.min_class_clips]
            if scored.empty:
                lines.append(
                    f"No verb reaches {args.min_class_clips} validation clips, so the per-verb "
                    "breakdown is omitted."
                )
                lines.append("")
                continue
            # Cap each half at len/2 so a verb can never appear as both best and worst.
            half = max(1, min(8, len(scored) // 2))
            ranked = scored.sort_values("accuracy", ascending=False)
            best_verbs = ranked.head(half)
            worst_verbs = ranked.tail(half).iloc[::-1]
            lines.append(
                f"Verbs with at least {args.min_class_clips} validation clips, best and worst "
                f"{half} ({len(scored)} of {len(frame)} verbs qualify):"
            )
            lines.append("")
            lines.append("| Best verbs | Acc (%) | Val clips | Train clips | | Worst verbs | Acc (%) | Val clips | Train clips |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for (_, good), (_, bad) in zip(best_verbs.iterrows(), worst_verbs.iterrows()):
                lines.append(
                    f"| {good['class_name']} | {fmt_pct(good['accuracy'])} | {good['val_clips']} | "
                    f"{good['train_clips']} | | {bad['class_name']} | {fmt_pct(bad['accuracy'])} | "
                    f"{bad['val_clips']} | {bad['train_clips']} |"
                )
            lines.append("")
            conf = confusions.get(model_name)
            if conf is not None and not conf.empty:
                lines.append("Most frequent confusions:")
                lines.append("")
                lines.append("| True verb | Predicted verb | Errors | Share of all errors (%) |")
                lines.append("| --- | --- | --- | --- |")
                for row in conf.itertuples():
                    lines.append(
                        f"| {row.true_class} | {row.predicted_class} | {row.count} | "
                        f"{fmt_pct(row.share_of_errors)} |"
                    )
                lines.append("")

    # -- Caveats
    lines.append("## How to read these numbers")
    lines.append("")
    lines.append(
        "- **This is not the supervised eval number.** The attentive-probe eval trains a 4-block "
        "attentive classifier for 20 epochs and sweeps 20 lr/wd settings; here nothing is trained "
        "and all spatial/temporal detail is collapsed by a single mean. Expect these numbers to sit "
        "well below the supervised probe - they are a *cheap, comparable* diagnostic, not a "
        "state-of-the-art claim."
    )
    lines.append(
        "- **Mean-pooling is deliberately blunt.** It discards where and when things happen, which is "
        "exactly the information verbs depend on. A model can therefore look weak here and still be a "
        "good backbone for a probe that attends over tokens."
    )
    lines.append(
        "- **Top-1 vs mean per-verb.** EPIC-KITCHENS verbs are heavily imbalanced. Top-1 rewards "
        "getting `take`/`put` right; mean per-verb weights each of the "
        f"{split_infos['val']['num_verbs_present']} verbs present in the validation split equally, "
        "and is the harder, more honest number. The 'verbs predicted' column shows how many verbs "
        "the probe is willing to answer at all - a configuration that wins on top-1 while naming "
        "only a third of the vocabulary is winning by declining to guess."
    )
    if split_infos["train"]["num_clips"] < split_infos["train"]["num_clips_available"]:
        lines.append(
            f"- **Subsampled splits.** {split_infos['train']['num_clips']} of "
            f"{split_infos['train']['num_clips_available']} train clips and "
            f"{split_infos['val']['num_clips']} of {split_infos['val']['num_clips_available']} "
            "validation clips were used. kNN accuracy grows with the size of the train bank, so "
            "absolute values would rise with the full manifests; the comparison between models "
            "(identical clips) is unaffected."
        )
    lines.append("")

    # -- Artifacts
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Full metric grid: `{(output_dir / 'knn_results.csv').name}`")
    lines.append(f"- Per-verb accuracies: `{(output_dir / 'per_verb_accuracy.csv').name}`")
    lines.append(f"- Sampled clip lists: `splits/train_manifest.csv`, `splits/val_manifest.csv`")
    lines.append(
        "- Per-model pooled embeddings and labels: "
        "`<model-slug>/{train,val}_{features,labels}.npy`, described by `<model-slug>/feature_meta.json`"
    )
    lines.append(
        "- Predictions of the two selected configurations: "
        "`<model-slug>/val_predictions.npy` (top-1-best) and "
        "`<model-slug>/val_predictions_balanced.npy` (mean-per-verb-best)"
    )
    lines.append(f"- Run log: `{(output_dir / 'knn_probe_report.log').name}`")
    lines.append("")
    lines.append(
        f"Reproduce with `python evals/generate_knn_probe_report.py --train-csv {args.train_csv} "
        f"--val-csv {args.val_csv} --num-train-clips {args.num_train_clips} "
        f"--num-val-clips {args.num_val_clips} --batch-size {args.batch_size} --seed {args.seed}`. "
        "Add `--reuse-features` to re-sweep the probe on the cached embeddings without re-running "
        "the encoders."
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a frozen-encoder verb kNN probing report")
    parser.add_argument(
        "--train-csv",
        default="data/EK100_action_recognition_train_verbs.csv",
        help="Action-recognition train manifest (video_path verb_class).",
    )
    parser.add_argument(
        "--val-csv",
        default="data/EK100_action_recognition_validation_verbs.csv",
        help="Action-recognition validation manifest (video_path verb_class).",
    )
    parser.add_argument(
        "--verb-labels-json",
        default="preliminary_experiments/evals/vitl/vjepa_ek100_ar/qualitative_eval/verb_labels.json",
        help="JSON mapping verb class index -> verb name (used for the per-verb tables).",
    )
    parser.add_argument(
        "--output-dir",
        default="preliminary_experiments/evals/vitl/vjepa_ek100_knn_probe",
        help="Directory where the report artifacts will be saved.",
    )
    parser.add_argument("--num-classes", type=int, default=97, help="Number of verb classes.")
    parser.add_argument(
        "--num-train-clips",
        type=int,
        default=8000,
        help="Number of train clips to encode (<=0 uses the full manifest).",
    )
    parser.add_argument(
        "--num-val-clips",
        type=int,
        default=4000,
        help="Number of validation clips to encode (<=0 uses the full manifest).",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 50],
        help="Neighbour counts to evaluate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for similarity-weighted votes.",
    )
    parser.add_argument(
        "--min-class-clips",
        type=int,
        default=20,
        help="Minimum validation clips for a verb to appear in the best/worst per-verb tables.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for clip sampling.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of clips per encoder forward pass.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the dataloader worker count from the model's pretraining config.",
    )
    parser.add_argument(
        "--allow-variable-length",
        dest="allow_variable_length",
        action="store_true",
        default=True,
        help="Keep action segments at their native length (masked), as the supervised eval does.",
    )
    parser.add_argument(
        "--no-allow-variable-length",
        dest="allow_variable_length",
        action="store_false",
        help="Pad/truncate every clip to a fixed length instead.",
    )
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", default=True, help="Disable fp16 autocast.")
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help=(
            "Reuse pooled embeddings already saved under --output-dir instead of re-encoding, "
            "when they were produced from the same manifests and clip counts. Lets the kNN grid "
            "be re-swept in seconds."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(str(output_dir / "knn_probe_report.log"))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    verb_map = load_label_map(args.verb_labels_json)

    # Sample both splits once so every model is evaluated on identical clips.
    split_infos = {
        "train": prepare_split_manifest(args.train_csv, args.num_train_clips, args.seed, output_dir, "train"),
        "val": prepare_split_manifest(args.val_csv, args.num_val_clips, args.seed, output_dir, "val"),
    }
    logger.info(
        "Train clips: %d (of %d) | Val clips: %d (of %d)",
        split_infos["train"]["num_clips"],
        split_infos["train"]["num_clips_available"],
        split_infos["val"]["num_clips"],
        split_infos["val"]["num_clips_available"],
    )

    train_verbs = set(split_infos["train"]["labels"].tolist())
    val_labels_manifest = split_infos["val"]["labels"]
    unseen_mask = ~np.isin(val_labels_manifest, list(train_verbs))
    split_infos["val_unseen_verbs"] = {
        "clips": int(unseen_mask.sum()),
        "share": 100.0 * float(unseen_mask.mean()) if val_labels_manifest.size else 0.0,
    }

    results_rows: List[dict] = []
    model_meta: Dict[str, dict] = {}
    per_class_frames: Dict[str, pd.DataFrame] = {}
    confusions: Dict[str, pd.DataFrame] = {}
    per_class_rows: List[pd.DataFrame] = []
    skipped_models: List[Tuple[str, str]] = []
    baselines: dict = {}

    for model_cfg in MODELS_CONFIG:
        model_name = model_cfg["name"]
        model_slug = slugify(model_name)
        logger.info("%s", "=" * 80)
        logger.info("Processing model: %s", model_name)
        logger.info("%s", "=" * 80)

        model_dir = output_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        feature_sets = load_cached_features(model_dir, split_infos) if args.reuse_features else None
        if feature_sets is not None:
            logger.info("Reusing cached features for %s from %s", model_name, model_dir)
        else:
            try:
                config = load_config(model_cfg["config"])
                # The pretraining config drives clip sampling, but it never had to
                # cope with variable-length action segments; the supervised eval
                # config turns this on, so mirror it here.
                config["data"]["allow_variable_length"] = args.allow_variable_length
                if args.num_workers is not None:
                    config["data"]["num_workers"] = args.num_workers
                encoder, _ = prepare_encoder(config, device)
            except Exception as exc:
                logger.error("Failed to load %s: %s", model_name, exc)
                skipped_models.append((model_name, f"could not be loaded ({exc})"))
                continue

            feature_sets = {}
            for split in ("train", "val"):
                feature_sets[split] = extract_features(
                    encoder,
                    config,
                    split_infos[split]["csv"],
                    batch_size=args.batch_size,
                    device=device,
                    model_name=model_name,
                    split=split,
                    use_amp=args.use_amp,
                )
                logger.info(
                    "%s [%s]: %d clips encoded (%d skipped) in %.1f min",
                    model_name,
                    split,
                    feature_sets[split].features.shape[0],
                    feature_sets[split].num_clips_skipped,
                    feature_sets[split].elapsed_s / 60.0,
                )

            del encoder
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        train_set, val_set = feature_sets["train"], feature_sets["val"]
        if train_set.features.shape[0] < 2 or val_set.features.shape[0] < 1:
            logger.error("Not enough features for %s - skipping", model_name)
            skipped_models.append((model_name, "produced too few usable clip embeddings"))
            continue

        save_features(model_dir, split_infos, train_set, val_set)

        rows, winners = evaluate_model_knn(
            train_set,
            val_set,
            k_values=args.k_values,
            temperature=args.temperature,
            num_classes=args.num_classes,
            device=device,
        )
        if not rows or not winners:
            logger.error("kNN evaluation produced no results for %s", model_name)
            skipped_models.append((model_name, "kNN evaluation produced no results"))
            continue

        for row in rows:
            results_rows.append({"model": model_name, **row})

        best = winners["top1"]
        balanced = winners["balanced"]
        np.save(model_dir / "val_predictions.npy", best["predictions"])
        np.save(model_dir / "val_predictions_balanced.npy", balanced["predictions"])
        if not baselines:
            baselines = baseline_metrics(train_set.labels, val_set.labels, args.num_classes)

        frame = per_class_table(val_set.labels, best["predictions"], train_set.labels, verb_map)
        per_class_frames[model_name] = frame
        per_class_rows.append(frame.assign(model=model_name))
        confusions[model_name] = confusion_pairs(val_set.labels, best["predictions"], verb_map)

        all_tokens = train_set.token_counts + val_set.token_counts
        model_meta[model_name] = {
            "config": model_cfg["config"],
            "checkpoint": model_cfg.get("checkpoint", "n/a"),
            "embed_dim": train_set.embed_dim,
            "num_train": int(train_set.features.shape[0]),
            "num_val": int(val_set.features.shape[0]),
            "mean_tokens": float(np.mean(all_tokens)) if all_tokens else float("nan"),
            "encode_minutes": (train_set.elapsed_s + val_set.elapsed_s) / 60.0,
            "best_row": winner_row(model_name, best),
            "balanced_row": winner_row(model_name, balanced),
        }

        for criterion, winner in (("top-1", best), ("mean-per-verb", balanced)):
            logger.info(
                "%s best by %s: top1=%.2f%% top5=%.2f%% mean-per-verb=%.2f%% verbs-predicted=%d "
                "(k=%d, %s, %s)",
                model_name,
                criterion,
                winner["metrics"]["top1"],
                winner["metrics"]["topk"],
                winner["metrics"]["mean_class_acc"],
                winner["metrics"]["classes_predicted"],
                winner["k"],
                winner["weighting"],
                winner["preproc"],
            )

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(output_dir / "knn_results.csv", index=False)
    if per_class_rows:
        pd.concat(per_class_rows, ignore_index=True).to_csv(output_dir / "per_verb_accuracy.csv", index=False)

    charts: Dict[str, Optional[Path]] = {"accuracy_vs_k": None, "headline": None, "per_class": None}
    if not results_df.empty:
        charts["accuracy_vs_k"] = output_dir / "accuracy_vs_k.png"
        save_accuracy_vs_k_chart(results_df, charts["accuracy_vs_k"])

        best_rows = [meta["best_row"] for meta in model_meta.values() if meta.get("best_row")]
        best_rows.sort(key=lambda row: row["top1"], reverse=True)
        charts["headline"] = output_dir / "best_per_model.png"
        save_headline_chart(best_rows, baselines, charts["headline"])

    if per_class_frames:
        charts["per_class"] = output_dir / "per_verb_accuracy.png"
        save_per_class_heatmap(per_class_frames, charts["per_class"])

    report_path = output_dir / "knn_probe_report.md"
    generate_markdown_report(
        report_path=report_path,
        output_dir=output_dir,
        results_df=results_df,
        model_meta=model_meta,
        split_infos=split_infos,
        baselines=baselines,
        per_class_frames=per_class_frames,
        confusions=confusions,
        charts=charts,
        skipped_models=skipped_models,
        args=args,
    )
    logger.info("Markdown report saved to %s", report_path)


if __name__ == "__main__":
    main()
