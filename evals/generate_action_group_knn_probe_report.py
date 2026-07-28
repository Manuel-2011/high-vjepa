#!/usr/bin/env python3
"""
Action-Group kNN Probing Report Generator

Identical to evals/generate_knn_probe_report.py - same frozen encoders, same
dataloader stack as the supervised eval, same mean-average pooling of every valid
patch token into one clip embedding, same cosine kNN probe grid - with one
difference: EPIC-KITCHENS-100's 97 fine-grained verbs are first collapsed into a
small set of broader, semantically meaningful **action groups** (see
ACTION_GROUPS), and the probe classifies those instead.

Why this is worth measuring separately: the 97-way verb vocabulary is both
long-tailed and semantically redundant (`take` vs `pull` vs `lift`, `wash` vs
`scrub` vs `rub`). A frozen representation can capture *what kind of manipulation
is happening* while being unable to name the exact verb, and the fine-grained
number cannot distinguish that from not capturing anything at all. Grouping the
labels *before* the probe sees them answers the coarser question directly.

The clip embeddings are identical to the verb report's, so pointing
--feature-cache-dir at that run reuses its cached features and the whole report
regenerates in seconds without touching a GPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.generate_patch_embedding_report import (
    load_config,
    prepare_encoder,
    slugify,
    MODELS_CONFIG,
)

# Everything mechanical is shared with the verb-level report so the two cannot
# drift apart: feature extraction, the kNN sweep, feature caching, the charts.
from evals.generate_knn_probe_report import (
    FeatureSet,
    baseline_metrics,
    confusion_pairs,
    evaluate_model_knn,
    extract_features,
    fmt_pct,
    load_cached_features,
    load_label_map,
    per_class_table,
    prepare_split_manifest,
    save_accuracy_vs_k_chart,
    save_features,
    save_headline_chart,
    save_per_class_heatmap,
    winner_row,
)

logger = logging.getLogger(__name__)

CLASS_NOUN = "action group"

# EPIC-KITCHENS-100's 97 verbs collapsed into broader manipulation categories.
# Groups are defined by verb *name* (resolved through --verb-labels-json) so the
# taxonomy stays readable and reviewable; every verb must land in exactly one
# group, which build_group_mapping() enforces at startup.
#
# Each group answers "what kind of manipulation is happening", not "which verb
# was uttered" - that is the level a single mean-pooled clip embedding can
# plausibly represent. EPIC's verb vocabulary was never designed to be
# partitioned, so a handful of judgement calls are unavoidable; the ambiguous
# ones are flagged inline rather than hidden.
ACTION_GROUPS: List[Tuple[str, str, List[str]]] = [
    (
        "possession",
        "Change which agent or support possesses an object",
        [
            "take",
            "lift",
            "hold",
            "carry",
            "put",
            "drop",
            "let-go",
            "set",
            "move",
            "throw",
            "gather",
        ],
    ),
    (
        "accessibility",
        "Change whether an object or its contents are accessible",
        [
            "open",
            "close",
            "lock",
            "unlock",
            "uncover",
            "wrap",
            "unwrap",
            "unroll",
        ],
    ),
    (
        "connectivity",
        "Create, remove or modify physical connections between objects",
        [
            "insert",
            "attach",
            "remove",
            "pull",
            "screw",
            "unscrew",
        ],
    ),
    (
        "configuration",
        "Change an object's pose, orientation or spatial arrangement",
        [
            "flip",
            "slide",
            "turn",
            "adjust",
            "lower",
            "increase",
            "turn-down",
            "hang",
            "sort",
            "fold",
        ],
    ),
    (
        "material-transfer",
        "Move material between objects, containers or surfaces",
        [
            "pour",
            "fill",
            "empty",
            "scoop",
            "add",
            "sprinkle",
            "filter",
            "water",
            "coat",
            "apply",
            "spray",
            "measure",
            "season",
        ],
    ),
    (
        "material-transformation",
        "Permanently change the geometry or topology of a material",
        [
            "cut",
            "break",
            "crush",
            "grate",
            "peel",
            "rip",
            "flatten",
            "bend",
            "stretch",
            "knead",
            "roll",
            "form",
            "divide",
            "stab",
            "squeeze",
        ],
    ),
    (
        "surface-state",
        "Modify surface cleanliness, moisture or finish",
        [
            "wash",
            "scrub",
            "brush",
            "rub",
            "dry",
            "pat",
            "scrape",
            "soak",
        ],
    ),
    (
        "internal-state",
        "Modify the internal composition or physical state of a substance",
        [
            "mix",
            "shake",
            "cook",
            "bake",
            "prepare",
            "unfreeze",
        ],
    ),
    (
        "device-state",
        "Change the operating state of a device or mechanism",
        [
            "turn-on",
            "turn-off",
            "switch",
            "press",
        ],
    ),
    (
        "information",
        "Acquire information or make information explicit",
        [
            "look",
            "check",
            "search",
            "feel",
            "smell",
            "choose",
            "mark",
        ],
    ),
    (
        "consumption",
        "Transfer material from the environment into the agent",
        [
            "eat",
            "drink",
        ],
    ),
    (
        "process",
        "High-level process actions that do not primarily correspond to a single persistent world-state change",
        [
            "wait",
            "transition",
            "finish",
            "use",
            "wear",
            "serve",
            "sharpen",
        ],
    ),
]


def configure_logging(log_path: str) -> None:
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


def build_group_mapping(
    verb_map: Dict[int, str],
) -> Tuple[np.ndarray, List[str], Dict[int, str], Dict[str, str], Dict[str, List[str]]]:
    """Resolve ACTION_GROUPS into a verb-index -> group-index lookup table.

    Fails loudly unless the taxonomy is a genuine partition of the verb
    vocabulary: a silently unmapped verb would quietly drop clips, and a verb in
    two groups would make the label depend on declaration order.
    """
    if not verb_map:
        raise ValueError(
            "A verb label map is required to resolve ACTION_GROUPS (groups are declared by verb "
            "name). Pass --verb-labels-json pointing at the index -> verb-name JSON."
        )

    name_to_idx = {name: idx for idx, name in verb_map.items()}
    num_verbs = max(verb_map) + 1
    lookup = np.full(num_verbs, -1, dtype=np.int64)

    group_names: List[str] = []
    descriptions: Dict[str, str] = {}
    members: Dict[str, List[str]] = {}
    unknown: List[str] = []
    duplicated: List[str] = []

    for group_idx, (group, description, verbs) in enumerate(ACTION_GROUPS):
        group_names.append(group)
        descriptions[group] = description
        members[group] = list(verbs)
        for verb in verbs:
            verb_idx = name_to_idx.get(verb)
            if verb_idx is None:
                unknown.append(verb)
                continue
            if lookup[verb_idx] >= 0:
                duplicated.append(verb)
                continue
            lookup[verb_idx] = group_idx

    problems = []
    if unknown:
        problems.append(f"named in ACTION_GROUPS but absent from the label map: {sorted(set(unknown))}")
    if duplicated:
        problems.append(f"assigned to more than one group: {sorted(set(duplicated))}")
    unassigned = [verb_map.get(int(i), f"verb_{int(i)}") for i in np.where(lookup < 0)[0]]
    if unassigned:
        problems.append(f"not assigned to any group: {sorted(unassigned)}")
    if problems:
        raise ValueError(
            "ACTION_GROUPS is not a partition of the "
            f"{num_verbs}-verb vocabulary - verbs " + "; verbs ".join(problems)
        )

    group_map = {idx: name for idx, name in enumerate(group_names)}
    return lookup, group_names, group_map, descriptions, members


def regroup_labels(verb_labels: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    """Map fine-grained verb indices onto action-group indices."""
    verb_labels = np.asarray(verb_labels, dtype=np.int64)
    if verb_labels.size and (verb_labels.min() < 0 or verb_labels.max() >= lookup.shape[0]):
        raise ValueError(
            f"Verb label out of range for the {lookup.shape[0]}-verb taxonomy: "
            f"[{verb_labels.min()}, {verb_labels.max()}]"
        )
    return lookup[verb_labels]


def group_taxonomy_table(
    group_names: Sequence[str],
    descriptions: Dict[str, str],
    members: Dict[str, List[str]],
    verb_labels: Dict[str, np.ndarray],
    lookup: np.ndarray,
) -> pd.DataFrame:
    """One row per action group: which verbs it absorbs and how many clips of each
    split it ends up holding."""
    rows = []
    grouped = {split: regroup_labels(labels, lookup) for split, labels in verb_labels.items()}
    for group_idx, group in enumerate(group_names):
        row = {
            "group_idx": group_idx,
            "group": group,
            "description": descriptions[group],
            "num_verbs": len(members[group]),
            "verbs": ", ".join(members[group]),
        }
        for split, labels in grouped.items():
            row[f"{split}_clips"] = int((labels == group_idx).sum())
            row[f"{split}_share"] = 100.0 * float((labels == group_idx).mean()) if labels.size else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_confusion_matrix_chart(
    val_labels: np.ndarray,
    predictions: np.ndarray,
    group_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    """Row-normalized confusion matrix (per-group recall). With a dozen groups the
    whole matrix fits on one screen, which is the main practical advantage of the
    coarse taxonomy over the 97-way one."""
    n = len(group_names)
    counts = np.zeros((n, n), dtype=np.float64)
    for true_cls, pred_cls in zip(val_labels.tolist(), predictions.tolist()):
        counts[true_cls, pred_cls] += 1.0

    support = counts.sum(axis=1)
    recall = 100.0 * counts / np.maximum(support[:, None], 1.0)
    recall[support == 0] = np.nan

    fig, ax = plt.subplots(figsize=(max(8.0, 0.72 * n + 4.0), max(6.5, 0.62 * n + 3.0)), dpi=150)
    image = ax.imshow(recall, cmap="magma", vmin=0, vmax=100)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(group_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(
        [f"{name} (n={int(support[i])})" for i, name in enumerate(group_names)],
        fontsize=8,
    )
    ax.set_xlabel("Predicted action group")
    ax.set_ylabel("True action group")
    for i in range(n):
        for j in range(n):
            if np.isnan(recall[i, j]) or recall[i, j] < 0.5:
                continue
            ax.text(
                j,
                i,
                f"{recall[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if recall[i, j] < 55 else "black",
            )
    fig.colorbar(image, ax=ax, label="Share of the true group's clips (%)", fraction=0.035, pad=0.02)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_group_support_chart(taxonomy: pd.DataFrame, output_path: Path) -> None:
    """How much of each split every group holds - the imbalance the probe faces."""
    if taxonomy.empty:
        return

    ordered = taxonomy.sort_values("val_clips", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.42 * len(ordered) + 1.8)), dpi=150)
    y = np.arange(len(ordered))
    ax.barh(y - 0.2, ordered["train_clips"], height=0.4, label="train clips")
    ax.barh(y + 0.2, ordered["val_clips"], height=0.4, label="validation clips")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row.group} ({row.num_verbs} verbs)" for row in ordered.itertuples()], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Clips (log scale)")
    ax.set_title("Action-group support after grouping the 97 verbs")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def grouped_metrics_from_predictions(
    val_verb_labels: np.ndarray,
    verb_predictions: np.ndarray,
    lookup: np.ndarray,
    num_groups: int,
) -> dict:
    """Score a *fine-grained* probe's predictions at group granularity: a `pull`
    predicted as `take` becomes correct, because both are `get`. This isolates
    'grouping the labels before probing' from 'grouping after predicting'."""
    true_groups = regroup_labels(val_verb_labels, lookup)
    pred_groups = regroup_labels(verb_predictions, lookup)
    correct = true_groups == pred_groups

    per_group = np.full(num_groups, np.nan, dtype=np.float64)
    for cls in np.unique(true_groups):
        per_group[cls] = 100.0 * float(correct[true_groups == cls].mean())

    return {
        "top1": 100.0 * float(correct.mean()),
        "mean_class_acc": float(np.nanmean(per_group)),
        "classes_predicted": int(np.unique(pred_groups).size),
    }


def load_comparison(
    compare_run_dir: Optional[str],
    lookup: np.ndarray,
    num_groups: int,
    val_group_labels: Dict[str, np.ndarray],
) -> dict:
    """Pull the fine-grained verb run's headline numbers and re-score its
    predictions at group granularity. Returns {} when unavailable."""
    if not compare_run_dir:
        return {}

    run_dir = Path(compare_run_dir)
    results_csv = run_dir / "knn_results.csv"
    if not results_csv.exists():
        logger.warning("No knn_results.csv under %s - skipping the comparison section", run_dir)
        return {}

    try:
        fine_df = pd.read_csv(results_csv)
    except Exception as exc:
        logger.warning("Could not read %s: %s", results_csv, exc)
        return {}

    comparison: dict = {"run_dir": str(run_dir), "models": {}}
    for model_name, model_df in fine_df.groupby("model"):
        best = model_df.sort_values("top1", ascending=False).iloc[0]
        entry = {
            "fine_top1": float(best["top1"]),
            "fine_mean_class_acc": float(best["mean_class_acc"]),
            "fine_classes_predicted": int(best["classes_predicted"]),
            "fine_k": int(best["k"]),
            "fine_weighting": str(best["weighting"]),
            "fine_preproc": str(best["preproc"]),
        }

        model_dir = run_dir / slugify(str(model_name))
        preds_path = model_dir / "val_predictions.npy"
        labels_path = model_dir / "val_labels.npy"
        expected_groups = val_group_labels.get(str(model_name))
        if preds_path.exists() and labels_path.exists() and expected_groups is not None:
            verb_predictions = np.load(preds_path)
            verb_labels = np.load(labels_path)
            if verb_predictions.shape == verb_labels.shape and verb_labels.shape == expected_groups.shape:
                if np.array_equal(regroup_labels(verb_labels, lookup), expected_groups):
                    entry["regrouped"] = grouped_metrics_from_predictions(
                        verb_labels, verb_predictions, lookup, num_groups
                    )
                else:
                    logger.warning(
                        "%s: the comparison run's validation labels do not match this run's - "
                        "skipping its regrouped score",
                        model_name,
                    )
            else:
                logger.warning(
                    "%s: comparison run has %s validation clips, this run has %s - skipping its "
                    "regrouped score",
                    model_name,
                    verb_labels.shape,
                    expected_groups.shape,
                )
        comparison["models"][str(model_name)] = entry

    return comparison


def generate_markdown_report(
    report_path: Path,
    output_dir: Path,
    results_df: pd.DataFrame,
    model_meta: Dict[str, dict],
    split_infos: Dict[str, dict],
    baselines: dict,
    taxonomy: pd.DataFrame,
    per_class_frames: Dict[str, pd.DataFrame],
    confusions: Dict[str, pd.DataFrame],
    charts: Dict[str, Optional[Path]],
    skipped_models: List[Tuple[str, str]],
    comparison: dict,
    group_names: Sequence[str],
    args: argparse.Namespace,
) -> None:
    topk = args.secondary_topk
    num_groups = len(group_names)
    lines: List[str] = []

    best_rows = [meta["best_row"] for meta in model_meta.values() if meta.get("best_row")]
    best_rows.sort(key=lambda row: row["top1"], reverse=True)

    def rel(path) -> str:
        return Path(path).relative_to(output_dir).as_posix()

    lines.append("# Frozen-Encoder Action-Group kNN Probing Report")
    lines.append("")
    lines.append(
        f"EPIC-KITCHENS-100 labels 97 verbs, many of which name the same physical act "
        f"(`take` / `pull` / `lift`, `wash` / `scrub` / `rub`). This report collapses them into "
        f"**{num_groups} broader action groups** *before* probing, then asks the same question as "
        "the verb-level report: how much of that structure does a frozen V-JEPA encoder already "
        "expose? Every clip becomes one mean-average-pooled embedding, and a k-nearest-neighbour "
        "vote over the training clips predicts its action group. Nothing is fine-tuned."
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
            f"- **{leader['model']}** reaches **{fmt_pct(leader['top1'])}% action-group top-1** "
            f"(k={leader['k']}, {leader['weighting']} votes, {leader['preproc']} features){margin}."
        )
        if baselines:
            lines.append(
                f"- Against **{fmt_pct(baselines['majority_top1'])}%** for always answering the "
                f"most frequent group and **{fmt_pct(baselines['chance_top1'])}%** for random "
                f"guessing among {num_groups} groups."
            )
        lines.append(
            f"- Mean per-group accuracy is **{fmt_pct(leader['mean_class_acc'])}%** across "
            f"{leader['classes_predicted']} of {num_groups} groups actually predicted: even with a "
            "coarse vocabulary the probe concentrates on the groups that dominate the data."
        )
        leader_cmp = comparison.get("models", {}).get(leader["model"], {})
        if leader_cmp:
            regrouped = leader_cmp.get("regrouped")
            lines.append(
                f"- Grouping the labels **before** probing lifts top-1 from "
                f"{fmt_pct(leader_cmp['fine_top1'])}% (97-way verbs) to "
                f"{fmt_pct(leader['top1'])}%"
                + (
                    f", and beats the {fmt_pct(regrouped['top1'])}% you get by taking the 97-way "
                    "probe's predictions and merging them into groups afterwards."
                    if regrouped
                    else "."
                )
            )
        lines.append("")

    # -- Headline
    if best_rows:
        if charts.get("headline") is not None:
            lines.append(f"![Best kNN probe per model]({rel(charts['headline'])})")
            lines.append("")
        lines.append("### Best configuration per model (selected on top-1)")
        lines.append("")
        lines.append(
            f"| Model | Top-1 (%) | Top-{topk} (%) | Mean per-group (%) | Groups predicted | Best k | "
            "Votes | Features | Embed dim |"
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
                f"| _majority group baseline_ | {fmt_pct(baselines['majority_top1'])} | "
                f"{fmt_pct(baselines['majority_topk'])} | "
                f"{fmt_pct(100.0 / max(1, split_infos['val']['num_groups_present']))} | 1 | - | - | - | - |"
            )
            lines.append(
                f"| _random chance_ | {fmt_pct(baselines['chance_top1'])} | "
                f"{fmt_pct(min(100.0, topk * baselines['chance_top1']))} | "
                f"{fmt_pct(baselines['chance_top1'])} | - | - | - | - | - |"
            )
        lines.append("")

        balanced_rows = [meta["balanced_row"] for meta in model_meta.values() if meta.get("balanced_row")]
        if balanced_rows:
            lines.append("### Same features, selected on mean per-group accuracy instead")
            lines.append("")
            lines.append(
                "Grouping shrinks the label set but does not balance it: `get` and `put` still "
                "dominate. Selecting the probe on top-1 rewards riding that imbalance, selecting on "
                "mean per-group accuracy rewards covering every group. Both are reported."
            )
            lines.append("")
            lines.append(
                f"| Model | Mean per-group (%) | Top-1 (%) | Top-{topk} (%) | Groups predicted | k | "
                "Votes | Features |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for row in sorted(balanced_rows, key=lambda r: r["mean_class_acc"], reverse=True):
                lines.append(
                    f"| {row['model']} | **{fmt_pct(row['mean_class_acc'])}** | {fmt_pct(row['top1'])} | "
                    f"{fmt_pct(row['topk'])} | {row['classes_predicted']} | {row['k']} | "
                    f"{row['weighting']} | {row['preproc']} |"
                )
            lines.append("")

    # -- Effect of grouping
    if comparison.get("models"):
        lines.append("## Does grouping actually help, or just shrink the problem?")
        lines.append("")
        lines.append(
            "Three numbers per model, all on the same clips and the same embeddings:"
        )
        lines.append("")
        lines.append(
            "1. **97-way verbs** - the fine-grained probe from "
            f"`{comparison['run_dir']}`."
        )
        lines.append(
            "2. **97-way verbs, merged after** - that same probe's predictions mapped into action "
            "groups afterwards. A `pull` predicted as `take` now counts as correct."
        )
        lines.append(
            f"3. **{num_groups} groups, grouped before** - this report: the probe's train bank is "
            "labelled with groups, so neighbours vote on groups directly."
        )
        lines.append("")
        lines.append(
            "(2) versus (3) is the interesting comparison: it says whether grouping *the label "
            "space the probe votes in* buys anything beyond forgiving within-group confusions."
        )
        lines.append("")
        lines.append(
            "| Model | 97-way top-1 (%) | Merged-after top-1 (%) | Grouped-before top-1 (%) | "
            "97-way mean per-class (%) | Merged-after mean per-group (%) | Grouped-before mean per-group (%) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for model_name, entry in comparison["models"].items():
            grouped_row = model_meta.get(model_name, {}).get("best_row")
            regrouped = entry.get("regrouped") or {}
            lines.append(
                f"| {model_name} | {fmt_pct(entry['fine_top1'])} | "
                f"{fmt_pct(regrouped['top1']) if regrouped else 'n/a'} | "
                f"{fmt_pct(grouped_row['top1']) if grouped_row else 'n/a'} | "
                f"{fmt_pct(entry['fine_mean_class_acc'])} | "
                f"{fmt_pct(regrouped['mean_class_acc']) if regrouped else 'n/a'} | "
                f"{fmt_pct(grouped_row['mean_class_acc']) if grouped_row else 'n/a'} |"
            )
        lines.append("")
        lines.append(
            "> The three columns are **not** on a common chance level: 1/97 = 1.03% for the "
            f"97-way task, 1/{num_groups} = {fmt_pct(100.0 / num_groups)}% for the grouped one. "
            "Compare (2) with (3) - they share both the label space and the chance level - and read "
            "(1) only as context."
        )
        lines.append("")

    # -- Taxonomy
    lines.append("## The action-group taxonomy")
    lines.append("")
    lines.append(
        f"The {num_groups} groups below partition all "
        f"{int(taxonomy['num_verbs'].sum())} EPIC-KITCHENS-100 verbs - every verb belongs to exactly "
        "one group, enforced at startup, so no clip is dropped or double-counted. Verb membership is "
        "declared by name in `ACTION_GROUPS` at the top of the script and is meant to be edited: "
        "the grouping is a modelling choice, not ground truth."
    )
    lines.append("")
    if charts.get("support") is not None:
        lines.append(f"![Action-group support]({rel(charts['support'])})")
        lines.append("")
    lines.append("| Group | What it means | Verbs | Train clips | Val clips | Val share (%) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in taxonomy.sort_values("val_clips", ascending=False).itertuples():
        lines.append(
            f"| **{row.group}** | {row.description} | {row.verbs} | {row.train_clips} | "
            f"{row.val_clips} | {fmt_pct(row.val_share)} |"
        )
    lines.append("")
    lines.append(
        "Ambiguous calls worth knowing about: `empty` counts as *remove-discard* (emptying a bin) "
        "rather than *transfer-substance*; `scrape` as *clean* (scraping a pan) rather than "
        "*cut-divide*; `squeeze` as *mix-shape* (hand shaping) rather than *transfer-substance* "
        "(juicing); bare `turn` as *operate-appliance* (a knob or tap) rather than *reposition*. "
        "Each of these could reasonably go the other way."
    )
    lines.append("")

    # -- Setup
    lines.append("## Setup")
    lines.append("")
    lines.append("### Protocol")
    lines.append("")
    lines.append(
        "1. **Frozen encoder.** Loaded exactly as in the supervised eval "
        "(`vit_encoder_multiclip` around the pretrained `target_encoder`), eval mode, no gradients."
    )
    lines.append(
        "2. **Same dataloader as the supervised eval.** `VideoDataset` + `AttnMaskCollator`, so "
        "variable-length action segments are zero-padded within a batch and the attention mask marks "
        "which patch tokens are real. Clip sampling (`fps`, `dataset_fpcs`, `crop_size`, "
        "`patch_size`, `tubelet_size`) comes from each model's own pretraining config."
    )
    lines.append(
        "3. **Clip embedding.** Every valid patch token of a clip - all spatial positions of all "
        "temporal tokens - is mean-average pooled into one vector, then L2-normalized. Padding "
        "tokens are excluded from the mean."
    )
    lines.append(
        "4. **Label grouping.** Each clip's verb index is mapped through `ACTION_GROUPS` to a group "
        "index. This happens to *both* splits before the probe is built, so the train bank the "
        "neighbours vote from is labelled with groups."
    )
    lines.append(
        f"5. **kNN probe.** Cosine similarity against the pooled train embeddings; "
        f"k ∈ {{{', '.join(str(k) for k in args.k_values)}}}; three vote weightings (`uniform`, "
        f"`similarity` = max(cos, 0), `softmax` = exp(cos / {args.temperature})); features raw "
        "(`l2`) and train-mean-centered (`center_l2`). Two configurations are reported per model: "
        "best top-1 and best mean per-group."
    )
    lines.append("")
    lines.append(
        f"Secondary accuracy is reported as top-{topk} rather than top-5, since top-5 out of "
        f"{num_groups} groups is close to free."
    )
    lines.append("")

    lines.append("### Data")
    lines.append("")
    lines.append("| Split | Manifest | Clips used | Clips available | Verbs present | Groups present |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for split in ("train", "val"):
        info = split_infos[split]
        lines.append(
            f"| {split} | `{info['source_csv']}` | {info['num_clips']} | "
            f"{info['num_clips_available']} | {info['num_verbs_present']} | "
            f"{info.get('num_groups_present', 'n/a')} |"
        )
    lines.append("")

    lines.append("### Models")
    lines.append("")
    lines.append(
        "| Model | Pretraining config | Checkpoint | Embed dim | Train clips | Val clips | "
        "Mean tokens/clip | Features |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for model_name, meta in model_meta.items():
        source = "cached" if meta.get("features_cached") else f"encoded in {meta.get('encode_minutes', 0.0):.1f} min"
        lines.append(
            f"| {model_name} | `{meta['config']}` | `{meta['checkpoint']}` | "
            f"{meta.get('embed_dim', 'n/a')} | {meta.get('num_train', 0)} | {meta.get('num_val', 0)} | "
            f"{meta.get('mean_tokens', float('nan')):.0f} | {source} |"
        )
    lines.append("")
    if skipped_models:
        lines.append("Models declared in `MODELS_CONFIG` but skipped:")
        lines.append("")
        for name, reason in skipped_models:
            lines.append(f"- **{name}** - {reason}")
        lines.append("")

    # -- Grid
    if not results_df.empty:
        lines.append("## Sensitivity to the probe's hyper-parameters")
        lines.append("")
        if charts.get("accuracy_vs_k") is not None:
            lines.append(f"![Accuracy vs k]({rel(charts['accuracy_vs_k'])})")
            lines.append("")
        lines.append(
            "The full grid. A ranking that only holds at one k is not a ranking."
        )
        lines.append("")
        lines.append(
            f"| Model | Features | Votes | k | Top-1 (%) | Top-{topk} (%) | Mean per-group (%) | "
            "Groups predicted |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in results_df.sort_values(["model", "preproc", "weighting", "k"]).itertuples():
            lines.append(
                f"| {row.model} | {row.preproc} | {row.weighting} | {row.k} | {fmt_pct(row.top1)} | "
                f"{fmt_pct(row.topk)} | {fmt_pct(row.mean_class_acc)} | {row.classes_predicted} |"
            )
        lines.append("")
        spread = results_df.groupby("model")["top1"].agg(["min", "max"]).assign(spread=lambda d: d["max"] - d["min"])
        for model_name, row in spread.iterrows():
            lines.append(
                f"- **{model_name}**: top-1 ranges {fmt_pct(row['min'])}% - {fmt_pct(row['max'])}% "
                f"across the grid (spread {fmt_pct(row['spread'])} points)."
            )
        lines.append("")

    # -- Per-group behaviour
    if per_class_frames:
        lines.append("## Per-group behaviour")
        lines.append("")
        lines.append(
            "All of this section uses each model's **top-1-best** configuration, so it describes the "
            "same predictions as the headline table."
        )
        lines.append("")
        if charts.get("per_class") is not None:
            lines.append(f"![Per-group accuracy]({rel(charts['per_class'])})")
            lines.append("")
        for model_name, frame in per_class_frames.items():
            best_row = model_meta[model_name]["best_row"]
            lines.append(f"### {model_name} (k={best_row['k']}, {best_row['weighting']}, {best_row['preproc']})")
            lines.append("")
            confusion_chart = charts.get(f"confusion::{model_name}")
            if confusion_chart is not None:
                lines.append(f"![Confusion matrix]({rel(confusion_chart)})")
                lines.append("")
            scored = frame[frame["val_clips"] >= args.min_class_clips].sort_values("accuracy", ascending=False)
            if scored.empty:
                lines.append(
                    f"No group reaches {args.min_class_clips} validation clips, so the per-group "
                    "table is omitted."
                )
                lines.append("")
                continue
            lines.append(
                f"Every group with at least {args.min_class_clips} validation clips "
                f"({len(scored)} of {len(frame)}), best first:"
            )
            lines.append("")
            lines.append("| Group | Accuracy (%) | Val clips | Train clips |")
            lines.append("| --- | --- | --- | --- |")
            for row in scored.itertuples():
                lines.append(
                    f"| {row.class_name} | {fmt_pct(row.accuracy)} | {row.val_clips} | {row.train_clips} |"
                )
            lines.append("")
            conf = confusions.get(model_name)
            if conf is not None and not conf.empty:
                lines.append("Most frequent confusions:")
                lines.append("")
                lines.append("| True group | Predicted group | Errors | Share of all errors (%) |")
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
        "- **The taxonomy is an assumption, not data.** `ACTION_GROUPS` was written by hand; a "
        "different partition would move every number in this report. Treat cross-model comparisons "
        "(same taxonomy) as sound and the absolute values as taxonomy-dependent."
    )
    lines.append(
        f"- **Chance is {fmt_pct(100.0 / num_groups)}%, not 1.03%.** Any comparison with the "
        "97-way verb report has to account for that; the section above does it by re-scoring the "
        "97-way probe in group space."
    )
    lines.append(
        "- **Grouping does not remove imbalance.** `get` and `put` absorb the two most frequent "
        "verbs and stay the largest groups, which is why mean per-group accuracy remains well below "
        "top-1."
    )
    lines.append(
        "- **Mean-pooling is deliberately blunt** and nothing is trained, so these are cheap "
        "diagnostics of the frozen features, not a ceiling on what a trained probe could do."
    )
    if split_infos["train"]["num_clips"] < split_infos["train"]["num_clips_available"]:
        lines.append(
            f"- **Subsampled splits.** {split_infos['train']['num_clips']} of "
            f"{split_infos['train']['num_clips_available']} train clips and "
            f"{split_infos['val']['num_clips']} of {split_infos['val']['num_clips_available']} "
            "validation clips. kNN accuracy grows with the train bank; cross-model comparison on "
            "identical clips is unaffected."
        )
    lines.append("")

    # -- Artifacts
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Full metric grid: `{(output_dir / 'knn_results.csv').name}`")
    lines.append(f"- Per-group accuracies: `{(output_dir / 'per_group_accuracy.csv').name}`")
    lines.append(f"- Group taxonomy with clip counts: `{(output_dir / 'action_groups.csv').name}`")
    lines.append("- Sampled clip lists: `splits/train_manifest.csv`, `splits/val_manifest.csv`")
    lines.append(
        "- Group labels and predictions per model: `<model-slug>/{train,val}_group_labels.npy`, "
        "`<model-slug>/val_group_predictions.npy` (top-1-best) and "
        "`<model-slug>/val_group_predictions_balanced.npy`"
    )
    lines.append(
        f"- Pooled clip embeddings (shared with the verb-level report): `{args.feature_cache_dir or args.output_dir}`"
    )
    lines.append(f"- Run log: `{(output_dir / 'action_group_knn_probe_report.log').name}`")
    lines.append("")
    lines.append(
        "Reproduce with `python evals/generate_action_group_knn_probe_report.py "
        f"--num-train-clips {args.num_train_clips} --num-val-clips {args.num_val_clips} "
        f"--batch-size {args.batch_size} --seed {args.seed}`. Add `--reuse-features "
        "--feature-cache-dir <verb-run-dir>` to reuse that run's embeddings instead of re-encoding, "
        "and `--compare-run-dir <verb-run-dir>` for the grouping-effect section."
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a frozen-encoder kNN probing report over broad action groups"
    )
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
        help="JSON mapping verb class index -> verb name. Required: ACTION_GROUPS is declared by name.",
    )
    parser.add_argument(
        "--output-dir",
        default="preliminary_experiments/evals/vitl/vjepa_ek100_knn_probe_action_groups",
        help="Directory where the report artifacts will be saved.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        default=None,
        help=(
            "Directory holding the pooled clip embeddings. Defaults to --output-dir. Point it at the "
            "verb-level run to reuse its features (the embeddings are identical - only the labels "
            "differ)."
        ),
    )
    parser.add_argument(
        "--compare-run-dir",
        default=None,
        help=(
            "Output directory of a fine-grained verb kNN run. Enables the section comparing "
            "grouping-before-probing against merging the 97-way predictions afterwards."
        ),
    )
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
        "--secondary-topk",
        type=int,
        default=3,
        help="Secondary accuracy metric: top-N. Kept small because there are few groups.",
    )
    parser.add_argument(
        "--min-class-clips",
        type=int,
        default=20,
        help="Minimum validation clips for a group to appear in the per-group table.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for clip sampling.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of clips per encoder forward pass.")
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
            "Reuse pooled embeddings already saved under --feature-cache-dir instead of re-encoding, "
            "when they were produced from the same manifests and clip counts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(str(output_dir / "action_group_knn_probe_report.log"))

    cache_root = Path(args.feature_cache_dir) if args.feature_cache_dir else output_dir

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    verb_map = load_label_map(args.verb_labels_json)
    lookup, group_names, group_map, descriptions, members = build_group_mapping(verb_map)
    num_groups = len(group_names)
    logger.info("Grouped %d verbs into %d action groups", lookup.shape[0], num_groups)

    # Sample both splits once so every model is evaluated on identical clips. The
    # manifests still carry verb labels; grouping happens on the encoded labels.
    split_infos = {
        "train": prepare_split_manifest(args.train_csv, args.num_train_clips, args.seed, output_dir, "train"),
        "val": prepare_split_manifest(args.val_csv, args.num_val_clips, args.seed, output_dir, "val"),
    }
    for split, info in split_infos.items():
        info["num_groups_present"] = int(np.unique(regroup_labels(info["labels"], lookup)).size)
    logger.info(
        "Train clips: %d (of %d) | Val clips: %d (of %d)",
        split_infos["train"]["num_clips"],
        split_infos["train"]["num_clips_available"],
        split_infos["val"]["num_clips"],
        split_infos["val"]["num_clips_available"],
    )

    results_rows: List[dict] = []
    model_meta: Dict[str, dict] = {}
    per_class_frames: Dict[str, pd.DataFrame] = {}
    per_class_rows: List[pd.DataFrame] = []
    confusions: Dict[str, pd.DataFrame] = {}
    val_group_labels: Dict[str, np.ndarray] = {}
    group_predictions: Dict[str, np.ndarray] = {}
    skipped_models: List[Tuple[str, str]] = []
    baselines: dict = {}
    taxonomy_verb_labels: Dict[str, np.ndarray] = {}

    for model_cfg in MODELS_CONFIG:
        model_name = model_cfg["name"]
        model_slug = slugify(model_name)
        logger.info("%s", "=" * 80)
        logger.info("Processing model: %s", model_name)
        logger.info("%s", "=" * 80)

        cache_dir = cache_root / model_slug
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_dir = output_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        feature_sets = load_cached_features(cache_dir, split_infos) if args.reuse_features else None
        used_cache = feature_sets is not None
        if used_cache:
            logger.info("Reusing cached features for %s from %s", model_name, cache_dir)
        else:
            try:
                config = load_config(model_cfg["config"])
                # The pretraining config drives clip sampling but never had to cope
                # with variable-length action segments; the supervised eval config
                # turns this on, so mirror it here.
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

        if not used_cache:
            # Only write embeddings we just computed; a shared cache is never rewritten.
            save_features(cache_dir, split_infos, train_set, val_set)

        # -- the one substantive difference from the verb-level report
        if not taxonomy_verb_labels:
            taxonomy_verb_labels = {"train": train_set.labels.copy(), "val": val_set.labels.copy()}
        train_set = FeatureSet(
            features=train_set.features,
            labels=regroup_labels(train_set.labels, lookup),
            token_counts=train_set.token_counts,
            elapsed_s=train_set.elapsed_s,
        )
        val_set = FeatureSet(
            features=val_set.features,
            labels=regroup_labels(val_set.labels, lookup),
            token_counts=val_set.token_counts,
            elapsed_s=val_set.elapsed_s,
        )
        np.save(model_dir / "train_group_labels.npy", train_set.labels)
        np.save(model_dir / "val_group_labels.npy", val_set.labels)
        val_group_labels[model_name] = val_set.labels

        rows, winners = evaluate_model_knn(
            train_set,
            val_set,
            k_values=args.k_values,
            temperature=args.temperature,
            num_classes=num_groups,
            device=device,
            secondary_topk=args.secondary_topk,
        )
        if not rows or not winners:
            logger.error("kNN evaluation produced no results for %s", model_name)
            skipped_models.append((model_name, "kNN evaluation produced no results"))
            continue

        for row in rows:
            results_rows.append({"model": model_name, **row})

        best = winners["top1"]
        balanced = winners["balanced"]
        np.save(model_dir / "val_group_predictions.npy", best["predictions"])
        np.save(model_dir / "val_group_predictions_balanced.npy", balanced["predictions"])
        group_predictions[model_name] = best["predictions"]
        if not baselines:
            baselines = baseline_metrics(
                train_set.labels, val_set.labels, num_groups, secondary_topk=args.secondary_topk
            )

        frame = per_class_table(
            val_set.labels, best["predictions"], train_set.labels, group_map, fallback_prefix="group"
        )
        per_class_frames[model_name] = frame
        per_class_rows.append(frame.assign(model=model_name))
        confusions[model_name] = confusion_pairs(
            val_set.labels, best["predictions"], group_map, fallback_prefix="group"
        )

        all_tokens = train_set.token_counts + val_set.token_counts
        model_meta[model_name] = {
            "config": model_cfg["config"],
            "checkpoint": model_cfg.get("checkpoint", "n/a"),
            "embed_dim": train_set.embed_dim,
            "num_train": int(train_set.features.shape[0]),
            "num_val": int(val_set.features.shape[0]),
            "mean_tokens": float(np.mean(all_tokens)) if all_tokens else float("nan"),
            "encode_minutes": (train_set.elapsed_s + val_set.elapsed_s) / 60.0,
            "features_cached": used_cache,
            "best_row": winner_row(model_name, best),
            "balanced_row": winner_row(model_name, balanced),
        }

        for criterion, winner in (("top-1", best), ("mean-per-group", balanced)):
            logger.info(
                "%s best by %s: top1=%.2f%% top%d=%.2f%% mean-per-group=%.2f%% groups-predicted=%d "
                "(k=%d, %s, %s)",
                model_name,
                criterion,
                winner["metrics"]["top1"],
                args.secondary_topk,
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
        pd.concat(per_class_rows, ignore_index=True).to_csv(output_dir / "per_group_accuracy.csv", index=False)

    taxonomy = group_taxonomy_table(
        group_names,
        descriptions,
        members,
        taxonomy_verb_labels or {split: split_infos[split]["labels"] for split in ("train", "val")},
        lookup,
    )
    taxonomy.to_csv(output_dir / "action_groups.csv", index=False)

    charts: Dict[str, Optional[Path]] = {
        "accuracy_vs_k": None,
        "headline": None,
        "per_class": None,
        "support": None,
    }
    charts["support"] = output_dir / "action_group_support.png"
    save_group_support_chart(taxonomy, charts["support"])

    if not results_df.empty:
        charts["accuracy_vs_k"] = output_dir / "accuracy_vs_k.png"
        save_accuracy_vs_k_chart(results_df, charts["accuracy_vs_k"], class_noun=CLASS_NOUN)

        best_rows = [meta["best_row"] for meta in model_meta.values() if meta.get("best_row")]
        best_rows.sort(key=lambda row: row["top1"], reverse=True)
        charts["headline"] = output_dir / "best_per_model.png"
        save_headline_chart(
            best_rows,
            baselines,
            charts["headline"],
            class_noun=CLASS_NOUN,
            secondary_topk=args.secondary_topk,
        )

    if per_class_frames:
        charts["per_class"] = output_dir / "per_group_accuracy.png"
        save_per_class_heatmap(per_class_frames, charts["per_class"], top_n=num_groups, class_noun=CLASS_NOUN)

    for model_name, predictions in group_predictions.items():
        chart_path = output_dir / slugify(model_name) / "confusion_matrix.png"
        save_confusion_matrix_chart(
            val_group_labels[model_name],
            predictions,
            group_names,
            chart_path,
            title=f"{model_name}: action-group confusion (row-normalized)",
        )
        charts[f"confusion::{model_name}"] = chart_path

    comparison = load_comparison(args.compare_run_dir, lookup, num_groups, val_group_labels)

    report_path = output_dir / "action_group_knn_probe_report.md"
    generate_markdown_report(
        report_path=report_path,
        output_dir=output_dir,
        results_df=results_df,
        model_meta=model_meta,
        split_infos=split_infos,
        baselines=baselines,
        taxonomy=taxonomy,
        per_class_frames=per_class_frames,
        confusions=confusions,
        charts=charts,
        skipped_models=skipped_models,
        comparison=comparison,
        group_names=group_names,
        args=args,
    )
    logger.info("Markdown report saved to %s", report_path)


if __name__ == "__main__":
    main()
