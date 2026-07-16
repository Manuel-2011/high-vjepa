#!/usr/bin/env python3
"""
Action Recognition Frame Embedding Report Generator

For each model configuration:
- Load the frozen encoder (no classifier head)
- Sample random video clips from an action-recognition dataset manifest
  (video_path, "verb_class,noun_class")
- Encode each clip and pool the *final* frame/tubelet's patch tokens into a
  single embedding: L2-normalize each patch embedding, mean-pool, then
  L2-normalize the resulting vector (models with a single patch per frame
  just use that one embedding directly)
- Reduce the pooled embeddings to 2D with UMAP
- Generate an HTML report with two scatter plots per model: one colored by
  verb label, one colored by noun label
"""

from __future__ import annotations

import argparse
import colorsys
import json
import html
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.generate_patch_embedding_report import MODELS_CONFIG
from evals.video_classification_frozen.eval import AttnMaskCollator
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data

logger = logging.getLogger(__name__)

DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


@dataclass
class FrameEmbeddingSample:
    sample_index: int
    video_path: str
    frame_path: str
    embedding: np.ndarray
    verb_idx: int
    verb_name: str
    noun_idx: int
    noun_name: str
    umap_x: float = 0.0
    umap_y: float = 0.0


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
    logging.captureWarnings(True)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def load_manifest(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, sep=r"\s+", engine="python")
    if df.empty:
        raise ValueError(f"No rows found in manifest: {csv_path}")

    df = df.rename(columns={0: "video_path"})
    if 1 in df.columns:
        df = df.rename(columns={1: "label"})
    return df


def sample_rows(df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample_size = min(len(df), n_samples)
    indices = rng.choice(len(df), size=sample_size, replace=False)
    return df.iloc[indices].reset_index(drop=True)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "model"


def load_label_map(json_path: str) -> Dict[int, str]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {int(k): v for k, v in data.items()}


def label_name(idx: int, mapping: Dict[int, str], prefix: str) -> str:
    return mapping.get(idx, f"{prefix}_{idx}")


def label_color_hex(idx: int) -> str:
    hue = (idx * GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.88)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def denormalize_clip(clip: torch.Tensor, normalization: Tuple[Sequence[float], Sequence[float]]) -> torch.Tensor:
    mean, std = normalization
    mean_tensor = torch.tensor(mean, dtype=clip.dtype, device=clip.device).view(-1, 1, 1, 1)
    std_tensor = torch.tensor(std, dtype=clip.dtype, device=clip.device).view(-1, 1, 1, 1)
    return (clip * std_tensor + mean_tensor).clamp(0.0, 1.0)


def tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    frame = (frame * 255.0).round().clamp(0, 255).to(torch.uint8)
    return Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())


def extract_frame_image(
    clip: torch.Tensor,
    frame_index: int,
    normalization: Tuple[Sequence[float], Sequence[float]],
) -> Image.Image:
    clip = denormalize_clip(clip, normalization)
    frame_index = max(0, min(frame_index, clip.shape[1] - 1))
    return tensor_frame_to_pil(clip[:, frame_index])


def prepare_encoder(config: dict, device: str):
    args_data = config["data"]
    args_model = config["model"]

    resolution = args_data.get("crop_size", 224)
    frames_per_clip = max(args_data["dataset_fpcs"])
    patch_size = args_data["patch_size"]
    tubelet_size = args_data["tubelet_size"]

    checkpoint = os.path.join(config["folder"], "latest.pt")
    model_kwargs = {
        "encoder": {
            "checkpoint_key": "target_encoder",
            "img_temporal_dim_size": None,
            "model_name": args_model["model_name"],
            "patch_size": patch_size,
            "tubelet_size": tubelet_size,
            "uniform_power": args_model.get("uniform_power", False),
            "use_rope": args_model.get("use_rope", False),
        }
    }
    wrapper_kwargs = {"max_frames": frames_per_clip, "use_pos_embed": False}

    from evals.video_classification_frozen.models import init_module

    encoder = init_module(
        "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip",
        device,
        frames_per_clip,
        resolution,
        checkpoint,
        model_kwargs,
        wrapper_kwargs,
    )

    return encoder


def load_data_for_model(config: dict, data_path: str, batch_size: int = 1):
    args_data = config["data"]
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    resolution = args_data.get("crop_size", 224)
    frames_per_clip = max(args_data["dataset_fpcs"])
    fps = args_data.get("fps")
    patch_size = args_data["patch_size"]
    tubelet_size = args_data["tubelet_size"]
    normalization = args_data.get("normalization", DEFAULT_NORMALIZATION)
    allow_variable_length = args_data.get("allow_variable_length", False)
    num_workers = args_data.get("num_workers", 16)
    persistent_workers = args_data.get("persistent_workers", False)
    pin_mem = args_data.get("pin_mem", True)

    # A causal model continued from a pretrained checkpoint at a coarser fps
    # (meta.use_pretrained_model) samples raw frames at `previous_fps` over a
    # longer window, then keeps only the leading `previous_tubulet_size` frames
    # of every `frames_to_skip`-sized chunk so the encoder still sees clips
    # shaped like the ones its backbone was pretrained on. See SimpleCollator
    # in src/masks/multiseq_multiblock3d.py, mirrored by
    # apply_pretrained_frame_skip() below.
    use_pretrained_cfg = config.get("meta", {}).get("use_pretrained_model") or {}
    use_pretrained_model = use_pretrained_cfg.get("enabled", False)
    frame_skip_info = None
    if use_pretrained_model:
        previous_fps = use_pretrained_cfg["previous_fps"]
        previous_tubelet_size = use_pretrained_cfg.get("previous_tubulet_size", tubelet_size)
        frames_to_skip = int(previous_fps // fps)
        sampling_fps = previous_fps
        sampling_frames_per_clip = int(frames_per_clip * frames_to_skip / tubelet_size)
        frame_skip_info = (frames_to_skip, previous_tubelet_size)
    else:
        sampling_fps = fps
        sampling_frames_per_clip = frames_per_clip

    transform = make_transforms(
        training=False,
        num_views_per_clip=1,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
        normalize=normalization,
    )
    collator = AttnMaskCollator(patch_size, tubelet_size, num_classes=None)

    data_loader, _ = init_data(
        data=dataset_type,
        root_path=data_path,
        transform=transform,
        batch_size=batch_size,
        world_size=1,
        rank=0,
        clip_len=sampling_frames_per_clip,
        fps=sampling_fps,
        num_clips=1,
        allow_clip_overlap=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_mem=pin_mem,
        drop_last=False,
        collator=collator,
        training=False,
        allow_variable_length=allow_variable_length,
        shuffle=False,
    )

    return data_loader, frame_skip_info


def apply_pretrained_frame_skip(clip: torch.Tensor, frames_to_skip: int, previous_tubelet_size: int) -> torch.Tensor:
    """Reduce a clip sampled at `previous_fps` down to the tubelet arrangement
    a use_pretrained_model backbone actually expects, matching SimpleCollator's
    per-chunk frame-skip in src/masks/multiseq_multiblock3d.py."""
    batch_size, channels, num_frames, height, width = clip.shape
    return (
        clip.view(batch_size, channels, num_frames // frames_to_skip, frames_to_skip, height, width)[
            :, :, :, :previous_tubelet_size, :, :
        ].reshape(batch_size, channels, num_frames // frames_to_skip * previous_tubelet_size, height, width)
    )


def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def pool_final_frame_embedding(token_embeddings: np.ndarray) -> np.ndarray:
    """token_embeddings: [num_patches, D] embeddings of every patch in the
    clip's final frame/tubelet. Models with a single patch per frame just
    return that embedding untouched."""
    if token_embeddings.shape[0] > 1:
        normed = l2_normalize_embeddings(token_embeddings)
        pooled = normed.mean(axis=0)
        pooled = pooled / max(np.linalg.norm(pooled), 1e-12)
    else:
        pooled = token_embeddings[0].astype(np.float64)
    return pooled.astype(np.float32)


def standardize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    return (embeddings - mean) / np.maximum(std, 1e-12)


def umap_project(
    x: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    if x.shape[0] < 3:
        return np.zeros((x.shape[0], n_components), dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-12)
    n_neighbors = min(15, x.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=max(2, n_neighbors),
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(x)


def project_embeddings(samples: List[FrameEmbeddingSample], seed: int) -> None:
    if not samples:
        return

    embeddings = np.stack([sample.embedding for sample in samples], axis=0)
    umap_embeddings = standardize_embeddings(embeddings)
    coords = umap_project(umap_embeddings, random_state=seed)

    for sample, (x_coord, y_coord) in zip(samples, coords):
        sample.umap_x = float(x_coord)
        sample.umap_y = float(y_coord)


def save_scatter_plot(
    samples: List[FrameEmbeddingSample],
    color_attr: str,
    name_attr: str,
    output_path: Path,
    title: str,
    max_legend_entries: int = 30,
) -> None:
    if not samples:
        return

    fig, ax = plt.subplots(figsize=(16, 12), dpi=160)
    ax.set_title(title, fontsize=16, pad=18)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    xs = np.array([sample.umap_x for sample in samples], dtype=np.float64)
    ys = np.array([sample.umap_y for sample in samples], dtype=np.float64)
    x_pad = max(1e-3, 0.08 * (xs.max() - xs.min() if len(xs) > 1 else 1.0))
    y_pad = max(1e-3, 0.08 * (ys.max() - ys.min() if len(ys) > 1 else 1.0))
    ax.set_xlim(xs.min() - x_pad, xs.max() + x_pad)
    ax.set_ylim(ys.min() - y_pad, ys.max() + y_pad)
    ax.grid(True, alpha=0.15)

    seen_labels: Dict[int, str] = {}
    for sample in samples:
        label_idx = getattr(sample, color_attr)
        seen_labels[label_idx] = getattr(sample, name_attr)
        ax.scatter(
            sample.umap_x,
            sample.umap_y,
            s=28,
            color=label_color_hex(label_idx),
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
        )

    if 0 < len(seen_labels) <= max_legend_entries:
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=label_color_hex(idx), label=name)
            for idx, name in sorted(seen_labels.items(), key=lambda item: item[1])
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_view(
    samples: List[FrameEmbeddingSample],
    color_attr: str,
    idx_attr: str,
    name_attr: str,
    view_key: str,
    view_title: str,
    output_dir: Path,
) -> dict:
    legend_map: Dict[int, str] = {}
    for sample in samples:
        legend_map[getattr(sample, idx_attr)] = getattr(sample, name_attr)

    legend = [
        {"idx": idx, "name": name, "color": label_color_hex(idx)}
        for idx, name in sorted(legend_map.items(), key=lambda item: item[1])
    ]

    return {
        "key": view_key,
        "title": view_title,
        "legend": legend,
        "samples": [
            {
                "video_path": sample.video_path,
                "frame_rel_path": os.path.relpath(sample.frame_path, output_dir),
                "label_idx": getattr(sample, idx_attr),
                "label_name": getattr(sample, name_attr),
                "color": label_color_hex(getattr(sample, idx_attr)),
                "umap_x": sample.umap_x,
                "umap_y": sample.umap_y,
            }
            for sample in samples
        ],
    }


def generate_html_report(model_reports: List[dict], output_path: Path, dataset_csv: str) -> None:
    report_data = json.dumps(model_reports, ensure_ascii=True)
    parts = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"UTF-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
        "<title>Action Recognition Frame Embedding Report</title>",
        "<style>",
        "body { font-family: Arial, Helvetica, sans-serif; margin: 24px; background: linear-gradient(180deg, #f6f7fb 0%, #eef1f7 100%); color: #1f2937; }",
        "h1 { margin-bottom: 6px; }",
        ".subtitle { color: #4b5563; margin-top: 0; }",
        ".model-card { background: rgba(255,255,255,0.96); border-radius: 16px; padding: 20px; margin: 18px 0 28px; box-shadow: 0 12px 32px rgba(15,23,42,0.10); border: 1px solid rgba(148,163,184,0.25); }",
        ".model-meta { display: flex; flex-wrap: wrap; gap: 16px; color: #374151; margin-bottom: 14px; }",
        ".model-meta span { background: #eef2ff; padding: 6px 10px; border-radius: 999px; }",
        ".views-shell { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }",
        ".view-card { border: 1px solid rgba(148,163,184,0.3); border-radius: 14px; padding: 14px; }",
        ".view-card h3 { margin: 0 0 10px; }",
        ".scatter-shell { display: grid; grid-template-columns: minmax(0, 1fr) 260px; gap: 14px; align-items: start; }",
        ".plot-column { min-width: 0; }",
        ".plot-toolbar { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }",
        ".plot-toolbar button { background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe; border-radius: 8px; padding: 6px 12px; font-size: 13px; cursor: pointer; }",
        ".plot-toolbar button:hover { background: #e0e7ff; }",
        ".plot-toolbar .hint { color: #6b7280; font-size: 12px; margin-left: auto; }",
        ".scatter-plot { position: relative; width: 100%; height: 460px; border-radius: 14px; border: 1px solid #e5e7eb; background: #ffffff; overflow: hidden; }",
        ".scatter-plot canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }",
        ".scatter-plot canvas.dragging { cursor: grabbing; }",
        ".hover-panel { position: sticky; top: 18px; background: #ffffff; border-radius: 16px; border: 1px solid #dbe2ea; box-shadow: 0 10px 24px rgba(15,23,42,0.08); padding: 12px; display: flex; flex-direction: column; gap: 10px; }",
        ".hover-panel h4 { margin: 0; font-size: 14px; }",
        ".hover-preview { width: 100%; aspect-ratio: 4 / 3; border-radius: 12px; background: #f8fafc; border: 1px dashed #cbd5e1; display: grid; place-items: center; overflow: hidden; }",
        ".hover-preview img { width: 100%; height: 100%; object-fit: contain; image-rendering: auto; }",
        ".hover-preview-placeholder { color: #64748b; font-size: 12px; text-align: center; line-height: 1.35; padding: 10px; }",
        ".hover-meta { font-size: 12px; color: #475569; display: grid; gap: 4px; }",
        ".hover-meta code { background: #f1f5f9; padding: 2px 6px; border-radius: 6px; word-break: break-all; }",
        ".legend { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; max-height: 130px; overflow-y: auto; }",
        ".legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 999px; padding: 3px 8px; }",
        ".legend-swatch { width: 10px; height: 10px; border-radius: 50%; flex: none; }",
        ".small { font-size: 13px; color: #6b7280; }",
        "@media (max-width: 1100px) { .views-shell { grid-template-columns: 1fr; } .scatter-shell { grid-template-columns: 1fr; } .hover-panel { position: static; } }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Action Recognition Frame Embedding Report</h1>",
        f"<p class=\"subtitle\">Dataset manifest: {html.escape(dataset_csv)}</p>",
        f"<script id=\"report-data\" type=\"application/json\">{report_data}</script>",
        "<script>",
        "function setPreview(panel, sample) {",
        "  const img = panel.querySelector('.hover-preview-img');",
        "  const placeholder = panel.querySelector('.hover-preview-placeholder');",
        "  const title = panel.querySelector('.hover-preview-title');",
        "  const meta = panel.querySelector('.hover-meta');",
        "  img.src = sample.frame_rel_path;",
        "  img.alt = 'Final frame preview for ' + sample.video_path;",
        "  img.hidden = false;",
        "  placeholder.hidden = true;",
        "  title.textContent = sample.video_path.split('/').pop();",
        "  meta.innerHTML = '<div><strong>Label:</strong> ' + sample.label_name + ' (' + sample.label_idx + ')</div>' +",
        "    '<div><strong>Video:</strong> <code>' + sample.video_path + '</code></div>';",
        "}",
        "function resetPreview(panel) {",
        "  const img = panel.querySelector('.hover-preview-img');",
        "  const placeholder = panel.querySelector('.hover-preview-placeholder');",
        "  const title = panel.querySelector('.hover-preview-title');",
        "  const meta = panel.querySelector('.hover-meta');",
        "  img.hidden = true;",
        "  img.removeAttribute('src');",
        "  placeholder.hidden = false;",
        "  title.textContent = 'Hover a point';",
        "  meta.innerHTML = '<div>Move the pointer over a point in the scatter plot to inspect the clip.</div>';",
        "}",
        "function createScatterPlot(container, view, panel) {",
        "  const samples = view.samples;",
        "  const canvas = document.createElement('canvas');",
        "  container.innerHTML = '';",
        "  container.appendChild(canvas);",
        "  const ctx = canvas.getContext('2d');",
        "  const margin = { top: 20, right: 16, bottom: 36, left: 44 };",
        "  const xs = samples.map(function(s) { return s.umap_x; });",
        "  const ys = samples.map(function(s) { return s.umap_y; });",
        "  const xMin = Math.min.apply(null, xs);",
        "  const xMax = Math.max.apply(null, xs);",
        "  const yMin = Math.min.apply(null, ys);",
        "  const yMax = Math.max.apply(null, ys);",
        "  const xPad = Math.max(1e-3, 0.08 * ((xMax - xMin) || 1));",
        "  const yPad = Math.max(1e-3, 0.08 * ((yMax - yMin) || 1));",
        "  const dataXLo = xMin - xPad, dataXHi = xMax + xPad;",
        "  const dataYLo = yMin - yPad, dataYHi = yMax + yPad;",
        "  const view_ = { scale: 1, tx: 0, ty: 0 };",
        "  let hoverIndex = -1;",
        "  let cssWidth = 0, cssHeight = 0;",
        "  const dpr = window.devicePixelRatio || 1;",
        "  function plotWidth() { return cssWidth - margin.left - margin.right; }",
        "  function plotHeight() { return cssHeight - margin.top - margin.bottom; }",
        "  function baseX(x) { return margin.left + ((x - dataXLo) / (dataXHi - dataXLo)) * plotWidth(); }",
        "  function baseY(y) { return margin.top + (1 - (y - dataYLo) / (dataYHi - dataYLo)) * plotHeight(); }",
        "  function toScreen(x, y) {",
        "    const bx = baseX(x), by = baseY(y);",
        "    return [bx * view_.scale + view_.tx, by * view_.scale + view_.ty];",
        "  }",
        "  let pending = false;",
        "  function draw() {",
        "    pending = false;",
        "    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);",
        "    ctx.clearRect(0, 0, cssWidth, cssHeight);",
        "    ctx.fillStyle = '#ffffff';",
        "    ctx.fillRect(0, 0, cssWidth, cssHeight);",
        "    ctx.save();",
        "    ctx.beginPath();",
        "    ctx.rect(margin.left, margin.top, plotWidth(), plotHeight());",
        "    ctx.clip();",
        "    ctx.strokeStyle = '#e2e8f0';",
        "    ctx.lineWidth = 1;",
        "    const ticks = 5;",
        "    for (let i = 0; i < ticks; i++) {",
        "      const t = i / (ticks - 1);",
        "      const gx = margin.left + t * plotWidth();",
        "      const gy = margin.top + t * plotHeight();",
        "      ctx.beginPath();",
        "      ctx.moveTo(gx, margin.top);",
        "      ctx.lineTo(gx, margin.top + plotHeight());",
        "      ctx.stroke();",
        "      ctx.beginPath();",
        "      ctx.moveTo(margin.left, gy);",
        "      ctx.lineTo(margin.left + plotWidth(), gy);",
        "      ctx.stroke();",
        "    }",
        "    const radius = Math.max(1.6, Math.min(5, 5 * Math.sqrt(1 / Math.max(1, view_.scale))));",
        "    for (let idx = 0; idx < samples.length; idx++) {",
        "      const sample = samples[idx];",
        "      const pos = toScreen(sample.umap_x, sample.umap_y);",
        "      const sx = pos[0], sy = pos[1];",
        "      if (sx < margin.left - radius || sx > margin.left + plotWidth() + radius ||",
        "          sy < margin.top - radius || sy > margin.top + plotHeight() + radius) {",
        "        continue;",
        "      }",
        "      const isHover = idx === hoverIndex;",
        "      ctx.beginPath();",
        "      ctx.arc(sx, sy, isHover ? radius + 2.5 : radius, 0, Math.PI * 2);",
        "      ctx.fillStyle = sample.color;",
        "      ctx.globalAlpha = isHover ? 1 : 0.85;",
        "      ctx.fill();",
        "      ctx.lineWidth = isHover ? 2 : 1;",
        "      ctx.strokeStyle = isHover ? '#0f172a' : '#ffffff';",
        "      ctx.stroke();",
        "      ctx.globalAlpha = 1;",
        "    }",
        "    ctx.restore();",
        "    ctx.strokeStyle = '#94a3b8';",
        "    ctx.lineWidth = 1;",
        "    ctx.beginPath();",
        "    ctx.moveTo(margin.left, margin.top + plotHeight());",
        "    ctx.lineTo(margin.left + plotWidth(), margin.top + plotHeight());",
        "    ctx.moveTo(margin.left, margin.top);",
        "    ctx.lineTo(margin.left, margin.top + plotHeight());",
        "    ctx.stroke();",
        "    ctx.fillStyle = '#475569';",
        "    ctx.font = '12px Arial, Helvetica, sans-serif';",
        "    ctx.textAlign = 'center';",
        "    ctx.fillText('UMAP 1', margin.left + plotWidth() / 2, cssHeight - 8);",
        "    ctx.save();",
        "    ctx.translate(12, margin.top + plotHeight() / 2);",
        "    ctx.rotate(-Math.PI / 2);",
        "    ctx.fillText('UMAP 2', 0, 0);",
        "    ctx.restore();",
        "  }",
        "  function requestDraw() {",
        "    if (!pending) {",
        "      pending = true;",
        "      requestAnimationFrame(draw);",
        "    }",
        "  }",
        "  function clampScale(scale) { return Math.min(60, Math.max(0.5, scale)); }",
        "  function zoomAt(px, py, factor) {",
        "    const newScale = clampScale(view_.scale * factor);",
        "    const actualFactor = newScale / view_.scale;",
        "    view_.tx = px - (px - view_.tx) * actualFactor;",
        "    view_.ty = py - (py - view_.ty) * actualFactor;",
        "    view_.scale = newScale;",
        "    requestDraw();",
        "  }",
        "  function resetView() { view_.scale = 1; view_.tx = 0; view_.ty = 0; requestDraw(); }",
        "  function resize() {",
        "    const rect = container.getBoundingClientRect();",
        "    cssWidth = Math.max(1, rect.width);",
        "    cssHeight = Math.max(1, rect.height);",
        "    canvas.width = Math.round(cssWidth * dpr);",
        "    canvas.height = Math.round(cssHeight * dpr);",
        "    draw();",
        "  }",
        "  canvas.addEventListener('wheel', function(e) {",
        "    e.preventDefault();",
        "    const rect = canvas.getBoundingClientRect();",
        "    const px = e.clientX - rect.left;",
        "    const py = e.clientY - rect.top;",
        "    const factor = Math.exp(-e.deltaY * 0.0015);",
        "    zoomAt(px, py, factor);",
        "  }, { passive: false });",
        "  let dragging = false;",
        "  let lastX = 0, lastY = 0;",
        "  canvas.addEventListener('mousedown', function(e) {",
        "    dragging = true;",
        "    lastX = e.clientX;",
        "    lastY = e.clientY;",
        "    canvas.classList.add('dragging');",
        "  });",
        "  window.addEventListener('mouseup', function() {",
        "    dragging = false;",
        "    canvas.classList.remove('dragging');",
        "  });",
        "  window.addEventListener('mousemove', function(e) {",
        "    if (!dragging) return;",
        "    view_.tx += e.clientX - lastX;",
        "    view_.ty += e.clientY - lastY;",
        "    lastX = e.clientX;",
        "    lastY = e.clientY;",
        "    requestDraw();",
        "  });",
        "  canvas.addEventListener('mousemove', function(e) {",
        "    if (dragging) return;",
        "    const rect = canvas.getBoundingClientRect();",
        "    const mx = e.clientX - rect.left;",
        "    const my = e.clientY - rect.top;",
        "    let best = -1;",
        "    let bestDist = 144;",
        "    for (let i = 0; i < samples.length; i++) {",
        "      const pos = toScreen(samples[i].umap_x, samples[i].umap_y);",
        "      const dx = pos[0] - mx;",
        "      const dy = pos[1] - my;",
        "      const d = dx * dx + dy * dy;",
        "      if (d < bestDist) { bestDist = d; best = i; }",
        "    }",
        "    if (best !== hoverIndex) { hoverIndex = best; requestDraw(); }",
        "    if (best >= 0) { setPreview(panel, samples[best]); } else { resetPreview(panel); }",
        "  });",
        "  canvas.addEventListener('mouseleave', function() {",
        "    if (!dragging) { hoverIndex = -1; resetPreview(panel); requestDraw(); }",
        "  });",
        "  const toolbar = container.parentElement.querySelector('.plot-toolbar');",
        "  if (toolbar) {",
        "    toolbar.querySelectorAll('button').forEach(function(button) {",
        "      button.addEventListener('click', function() {",
        "        const action = button.getAttribute('data-action');",
        "        const cx = margin.left + plotWidth() / 2;",
        "        const cy = margin.top + plotHeight() / 2;",
        "        if (action === 'zoom-in') zoomAt(cx, cy, 1.4);",
        "        else if (action === 'zoom-out') zoomAt(cx, cy, 1 / 1.4);",
        "        else if (action === 'reset') resetView();",
        "      });",
        "    });",
        "  }",
        "  window.addEventListener('resize', resize);",
        "  resize();",
        "  resetPreview(panel);",
        "}",
        "function initReport() {",
        "  const models = JSON.parse(document.getElementById('report-data').textContent);",
        "  document.querySelectorAll('.model-card').forEach(function(card, modelIndex) {",
        "    const model = models[modelIndex];",
        "    card.querySelectorAll('.view-card').forEach(function(viewCard, viewIndex) {",
        "      const view = model.views[viewIndex];",
        "      const plot = viewCard.querySelector('.scatter-plot');",
        "      const panel = viewCard.querySelector('.hover-panel');",
        "      createScatterPlot(plot, view, panel);",
        "    });",
        "  });",
        "}",
        "if (document.readyState === 'loading') {",
        "  document.addEventListener('DOMContentLoaded', initReport);",
        "} else {",
        "  initReport();",
        "}",
        "</script>",
    ]

    for report in model_reports:
        parts.extend(
            [
                "<div class=\"model-card\">",
                f"<h2>{html.escape(report['model_name'])}</h2>",
                "<div class=\"model-meta\">",
                f"<span>Samples: {report['num_samples']}</span>",
                f"<span>Embedding dim: {report['embedding_dim']}</span>",
                "</div>",
                "<div class=\"views-shell\">",
            ]
        )

        for view in report["views"]:
            legend_items = "".join(
                f"<span class=\"legend-item\"><span class=\"legend-swatch\" style=\"background:{entry['color']}\"></span>{html.escape(entry['name'])}</span>"
                for entry in view["legend"]
            )
            parts.extend(
                [
                    "<div class=\"view-card\">",
                    f"<h3>{html.escape(view['title'])}</h3>",
                    "<div class=\"scatter-shell\">",
                    "<div class=\"plot-column\">",
                    "<div class=\"plot-toolbar\">",
                    "<button type=\"button\" data-action=\"zoom-in\">Zoom in</button>",
                    "<button type=\"button\" data-action=\"zoom-out\">Zoom out</button>",
                    "<button type=\"button\" data-action=\"reset\">Reset view</button>",
                    "<span class=\"hint\">Scroll to zoom &middot; drag to pan &middot; hover to preview</span>",
                    "</div>",
                    "<div class=\"scatter-plot\"></div>",
                    "</div>",
                    "<aside class=\"hover-panel\">",
                    "<h4 class=\"hover-preview-title\">Hover a point</h4>",
                    "<div class=\"hover-preview\">",
                    "<img class=\"hover-preview-img\" alt=\"Hovered frame preview\" hidden>",
                    "<div class=\"hover-preview-placeholder\">Move the pointer over a point to preview the clip's final frame.</div>",
                    "</div>",
                    "<div class=\"hover-meta\">",
                    "<div>Move the pointer over a point in the scatter plot to inspect the clip.</div>",
                    "</div>",
                    "</aside>",
                    "</div>",
                    f"<div class=\"legend\">{legend_items}</div>",
                    "</div>",
                ]
            )

        parts.extend(
            [
                "</div>",
                f"<p class=\"small\">Raw embeddings: {html.escape(report['embeddings_path'])} | Metadata: {html.escape(report['metadata_path'])}</p>",
                "</div>",
            ]
        )

    parts.extend(["</body>", "</html>"])
    output_path.write_text("".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an action-recognition frame embedding report")
    parser.add_argument(
        "--dataset-csv",
        default="data/EK100_action_recognition_validation.csv",
        help="Path to a whitespace-separated manifest with 'video_path verb_class,noun_class' rows.",
    )
    parser.add_argument(
        "--verb-labels-json",
        default="preliminary_experiments/evals/vitl/vjepa_ek100_ar/qualitative_eval/verb_labels.json",
        help="JSON mapping verb class index -> verb name.",
    )
    parser.add_argument(
        "--noun-labels-json",
        default="preliminary_experiments/evals/vitl/vjepa_ek100_ar/qualitative_eval/noun_labels.json",
        help="JSON mapping noun class index -> noun name.",
    )
    parser.add_argument(
        "--output-dir",
        default="preliminary_experiments/evals/vitl/vjepa_ek100_frame_embeddings",
        help="Directory where the report artifacts will be saved.",
    )
    parser.add_argument("--num-samples", type=int, default=200, help="Number of random videos to sample per model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling and visualization.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of clips to run through the encoder per forward pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(str(output_dir / "action_embedding_report.log"))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    verb_map = load_label_map(args.verb_labels_json)
    noun_map = load_label_map(args.noun_labels_json)

    manifest = load_manifest(args.dataset_csv)
    sampled_manifest = sample_rows(manifest, args.num_samples, args.seed)
    logger.info("Loaded %d manifest rows, using %d candidates per model", len(manifest), len(sampled_manifest))
    sampled_csv = output_dir / "sampled_manifest.csv"
    sampled_manifest.to_csv(sampled_csv, sep=" ", header=False, index=False)

    model_reports = []
    for model_cfg in MODELS_CONFIG:
        model_name = model_cfg["name"]
        model_slug = slugify(model_name)
        logger.info("%s", "=" * 80)
        logger.info("Processing model: %s", model_name)
        logger.info("%s", "=" * 80)

        try:
            config = load_config(model_cfg["config"])
            encoder = prepare_encoder(config, device)
        except Exception as exc:
            logger.error("Failed to load %s: %s", model_name, exc)
            continue

        args_data = config["data"]
        patch_size = args_data["patch_size"]
        tubelet_size = args_data["tubelet_size"]
        normalization = args_data.get("normalization", DEFAULT_NORMALIZATION)

        data_loader, frame_skip_info = load_data_for_model(config, str(sampled_csv), batch_size=args.batch_size)

        samples: List[FrameEmbeddingSample] = []
        for batch_idx, data in enumerate(tqdm(data_loader, total=len(data_loader), desc=f"{model_name}")):
            try:
                clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
                clip_indices = data[2]
                if clip_indices is not None:
                    clip_indices = [d.to(device, non_blocking=True) for d in clip_indices]
                attn_mask = [[dij.to(device, non_blocking=True) for dij in di] for di in data[3]]

                if frame_skip_info is not None:
                    frames_to_skip, previous_tubelet_size = frame_skip_info
                    clips = [[apply_pretrained_frame_skip(view, frames_to_skip, previous_tubelet_size) for view in clip] for clip in clips]
                    # AttnMaskCollator's mask was sized for the pre-skip clip length; every
                    # sample in this fixed-length path is fully valid post-skip, so rebuild
                    # a fully-valid mask at the new (smaller) token count instead of slicing it.
                    b, _, t, h, w = clips[0][0].shape
                    num_tokens = (t // tubelet_size) * (h // patch_size) * (w // patch_size)
                    attn_mask = [[torch.ones(b, 1, num_tokens, num_tokens, dtype=torch.bool, device=device)]]

                with torch.no_grad():
                    outputs, attn_mask = encoder(clips, clip_indices, attn_mask)

                view_output = outputs[0]
                view_attn_mask = attn_mask[0]
                videos = clips[0][0]
                batch_size = videos.shape[0]
                token_validity = torch.diagonal(view_attn_mask[:, 0], dim1=-2, dim2=-1)

                for batch_index in range(batch_size):
                    # Order-preserving map back into sampled_manifest: shuffle=False and
                    # drop_last=False mean every batch but the last is exactly
                    # args.batch_size long, so this recovers the flat sample position.
                    global_idx = batch_idx * args.batch_size + batch_index
                    valid_token_indices = torch.where(token_validity[batch_index])[0]
                    num_valid = len(valid_token_indices)
                    if num_valid == 0:
                        logger.warning("No valid tokens found for sample %d in %s", global_idx, model_name)
                        continue

                    _, _, height, width = videos[batch_index].shape
                    grid_h = height // patch_size
                    grid_w = width // patch_size
                    num_spatial_tokens = grid_h * grid_w

                    if num_valid < num_spatial_tokens:
                        logger.warning("Fewer valid tokens than a full frame for sample %d in %s", global_idx, model_name)
                        continue

                    # Tokens are ordered temporal-major (t * S + s) and the valid prefix is
                    # contiguous from t=0, so the last `num_spatial_tokens` valid indices are
                    # exactly the patches of the clip's final frame/tubelet.
                    final_indices = valid_token_indices[num_valid - num_spatial_tokens : num_valid]
                    patch_embeddings = (
                        view_output[batch_index, final_indices].detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                    embedding = pool_final_frame_embedding(patch_embeddings)

                    num_valid_temporal = num_valid // num_spatial_tokens
                    final_frame_index = num_valid_temporal * tubelet_size - 1
                    frame_image = extract_frame_image(
                        videos[batch_index].detach().cpu(),
                        frame_index=final_frame_index,
                        normalization=normalization,
                    )

                    frame_dir = output_dir / model_slug / "frames"
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    frame_path = frame_dir / f"sample_{global_idx:04d}.png"
                    frame_image.save(frame_path)

                    source_row = sampled_manifest.iloc[global_idx]
                    verb_idx, noun_idx = (int(part) for part in str(source_row["label"]).split(","))

                    samples.append(
                        FrameEmbeddingSample(
                            sample_index=global_idx,
                            video_path=str(source_row["video_path"]),
                            frame_path=str(frame_path),
                            embedding=embedding,
                            verb_idx=verb_idx,
                            verb_name=label_name(verb_idx, verb_map, "verb"),
                            noun_idx=noun_idx,
                            noun_name=label_name(noun_idx, noun_map, "noun"),
                        )
                    )

                if len(samples) >= args.num_samples:
                    break
            except Exception as exc:
                logger.error("Failed processing batch %d for %s: %s", batch_idx, model_name, exc)

        if not samples:
            logger.warning("No samples collected for %s", model_name)
            continue

        project_embeddings(samples, seed=args.seed)

        model_dir = output_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)
        embeddings = np.stack([sample.embedding for sample in samples], axis=0)
        metadata = pd.DataFrame(
            [
                {
                    "sample_index": sample.sample_index,
                    "video_path": sample.video_path,
                    "frame_path": sample.frame_path,
                    "verb_idx": sample.verb_idx,
                    "verb_name": sample.verb_name,
                    "noun_idx": sample.noun_idx,
                    "noun_name": sample.noun_name,
                    "umap_x": sample.umap_x,
                    "umap_y": sample.umap_y,
                }
                for sample in samples
            ]
        )

        embeddings_path = model_dir / "embeddings.npy"
        metadata_path = model_dir / "metadata.csv"
        verb_plot_path = model_dir / "umap_by_verb.png"
        noun_plot_path = model_dir / "umap_by_noun.png"
        metadata.to_csv(metadata_path, index=False)
        np.save(embeddings_path, embeddings)
        save_scatter_plot(
            samples, "verb_idx", "verb_name", verb_plot_path, title=f"{model_name} frame embeddings (by verb)"
        )
        save_scatter_plot(
            samples, "noun_idx", "noun_name", noun_plot_path, title=f"{model_name} frame embeddings (by noun)"
        )

        logger.info("Model %s produced %d samples", model_name, len(samples))

        model_reports.append(
            {
                "model_name": model_name,
                "num_samples": len(samples),
                "embedding_dim": int(embeddings.shape[1]),
                "embeddings_path": str(embeddings_path),
                "metadata_path": str(metadata_path),
                "views": [
                    build_view(samples, "verb_idx", "verb_idx", "verb_name", "verb", "Colored by verb", output_dir),
                    build_view(samples, "noun_idx", "noun_idx", "noun_name", "noun", "Colored by noun", output_dir),
                ],
            }
        )

    if not model_reports:
        logger.error("No model reports were generated")
        return

    report_path = output_dir / "action_embedding_report.html"
    generate_html_report(model_reports, report_path, args.dataset_csv)
    logger.info("HTML report saved to %s", report_path)


if __name__ == "__main__":
    main()
