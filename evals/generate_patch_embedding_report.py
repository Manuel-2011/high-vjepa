#!/usr/bin/env python3
"""
Patch Embedding Cluster Report Generator

For each model configuration:
- Load the frozen encoder
- Sample random video clips from a dataset manifest
- Extract one random patch embedding per sample
- Save the original patch crop for later inspection
- Cluster embeddings with HDBSCAN
- Reduce embeddings to 2D with UMAP
- Generate an HTML report with a thumbnail scatter plot per model
"""

from __future__ import annotations

import argparse
import json
import html
import logging
import os
import re
import sys
from collections import Counter
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
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.eval import AttnMaskCollator
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data

logger = logging.getLogger(__name__)

DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

MODELS_CONFIG = [
    {
        "name": "V-JEPA2 (baseline)",
        "checkpoint": "preliminary_experiments/EK100-vjepa-16f-4pfs/latest.pt",
        "config": "configs/train/vitl16-EK100/pretrain-vjepa.yaml",
    },
    # {
    #     "name": "High V-JEPA (Same data as baseline)",
    #     "checkpoint": "preliminary_experiments/EK100-long-vjepa-16f_extended/latest.pt",
    #     "config": "configs/train/vitl16-EK100/pretrain-long-vjepa_extended.yaml",
    # },
    # {
    #     "name": "High V-JEPA (Same data + patches)",
    #     "checkpoint": "preliminary_experiments/EK100-long-vjepa-16f-16x16/latest.pt",
    #     "config": "configs/train/vitl16-EK100/pretrain-long-vjepa_extended_16x16.yaml",
    # },
    # {
    #     "name": "V-JEPA2 - Causal learning",
    #     "checkpoint": "preliminary_experiments/EK100-vjepa-16f-4pfs-future-prediction/latest.pt",
    #     "config": "configs/train/vitl16-EK100/pretrain-vjepa-future-prediction-task.yaml",
    # },
    # {
    #     "name": "High V-JEPA on V-JEPA2",
    #     "checkpoint": "preliminary_experiments/EK100-long-vjepa-16f-16x16_post_training/latest.pt",
    #     "config": "configs/train/vitl16-EK100/pretrain-long-vjepa_16x16_post_training.yaml",
    # },
]


@dataclass
class PatchSample:
    sample_index: int
    video_path: str
    patch_path: str
    embedding: np.ndarray
    token_index: int
    temporal_token_index: int
    spatial_token_index: int
    cluster: int = -1
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


def load_video_clip(
    video_path: str,
    frames_per_clip: int,
    frame_step: int,
    seed: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        reader = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        if len(reader) == 0:
            logger.warning("Video is empty: %s", video_path)
            return None, None

        clip_len = frames_per_clip * frame_step
        if len(reader) <= clip_len:
            indices = np.linspace(0, len(reader) - 1, frames_per_clip).astype(np.int32)
        else:
            rng = np.random.default_rng(seed)
            start = int(rng.integers(0, len(reader) - clip_len + 1))
            indices = start + np.arange(0, clip_len, frame_step, dtype=np.int32)

        frames = reader.get_batch(indices).asnumpy()
        return frames, indices
    except Exception as exc:
        logger.warning("Failed to load video %s: %s", video_path, exc)
        return None, None


def denormalize_clip(clip: torch.Tensor, normalization: Tuple[Sequence[float], Sequence[float]]) -> torch.Tensor:
    mean, std = normalization
    mean_tensor = torch.tensor(mean, dtype=clip.dtype, device=clip.device).view(-1, 1, 1, 1)
    std_tensor = torch.tensor(std, dtype=clip.dtype, device=clip.device).view(-1, 1, 1, 1)
    return (clip * std_tensor + mean_tensor).clamp(0.0, 1.0)


def tensor_patch_to_pil(patch: torch.Tensor) -> Image.Image:
    patch = (patch * 255.0).round().clamp(0, 255).to(torch.uint8)
    return Image.fromarray(patch.permute(1, 2, 0).cpu().numpy())


def extract_patch_image(
    clip: torch.Tensor,
    temporal_token_index: int,
    spatial_token_index: int,
    patch_size: int,
    tubelet_size: int,
    normalization: Tuple[Sequence[float], Sequence[float]],
) -> Image.Image:
    clip = denormalize_clip(clip, normalization)
    _, num_frames, height, width = clip.shape
    grid_w = width // patch_size

    row = spatial_token_index // grid_w
    col = spatial_token_index % grid_w
    frame_index = min(temporal_token_index * tubelet_size, num_frames - 1)

    frame = clip[:, frame_index]
    y0 = row * patch_size
    x0 = col * patch_size
    patch = frame[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
    return tensor_patch_to_pil(patch)


def load_model_checkpoint(config: dict, encoder_emb_dim: int, device: str = "cuda:0"):
    args_exp = config.get("experiment")
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    args_data = args_exp.get("data")
    num_classes = args_data.get("num_classes")

    pretrain_folder = config.get("folder", None)
    eval_tag = config.get("tag", None)
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    checkpoint_path = os.path.join(folder, "latest.pt")
    logger.info("Loading checkpoint from %s", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    classifiers = []
    if "classifiers" in checkpoint:
        from src.models.attentive_pooler import AttentiveClassifier

        for state_dict in checkpoint["classifiers"]:
            classifier = AttentiveClassifier(
                embed_dim=encoder_emb_dim,
                num_heads=num_heads,
                depth=num_probe_blocks,
                num_classes=num_classes,
            )
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            msg = classifier.load_state_dict(state_dict)
            logger.info("loaded pretrained classifier with msg: %s", msg)
            classifier.train(mode=False)
            classifiers.append(classifier.to(device))

    return classifiers


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

    encoder = init_module(
        "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip",
        device,
        frames_per_clip,
        resolution,
        checkpoint,
        model_kwargs,
        wrapper_kwargs,
    )
    classifiers = None

    return encoder, classifiers


def build_transform(config: dict):
    args_data = config.get("experiment").get("data")
    resolution = args_data.get("resolution", 224)
    normalization = args_data.get("normalization", DEFAULT_NORMALIZATION)
    return make_transforms(
        training=False,
        num_views_per_clip=1,
        crop_size=resolution,
        normalize=normalization,
    ), normalization


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

    # Mirrors evals.video_classification_frozen.eval.make_dataloader, but passes
    # `fps` straight through to VideoDataset instead of a fixed `frame_step`, so
    # the per-video frame stride is derived from each clip's native fps exactly
    # like it is during pretraining (see src/datasets/video_dataset.py).
    #
    # Note: config["data"]["dataset_fpcs"]/"datasets_weights" describe the
    # original multi-dataset training mix and aren't forwarded here — root_path
    # is a single flattened sampled-manifest CSV, and VideoDataset requires
    # dataset_fpcs/datasets_weights to have one entry per root_path dataset.
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


def encode_patch_sample(
    encoder,
    transform,
    normalization,
    video_path: str,
    sample_index: int,
    output_dir: Path,
    frames_per_clip: int,
    frame_step: int,
    patch_size: int,
    tubelet_size: int,
    device: str,
    seed: int,
    model_slug: str,
) -> Optional[PatchSample]:
    frames, clip_indices = load_video_clip(video_path, frames_per_clip, frame_step, seed)
    if frames is None:
        return None

    transformed_views = transform(frames)
    if not transformed_views:
        logger.warning("No transformed views produced for %s", video_path)
        return None

    clip = transformed_views[0]
    clip = clip.unsqueeze(0).to(device)
    clip_indices_tensor = torch.from_numpy(clip_indices).long().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = encoder([[clip]], clip_indices=[clip_indices_tensor], attn_mask=None)

    view_output = outputs[0]
    if view_output.ndim != 3 or view_output.shape[0] != 1:
        raise RuntimeError(f"Unexpected encoder output shape: {tuple(view_output.shape)}")

    token_embeddings = view_output[0].detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    token_index = int(rng.integers(0, token_embeddings.shape[0]))
    embedding = token_embeddings[token_index].astype(np.float32, copy=False)

    grid_size = (clip.shape[-1] // patch_size) * (clip.shape[-2] // patch_size)
    temporal_token_index = token_index // grid_size
    spatial_token_index = token_index % grid_size

    patch_image = extract_patch_image(
        clip[0].cpu(),
        temporal_token_index=temporal_token_index,
        spatial_token_index=spatial_token_index,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        normalization=normalization,
    )

    patch_dir = output_dir / model_slug / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / f"sample_{sample_index:04d}.png"
    patch_image.save(patch_path)

    return PatchSample(
        sample_index=sample_index,
        video_path=video_path,
        patch_path=str(patch_path),
        embedding=embedding,
        token_index=token_index,
        temporal_token_index=temporal_token_index,
        spatial_token_index=spatial_token_index,
    )


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    squared_norms = np.sum(x * x, axis=1, keepdims=True)
    distances_sq = squared_norms + squared_norms.T - 2.0 * (x @ x.T)
    np.maximum(distances_sq, 0.0, out=distances_sq)
    return np.sqrt(distances_sq, out=distances_sq)


def hdbscan_cluster(
    x: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    if x.shape[0] == 0:
        return np.empty((0,), dtype=int)

    min_cluster_size = max(2, min(int(min_cluster_size), x.shape[0]))
    if min_samples is not None:
        min_samples = max(1, min(int(min_samples), x.shape[0]))

    return SklearnHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        copy=False,
    ).fit_predict(x)


def pca_init(x: np.ndarray, n_components: int = 50) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2 or x.shape[1] <= n_components:
        return x

    u, s, _ = np.linalg.svd(x, full_matrices=False)
    n_components = min(n_components, u.shape[1], s.shape[0])
    return (u[:, :n_components] * s[:n_components]).astype(np.float64, copy=False)


def _hbeta(distances: np.ndarray, beta: float) -> Tuple[float, np.ndarray]:
    probs = np.exp(-distances * beta)
    probs[distances == 0.0] = 0.0
    total = probs.sum()
    if total <= 0.0:
        return 0.0, np.zeros_like(probs)
    probs /= total
    entropy = np.log(total) + beta * np.sum(distances * probs)
    return entropy, probs


def _binary_search_perplexity(distances: np.ndarray, target_perplexity: float, tol: float = 1e-5) -> np.ndarray:
    beta = 1.0
    beta_min = -np.inf
    beta_max = np.inf
    target_entropy = np.log(target_perplexity)

    for _ in range(50):
        entropy, probs = _hbeta(distances, beta)
        entropy_diff = entropy - target_entropy
        if abs(entropy_diff) <= tol:
            return probs

        if entropy_diff > 0:
            beta_min = beta
            beta = 2.0 * beta if np.isinf(beta_max) else 0.5 * (beta + beta_max)
        else:
            beta_max = beta
            beta = 0.5 * beta if np.isinf(beta_min) else 0.5 * (beta + beta_min)

    _, probs = _hbeta(distances, beta)
    return probs


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


def cluster_and_project(
    samples: List[PatchSample],
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: Optional[int],
    seed: int,
) -> None:
    if not samples:
        return

    embeddings = np.stack([sample.embedding for sample in samples], axis=0)
    cluster_embeddings = pca_init(l2_normalize_embeddings(embeddings), n_components=50)
    umap_embeddings = standardize_embeddings(embeddings)
    clusters = hdbscan_cluster(
        cluster_embeddings,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
    )
    coords = umap_project(umap_embeddings, random_state=seed)

    for sample, cluster, (x_coord, y_coord) in zip(samples, clusters, coords):
        sample.cluster = int(cluster)
        sample.umap_x = float(x_coord)
        sample.umap_y = float(y_coord)


def cluster_color(cluster_id: int):
    if cluster_id < 0:
        return (0.5, 0.5, 0.5, 1.0)
    cmap = plt.get_cmap("tab20")
    return cmap(cluster_id % 20)


def save_umap_plot(samples: List[PatchSample], output_path: Path, title: str) -> None:
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

    for sample in samples:
        border_color = cluster_color(sample.cluster)
        ax.scatter(
            sample.umap_x,
            sample.umap_y,
            s=28,
            color=border_color,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def summarize_clusters(samples: List[PatchSample]) -> Dict[int, int]:
    counts = Counter(sample.cluster for sample in samples)
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def standardize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    return (embeddings - mean) / np.maximum(std, 1e-12)


def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def generate_html_report(model_reports: List[dict], output_path: Path, dataset_csv: str) -> None:
    report_data = json.dumps(model_reports, ensure_ascii=True)
    parts = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"UTF-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
        "<title>Patch Embedding Cluster Report</title>",
        "<style>",
        "body { font-family: Arial, Helvetica, sans-serif; margin: 24px; background: linear-gradient(180deg, #f6f7fb 0%, #eef1f7 100%); color: #1f2937; }",
        "h1 { margin-bottom: 6px; }",
        ".subtitle { color: #4b5563; margin-top: 0; }",
        ".model-card { background: rgba(255,255,255,0.96); border-radius: 16px; padding: 20px; margin: 18px 0 28px; box-shadow: 0 12px 32px rgba(15,23,42,0.10); border: 1px solid rgba(148,163,184,0.25); }",
        ".model-meta { display: flex; flex-wrap: wrap; gap: 16px; color: #374151; margin-bottom: 14px; }",
        ".model-meta span { background: #eef2ff; padding: 6px 10px; border-radius: 999px; }",
        ".scatter-shell { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 18px; align-items: start; }",
        ".plot-column { min-width: 0; }",
        ".plot-toolbar { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }",
        ".plot-toolbar button { background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe; border-radius: 8px; padding: 6px 12px; font-size: 13px; cursor: pointer; }",
        ".plot-toolbar button:hover { background: #e0e7ff; }",
        ".plot-toolbar .hint { color: #6b7280; font-size: 12px; margin-left: auto; }",
        ".scatter-plot { position: relative; width: 100%; height: 560px; border-radius: 14px; border: 1px solid #e5e7eb; background: #ffffff; overflow: hidden; }",
        ".scatter-plot canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }",
        ".scatter-plot canvas.dragging { cursor: grabbing; }",
        ".hover-panel { position: sticky; top: 18px; background: #ffffff; border-radius: 16px; border: 1px solid #dbe2ea; box-shadow: 0 10px 24px rgba(15,23,42,0.08); padding: 14px; display: flex; flex-direction: column; gap: 12px; }",
        ".hover-panel h3 { margin: 0; font-size: 16px; }",
        ".hover-preview { width: 100%; aspect-ratio: 1 / 1; border-radius: 12px; background: #f8fafc; border: 1px dashed #cbd5e1; display: grid; place-items: center; overflow: hidden; }",
        ".hover-preview img { width: 100%; height: 100%; object-fit: contain; image-rendering: auto; }",
        ".hover-preview-placeholder { color: #64748b; font-size: 13px; text-align: center; line-height: 1.35; padding: 12px; }",
        ".hover-meta { font-size: 13px; color: #475569; display: grid; gap: 6px; }",
        ".hover-meta code { background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }",
        "table { border-collapse: collapse; width: 100%; margin-top: 14px; }",
        "th, td { border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; font-size: 14px; }",
        "th { background: #f9fafb; }",
        ".noise { color: #6b7280; }",
        ".cluster { white-space: nowrap; }",
        ".small { font-size: 13px; color: #6b7280; }",
        "@media (max-width: 1000px) { .scatter-shell { grid-template-columns: 1fr; } .hover-panel { position: static; } }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Patch Embedding Cluster Report</h1>",
        f"<p class=\"subtitle\">Dataset manifest: {html.escape(dataset_csv)}</p>",
        f"<script id=\"report-data\" type=\"application/json\">{report_data}</script>",
        "<script>",
        "function escapeHtml(text) {",
        "  return String(text).replace(/[&<>\"']/g, function(character) {",
        "    return ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', \"'\": '&#39;'}[character]);",
        "  });",
        "}",
        "function clusterColor(clusterId) {",
        "  if (clusterId < 0) return '#9ca3af';",
        "  const palette = ['#2563eb', '#16a34a', '#dc2626', '#d97706', '#7c3aed', '#0f766e', '#db2777', '#4f46e5', '#0891b2', '#65a30d', '#ea580c', '#9333ea', '#0284c7', '#be123c', '#059669', '#ca8a04', '#1d4ed8', '#a855f7', '#0d9488', '#fb7185'];",
        "  return palette[clusterId % palette.length];",
        "}",
        "function setPreview(panel, sample) {",
        "  const img = panel.querySelector('.hover-preview-img');",
        "  const placeholder = panel.querySelector('.hover-preview-placeholder');",
        "  const title = panel.querySelector('.hover-preview-title');",
        "  const meta = panel.querySelector('.hover-meta');",
        "  img.src = sample.patch_rel_path;",
        "  img.alt = 'Patch preview for ' + sample.video_path;",
        "  img.hidden = false;",
        "  placeholder.hidden = true;",
        "  title.textContent = sample.video_path.split('/').pop();",
        "  meta.innerHTML = '<div><strong>Cluster:</strong> ' + escapeHtml(sample.cluster_label) + '</div>' +",
        "    '<div><strong>Token:</strong> ' + sample.token_index + ' (t=' + sample.temporal_token_index + ', s=' + sample.spatial_token_index + ')</div>' +",
        "    '<div><strong>Video:</strong> <code>' + escapeHtml(sample.video_path) + '</code></div>';",
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
        "  meta.innerHTML = '<div>Move the pointer over a point in the scatter plot to inspect the crop.</div>';",
        "}",
        "function createScatterPlot(container, model, panel) {",
        "  const samples = model.samples;",
        "  const canvas = document.createElement('canvas');",
        "  container.innerHTML = '';",
        "  container.appendChild(canvas);",
        "  const ctx = canvas.getContext('2d');",
        "  const margin = { top: 30, right: 20, bottom: 40, left: 48 };",
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
        "  const view = { scale: 1, tx: 0, ty: 0 };",
        "  let hoverIndex = -1;",
        "  let cssWidth = 0, cssHeight = 0;",
        "  const dpr = window.devicePixelRatio || 1;",
        "  function plotWidth() { return cssWidth - margin.left - margin.right; }",
        "  function plotHeight() { return cssHeight - margin.top - margin.bottom; }",
        "  function baseX(x) { return margin.left + ((x - dataXLo) / (dataXHi - dataXLo)) * plotWidth(); }",
        "  function baseY(y) { return margin.top + (1 - (y - dataYLo) / (dataYHi - dataYLo)) * plotHeight(); }",
        "  function toScreen(x, y) {",
        "    const bx = baseX(x), by = baseY(y);",
        "    return [bx * view.scale + view.tx, by * view.scale + view.ty];",
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
        "    const radius = Math.max(1.6, Math.min(5, 5 * Math.sqrt(1 / Math.max(1, view.scale))));",
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
        "      ctx.fillStyle = clusterColor(sample.cluster);",
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
        "    ctx.font = '13px Arial, Helvetica, sans-serif';",
        "    ctx.textAlign = 'center';",
        "    ctx.fillText(model.x_label || 'UMAP 1', margin.left + plotWidth() / 2, cssHeight - 8);",
        "    ctx.save();",
        "    ctx.translate(14, margin.top + plotHeight() / 2);",
        "    ctx.rotate(-Math.PI / 2);",
        "    ctx.fillText(model.y_label || 'UMAP 2', 0, 0);",
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
        "    const newScale = clampScale(view.scale * factor);",
        "    const actualFactor = newScale / view.scale;",
        "    view.tx = px - (px - view.tx) * actualFactor;",
        "    view.ty = py - (py - view.ty) * actualFactor;",
        "    view.scale = newScale;",
        "    requestDraw();",
        "  }",
        "  function resetView() { view.scale = 1; view.tx = 0; view.ty = 0; requestDraw(); }",
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
        "    view.tx += e.clientX - lastX;",
        "    view.ty += e.clientY - lastY;",
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
        "  document.querySelectorAll('.model-card').forEach(function(card, index) {",
        "    const plot = card.querySelector('.scatter-plot');",
        "    const panel = card.querySelector('.hover-panel');",
        "    createScatterPlot(plot, models[index], panel);",
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
        summary = report["cluster_summary"]
        cluster_rows = []
        for cluster_id, count in summary.items():
            label = "noise" if cluster_id < 0 else f"cluster {cluster_id}"
            cluster_rows.append(
                f"<tr><td class='cluster'>{html.escape(label)}</td><td>{count}</td></tr>"
            )

        parts.extend(
            [
                f"<div class=\"model-card\">",
                f"<h2>{html.escape(report['model_name'])}</h2>",
                "<div class=\"model-meta\">",
                f"<span>Samples: {report['num_samples']}</span>",
                f"<span>Embedding dim: {report['embedding_dim']}</span>",
                f"<span>HDBSCAN clusters: {report['num_clusters']} (+ noise)</span>",
                f"<span>Plot: {html.escape(Path(report['plot_path']).name)}</span>",
                "</div>",
                "<div class=\"scatter-shell\">",
                "<div class=\"plot-column\">",
                "<div class=\"plot-toolbar\">",
                "<button type=\"button\" data-action=\"zoom-in\">Zoom in</button>",
                "<button type=\"button\" data-action=\"zoom-out\">Zoom out</button>",
                "<button type=\"button\" data-action=\"reset\">Reset view</button>",
                "<span class=\"hint\">Scroll to zoom &middot; drag to pan &middot; hover a point to preview</span>",
                "</div>",
                "<div class=\"scatter-plot\"></div>",
                "</div>",
                "<aside class=\"hover-panel\">",
                "<h3 class=\"hover-preview-title\">Hover a point</h3>",
                "<div class=\"hover-preview\">",
                "<img class=\"hover-preview-img\" alt=\"Hovered patch preview\" hidden>",
                "<div class=\"hover-preview-placeholder\">Move the pointer over a point to preview the corresponding patch at a larger size.</div>",
                "</div>",
                "<div class=\"hover-meta\">",
                "<div>Move the pointer over a point in the scatter plot to inspect the crop.</div>",
                "</div>",
                "</aside>",
                "</div>",
                "<table>",
                "<thead><tr><th>Cluster</th><th>Count</th></tr></thead>",
                f"<tbody>{''.join(cluster_rows)}</tbody>",
                "</table>",
                f"<p class=\"small\">Raw embeddings: {html.escape(report['embeddings_path'])} | Metadata: {html.escape(report['metadata_path'])}</p>",
                "</div>",
            ]
        )

    parts.extend(["</body>", "</html>"])
    output_path.write_text("".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a patch embedding cluster report")
    parser.add_argument(
        "--dataset-csv",
        default="data/EK100_action_recognition_validation.csv",
        help="Path to a whitespace-separated dataset manifest with video paths in the first column.",
    )
    parser.add_argument(
        "--output-dir",
        default="preliminary_experiments/evals/vitl/vjepa_ek100_patch_embeddings",
        help="Directory where the report artifacts will be saved.",
    )
    parser.add_argument("--num-samples", type=int, default=200, help="Number of random videos to sample per model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling and visualization.")
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=10,
        help="Minimum cluster size for HDBSCAN.",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="Optional HDBSCAN min_samples value. If omitted, HDBSCAN uses its default behavior.",
    )
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
    configure_logging(str(output_dir / "patch_embedding_report.log"))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

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
            encoder, _ = prepare_encoder(config, device)
        except Exception as exc:
            logger.error("Failed to load %s: %s", model_name, exc)
            continue

        args_data = config["data"]
        patch_size = args_data["patch_size"]
        tubelet_size = args_data["tubelet_size"]
        normalization = args_data.get("normalization", DEFAULT_NORMALIZATION)

        data_loader, frame_skip_info = load_data_for_model(config, str(sampled_csv), batch_size=args.batch_size)

        samples: List[PatchSample] = []
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
                rng = np.random.default_rng(args.seed + batch_idx)

                for batch_index in range(batch_size):
                    # Order-preserving map back into sampled_manifest: shuffle=False and
                    # drop_last=False mean every batch but the last is exactly
                    # args.batch_size long, so this recovers the flat sample position.
                    global_idx = batch_idx * args.batch_size + batch_index
                    valid_token_indices = torch.where(token_validity[batch_index])[0]
                    if len(valid_token_indices) == 0:
                        logger.warning("No valid tokens found for sample %d in %s", global_idx, model_name)
                        continue

                    token_index = int(valid_token_indices[int(rng.integers(0, len(valid_token_indices)))].item())
                    embedding = view_output[batch_index, token_index].detach().cpu().numpy().astype(np.float32, copy=False)

                    _, _, height, width = videos[batch_index].shape
                    grid_h = height // patch_size
                    grid_w = width // patch_size
                    num_spatial_tokens = grid_h * grid_w
                    temporal_token_index = token_index // num_spatial_tokens
                    spatial_token_index = token_index % num_spatial_tokens

                    patch_image = extract_patch_image(
                        videos[batch_index].detach().cpu(),
                        temporal_token_index=temporal_token_index,
                        spatial_token_index=spatial_token_index,
                        patch_size=patch_size,
                        tubelet_size=tubelet_size,
                        normalization=normalization,
                    )

                    patch_dir = output_dir / model_slug / "patches"
                    patch_dir.mkdir(parents=True, exist_ok=True)
                    patch_path = patch_dir / f"sample_{global_idx:04d}.png"
                    patch_image.save(patch_path)

                    source_row = sampled_manifest.iloc[global_idx]
                    samples.append(
                        PatchSample(
                            sample_index=global_idx,
                            video_path=str(source_row["video_path"]),
                            patch_path=str(patch_path),
                            embedding=embedding,
                            token_index=token_index,
                            temporal_token_index=temporal_token_index,
                            spatial_token_index=spatial_token_index,
                        )
                    )

                if len(samples) >= args.num_samples:
                    break
            except Exception as exc:
                logger.error("Failed processing batch %d for %s: %s", batch_idx, model_name, exc)

        if not samples:
            logger.warning("No samples collected for %s", model_name)
            continue

        cluster_and_project(
            samples,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_min_samples=args.hdbscan_min_samples,
            seed=args.seed,
        )

        model_dir = output_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)
        embeddings = np.stack([sample.embedding for sample in samples], axis=0)
        metadata = pd.DataFrame(
            [
                {
                    "sample_index": sample.sample_index,
                    "video_path": sample.video_path,
                    "patch_path": sample.patch_path,
                    "token_index": sample.token_index,
                    "temporal_token_index": sample.temporal_token_index,
                    "spatial_token_index": sample.spatial_token_index,
                    "cluster": sample.cluster,
                    "umap_x": sample.umap_x,
                    "umap_y": sample.umap_y,
                }
                for sample in samples
            ]
        )

        embeddings_path = model_dir / "embeddings.npy"
        metadata_path = model_dir / "metadata.csv"
        plot_path = model_dir / "umap_patch_clusters.png"
        metadata.to_csv(metadata_path, index=False)
        np.save(embeddings_path, embeddings)
        save_umap_plot(samples, plot_path, title=f"{model_name} patch embeddings")

        cluster_summary = summarize_clusters(samples)
        num_clusters = len([cluster_id for cluster_id in cluster_summary if cluster_id >= 0])
        logger.info("Model %s produced %d clusters", model_name, num_clusters)

        model_reports.append(
            {
                "model_name": model_name,
                "num_samples": len(samples),
                "embedding_dim": int(embeddings.shape[1]),
                "num_clusters": num_clusters,
                "cluster_summary": cluster_summary,
                "plot_path": str(plot_path),
                "embeddings_path": str(embeddings_path),
                "metadata_path": str(metadata_path),
                "x_label": "UMAP 1",
                "y_label": "UMAP 2",
                "samples": [
                    {
                        "video_path": sample.video_path,
                        "patch_rel_path": os.path.relpath(sample.patch_path, output_dir),
                        "token_index": sample.token_index,
                        "temporal_token_index": sample.temporal_token_index,
                        "spatial_token_index": sample.spatial_token_index,
                        "cluster": sample.cluster,
                        "cluster_label": "noise" if sample.cluster < 0 else f"cluster {sample.cluster}",
                        "umap_x": sample.umap_x,
                        "umap_y": sample.umap_y,
                    }
                    for sample in samples
                ],
            }
        )

    if not model_reports:
        logger.error("No model reports were generated")
        return

    report_path = output_dir / "patch_embedding_report.html"
    generate_html_report(model_reports, report_path, args.dataset_csv)
    logger.info("HTML report saved to %s", report_path)


if __name__ == "__main__":
    main()