#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Latent caching / shard writing for the flow-matching decoder.

What a cached sample is. One training example for `D(x_t, z_{t+1})`:

    frame_prev  (3, H, W) uint8  - x_t,     the last frame of temporal token j-1
    frame_next  (3, H, W) uint8  - x_{t+1}, the last frame of temporal token j
    z_target    (S, d_m)  fp16   - the frozen world model's latents for token j
    z_pred      (S, d_m)  fp16   - optional: the predictor's one-step *prediction*
                                   of token j from the real window ending at j-1

Token j spans `tubelet_size` frames; the *last* frame of the tubelet is taken as
the frame that token represents, so `x_t -> x_{t+1}` is exactly one
autoregressive step of the world model (`geom.step_seconds` of video time) and
nothing is being asked of the decoder that the world model was not trained to
predict. `--target-frame-in-tubelet first` is available for a control run; it is
a global choice, never per world model.

`z_target` is produced the same way `encode_ground_truth` in
`evals/generate_world_model_report.py` produces its targets: the target encoder
sees token j inside a *full-length* clip window ending at j, because that is the
only way a token is ever produced during pretraining and a token embedded in
isolation is a different vector. This is what makes the cached latents genuinely
"the world model's latent space" rather than an artifact of how this script
batched them.

What is deliberately NOT cached: codec (VAE) latents. Encoding two 256px frames
is a millisecond and the VAE is frozen and swappable, whereas re-running a ViT-L
world model over a few thousand clips is tens of minutes. Keeping pixels in the
shard means a codec can be changed - or the reconstruction ceiling re-measured -
without touching the expensive half of the pipeline. The training loop encodes on
the fly (see `app/flow_decoder/shard_dataset.py`).

Normalization statistics are fitted in a second pass over the written shards and
stored in the shard-set manifest, from where `LatentNormalization.fit` installs
them as frozen buffers. They are one of the three things the spec allows to
differ between world models, and fitting them here - once, from data, never from
a validation curve - is what keeps them out of the hyperparameter budget.

Example
-------
    python -m app.flow_decoder.latent_cache \
        --model-name "V-JEPA2 (baseline)" \
        --output-dir data/flow_decoder_shards/vjepa2-baseline/train \
        --dataset-csv data/ek55_4fps_train.csv \
        --num-clips 512 --targets-per-clip 6 --store-predictor-latents
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.generate_patch_embedding_report import (
    DEFAULT_NORMALIZATION,
    MODELS_CONFIG,
    denormalize_clip,
    load_config,
    load_manifest,
    slugify,
)
from evals.generate_world_model_report import (
    RolloutGeometry,
    autocast_config,
    build_geometry,
    encode_context,
    layer_norm_last,
    load_long_clip,
    predict_next_token,
    predictor_masks,
    prepare_world_model,
    resolve_checkpoint,
    select_videos,
    temporal_slice,
)
from evals.video_classification_frozen.utils import make_transforms

logger = logging.getLogger(__name__)

SHARD_FORMAT_VERSION = 2
TARGET_FRAME_CHOICES = ("last", "first")
LATENT_SOURCES = ("target_encoder", "predictor")


def configure_logging(log_path: Path) -> None:
    logger.setLevel(logging.INFO)
    # The world-model modules imported above configure the root logger, so without
    # this every line would be emitted twice.
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


@dataclass
class CacheHeader:
    """Everything a consumer needs to interpret a shard set without a world model.

    Written once into `manifest.json`. The decoder is built straight from this -
    `latent_dim`, `latent_grid` and the fitted statistics are precisely the three
    things the spec lets vary between world models, so a training run never has
    to be told which world model it is looking at.
    """

    format_version: int
    model_name: str
    model_slug: str
    checkpoint: str
    config: str
    epoch: int
    latent_dim: int
    latent_grid: Tuple[int, int, int]
    spatial_tokens: int
    crop_size: int
    patch_size: int
    tubelet_size: int
    fps: float
    sampling_fps: float
    step_seconds: float
    window_tokens: int
    context_tokens: int
    is_causal: bool
    target_frame_in_tubelet: str
    has_predictor_latents: bool
    normalization: Tuple[Sequence[float], Sequence[float]]
    num_samples: int
    num_shards: int
    dataset_csv: str
    seed: int


@dataclass
class SampleBuffer:
    """Accumulates samples until a shard is full."""

    frame_prev: List[torch.Tensor]
    frame_next: List[torch.Tensor]
    z_target: List[torch.Tensor]
    z_pred: List[torch.Tensor]
    meta: List[dict]

    @classmethod
    def empty(cls) -> "SampleBuffer":
        return cls([], [], [], [], [])

    def __len__(self) -> int:
        return len(self.meta)

    def clear(self) -> None:
        self.frame_prev.clear()
        self.frame_next.clear()
        self.z_target.clear()
        self.z_pred.clear()
        self.meta.clear()


def frame_in_tubelet(clip: torch.Tensor, token: int, tubelet_size: int, which: str) -> torch.Tensor:
    """The (C, H, W) frame representing temporal token `token` of a (C, T, H, W) clip."""
    offset = tubelet_size - 1 if which == "last" else 0
    return clip[:, token * tubelet_size + offset]


def to_uint8(frame: torch.Tensor, normalization) -> torch.Tensor:
    """One model-normalized (C, H, W) frame -> uint8 RGB in [0, 255].

    Goes through the repo's own `denormalize_clip` (which wants a temporal axis)
    so the inverse of the eval transform is applied in exactly one place.
    Round-tripping through uint8 loses less than a quantization step of the
    original 8-bit video, and it is a 4x saving on the biggest thing in the shard
    after the latents.
    """
    rgb = denormalize_clip(frame.unsqueeze(1), normalization).squeeze(1)  # (C, H, W) in [0, 1]
    return (rgb * 255.0).round().to(torch.uint8)


@torch.no_grad()
def extract_clip_samples(
    bundle,
    geom: RolloutGeometry,
    clip: torch.Tensor,
    autocast_kwargs: dict,
    target_frame: str,
    store_predictor: bool,
    mask_token_index: int,
) -> Tuple[List[dict], List[torch.Tensor], List[torch.Tensor]]:
    """All cacheable samples from one clip.

    `clip` is (1, C, T, H, W), model-normalized, with `geom.total_tokens` temporal
    tokens. For step k in [0, num_steps):

        window w = k + 1  spans tokens [w, w + N_t - 1]
        target token      j = w + N_t - 1 = k + N_t
        x_{t+1}           = frame of token j
        x_t               = frame of token j - 1

    which is index-for-index the layout `encode_ground_truth` uses, so a latent
    cached here and a latent produced by the world-model report for the same
    (video, step) are the same tensor.

    Returns (per-sample metadata, z_target list, z_pred list).
    """
    metas: List[dict] = []
    z_targets: List[torch.Tensor] = []
    z_preds: List[torch.Tensor] = []

    for step in range(geom.num_steps):
        window = step + 1
        target_token = window + geom.window_tokens - 1

        frames = temporal_slice(clip, window, geom.window_tokens, geom.tubelet_size)
        with torch.autocast(**autocast_kwargs):
            tokens = bundle.target_encoder([frames])[0]
        # Last S tokens of the window = the target token's patches.
        z_target = tokens.float()[:, -geom.spatial_tokens :]

        z_pred = None
        if store_predictor:
            # The predictor's own one-step prediction, teacher-forced from the
            # real window that *ends* at target_token - 1: context window
            # [window, window + n_ctx - 1].
            context = encode_context(bundle, geom, clip, window, autocast_kwargs)
            masks = (
                None
                if geom.is_causal
                else predictor_masks(geom, clip.size(0), str(clip.device))
            )
            z_pred = predict_next_token(
                bundle, geom, context, masks, mask_token_index, autocast_kwargs
            )

        metas.append(
            {
                "step": step,
                "target_token": target_token,
                "lead_seconds": geom.step_seconds,
            }
        )
        z_targets.append(z_target)
        z_preds.append(z_pred)

    return metas, z_targets, z_preds


def write_shard(path: Path, buffer: SampleBuffer, header_fields: dict) -> int:
    """Write one shard. Returns the number of samples in it."""
    payload = {
        "format_version": SHARD_FORMAT_VERSION,
        "frame_prev": torch.stack(buffer.frame_prev),
        "frame_next": torch.stack(buffer.frame_next),
        "z_target": torch.stack(buffer.z_target),
        "meta": list(buffer.meta),
        **header_fields,
    }
    if buffer.z_pred and buffer.z_pred[0] is not None:
        payload["z_pred"] = torch.stack(buffer.z_pred)
    torch.save(payload, path)
    return len(buffer)


@torch.no_grad()
def fit_statistics(shard_paths: Sequence[Path], key: str = "z_target") -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std over every cached latent, in one streaming pass.

    Accumulated in float64 over token-and-sample sums rather than by averaging
    per-shard means, so the result does not depend on how the samples happened to
    be split into shards - a shard set rewritten with a different
    `--samples-per-shard` must produce identical buffers or the comparison
    between two world models is not controlled.
    """
    total = None
    total_sq = None
    count = 0
    for path in tqdm(shard_paths, desc=f"fitting {key} statistics", leave=False):
        z = torch.load(path, map_location="cpu", weights_only=False)[key].double()
        flat = z.reshape(-1, z.size(-1))
        total = flat.sum(dim=0) if total is None else total + flat.sum(dim=0)
        total_sq = (flat * flat).sum(dim=0) if total_sq is None else total_sq + (flat * flat).sum(dim=0)
        count += flat.size(0)
    if not count:
        raise ValueError("no latents found; cannot fit statistics")
    mean = total / count
    var = (total_sq / count - mean * mean).clamp_min(0.0)
    return mean.float(), var.sqrt().float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen world-model latents and their frame pairs into shards for the "
        "flow-matching decoder."
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Name of the entry in evals.generate_patch_embedding_report.MODELS_CONFIG to cache. "
        "Defaults to the single enabled entry if there is exactly one.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write shards and manifest.json into.")
    parser.add_argument(
        "--dataset-csv",
        default="data/ek55_4fps_test.csv",
        help="Whitespace-separated manifest of long videos. Use disjoint splits for train and eval "
        "shard sets; the default is the held-out EK55 test split.",
    )
    parser.add_argument("--num-clips", type=int, default=256, help="Videos to draw clips from.")
    parser.add_argument(
        "--targets-per-clip",
        type=int,
        default=4,
        help="Consecutive one-step targets to extract per clip. Each costs one target-encoder forward "
        "over a full window, so this is the main runtime knob.",
    )
    parser.add_argument("--samples-per-shard", type=int, default=256, help="Samples per shard file.")
    parser.add_argument(
        "--target-frame-in-tubelet",
        choices=TARGET_FRAME_CHOICES,
        default="last",
        help="Which frame of a tubelet stands for its temporal token. Global, not per world model.",
    )
    parser.add_argument(
        "--store-predictor-latents",
        action="store_true",
        help="Also cache the predictor's teacher-forced one-step prediction of the target token. "
        "Doubles latent storage; needed by the panel harness to decode predictions alongside targets.",
    )
    parser.add_argument(
        "--layer-norm-latents",
        action="store_true",
        help="Store target-encoder latents layer-normalized, matching the pretraining loss's target "
        "space exactly. Off by default: the channel statistics fitted here already remove the scale "
        "difference, and layer norm additionally discards each token's own norm.",
    )
    parser.add_argument("--mask-token-index", type=int, default=0, help="Predictor mask token to use.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for video and start-offset sampling.")
    parser.add_argument("--batch-size", type=int, default=1, help="Clips encoded at once (memory bound).")
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        default=True,
        help="Run the world model in fp32 instead of the config's meta.dtype.",
    )
    return parser.parse_args()


def resolve_model_cfg(model_name: Optional[str]) -> dict:
    if model_name is None:
        if len(MODELS_CONFIG) != 1:
            raise SystemExit(
                f"MODELS_CONFIG has {len(MODELS_CONFIG)} enabled entries; pass --model-name to pick one."
            )
        return MODELS_CONFIG[0]
    for cfg in MODELS_CONFIG:
        if cfg["name"] == model_name:
            return cfg
    names = ", ".join(repr(c["name"]) for c in MODELS_CONFIG)
    raise SystemExit(f"no model named {model_name!r} in MODELS_CONFIG; enabled entries: {names}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "latent_cache.log")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_cfg = resolve_model_cfg(args.model_name)
    config = load_config(model_cfg["config"])
    checkpoint = resolve_checkpoint(model_cfg, config)

    # `num_steps` is what decides how many one-step targets a clip yields, and
    # `total_tokens` already includes the leading window every target needs behind
    # it. Build once to learn `step_seconds`, then set the count directly rather
    # than back-solving a horizon in seconds, which would round.
    geom = replace(build_geometry(config, horizon_seconds=1.0), num_steps=int(args.targets_per_clip))

    logger.info("model: %s (%s)", model_cfg["name"], checkpoint)
    logger.info(
        "geometry: %d temporal token(s) per clip, %d spatial token(s), tubelet %d, step %.3fs, "
        "crop %d, causal=%s",
        geom.total_tokens,
        geom.spatial_tokens,
        geom.tubelet_size,
        geom.step_seconds,
        geom.crop_size,
        geom.is_causal,
    )

    bundle = prepare_world_model(config, checkpoint, device)
    autocast_kwargs = autocast_config(config, device, args.use_amp)

    normalization = config["data"].get("normalization", DEFAULT_NORMALIZATION)
    transform = make_transforms(
        training=False,
        crop_size=geom.crop_size,
        num_views_per_clip=1,
        normalize=tuple(tuple(v) for v in normalization),
    )

    manifest = load_manifest(args.dataset_csv)
    videos = select_videos(manifest, {model_cfg["name"]: geom}, args.num_clips)
    if not videos:
        raise SystemExit(
            f"no video in {args.dataset_csv} is long enough for {geom.total_tokens} temporal tokens at "
            f"{geom.sampling_fps:g} fps."
        )
    logger.info("selected %d/%d video(s)", len(videos), len(manifest))

    side = int(round(math.sqrt(geom.spatial_tokens)))
    if side * side != geom.spatial_tokens:
        raise SystemExit(f"{geom.spatial_tokens} spatial tokens is not a square grid; cannot build coordinates.")
    latent_grid = (1, side, side)

    shard_fields = {
        "model_name": model_cfg["name"],
        "latent_dim": int(bundle.embed_dim),
        "latent_grid": latent_grid,
        "crop_size": geom.crop_size,
        "normalization": normalization,
    }

    buffer = SampleBuffer.empty()
    shard_paths: List[Path] = []
    num_samples = 0
    started = time.time()

    for clip_id, (row_idx, video_path) in enumerate(tqdm(videos, desc="caching clips")):
        loaded = load_long_clip(video_path, geom, transform, np.random.default_rng(args.seed + row_idx))
        if loaded is None:
            continue
        clip, start_frame = loaded
        clip = clip.unsqueeze(0).to(device, non_blocking=True)

        try:
            metas, z_targets, z_preds = extract_clip_samples(
                bundle,
                geom,
                clip,
                autocast_kwargs,
                args.target_frame_in_tubelet,
                args.store_predictor_latents,
                args.mask_token_index,
            )
        except torch.cuda.OutOfMemoryError:
            logger.error("OOM on %s; skipping clip. Lower --targets-per-clip.", video_path)
            torch.cuda.empty_cache()
            continue

        clip_cpu = clip.squeeze(0).cpu()
        for meta, z_target, z_pred in zip(metas, z_targets, z_preds):
            token = meta["target_token"]
            prev = frame_in_tubelet(clip_cpu, token - 1, geom.tubelet_size, args.target_frame_in_tubelet)
            nxt = frame_in_tubelet(clip_cpu, token, geom.tubelet_size, args.target_frame_in_tubelet)

            latent = z_target[0]
            if args.layer_norm_latents:
                latent = layer_norm_last(latent)

            buffer.frame_prev.append(to_uint8(prev, normalization))
            buffer.frame_next.append(to_uint8(nxt, normalization))
            buffer.z_target.append(latent.half().cpu())
            buffer.z_pred.append(z_pred[0].half().cpu() if z_pred is not None else None)
            buffer.meta.append(
                {**meta, "video_path": video_path, "start_frame": int(start_frame), "clip_id": clip_id}
            )

            if len(buffer) >= args.samples_per_shard:
                path = output_dir / f"shard-{len(shard_paths):05d}.pt"
                num_samples += write_shard(path, buffer, shard_fields)
                shard_paths.append(path)
                buffer.clear()

        del clip, clip_cpu, z_targets, z_preds

    if len(buffer):
        path = output_dir / f"shard-{len(shard_paths):05d}.pt"
        num_samples += write_shard(path, buffer, shard_fields)
        shard_paths.append(path)
        buffer.clear()

    if not shard_paths:
        raise SystemExit("no samples were written; every clip failed to load or encode.")

    logger.info(
        "wrote %d sample(s) across %d shard(s) in %.1fs", num_samples, len(shard_paths), time.time() - started
    )

    mean, std = fit_statistics(shard_paths, key="z_target")
    stats = {"z_target": {"mean": mean, "std": std}}
    logger.info(
        "z_target statistics: mean in [%.4f, %.4f], std in [%.4f, %.4f]",
        mean.min().item(),
        mean.max().item(),
        std.min().item(),
        std.max().item(),
    )
    if args.store_predictor_latents:
        pred_mean, pred_std = fit_statistics(shard_paths, key="z_pred")
        stats["z_pred"] = {"mean": pred_mean, "std": pred_std}
        # Worth logging together: the gap between these two is exactly the
        # distribution shift a decoder trained on target-encoder latents faces
        # when it is asked to decode a predictor output.
        logger.info(
            "z_pred statistics: mean in [%.4f, %.4f], std in [%.4f, %.4f]; mean |std ratio - 1| vs "
            "z_target = %.4f",
            pred_mean.min().item(),
            pred_mean.max().item(),
            pred_std.min().item(),
            pred_std.max().item(),
            (pred_std / std.clamp_min(1e-6) - 1.0).abs().mean().item(),
        )
    torch.save(stats, output_dir / "statistics.pt")

    header = CacheHeader(
        format_version=SHARD_FORMAT_VERSION,
        model_name=model_cfg["name"],
        model_slug=slugify(model_cfg["name"]),
        checkpoint=checkpoint,
        config=model_cfg["config"],
        epoch=int(bundle.epoch),
        latent_dim=int(bundle.embed_dim),
        latent_grid=latent_grid,
        spatial_tokens=int(geom.spatial_tokens),
        crop_size=int(geom.crop_size),
        patch_size=int(geom.patch_size),
        tubelet_size=int(geom.tubelet_size),
        fps=float(geom.fps),
        sampling_fps=float(geom.sampling_fps),
        step_seconds=float(geom.step_seconds),
        window_tokens=int(geom.window_tokens),
        context_tokens=int(geom.context_tokens),
        is_causal=bool(geom.is_causal),
        target_frame_in_tubelet=args.target_frame_in_tubelet,
        has_predictor_latents=bool(args.store_predictor_latents),
        normalization=normalization,
        num_samples=num_samples,
        num_shards=len(shard_paths),
        dataset_csv=args.dataset_csv,
        seed=args.seed,
    )
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({**asdict(header), "shards": [p.name for p in shard_paths]}, handle, indent=2)
    logger.info("manifest written to %s", output_dir / "manifest.json")


if __name__ == "__main__":
    main()
