#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Qualitative comparison harness for the flow-matching decoder.

Six diagnostic panels, each answering one question about a frozen world model's
latent space. The decoder is a microscope, so every panel is built to keep
everything except the thing under study bit-for-bit fixed - same clip, same
noise seed, same solver, same step count, same guidance:

  1. `reconstruction`  What does the latent actually pin down? Columns:
       x_t | ground-truth x_{t+1} | codec round-trip of x_{t+1} (the ceiling no
       decoder can beat) | D(x_t, z_target) | D(x_t, z_pred) when predictor
       latents were cached. The ceiling column is the point: a blurry sample next
       to a sharp ceiling is the decoder or the latent, a blurry sample next to a
       blurry ceiling is the VAE and means nothing about the world model.

  2. `guidance`        How much does the latent add over persistence? A CFG sweep
       from w = 0 (the latent fully ignored - the decoder's prior for "next frame
       given this frame") through w = 1 (the honest conditional) to w = 5. Because
       only `z` is dropped and never `x_t`, this ladder isolates the latent's
       contribution. A latent space that carries little beyond persistence looks
       nearly static across the whole row.

  3. `seeds`           What does the latent leave free? The same latent decoded
       from K different noise seeds, plus a per-pixel standard-deviation map.
       Bright regions are where the latent did not determine the pixels. This is
       the most direct read-out of latent informativeness in the whole harness,
       and it needs no ground truth.

  4. `lead_time`       How does legibility decay with prediction distance? A
       ladder over consecutive one-step targets of one clip, or - with
       `--rollout-latents` - over caller-supplied autoregressive rollout latents.
       This harness never rolls a world model forward itself; that is the
       world-model report's job and explicitly out of scope here.

  5. `token_ablation`  Which pixels does which token control? A spatial block of
       latent tokens is replaced by the fitted channel mean (the "no information"
       token) and the sample is re-decoded from the same seed; the difference map
       shows the region that block governs. A latent space whose tokens have
       drifted away from their spatial position produces diffuse, misplaced
       difference maps.

  6. `crossmodel`      The actual comparison. One row per world model, same clip,
       same seed, same everything - plus a latent-swap control row in which the
       latent comes from a *different* clip. If the swap row still looks like the
       original clip, the decoder is ignoring the latent and every other panel in
       the report is meaningless. Also writes `pairs_2afc.json` for the Gradio
       forced-choice app.

Controlled comparison is enforced, not assumed. Every checkpoint loaded is
audited against the first one: `model_config`, `flow_config`, `total_steps`, the
codec identity and the latent source must match exactly, and the only permitted
differences are `latent_dim` and `latent_grid`. A mismatch aborts the report
rather than producing a picture that looks like a finding.

Example
-------
    python evals/generate_flow_decoder_panels.py \
        --decoder "V-JEPA2 (baseline)=preliminary_experiments/flow-decoder/vjepa2-baseline-configA/final.pt" \
        --decoder "High V-JEPA=preliminary_experiments/flow-decoder/high-vjepa-configA/final.pt" \
        --eval-shards "V-JEPA2 (baseline)=data/flow_decoder_shards/vjepa2-baseline/eval" \
        --eval-shards "High V-JEPA=data/flow_decoder_shards/high-vjepa/eval" \
        --output-dir preliminary_experiments/evals/vitl/flow_decoder_panels
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.flow_decoder.shard_dataset import ShardSet, frames_to_unit, load_shard_set
from src.models.flow_decoder.decoder import FlowMatchingDecoder
from src.models.flow_decoder.flow import sample_ode
from src.models.flow_decoder.latent_adapter import build_token_coords
from src.models.flow_decoder.vae import build_codec, roundtrip_psnr

logger = logging.getLogger(__name__)

PANELS = ("reconstruction", "guidance", "seeds", "lead_time", "token_ablation", "crossmodel")

# Config keys that a comparison is allowed to see differ between world models.
# Everything else differing is a bug by the spec's own definition.
PERMITTED_DIFFERENCES = ("latent_dim", "latent_grid")

DEFAULT_GUIDANCE_LADDER = (0.0, 1.0, 1.5, 2.0, 3.0, 5.0)


def configure_logging(log_path: Path) -> None:
    logger.setLevel(logging.INFO)
    # The world-model modules imported above configure the root logger, so without
    # this every line would be emitted twice.
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


# --------------------------------------------------------------------------- #
# Loading and auditing
# --------------------------------------------------------------------------- #


@dataclass
class DecoderBundle:
    """A trained decoder plus the shard set and provenance it came from."""

    name: str
    decoder: FlowMatchingDecoder
    shard_set: ShardSet
    coords: torch.Tensor
    checkpoint_path: Path
    provenance: dict
    codec_ceiling_db: float = float("nan")

    @property
    def latent_dim(self) -> int:
        return self.decoder.latent_dim


def read_provenance(name: str, checkpoint_path: Path) -> dict:
    """Read a checkpoint's metadata without materializing its weights.

    `mmap=True` maps the tensor storages lazily, so this touches only the small
    Python objects. That matters because the audit below has to run BEFORE any
    decoder is constructed: a checkpoint whose `model_config` is wrong is exactly
    the checkpoint you must not hand to `FlowMatchingDecoder`, and a bad `depth`
    would allocate a pathological model before the audit ever got a chance to
    reject it.
    """
    if not checkpoint_path.exists():
        raise SystemExit(f"{name}: checkpoint not found at {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    provenance = {
        "name": name,
        "checkpoint_path": checkpoint_path,
        "model_config": state.get("model_config"),
        "flow_config": state.get("flow_config"),
        "total_steps": state.get("total_steps"),
        "step": state.get("step"),
        "latent_source": state.get("latent_source", "target_encoder"),
        "codec": state.get("codec", {}),
        "latent_dim": int(state["latent_dim"]),
        "latent_grid": list(state["latent_grid"]),
        "crop_size": int(state["crop_size"]),
        "world_model": state.get("world_model", {}),
        "config_folder": (state.get("config") or {}).get("folder"),
    }
    del state
    return provenance


def audit_provenance(provenances: Sequence[dict]) -> List[str]:
    """Abort unless every decoder differs only in the permitted ways.

    Runs on metadata alone, before any decoder is built. Returns the audit lines
    for the report so the guarantee is visible to a reader, not just enforced in
    code.
    """
    reference = provenances[0]
    lines = [
        f"Reference run: `{reference['name']}` from `{reference['checkpoint_path']}`",
        f"- fixed step budget: {reference.get('total_steps')} step(s)",
        f"- latent source: {reference.get('latent_source')}",
        f"- codec: {reference.get('codec', {}).get('model_id')}",
    ]
    problems: List[str] = []
    for other in provenances[1:]:
        for key in ("model_config", "flow_config", "total_steps", "latent_source"):
            a, b = reference.get(key), other.get(key)
            if a != b:
                problems.append(f"{other['name']}: {key} differs from {reference['name']} ({b!r} vs {a!r})")
        a_codec, b_codec = dict(reference.get("codec", {})), dict(other.get("codec", {}))
        if a_codec != b_codec:
            problems.append(f"{other['name']}: codec differs from {reference['name']} ({b_codec} vs {a_codec})")
        differences = [key for key in PERMITTED_DIFFERENCES if reference.get(key) != other.get(key)]
        lines.append(
            f"`{other['name']}`: matches the reference; permitted differences: "
            + (", ".join(f"{k}={other.get(k)}" for k in differences) or "none")
        )
    if problems:
        raise SystemExit(
            "the decoders being compared are not a controlled comparison:\n  - "
            + "\n  - ".join(problems)
            + "\n\nPer the spec, any config difference between world models other than d_m, grid shape "
            "and normalization buffers is a bug. Retrain with a shared config, or drop the odd run."
        )
    return lines


def load_decoder(
    provenance: dict,
    shard_root: Path,
    device: str,
    use_ema: bool,
) -> Tuple[DecoderBundle, object]:
    """Rebuild a decoder from its checkpoint, with its codec and eval shards.

    Only called for checkpoints that already passed `audit_provenance`.
    """
    name = provenance["name"]
    checkpoint_path = provenance["checkpoint_path"]
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    codec_info = state.get("codec", {})
    codec = build_codec(
        kind=codec_info.get("kind", "kl"),
        model_id=codec_info.get("model_id", "stabilityai/sd-vae-ft-mse"),
        downsample_factor=int(codec_info.get("downsample_factor", 8)),
        device=device,
    )

    info = load_shard_set(shard_root)
    latent_source = state.get("latent_source", "target_encoder")
    model_cfg = state["model_config"]
    decoder = FlowMatchingDecoder(
        latent_dim=int(state["latent_dim"]),
        latent_grid=tuple(state["latent_grid"]),
        codec_channels=codec.latent_channels,
        codec_size=codec.latent_size(int(state["crop_size"])),
        patch_size=int(model_cfg.get("patch_size", 2)),
        dim=int(model_cfg.get("dim", 768)),
        depth=int(model_cfg.get("depth", 12)),
        num_heads=int(model_cfg.get("num_heads", 12)),
        cond_dim=int(model_cfg.get("cond_dim", 768)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        conditioning=str(model_cfg.get("conditioning", "perceiver")),
        frame_conditioning=str(model_cfg.get("frame_conditioning", "none")),
        num_queries=int(model_cfg.get("num_queries", 128)),
        resampler_depth=int(model_cfg.get("resampler_depth", 2)),
        resampler_heads=int(model_cfg.get("resampler_heads", 8)),
        latent_norm_mode=str(model_cfg.get("latent_norm_mode", "channel")),
        num_coord_bands=int(model_cfg.get("num_coord_bands", 8)),
    ).to(device)

    weights = state["ema"]["shadow"] if (use_ema and "ema" in state) else state["decoder"]
    missing, unexpected = decoder.load_state_dict({k: v for k, v in weights.items()}, strict=False)
    if missing or unexpected:
        logger.warning(
            "%s: %d missing / %d unexpected key(s) when loading weights", name, len(missing), len(unexpected)
        )
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False

    shard_set = ShardSet(info, latent_source=latent_source)
    coords = build_token_coords(info.latent_grid, device=torch.device(device))
    provenance = dict(provenance)
    provenance["weights"] = "ema" if (use_ema and "ema" in state) else "raw"
    provenance["shard_set"] = info.describe()
    logger.info("%s: loaded %s (%s weights); %s", name, checkpoint_path, provenance["weights"], info.describe())
    return DecoderBundle(name, decoder, shard_set, coords, checkpoint_path, provenance), codec


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) float in [0, 1] -> (H, W, 3) numpy for imshow."""
    return tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def save_panel(
    rows: Sequence[Sequence[np.ndarray]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    path: Path,
    title: str,
    caption: str = "",
    cell_size: float = 2.2,
) -> None:
    """Write a labelled image grid.

    Labels are drawn rather than left to a caption because these panels get
    pasted into notes and dragged into a forced-choice app, and an unlabelled
    grid of near-identical frames is worse than no panel at all.
    """
    num_rows, num_cols = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(cell_size * num_cols, cell_size * num_rows + 0.9),
        squeeze=False,
    )
    for r in range(num_rows):
        for c in range(num_cols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(rows[r]):
                image = rows[r][c]
                ax.imshow(image, cmap=None if image.ndim == 3 else "magma", vmin=None if image.ndim == 3 else 0)
            else:
                ax.axis("off")
            if r == 0 and c < len(col_labels):
                ax.set_title(col_labels[c], fontsize=8)
            if c == 0 and r < len(row_labels):
                ax.set_ylabel(row_labels[r], fontsize=8, rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle(title, fontsize=11)
    if caption:
        fig.text(0.5, 0.005, caption, ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10.0 * float(np.log10(1.0 / mse))


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #


@dataclass
class SamplerSettings:
    """Solver settings shared by every panel, so nothing varies by accident."""

    num_steps: int = 50
    guidance: float = 1.5
    solver: str = "heun"
    shift: float = 1.0
    seed: int = 0

    def describe(self) -> str:
        return (
            f"{self.solver} solver, {self.num_steps} step(s), guidance {self.guidance:g}, "
            f"shift {self.shift:g}, seed {self.seed}"
        )


@torch.no_grad()
def decode(
    bundle: DecoderBundle,
    codec,
    cond_frames: torch.Tensor,
    z: torch.Tensor,
    settings: SamplerSettings,
    guidance: Optional[float] = None,
    seed: Optional[int] = None,
    coords: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One decode in pixel space.

    `cond_frames` (B, 3, H, W) in [0, 1] is passed to the decoder only under
    `frame_conditioning='current_frame'`; under `none` it is ignored here, so the
    panels can hold one call signature while the reconstruction genuinely has no
    access to it.
    """
    uses_frame = bundle.decoder.uses_frame_conditioning
    latent = sample_ode(
        bundle.decoder,
        z=z,
        cond_latent=codec.encode(cond_frames).float() if uses_frame else None,
        coords=bundle.coords if coords is None else coords,
        num_steps=settings.num_steps,
        guidance=settings.guidance if guidance is None else guidance,
        solver=settings.solver,
        shift=settings.shift,
        seed=settings.seed if seed is None else seed,
    )
    return codec.decode(latent).float()


def fetch(bundle: DecoderBundle, index: int, device: str) -> dict:
    """One eval sample, on device, with frames in [0, 1]."""
    item = bundle.shard_set[index]
    out = {
        "prev": frames_to_unit(item["frame_prev"].unsqueeze(0).to(device)),
        "next": frames_to_unit(item["frame_next"].unsqueeze(0).to(device)),
        "z": item["z"].unsqueeze(0).float().to(device),
        "z_target": item["z_target"].unsqueeze(0).float().to(device),
        "meta": item["meta"],
    }
    if "z_pred" in item:
        out["z_pred"] = item["z_pred"].unsqueeze(0).float().to(device)
    return out


# --------------------------------------------------------------------------- #
# Panel 1: reconstruction
# --------------------------------------------------------------------------- #


@torch.no_grad()
def panel_reconstruction(
    bundle: DecoderBundle,
    codec,
    indices: Sequence[int],
    settings: SamplerSettings,
    output_dir: Path,
    device: str,
) -> dict:
    rows, row_labels = [], []
    stats = {"psnr_decoded": [], "psnr_copy": [], "psnr_ceiling": []}
    has_pred = False

    for index in indices:
        sample = fetch(bundle, index, device)
        ceiling = codec.decode(codec.encode(sample["next"])).float()
        decoded = decode(bundle, codec, sample["prev"], sample["z_target"], settings)

        row = [
            to_numpy_image(sample["prev"][0]),
            to_numpy_image(sample["next"][0]),
            to_numpy_image(ceiling[0]),
            to_numpy_image(decoded[0]),
        ]
        if "z_pred" in sample:
            has_pred = True
            row.append(to_numpy_image(decode(bundle, codec, sample["prev"], sample["z_pred"], settings)[0]))
        rows.append(row)
        row_labels.append(f"clip {sample['meta'].get('clip_id')}\nstep {sample['meta'].get('step')}")

        stats["psnr_decoded"].append(psnr(decoded, sample["next"]))
        stats["psnr_copy"].append(psnr(sample["prev"], sample["next"]))
        stats["psnr_ceiling"].append(psnr(ceiling, sample["next"]))

    uses_frame = bundle.decoder.uses_frame_conditioning
    first = "x_t (decoder input)" if uses_frame else "x_t (context only)"
    signature = "D(x_t, z" if uses_frame else "D(z"
    columns = [first, "x_{t+1} (truth)", "codec ceiling", f"{signature}_target)"]
    if has_pred:
        columns.append(f"{signature}_pred)")

    path = output_dir / f"panel1-reconstruction-{slug(bundle.name)}.png"
    save_panel(
        rows,
        row_labels,
        columns,
        path,
        f"Panel 1 - reconstruction: {bundle.name}",
        caption=(
            f"{settings.describe()}. The `codec ceiling` column is decode(encode(truth)) and bounds every "
            "column to its right; compare against it, not against the truth. `copy` PSNR "
            f"{np.mean(stats['psnr_copy']):.2f} dB is the persistence baseline"
            + (
                " - and, since the decoder is given x_t, a floor it could reach by echoing its input."
                if uses_frame
                else " - a reference for how much the scene moved, NOT something this decoder could "
                "reach by copying: it never sees x_t."
            )
        ),
    )
    summary = {k: float(np.mean(v)) for k, v in stats.items()}
    logger.info(
        "panel 1 (%s): decoded %.2f dB, copy baseline %.2f dB, codec ceiling %.2f dB",
        bundle.name,
        summary["psnr_decoded"],
        summary["psnr_copy"],
        summary["psnr_ceiling"],
    )
    return {"path": path, **summary, "has_predictor_latents": has_pred}


# --------------------------------------------------------------------------- #
# Panel 2: guidance ladder
# --------------------------------------------------------------------------- #


@torch.no_grad()
def panel_guidance(
    bundle: DecoderBundle,
    codec,
    indices: Sequence[int],
    settings: SamplerSettings,
    ladder: Sequence[float],
    output_dir: Path,
    device: str,
) -> dict:
    rows, row_labels = [], []
    drift = []
    for index in indices:
        sample = fetch(bundle, index, device)
        row = [to_numpy_image(sample["next"][0])]
        samples = []
        for guidance in ladder:
            decoded = decode(bundle, codec, sample["prev"], sample["z_target"], settings, guidance=guidance)
            samples.append(decoded)
            row.append(to_numpy_image(decoded[0]))
        rows.append(row)
        row_labels.append(f"clip {sample['meta'].get('clip_id')}")
        # How far w = 0 (latent ignored) sits from w = 1 (honest conditional):
        # the single number that says whether the latent is doing anything.
        if 1.0 in ladder:
            drift.append((samples[0] - samples[ladder.index(1.0)]).abs().mean().item())
        else:
            drift.append(float("nan"))

    path = output_dir / f"panel2-guidance-{slug(bundle.name)}.png"
    save_panel(
        rows,
        row_labels,
        ["x_{t+1} (truth)"] + [f"w = {g:g}" for g in ladder],
        path,
        f"Panel 2 - guidance ladder: {bundle.name}",
        caption=(
            f"{settings.solver} solver, {settings.num_steps} step(s), seed {settings.seed} held fixed across the "
            "row. Only z is dropped for the unconditional branch, so w = 0 is "
            + (
                "'next frame from this frame alone' and the row measures what the latent adds over "
                "persistence."
                if bundle.decoder.uses_frame_conditioning
                else "the decoder's unconditional prior over frames of this dataset - it sees NOTHING about "
                "this clip - and the row measures what the latent adds over a generic frame."
            )
            + f" Mean |w=0 - w=1| = {np.nanmean(drift):.4f}."
        ),
    )
    logger.info("panel 2 (%s): mean |w=0 - w=1| = %.4f", bundle.name, float(np.nanmean(drift)))
    return {"path": path, "uncond_to_cond_l1": float(np.nanmean(drift))}


# --------------------------------------------------------------------------- #
# Panel 3: seed spread
# --------------------------------------------------------------------------- #


@torch.no_grad()
def panel_seeds(
    bundle: DecoderBundle,
    codec,
    indices: Sequence[int],
    settings: SamplerSettings,
    num_seeds: int,
    output_dir: Path,
    device: str,
) -> dict:
    rows, row_labels = [], []
    spreads = []
    for index in indices:
        sample = fetch(bundle, index, device)
        draws = [
            decode(bundle, codec, sample["prev"], sample["z_target"], settings, seed=settings.seed + s)
            for s in range(num_seeds)
        ]
        stack = torch.cat(draws, dim=0)
        std_map = stack.std(dim=0).mean(dim=0)  # (H, W), averaged over colour
        spreads.append(std_map.mean().item())
        row = [to_numpy_image(sample["next"][0])]
        row += [to_numpy_image(d[0]) for d in draws]
        # Normalized so the structure is visible; the absolute level is in the caption.
        row.append((std_map / std_map.max().clamp_min(1e-8)).cpu().numpy())
        rows.append(row)
        row_labels.append(f"clip {sample['meta'].get('clip_id')}\nspread {std_map.mean().item():.4f}")

    path = output_dir / f"panel3-seeds-{slug(bundle.name)}.png"
    save_panel(
        rows,
        row_labels,
        ["x_{t+1} (truth)"] + [f"seed {settings.seed + s}" for s in range(num_seeds)] + ["per-pixel std"],
        path,
        f"Panel 3 - what the latent leaves free: {bundle.name}",
        caption=(
            f"{settings.describe()} with the seed varied and everything else fixed. Bright regions of the "
            "std map are pixels the latent did not determine. Mean spread over clips "
            f"{np.mean(spreads):.4f} (in [0, 1] pixel units); the map is max-normalized per row."
        ),
    )
    logger.info("panel 3 (%s): mean per-pixel seed spread %.4f", bundle.name, float(np.mean(spreads)))
    return {"path": path, "seed_spread": float(np.mean(spreads))}


# --------------------------------------------------------------------------- #
# Panel 4: lead-time ladder
# --------------------------------------------------------------------------- #


@torch.no_grad()
def panel_lead_time(
    bundle: DecoderBundle,
    codec,
    settings: SamplerSettings,
    output_dir: Path,
    device: str,
    rollout_latents: Optional[Path],
    max_steps: int,
) -> dict:
    """Consecutive one-step targets of one clip, or caller-supplied rollout latents.

    The two cases are labelled differently on purpose. Consecutive cached steps
    are each a *fresh* one-step problem from real context - difficulty rises only
    because the scene moves. Rollout latents compound the world model's own error,
    which is a different and much harder question; conflating them would flatter
    the world model.
    """
    if rollout_latents is not None:
        return _panel_lead_time_rollout(bundle, codec, settings, output_dir, device, rollout_latents, max_steps)

    # Group cached samples by clip and take the longest run of consecutive steps.
    by_clip: Dict[int, List[int]] = {}
    for i in range(len(bundle.shard_set)):
        meta = bundle.shard_set[i]["meta"]
        by_clip.setdefault(int(meta.get("clip_id", 0)), []).append(i)
    clip_id, indices = max(by_clip.items(), key=lambda kv: len(kv[1]))
    indices = indices[:max_steps]
    if len(indices) < 2:
        logger.warning("panel 4 skipped: no clip has 2+ cached steps (raise --targets-per-clip when caching)")
        return {}

    first = fetch(bundle, indices[0], device)
    truths, decoded_row = [], []
    for index in indices:
        sample = fetch(bundle, index, device)
        truths.append(to_numpy_image(sample["next"][0]))
        latent = sample.get("z_pred", sample["z_target"])
        decoded_row.append(to_numpy_image(decode(bundle, codec, sample["prev"], latent, settings)[0]))

    source = "z_pred (teacher-forced)" if "z_pred" in first else "z_target"
    path = output_dir / f"panel4-lead-time-{slug(bundle.name)}.png"
    save_panel(
        [truths, decoded_row],
        ["truth", f"D(x_t, {source})"],
        [f"step {i}" for i in range(len(indices))],
        path,
        f"Panel 4 - lead-time ladder (teacher-forced): {bundle.name}",
        caption=(
            f"Clip {clip_id}, consecutive one-step targets. Each column is an INDEPENDENT one-step problem "
            "from real context, so this shows how scene difficulty varies along a clip - it is NOT an "
            "autoregressive rollout. Pass --rollout-latents for that; this harness never rolls a world "
            f"model forward itself. {settings.describe()}."
        ),
    )
    logger.info(
        "panel 4 (%s): %d consecutive step(s) of clip %d, source %s",
        bundle.name,
        len(indices),
        clip_id,
        source,
    )
    return {"path": path, "mode": "teacher_forced", "num_steps": len(indices), "clip_id": clip_id}


@torch.no_grad()
def _panel_lead_time_rollout(
    bundle: DecoderBundle,
    codec,
    settings: SamplerSettings,
    output_dir: Path,
    device: str,
    rollout_path: Path,
    max_steps: int,
) -> dict:
    """Decode caller-supplied rollout latents.

    Expected file: a torch save with `latents` (K, S, d_m) - one latent per
    autoregressive step, in order - and `frame_prev` (3, H, W) uint8, the last
    frame the world model actually observed. Optional `truth` (K, 3, H, W) uint8
    for a comparison row. Producing this file is the caller's job (see
    `evals/generate_world_model_report.py`, which already does the rollout).
    """
    payload = torch.load(rollout_path, map_location="cpu", weights_only=False)
    latents = payload["latents"].float()[:max_steps]
    if latents.dim() != 3 or latents.size(-1) != bundle.latent_dim:
        raise SystemExit(
            f"{rollout_path}: expected latents (K, S, {bundle.latent_dim}); got {tuple(latents.shape)}"
        )
    prev = frames_to_unit(payload["frame_prev"].unsqueeze(0).to(device))

    decoded_row = []
    for step in range(latents.size(0)):
        z = latents[step].unsqueeze(0).to(device)
        decoded_row.append(to_numpy_image(decode(bundle, codec, prev, z, settings)[0]))

    rows, row_labels = [], []
    if "truth" in payload:
        truth = frames_to_unit(payload["truth"][: latents.size(0)].to(device))
        rows.append([to_numpy_image(truth[i]) for i in range(truth.size(0))])
        row_labels.append("truth")
    rows.append(decoded_row)
    row_labels.append("D(x_t_obs, z_k)")

    path = output_dir / f"panel4-rollout-{slug(bundle.name)}.png"
    save_panel(
        rows,
        row_labels,
        [f"k = {k + 1}" for k in range(latents.size(0))],
        path,
        f"Panel 4 - rollout ladder: {bundle.name}",
        caption=(
            f"Caller-supplied rollout latents from {rollout_path.name}, decoded against the LAST OBSERVED "
            "frame throughout, so degradation along the row is the world model's compounding error and not "
            f"the decoder's. {settings.describe()}."
        ),
    )
    logger.info("panel 4 (%s): decoded %d rollout latent(s) from %s", bundle.name, latents.size(0), rollout_path)
    return {"path": path, "mode": "rollout", "num_steps": int(latents.size(0)), "source": str(rollout_path)}


# --------------------------------------------------------------------------- #
# Panel 5: token ablation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def panel_token_ablation(
    bundle: DecoderBundle,
    codec,
    index: int,
    settings: SamplerSettings,
    block_grid: int,
    output_dir: Path,
    device: str,
) -> dict:
    """Ablate spatial blocks of latent tokens and show the pixels each governs.

    The replacement value is the fitted channel mean - the token that carries no
    information after normalization - rather than zero, which after
    normalization is a real and possibly extreme value.
    """
    sample = fetch(bundle, index, device)
    grid = bundle.decoder.latent_grid
    t_dim, h_dim, w_dim = grid
    z = sample["z_target"]
    if z.size(1) != t_dim * h_dim * w_dim:
        raise SystemExit(f"latent length {z.size(1)} does not match grid {grid}")

    baseline = decode(bundle, codec, sample["prev"], z, settings)
    norm = bundle.decoder.adapter.norm
    neutral = norm.mean.to(z.device).view(1, 1, -1) if norm.mode == "channel" else z.mean(dim=1, keepdim=True)

    block_h = max(1, h_dim // block_grid)
    block_w = max(1, w_dim // block_grid)
    ablated_images, difference_maps, labels = [], [], []
    view = z.view(1, t_dim, h_dim, w_dim, -1)

    for by in range(0, h_dim, block_h):
        for bx in range(0, w_dim, block_w):
            perturbed = view.clone()
            perturbed[:, :, by : by + block_h, bx : bx + block_w, :] = neutral.view(1, 1, 1, 1, -1)
            decoded = decode(bundle, codec, sample["prev"], perturbed.reshape(1, z.size(1), -1), settings)
            difference = (decoded - baseline).abs().mean(dim=1)[0]
            ablated_images.append(to_numpy_image(decoded[0]))
            difference_maps.append((difference / difference.max().clamp_min(1e-8)).cpu().numpy())
            labels.append(f"({by}:{by + block_h}, {bx}:{bx + block_w})")

    # Centre of mass of each difference map, in normalized image coordinates, vs
    # the centre of the ablated token block: the number that says whether tokens
    # still govern the region they sit in.
    displacements = []
    for maps, (by, bx) in zip(
        difference_maps,
        [(by, bx) for by in range(0, h_dim, block_h) for bx in range(0, w_dim, block_w)],
    ):
        weight = torch.from_numpy(maps)
        total = weight.sum().clamp_min(1e-8)
        ys = torch.arange(weight.size(0), dtype=torch.float32).view(-1, 1)
        xs = torch.arange(weight.size(1), dtype=torch.float32).view(1, -1)
        cy = (weight * ys).sum() / total / weight.size(0)
        cx = (weight * xs).sum() / total / weight.size(1)
        expect_y = (by + block_h / 2) / h_dim
        expect_x = (bx + block_w / 2) / w_dim
        displacements.append(float(((cy - expect_y) ** 2 + (cx - expect_x) ** 2) ** 0.5))

    rows = [
        [to_numpy_image(sample["next"][0]), to_numpy_image(baseline[0])] + ablated_images,
        [np.zeros_like(difference_maps[0]), np.zeros_like(difference_maps[0])] + difference_maps,
    ]
    path = output_dir / f"panel5-token-ablation-{slug(bundle.name)}.png"
    save_panel(
        rows,
        ["decoded", "|change| vs baseline"],
        ["x_{t+1} (truth)", "no ablation"] + labels,
        path,
        f"Panel 5 - which pixels does which token block govern: {bundle.name}",
        caption=(
            f"Each column past the second replaces one (h, w) block of latent tokens with the fitted channel "
            f"mean and re-decodes from the same seed. Mean displacement between a block's position and its "
            f"difference-map centre of mass: {np.mean(displacements):.3f} (normalized image units; lower is "
            f"more spatially faithful). {settings.describe()}."
        ),
    )
    logger.info(
        "panel 5 (%s): %d block(s) of %dx%d tokens, mean spatial displacement %.3f",
        bundle.name,
        len(labels),
        block_h,
        block_w,
        float(np.mean(displacements)),
    )
    return {"path": path, "spatial_displacement": float(np.mean(displacements)), "num_blocks": len(labels)}


# --------------------------------------------------------------------------- #
# Panel 6: cross-model comparison
# --------------------------------------------------------------------------- #


@torch.no_grad()
def panel_crossmodel(
    bundles: Sequence[DecoderBundle],
    codecs: Sequence[object],
    indices: Sequence[int],
    settings: SamplerSettings,
    output_dir: Path,
    device: str,
) -> dict:
    """One row per world model on identical inputs, plus a latent-swap control.

    Also emits `pairs_2afc.json`, the input to the Gradio forced-choice app: for
    every clip, every unordered pair of world models, with the ground-truth frame
    alongside. Written even for a single model (the pair list is then empty) so
    the app has a stable schema to read.
    """
    reference = bundles[0]
    rows: List[List[np.ndarray]] = []
    row_labels: List[str] = []
    per_model_files: Dict[str, Dict[int, str]] = {b.name: {} for b in bundles}

    frames_dir = output_dir / "crossmodel_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    truth_row, input_row = [], []
    for index in indices:
        sample = fetch(reference, index, device)
        truth_row.append(to_numpy_image(sample["next"][0]))
        input_row.append(to_numpy_image(sample["prev"][0]))
    rows.extend([input_row, truth_row])
    row_labels.extend(["x_t (input)", "x_{t+1} (truth)"])

    swap_rows: List[List[np.ndarray]] = []
    follows_latent: Dict[str, float] = {}
    for bundle, codec in zip(bundles, codecs):
        row, swap_row, follows = [], [], []
        for position, index in enumerate(indices):
            sample = fetch(bundle, index, device)
            decoded = decode(bundle, codec, sample["prev"], sample["z_target"], settings)
            row.append(to_numpy_image(decoded[0]))
            plt.imsave(frames_dir / f"{slug(bundle.name)}-{index}.png", to_numpy_image(decoded[0]))
            per_model_files[bundle.name][index] = f"crossmodel_frames/{slug(bundle.name)}-{index}.png"

            # Latent-swap control: this column's position, a DIFFERENT clip's
            # latent. Scored rather than merely displayed, because the visual
            # check only works when the decoder also has a frame to leak from:
            # with `frame_conditioning='none'` there is no x_t, so "does it still
            # look like this column" is vacuous. Asking instead whether the output
            # moved TOWARDS the donor clip is a control that works in both modes.
            donor = fetch(bundle, indices[(position + 1) % len(indices)], device)
            swapped = decode(bundle, codec, sample["prev"], donor["z_target"], settings)
            to_own = (swapped - sample["next"]).abs().mean().item()
            to_donor = (swapped - donor["next"]).abs().mean().item()
            follows.append(to_donor < to_own)
            swap_row.append(to_numpy_image(swapped[0]))
        rows.append(row)
        row_labels.append(bundle.name)
        swap_rows.append(swap_row)
        follows_latent[bundle.name] = float(np.mean(follows)) if follows else float("nan")
        logger.info(
            "panel 6 (%s): swapped output closer to the donor clip than to its own column in %.0f%% of "
            "cases (chance is 50%%; low values mean the decoder is not following its latent)",
            bundle.name,
            100.0 * follows_latent[bundle.name],
        )

    for bundle, swap_row in zip(bundles, swap_rows):
        rows.append(swap_row)
        row_labels.append(
            f"{bundle.name}\n[latent swapped]\nfollows {100.0 * follows_latent[bundle.name]:.0f}%"
        )

    for index in indices:
        sample = fetch(reference, index, device)
        plt.imsave(frames_dir / f"truth-{index}.png", to_numpy_image(sample["next"][0]))
        plt.imsave(frames_dir / f"input-{index}.png", to_numpy_image(sample["prev"][0]))

    path = output_dir / "panel6-crossmodel.png"
    save_panel(
        rows,
        row_labels,
        [f"clip {fetch(reference, i, device)['meta'].get('clip_id')}" for i in indices],
        path,
        "Panel 6 - cross-model comparison on identical inputs",
        caption=(
            f"{settings.describe()}, identical for every row. The `[latent swapped]` rows decode against a "
            "DIFFERENT clip's latent; `follows` is the fraction of swapped samples closer to the donor "
            "clip's true frame than to their own column's. Chance is 50%; well above it means the decoder "
            "tracks its latent, at or below it means the rows above say nothing about latent quality."
        ),
    )

    pairs = []
    names = [b.name for b in bundles]
    for index in indices:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append(
                    {
                        "index": int(index),
                        "truth": f"crossmodel_frames/truth-{index}.png",
                        "input": f"crossmodel_frames/input-{index}.png",
                        "left": {"model": names[i], "image": per_model_files[names[i]][index]},
                        "right": {"model": names[j], "image": per_model_files[names[j]][index]},
                    }
                )
    with open(output_dir / "pairs_2afc.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"sampler": settings.__dict__, "models": names, "pairs": pairs},
            handle,
            indent=2,
        )
    logger.info("panel 6: %d model(s) x %d clip(s); %d 2AFC pair(s) written", len(bundles), len(indices), len(pairs))
    return {
        "path": path,
        "num_pairs": len(pairs),
        "models": names,
        **{f"follows_latent[{name}]": value for name, value in follows_latent.items()},
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-").replace("--", "-")


def write_report(
    output_dir: Path,
    audit_lines: Sequence[str],
    settings: SamplerSettings,
    results: Dict[str, Dict[str, dict]],
    bundles: Sequence[DecoderBundle],
) -> Path:
    lines = [
        "# Flow-Matching Decoder - Qualitative Panels",
        "",
        "A decoder `D(x_t, z_{t+1}) -> x_hat_{t+1}` trained per frozen world model, used here as a "
        "microscope on those world models' latent spaces. **This report is qualitative.** The few numbers "
        "in it are scale references for reading the pictures, not a leaderboard: sample sharpness and "
        "controlled comparability are what it is for.",
        "",
        "## Controlled-comparison audit",
        "",
        "Every decoder below was checked against the first one. `model_config`, `flow_config`, the fixed "
        "step budget, the codec and the latent source must match exactly; only `d_m` and the token grid may "
        "differ. A mismatch aborts this report.",
        "",
    ]
    # Some audit lines are already list items (the reference run's sub-bullets), so
    # only prefix the ones that are not, or the report grows "- -" artifacts.
    lines += [line if line.lstrip().startswith("-") else f"- {line}" for line in audit_lines]
    lines += ["", f"Sampler, identical for every panel: {settings.describe()}.", ""]

    lines += [
        "## Runs",
        "",
        "| world model | d_m | grid | latents | frame cond. | weights | codec ceiling |",
        "|---|---|---|---|---|---|---|",
    ]
    for bundle in bundles:
        lines.append(
            f"| {bundle.name} | {bundle.latent_dim} | {tuple(bundle.decoder.latent_grid)} | "
            f"{bundle.provenance.get('latent_source')} | {bundle.decoder.frame_conditioning} | "
            f"{bundle.provenance.get('weights')} | {bundle.codec_ceiling_db:.2f} dB |"
        )
    lines.append("")

    latent_only = not bundles[0].decoder.uses_frame_conditioning
    lines += [
        "",
        "## What the decoder was given",
        "",
    ]
    if latent_only:
        lines += [
            "`frame_conditioning: none` - **the decoder reconstructs from the latent tokens alone.** Its "
            "only input is the world model's tokens for the target temporal step; it never sees the "
            "preceding frame, in training or at sampling time. So anything visible in a reconstruction was "
            "carried by those tokens, because there was no other source for it.",
            "",
            "Those tokens come from the **teacher (target) encoder** run over a **full training-length clip "
            "window** ending at the target step, exactly as `evals/generate_world_model_report.py` builds "
            "its targets - so a token here is the same vector the world model's own loss was computed "
            "against, not an artifact of encoding a frame in isolation. Only the **last temporal step's** "
            "tokens are handed to the decoder; the earlier steps of the window exist so that the encoder "
            "sees the clip length and temporal context it was trained on.",
            "",
            "Two consequences for reading the panels. Samples are markedly blurrier than a "
            "frame-conditioned decoder's, and that is the measurement rather than a defect - a JEPA "
            "representation is trained for prediction, not reconstruction, and discards appearance detail "
            "it does not need. And the persistence/`copy` reference in panel 1 is no longer a floor the "
            "decoder could reach by echoing an input, because it has no input to echo.",
        ]
    else:
        lines += [
            "`frame_conditioning: current_frame` - the decoder sees the preceding frame `x_t` as well as "
            "the latent, so the task is `D(x_t, z) -> x_hat`. Easier, and weaker as a measurement: a "
            "sample can look plausible while barely reading the latent. **Check panel 6's `follows` figure "
            "before reading anything else.**",
        ]
    lines.append("")

    descriptions = {
        "reconstruction": "**Panel 1 - reconstruction.** Does the latent contain the next frame? Read each "
        "sample against the `codec ceiling` column, never against the truth: the ceiling is the best any "
        "decoder using this frozen VAE can do.",
        "guidance": "**Panel 2 - guidance ladder.** What does the latent add over persistence? Only `z` is "
        "dropped for the unconditional branch, so `w = 0` is the decoder's prior given `x_t` alone and the "
        "row isolates the latent's contribution.",
        "seeds": "**Panel 3 - what the latent leaves free.** Same latent, different noise seeds. Bright "
        "regions of the std map are pixels the latent did not determine. Needs no ground truth, which makes "
        "it the panel to trust when the truth is ambiguous.",
        "lead_time": "**Panel 4 - lead time.** Legibility against prediction distance. Teacher-forced by "
        "default (each column an independent one-step problem); with `--rollout-latents`, the caller's "
        "autoregressive latents, where degradation is the world model compounding its own error.",
        "token_ablation": "**Panel 5 - token ablation.** Replace a block of latent tokens with the "
        "no-information token and re-decode from the same seed. A tight difference map over the block's own "
        "region means tokens still govern where they sit.",
        "crossmodel": "**Panel 6 - cross-model.** The comparison itself, plus the latent-swap control. "
        "`follows` is the fraction of swapped samples that land closer to the donor clip's true frame than "
        "to their own column's; chance is 50%, and a decoder tracking its latent should be well above it.",
    }

    for panel in PANELS:
        if panel not in results or not results[panel]:
            continue
        lines += ["", f"## {panel}", "", descriptions[panel], ""]
        for name, result in results[panel].items():
            if not result or "path" not in result:
                continue
            relative = Path(result["path"]).name
            lines.append(f"### {name}" if name != "__all__" else "")
            numbers = ", ".join(
                f"{k} = {v:.4f}" if isinstance(v, float) else f"{k} = {v}"
                for k, v in result.items()
                if k != "path" and not isinstance(v, (list, dict))
            )
            if numbers:
                lines.append(f"{numbers}")
            lines += ["", f"![{panel} {name}]({relative})", ""]

    lines += [
        "## How to read a null result",
        "",
        "Failure modes that look like findings and are not:",
        "",
        "1. A blurry sample whose `codec ceiling` neighbour is also blurry. That is the frozen VAE, and it "
        "says nothing about the world model. Always read panel 1 against the ceiling column.",
        "2. A panel-6 `follows` figure at or below 50%. That decoder is not tracking its latent, and every "
        "other panel for that run is void."
        + (
            " With `frame_conditioning: none` there is no frame to fall back on, so this failure shows up "
            "as reconstructions that are plausible but generic - the same scene regardless of latent - "
            "rather than as copies of the input."
            if latent_only
            else " With `frame_conditioning: current_frame` this failure shows up as reconstructions that "
            "are near-copies of `x_t`."
        ),
        "3. Blur that is uniform across every world model being compared. The comparison is *relative*; a "
        "shared blur level is a property of this decoder configuration and step budget, not of any one "
        "latent space.",
        "",
        "The decoder is also not a generative product model, and no attempt is made to make it one - no GAN "
        "loss, no perceptual loss, no discriminator. Sharpness that came from an adversarial term would be "
        "indistinguishable from detail the latent actually carried, which would defeat the whole exercise.",
        "",
    ]

    path = output_dir / "flow_decoder_panels.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("markdown report saved to %s", path)
    return path


def parse_key_values(items: Sequence[str], flag: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"malformed {flag} {item!r}; expected 'World Model Name=path'")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the six flow-matching decoder diagnostic panels.")
    parser.add_argument(
        "--decoder",
        action="append",
        required=True,
        help="'World Model Name=path/to/checkpoint.pt', repeatable. One trained decoder per world model.",
    )
    parser.add_argument(
        "--eval-shards",
        action="append",
        required=True,
        help="'World Model Name=path/to/eval/shard/dir', repeatable. Must name the same world models as "
        "--decoder, and should be a held-out split.",
    )
    parser.add_argument(
        "--output-dir",
        default="preliminary_experiments/evals/vitl/flow_decoder_panels",
        help="Directory for the report, panels and 2AFC pair list.",
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=PANELS + ("all",),
        default=["all"],
        help="Which panels to generate.",
    )
    parser.add_argument("--num-clips", type=int, default=4, help="Eval samples shown per panel.")
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit eval-shard indices to use, overriding --num-clips. Every world model is shown the "
        "same indices; the shard sets must therefore have been built with the same --seed and manifest.",
    )
    parser.add_argument("--num-steps", type=int, default=50, help="ODE solver steps.")
    parser.add_argument("--guidance", type=float, default=1.5, help="Default classifier-free guidance scale.")
    parser.add_argument("--solver", default="heun", choices=("euler", "midpoint", "heun"))
    parser.add_argument("--shift", type=float, default=1.0, help="Timestep-schedule shift (1.0 = uniform knots).")
    parser.add_argument("--seed", type=int, default=0, help="Noise seed, held fixed across every comparison.")
    parser.add_argument(
        "--guidance-ladder",
        type=float,
        nargs="+",
        default=list(DEFAULT_GUIDANCE_LADDER),
        help="Guidance scales for panel 2.",
    )
    parser.add_argument("--num-seeds", type=int, default=4, help="Seeds for panel 3.")
    parser.add_argument(
        "--ablation-block-grid",
        type=int,
        default=2,
        help="Panel 5 splits the token grid into this many blocks per axis.",
    )
    parser.add_argument("--lead-time-steps", type=int, default=6, help="Maximum columns in panel 4.")
    parser.add_argument(
        "--rollout-latents",
        default=None,
        help="Optional torch file of caller-supplied autoregressive rollout latents for panel 4 "
        "(keys: latents (K, S, d_m), frame_prev (3, H, W) uint8, optional truth (K, 3, H, W) uint8). "
        "This harness never rolls a world model forward itself.",
    )
    parser.add_argument(
        "--raw-weights",
        dest="use_ema",
        action="store_false",
        default=True,
        help="Sample from the raw weights instead of the EMA copy. A diagnostic; EMA is the default and is "
        "what a comparison should use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "flow_decoder_panels.log")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("using device %s", device)

    decoders = parse_key_values(args.decoder, "--decoder")
    shard_roots = parse_key_values(args.eval_shards, "--eval-shards")
    if set(decoders) != set(shard_roots):
        raise SystemExit(
            "--decoder and --eval-shards name different world models: "
            f"{sorted(decoders)} vs {sorted(shard_roots)}"
        )

    # Audit BEFORE building anything: a checkpoint that fails the audit is exactly
    # the one that must not be handed to FlowMatchingDecoder.
    provenances = [read_provenance(name, Path(decoders[name])) for name in decoders]
    audit_lines = audit_provenance(provenances)
    for line in audit_lines:
        logger.info("audit: %s", line)

    bundles: List[DecoderBundle] = []
    codecs: List[object] = []
    for provenance in provenances:
        bundle, codec = load_decoder(
            provenance, Path(shard_roots[provenance["name"]]), device, args.use_ema
        )
        bundles.append(bundle)
        codecs.append(codec)

    smallest = min(len(b.shard_set) for b in bundles)
    if args.sample_indices is not None:
        indices = [i for i in args.sample_indices if i < smallest]
        if len(indices) != len(args.sample_indices):
            logger.warning("dropped %d out-of-range index/indices", len(args.sample_indices) - len(indices))
    else:
        # Evenly spaced rather than the first N, so a panel is not four frames of
        # the same clip when a shard set has many steps per clip.
        indices = list(np.linspace(0, smallest - 1, min(args.num_clips, smallest)).astype(int))
    if not indices:
        raise SystemExit("no usable eval sample indices")
    logger.info("using eval indices %s (smallest shard set holds %d sample(s))", indices, smallest)

    settings = SamplerSettings(
        num_steps=args.num_steps,
        guidance=args.guidance,
        solver=args.solver,
        shift=args.shift,
        seed=args.seed,
    )
    logger.info("sampler: %s", settings.describe())

    for bundle, codec in zip(bundles, codecs):
        sample = fetch(bundle, indices[0], device)
        bundle.codec_ceiling_db = roundtrip_psnr(codec, sample["next"])

    wanted = set(PANELS) if "all" in args.panels else set(args.panels)
    results: Dict[str, Dict[str, dict]] = {panel: {} for panel in PANELS}

    for bundle, codec in zip(bundles, codecs):
        if "reconstruction" in wanted:
            results["reconstruction"][bundle.name] = panel_reconstruction(
                bundle, codec, indices, settings, output_dir, device
            )
        if "guidance" in wanted:
            results["guidance"][bundle.name] = panel_guidance(
                bundle, codec, indices, settings, list(args.guidance_ladder), output_dir, device
            )
        if "seeds" in wanted:
            results["seeds"][bundle.name] = panel_seeds(
                bundle, codec, indices, settings, args.num_seeds, output_dir, device
            )
        if "lead_time" in wanted:
            results["lead_time"][bundle.name] = panel_lead_time(
                bundle,
                codec,
                settings,
                output_dir,
                device,
                Path(args.rollout_latents) if args.rollout_latents else None,
                args.lead_time_steps,
            )
        if "token_ablation" in wanted:
            results["token_ablation"][bundle.name] = panel_token_ablation(
                bundle, codec, indices[0], settings, args.ablation_block_grid, output_dir, device
            )

    if "crossmodel" in wanted:
        results["crossmodel"]["__all__"] = panel_crossmodel(
            bundles, codecs, indices, settings, output_dir, device
        )

    write_report(output_dir, audit_lines, settings, results, bundles)


if __name__ == "__main__":
    main()
