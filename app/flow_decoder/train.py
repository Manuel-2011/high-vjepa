#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train the flow-matching decoder on a cached latent shard set.

Fixed step budget, no early stopping. This is a spec requirement, not a
default, and it is worth being explicit about why: the decoder exists to compare
frozen latent spaces, so two decoders must differ only in the latents they were
given. Stopping one run at the step where *its* validation loss bottomed out
would make training length a function of the latent space, and every visual
difference in the panels would then be confounded by "one model trained longer".
There is consequently no code path in this file that can end a run early, and the
validation loss is computed and logged for monitoring only - nothing reads it.

Everything that could constitute per-world-model tuning is therefore forbidden
here. The only things a run is allowed to learn from its shard set are the three
the spec permits: `d_m` (the adapter's input width), the token grid (which only
feeds continuous coordinates), and the fitted normalization buffers. All three
come out of the shard manifest automatically, so the *same config file* trains
against any world model - which the config files in `configs/train/flow_decoder/`
rely on.

An EMA copy of the weights is kept and is what gets sampled from. Rectified-flow
velocity fields are noisy late in training and the EMA weights are visibly
sharper at no cost; the decay is a global constant, applied identically to every
run.

Example
-------
    python -m app.flow_decoder.train --config configs/train/flow_decoder/configA-perceiver.yaml
    python -m app.flow_decoder.train --config configs/train/flow_decoder/configB-direct.yaml \
        --override data.train_shards=data/flow_decoder_shards/other-model/train
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.flow_decoder.shard_dataset import (
    ShardStream,
    collate,
    frames_to_unit,
    load_shard_set,
)
from src.models.flow_decoder.decoder import FlowMatchingDecoder
from src.models.flow_decoder.flow import FlowConfig, RectifiedFlow, sample_ode
from src.models.flow_decoder.latent_adapter import build_token_coords
from src.models.flow_decoder.vae import build_codec, roundtrip_psnr
from src.utils.schedulers import WarmupCosineSchedule

logger = logging.getLogger(__name__)


def configure_logging(log_path: Path) -> None:
    logger.setLevel(logging.INFO)
    # The world-model modules imported above configure the root logger, so without
    # this every line would be emitted twice.
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


class EMA:
    """Exponential moving average of the decoder's parameters.

    Kept on the training device in fp32. The warmup form `min(decay, (1+s)/(10+s))`
    stops the average from being dominated by the random initialization for the
    first few hundred steps, which otherwise wastes a visible chunk of a fixed
    budget.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Float tensors are averaged in fp32; integer/bool entries (the fitted-flag
        # buffer) are carried verbatim so a dtype does not silently change.
        self.shadow = {
            k: (v.detach().clone().float() if v.dtype.is_floating_point else v.detach().clone())
            for k, v in model.state_dict().items()
        }
        self.steps = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(d).add_(value.detach().float(), alpha=1.0 - d)
            else:
                shadow.copy_(value)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "steps": self.steps, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.steps = state["steps"]
        self.shadow = {
            k: (v.float() if v.dtype.is_floating_point else v) for k, v in state["shadow"].items()
        }

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict({k: v for k, v in self.shadow.items()}, strict=True)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def apply_overrides(config: dict, overrides: List[str]) -> dict:
    """`a.b=value` overrides, parsed as YAML scalars.

    Exists so a sweep over *shard sets* (i.e. over world models) can reuse one
    config file verbatim. Deliberately not a general escape hatch for tuning: the
    panel harness cross-checks the saved model config of every decoder it
    compares and refuses any difference outside `latent_dim` / `latent_grid`.
    """
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"malformed override {item!r}; expected key.path=value")
        key, raw = item.split("=", 1)
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise SystemExit(f"override {key!r} does not name a config section")
            node = node[part]
        if parts[-1] not in node:
            raise SystemExit(f"override {key!r} does not name an existing config key")
        node[parts[-1]] = yaml.safe_load(raw)
    return config


def build_decoder(config: dict, info, codec, device: str) -> FlowMatchingDecoder:
    """Build the decoder from the config plus the three permitted shard-set facts."""
    model_cfg = dict(config["model"])
    latent_source = config["data"].get("latent_source", "target_encoder")

    decoder = FlowMatchingDecoder(
        latent_dim=info.latent_dim,  # permitted: d_m
        latent_grid=info.latent_grid,  # permitted: grid shape
        codec_channels=codec.latent_channels,
        codec_size=codec.latent_size(info.crop_size),
        patch_size=int(model_cfg.get("patch_size", 2)),
        dim=int(model_cfg.get("dim", 768)),
        depth=int(model_cfg.get("depth", 12)),
        num_heads=int(model_cfg.get("num_heads", 12)),
        cond_dim=int(model_cfg.get("cond_dim", 768)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        conditioning=str(model_cfg.get("conditioning", "perceiver")),
        num_queries=int(model_cfg.get("num_queries", 128)),
        resampler_depth=int(model_cfg.get("resampler_depth", 2)),
        resampler_heads=int(model_cfg.get("resampler_heads", 8)),
        latent_norm_mode=str(model_cfg.get("latent_norm_mode", "channel")),
        num_coord_bands=int(model_cfg.get("num_coord_bands", 8)),
        use_activation_checkpointing=bool(model_cfg.get("use_activation_checkpointing", False)),
    ).to(device)

    stats = info.statistics(latent_source)
    if stats is None:
        if decoder.adapter.norm.mode == "channel":
            raise SystemExit(
                f"shard set {info.root} has no fitted statistics for latent source {latent_source!r}, "
                "but latent_norm_mode is 'channel'. Re-run latent_cache (it fits them), or set "
                "model.latent_norm_mode: token."
            )
        logger.warning("no fitted statistics found; normalization mode %r needs none", decoder.adapter.norm.mode)
    else:
        mean, std = stats
        decoder.adapter.norm.fit(mean.to(device), std.to(device))  # permitted: normalization buffers
        logger.info(
            "installed fitted %s statistics: |mean| mean %.4f, std mean %.4f",
            latent_source,
            mean.abs().mean().item(),
            std.mean().item(),
        )
    logger.info(decoder.describe())
    return decoder


@torch.no_grad()
def evaluate(
    flow: RectifiedFlow,
    codec,
    loader_iter,
    coords: torch.Tensor,
    device: str,
    num_batches: int,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    """Held-out velocity MSE. Monitoring ONLY - nothing branches on this value.

    Uses a fixed generator seed so the reported number is comparable step to step
    (the objective is stochastic in `tau`, `eps`, dropout and `sigma`; without a
    fixed seed the eval curve is mostly sampling noise).
    """
    flow.decoder.eval()
    totals: Dict[str, float] = {}
    counted = 0
    for _ in range(num_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        generator = torch.Generator(device=device)
        generator.manual_seed(1234 + counted)
        target, cond, z = prepare_batch(batch, codec, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            _, metrics = flow.loss(target, cond, z, coords=coords, generator=generator)
        for key, value in metrics.items():
            if value == value:  # skip NaN (an all-conditional or all-dropped batch)
                totals[key] = totals.get(key, 0.0) + value
        counted += 1
    flow.decoder.train()
    return {k: v / max(1, counted) for k, v in totals.items()}


def prepare_batch(batch: dict, codec, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """uint8 frames -> frozen codec latents, on the GPU. Returns (target, cond, z)."""
    prev = frames_to_unit(batch["frame_prev"].to(device, non_blocking=True))
    nxt = frames_to_unit(batch["frame_next"].to(device, non_blocking=True))
    target = codec.encode(nxt).float()
    cond = codec.encode(prev).float()
    return target, cond, batch["z"].to(device, non_blocking=True).float()


def save_checkpoint(
    path: Path,
    step: int,
    decoder: FlowMatchingDecoder,
    ema: EMA,
    optimizer: Optional[torch.optim.Optimizer],
    config: dict,
    info,
    codec,
    metrics: dict,
) -> None:
    """Checkpoint carrying enough provenance for the panel harness to audit it.

    `model_config`, `flow_config` and `total_steps` are saved so a comparison can
    *verify* that two decoders differ only in the permitted ways, rather than
    trusting that whoever launched the runs used the same config.
    """
    torch.save(
        {
            "step": step,
            "total_steps": int(config["optimization"]["steps"]),
            "decoder": decoder.state_dict(),
            "ema": ema.state_dict(),
            # Omitted from the final checkpoint: it is ~2x the model size and only
            # a resume needs it, which `latest.pt` covers.
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "config": config,
            "model_config": config["model"],
            "flow_config": config.get("flow", {}),
            "latent_dim": info.latent_dim,
            "latent_grid": list(info.latent_grid),
            "crop_size": info.crop_size,
            "codec": {
                "kind": config.get("codec", {}).get("kind", "kl"),
                "model_id": getattr(codec, "model_id", "unknown"),
                "latent_channels": codec.latent_channels,
                "downsample_factor": codec.downsample_factor,
            },
            "world_model": {"name": info.model_name, "shard_root": str(info.root)},
            "latent_source": config["data"].get("latent_source", "target_encoder"),
            "metrics": metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the flow-matching decoder on cached latents.")
    parser.add_argument("--config", required=True, help="Path to a configs/train/flow_decoder/*.yaml.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="key.path=value override, repeatable. Intended for pointing one config at a different "
        "shard set, not for tuning.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest.pt in the run folder.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override optimization.steps. Use only to shorten a smoke run; every run in a comparison "
        "must share one budget.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_yaml(args.config), args.override)
    if args.steps is not None:
        config["optimization"]["steps"] = int(args.steps)

    folder = Path(config["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    configure_logging(folder / "train.log")
    with open(folder / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(config.get("meta", {}).get("seed", 0)))
    dtype_name = str(config.get("meta", {}).get("dtype", "bfloat16")).lower()
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
    if not device.startswith("cuda"):
        amp_dtype = None
    logger.info("device %s, autocast %s", device, amp_dtype or "disabled")

    data_cfg = config["data"]
    train_info = load_shard_set(data_cfg["train_shards"])
    logger.info("train shards: %s", train_info.describe())
    eval_info = load_shard_set(data_cfg["eval_shards"]) if data_cfg.get("eval_shards") else None
    if eval_info is not None:
        logger.info("eval shards:  %s", eval_info.describe())
        if eval_info.latent_dim != train_info.latent_dim or eval_info.latent_grid != train_info.latent_grid:
            raise SystemExit(
                "train and eval shard sets describe different latent spaces "
                f"({train_info.latent_dim}/{train_info.latent_grid} vs "
                f"{eval_info.latent_dim}/{eval_info.latent_grid}); they were built from different world models."
            )

    codec_cfg = config.get("codec", {})
    codec = build_codec(
        kind=codec_cfg.get("kind", "kl"),
        model_id=codec_cfg.get("model_id", "stabilityai/sd-vae-ft-mse"),
        downsample_factor=int(codec_cfg.get("downsample_factor", 8)),
        device=device,
    )

    decoder = build_decoder(config, train_info, codec, device)
    flow = RectifiedFlow(decoder, FlowConfig(**config.get("flow", {})))
    coords = build_token_coords(train_info.latent_grid, device=torch.device(device))

    opt_cfg = config["optimization"]
    total_steps = int(opt_cfg["steps"])
    accum = int(opt_cfg.get("grad_accum", 1))
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(opt_cfg.get("lr", 1e-4)),
        betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
    )
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(opt_cfg.get("warmup", 1000)),
        start_lr=float(opt_cfg.get("start_lr", 1e-6)),
        ref_lr=float(opt_cfg.get("lr", 1e-4)),
        T_max=total_steps,
        final_lr=float(opt_cfg.get("final_lr", 1e-5)),
    )
    ema = EMA(decoder, decay=float(opt_cfg.get("ema_decay", 0.9999)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))
    latent_source = data_cfg.get("latent_source", "target_encoder")
    train_loader = DataLoader(
        ShardStream(train_info, latent_source=latent_source, seed=int(config.get("meta", {}).get("seed", 0))),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    eval_iter = None
    if eval_info is not None:
        eval_loader = DataLoader(
            ShardStream(eval_info, latent_source=latent_source, seed=7),
            batch_size=batch_size,
            num_workers=min(2, num_workers),
            collate_fn=collate,
            pin_memory=device.startswith("cuda"),
            persistent_workers=min(2, num_workers) > 0,
            drop_last=True,
        )
        eval_iter = iter(eval_loader)

    start_step = 0
    latest = folder / "latest.pt"
    if args.resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        decoder.load_state_dict(state["decoder"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        for _ in range(start_step):
            scheduler.step()
        logger.info("resumed from %s at step %d", latest, start_step)

    log_freq = int(config.get("meta", {}).get("log_freq", 100))
    eval_freq = int(config.get("meta", {}).get("eval_freq", 2000))
    eval_batches = int(config.get("meta", {}).get("eval_batches", 16))
    save_freq = int(config.get("meta", {}).get("save_freq", 2000))
    sample_freq = int(config.get("meta", {}).get("sample_freq", 5000))

    logger.info(
        "FIXED BUDGET: %d step(s) x batch %d x accum %d. There is no early stopping; the eval loss "
        "below is monitoring only and no decision reads it.",
        total_steps,
        batch_size,
        accum,
    )

    train_iter = iter(train_loader)
    decoder.train()
    running: Dict[str, float] = {}
    window = 0
    step_started = time.time()
    history: List[dict] = []

    for step in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = next(train_iter)
            target, cond, z = prepare_batch(batch, codec, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, metrics = flow.loss(target, cond, z, coords=coords)
            scaler.scale(loss / accum).backward()
            for key, value in metrics.items():
                if value == value:
                    running[key] = running.get(key, 0.0) + value
            window += 1

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), float(opt_cfg.get("clip_grad", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()
        ema.update(decoder)

        if (step + 1) % log_freq == 0:
            elapsed = time.time() - step_started
            averaged = {k: v / max(1, window) for k, v in running.items()}
            logger.info(
                "step %d/%d  loss %.4f (cond %.4f, uncond %.4f)  lr %.2e  grad %.3f  %.2f step/s",
                step + 1,
                total_steps,
                averaged.get("loss", float("nan")),
                averaged.get("loss_cond", float("nan")),
                averaged.get("loss_uncond", float("nan")),
                lr,
                float(grad_norm),
                log_freq / max(elapsed, 1e-6),
            )
            history.append({"step": step + 1, "lr": lr, "grad_norm": float(grad_norm), **averaged})
            running, window = {}, 0
            step_started = time.time()

        if eval_iter is not None and (step + 1) % eval_freq == 0:
            eval_metrics = evaluate(flow, codec, eval_iter, coords, device, eval_batches, amp_dtype)
            logger.info(
                "step %d/%d  [monitoring only] eval loss %.4f (cond %.4f)",
                step + 1,
                total_steps,
                eval_metrics.get("loss", float("nan")),
                eval_metrics.get("loss_cond", float("nan")),
            )
            history.append({"step": step + 1, **{f"eval_{k}": v for k, v in eval_metrics.items()}})

        if (step + 1) % save_freq == 0 or (step + 1) == total_steps:
            save_checkpoint(
                latest, step + 1, decoder, ema, optimizer, config, train_info, codec, history[-1] if history else {}
            )
            with open(folder / "history.json", "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)

        if sample_freq and (step + 1) % sample_freq == 0:
            dump_samples(decoder, ema, codec, eval_info or train_info, coords, folder, step + 1, device, config)

    save_checkpoint(
        folder / "final.pt", total_steps, decoder, ema, None, config, train_info, codec, history[-1] if history else {}
    )
    logger.info("done: %d step(s) completed, final checkpoint at %s", total_steps, folder / "final.pt")


@torch.no_grad()
def dump_samples(
    decoder: FlowMatchingDecoder,
    ema: EMA,
    codec,
    info,
    coords: torch.Tensor,
    folder: Path,
    step: int,
    device: str,
    config: dict,
) -> None:
    """Write a small qualitative strip during training, from the EMA weights.

    A progress check, not a panel: the real comparison harness is
    `evals/generate_flow_decoder_panels.py`. Kept cheap (4 samples, few solver
    steps) so it never becomes a reason to shorten a run.
    """
    from app.flow_decoder.shard_dataset import ShardSet

    try:
        shard_set = ShardSet(info, latent_source=config["data"].get("latent_source", "target_encoder"))
    except Exception as exc:  # pragma: no cover - diagnostics must not kill a run
        logger.warning("sample dump skipped: %s", exc)
        return

    indices = [min(i, len(shard_set) - 1) for i in (0, 1, 2, 3)]
    items = [shard_set[i] for i in indices]
    prev = frames_to_unit(torch.stack([it["frame_prev"] for it in items]).to(device))
    nxt = frames_to_unit(torch.stack([it["frame_next"] for it in items]).to(device))
    z = torch.stack([it["z"] for it in items]).float().to(device)

    shadow = copy.deepcopy(decoder.state_dict())
    ema.copy_to(decoder)
    decoder.eval()
    try:
        latent = sample_ode(
            decoder,
            cond_latent=codec.encode(prev).float(),
            z=z,
            coords=coords,
            num_steps=int(config.get("meta", {}).get("sample_steps", 24)),
            guidance=float(config.get("meta", {}).get("sample_guidance", 1.5)),
            solver="heun",
            seed=0,
        )
        recon = codec.decode(latent)
    finally:
        decoder.load_state_dict(shadow)
        decoder.train()

    grid = torch.cat([prev.cpu(), recon.float().cpu(), nxt.cpu()], dim=0)
    out = folder / "samples"
    out.mkdir(exist_ok=True)
    try:
        from torchvision.utils import save_image

        save_image(grid, out / f"step-{step:07d}.png", nrow=len(items))
        psnr = roundtrip_psnr(codec, nxt)
        logger.info("sample strip written to %s (codec ceiling %.2f dB)", out / f"step-{step:07d}.png", psnr)
    except Exception as exc:  # pragma: no cover
        logger.warning("could not write sample strip: %s", exc)


if __name__ == "__main__":
    main()
