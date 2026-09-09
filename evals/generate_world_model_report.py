#!/usr/bin/env python3
"""
Autoregressive World-Model Report Generator

Measures how far into the future each model in MODELS_CONFIG can keep predicting
its own representation of a video before the prediction stops carrying
information about what actually happens.

Protocol, per model and per long video clip:

- Load the *three* modules that make up a V-JEPA world model from the training
  checkpoint - the context `encoder`, the `predictor`, and the EMA
  `target_encoder` - built by exactly the same `init_video_model` call the
  training loop uses, so every architectural switch (`is_causal`, `use_rope`,
  `tubelet_size`, ...) comes from the model's own pretraining config.
- Sample one long clip per video at the model's own `fps`, long enough to cover
  the context window plus `--horizon-seconds` (~1 minute) of future.
- Roll the predictor forward one *temporal token* at a time - one tubelet, i.e.
  `tubelet_size` frames - never more, so each step is the same one-step-ahead
  problem the model was trained on. The prediction of step k is fed back in as
  context for step k+1, and the oldest temporal token is dropped, so the
  predictor always sees a window of the size it was trained on.
    * Non-causal models (`is_causal: false`) get an explicit mask: the context
      mask covers every patch of every temporal token but the last, and the
      prediction mask covers exactly the patches of the last temporal token.
    * Causal models (`is_causal: true`) need no mask - like an LLM they emit a
      next-token prediction at every position, and the last `S` outputs are the
      prediction of the next frame.
- Score each predicted frame against the ground-truth frame embedding produced
  by the target encoder (layer-normalized, exactly as in the training loss).

Because distances in an embedding space are only meaningful relative to the
scale of that space - a collapsed representation makes *every* distance small -
every distance is reported normalized by the *persistence baseline*: the
distance between the last frame the model actually observed and the same
ground-truth frame. A normalized value of 1.0 means the rollout is no more
informative than freezing the last observed frame; below 1.0 means the model
genuinely anticipates change; above 1.0 means it is actively worse than doing
nothing. Two explicit collapse diagnostics (spatial spread within a predicted
frame, and spread across different clips at the same horizon) are reported
alongside, both also as ratios against the ground truth.

The Markdown report gives the horizon-averaged metric per model *and* the metric
per predicted frame, so the degradation curve over prediction time is visible.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from decord import cpu, VideoReader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vjepa.utils import init_video_model, load_module_state_dict
from app.world_model.utils import (
    ChunkRolloutPredictor,
    ChunkWindowEncoder,
    init_frozen_backbone,
    init_world_model,
)
from evals.generate_patch_embedding_report import (
    DEFAULT_NORMALIZATION,
    MODELS_CONFIG,
    apply_pretrained_frame_skip,
    load_config,
    load_manifest,
    sample_rows,
    slugify,
)
from evals.video_classification_frozen.utils import make_transforms
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

logger = logging.getLogger(__name__)

# World models whose autoregressive unit is not a V-JEPA tubelet, and which therefore
# need their own geometry and loader. `kind` selects both; entries in the shared
# MODELS_CONFIG carry no `kind` and are read as plain V-JEPA world models.
#
# The goal-guided model is rolled out with its **goal withheld** (the learned null goal,
# i.e. the unconditional path `world_model.goal_drop_prob` trains). Told where it is
# going it would be solving a strictly easier problem than every other model here, and
# the comparison would say nothing. What this measures is whether learning *with* a goal
# produced a better world model, not whether a goal helps at inference.
# `goal_lead_seconds` turns the goal back on, held that many seconds ahead of the
# context window at *every* step. That reads real video past the rollout's frontier, so
# such an entry is PRIVILEGED - not comparable with the others as a measurement, only as
# a ceiling. Its value against the withheld entry above is the paired comparison: same
# weights, same clips, same steps, goal off vs goal on.
EXTRA_MODELS_CONFIG = [
    {
        "name": "Goal-guided world model (goal withheld)",
        "checkpoint": "preliminary_experiments/EK100-world-model-goal-256px/latest.pt",
        "config": "configs/train/vitl16-EK100/world-model-goal-256px.yaml",
        "kind": "goal_world_model",
    },
    {
        "name": "Goal-guided world model (goal 16s ahead)",
        "checkpoint": "preliminary_experiments/EK100-world-model-goal-256px/latest.pt",
        "config": "configs/train/vitl16-EK100/world-model-goal-256px.yaml",
        "kind": "goal_world_model",
        "goal_lead_seconds": 16.0,
    },
]

ALL_MODELS_CONFIG = list(MODELS_CONFIG) + EXTRA_MODELS_CONFIG

# Distances between a predicted frame embedding and its ground truth. Each is
# reported three ways: `_pred` (prediction vs ground truth), `_ref` (the
# persistence baseline - last observed frame vs ground truth) and `_norm`
# (`_pred / _ref`, the headline number).
#   l1      - mean absolute error over patch tokens and channels. This is the
#             quantity the pretraining loss minimizes (loss_exp: 1.0), so it is
#             the most faithful read-out of "did the predictor do its job".
#   l2      - per-token Euclidean distance, averaged over tokens.
#   cosine  - per-token (1 - cosine similarity), averaged over tokens. The only
#             one of the three that is invariant to the overall scale of the
#             embedding space even before normalization.
METRICS = ("l1", "l2", "cosine")

METRIC_LABELS = {
    "l1": "L1 (training loss)",
    "l2": "L2 per token",
    "cosine": "Cosine distance",
}

# How a prediction is mapped back into the space the predictor expects as input.
#
# The predictor is trained to *output* layer-normalized target-encoder features
# but to *consume* raw context-encoder features, so an autoregressive rollout has
# to bridge the two. There is no choice that is simultaneously faithful on both
# sides, so the bridge is explicit and switchable:
#   rescale     - feed the prediction back after restoring the per-token scale
#                 (mean/std across channels) of the real context tokens. Keeps
#                 the predictor's *input* distribution correct, which is what its
#                 first linear layer was fitted on. Default.
#   raw         - feed the prediction back untouched. Correct in spirit, but the
#                 fed-back tokens are ~unit-variance while real context tokens
#                 are not, so the predictor sees out-of-scale inputs.
#   layer_norm  - layer-normalize the *observed* context too, so the whole
#                 rollout lives in the predictor's output space. Consistent
#                 across steps, but every token is then out-of-scale.
#   rescale_running
#               - as `rescale`, but the mean/std are recomputed from the
#                 *current* window at every step instead of being frozen at the
#                 first real one. Uses no privileged information, and is exactly
#                 equivalent to `rescale` for as long as the window still holds
#                 real tokens. Past that point - after `context_tokens` steps,
#                 which is 7 of 120 for a 16-frame/tubelet-2 model at 4 fps -
#                 the statistics are derived entirely from previously rescaled
#                 predictions, so any systematic offset in the predictor's
#                 output scale compounds geometrically. Diagnostic: if it tracks
#                 `rescale`, the frozen anchor is not doing any work.
#   oracle_rescale
#               - ORACLE, LEAKS THE FUTURE. Restores the per-token mean/std of
#                 the *un-normalized target-encoder embedding of the frame being
#                 predicted*, which is the exact algebraic inverse of the layer
#                 norm the training loss applies. Not a legitimate rollout - a
#                 world model cannot see the frame it is predicting - but an
#                 upper bound: with a perfect predictor it reproduces the true
#                 embedding exactly, so it degenerates to teacher forcing. Run it
#                 to find out whether the feedback bridge is what limits the
#                 horizon. If it tracks `rescale`, the bridge is not the
#                 bottleneck and the report's conclusions are about the model.
FEEDBACK_MODES = ("rescale", "raw", "layer_norm", "rescale_running", "oracle_rescale")

# Modes that consume ground truth for the frame being predicted. Results from
# these are diagnostic ceilings, never measurements, and every artifact they
# produce says so.
ORACLE_FEEDBACK_MODES = frozenset({"oracle_rescale"})

# Horizons (seconds after the last observed frame) called out in the summary
# tables. Models with different `fps`/`tubelet_size` advance by different
# amounts per step, so each model reports the step nearest to each horizon.
SUMMARY_HORIZONS_S = (2.0, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0)

# A persistence baseline distance below this is treated as degenerate (the last
# observed frame is essentially identical to the ground-truth frame), because
# dividing by it would manufacture an enormous normalized value out of noise.
MIN_REFERENCE_DISTANCE = 1e-4


@dataclass
class RolloutGeometry:
    """Everything about a model's clip/token layout that the rollout needs.

    All of it is derived from the model's own pretraining config, so the rollout
    can never present the encoder with a clip shape it was not trained on.

    Frame spacing is *not* necessarily uniform. A model post-trained from a
    checkpoint that was pretrained at a higher frame rate
    (`meta.use_pretrained_model`) sees clips on two different timescales at once:
    the frames inside one tubelet stay at the backbone's original `previous_fps`,
    while consecutive tubelets are spaced at the new, coarser `fps`. The data
    pipeline builds this by sampling at `previous_fps` and keeping only the
    leading `tubelet_size` frames of every `frames_to_skip`-sized chunk (see
    `SimpleCollator` in src/masks/multiseq_multiblock3d.py). So for
    `previous_fps: 4`, `fps: 0.5`, `tubelet_size: 2` a tubelet is two frames
    0.25s apart, and tubelet starts are 2s apart - which is what an
    autoregressive step advances by.
    """

    fps: float  # config `data.fps`: the rate tubelet *starts* advance at
    sampling_fps: float  # rate frames are decoded at (previous_fps when post-training)
    tubelet_size: int  # frames the encoder groups into one temporal token
    frames_per_token: int  # sampled frames advanced per temporal token
    frames_to_skip: int  # sampled frames per chunk (1 when not post-training)
    patch_size: int
    crop_size: int
    frames_per_clip: int  # max(dataset_fpcs): the trained clip length, in frames
    spatial_tokens: int  # S: patch tokens per temporal token
    window_tokens: int  # N_t: temporal tokens in a trained clip
    context_tokens: int  # n_ctx = N_t - 1: temporal tokens the predictor conditions on
    num_steps: int  # autoregressive steps taken
    is_causal: bool
    uses_pretrained_backbone: bool
    # Temporal tokens decoded past the end of the rollout, for a privileged input that
    # reads ahead of the frontier (a goal). Zero for every ordinary rollout.
    extra_tokens: int = 0

    @property
    def step_seconds(self) -> float:
        """Video time advanced by one autoregressive step, i.e. between the start
        of one tubelet and the start of the next.

        Without post-training this is `tubelet_size / fps`; with it, the frame
        skip makes it `frames_to_skip / previous_fps`, which is `1 / fps`."""
        return self.frames_per_token / self.sampling_fps

    @property
    def intra_tubelet_seconds(self) -> float:
        """Gap between two consecutive frames *inside* one tubelet. Zero for a
        single-frame tubelet."""
        return 0.0 if self.tubelet_size < 2 else 1.0 / self.sampling_fps

    @property
    def tubelet_span_seconds(self) -> float:
        """Video time from the first to the last frame of one tubelet."""
        return (self.tubelet_size - 1) / self.sampling_fps

    @property
    def window_seconds(self) -> float:
        """Video time spanned by a full trained clip, first frame to last."""
        return (self.window_tokens - 1) * self.step_seconds + self.tubelet_span_seconds

    @property
    def context_seconds(self) -> float:
        """Video time spanned by the tokens the predictor conditions on."""
        return (self.context_tokens - 1) * self.step_seconds + self.tubelet_span_seconds

    @property
    def horizon_seconds(self) -> float:
        return self.num_steps * self.step_seconds

    @property
    def total_tokens(self) -> int:
        """Temporal tokens that must be decoded from the video.

        `window_tokens` of them are needed before the first prediction: the
        `context_tokens` the predictor sees, plus one extra leading token so the
        *last observed* frame also has a full ground-truth window behind it (see
        `encode_ground_truth`). `extra_tokens` covers a goal read ahead of the
        rollout's frontier."""
        return self.window_tokens + self.num_steps + self.extra_tokens

    @property
    def total_sampled_frames(self) -> int:
        """Frames to decode at `sampling_fps`, before any frame skip.

        Whole chunks are decoded even though the tail of the last one is
        discarded, because `apply_pretrained_frame_skip` drops a trailing partial
        chunk - shaving those frames off would cost a whole temporal token."""
        return self.total_tokens * self.frames_per_token

    @property
    def total_frames(self) -> int:
        """Frames the encoder actually sees, after the frame skip."""
        return self.total_tokens * self.tubelet_size

    def lead_seconds(self, step: int) -> float:
        """Video time between the last observed frame and the frame predicted at
        `step` (0-based)."""
        return (step + 1) * self.step_seconds


@dataclass
class GoalSchedule:
    """Where the goal chunk sits at every step of a goal-conditioned rollout.

    The goal is held a fixed `lead_seconds` ahead of the context window the step is
    conditioning on. That window slides forward one chunk per step, so the goals slide
    with it and step `k`'s goal is the `frames_per_chunk` frames at
    `first_frame + k * frames_per_chunk` - a contiguous run, which is why they can all
    be embedded in one call.

    Because both ends move together, the goal's position *relative to the window* never
    changes, which is what makes `position` a single number: `context_tokens` (the end
    of the context, in the predictor's local chunk units) plus the lead. It is exactly
    the `goal_pos` training computes for a goal sampled at this distance.
    """

    lead_seconds: float
    lead_frames: int
    first_frame: int  # first frame of step 0's goal chunk, within the decoded clip
    frames_per_chunk: int
    position: float  # goal position in the predictor's local chunk units, constant

    @property
    def extra_tokens(self) -> int:
        """Temporal tokens the clip needs beyond the rollout itself."""
        return math.ceil(self.lead_frames / self.frames_per_chunk)


@dataclass
class ModelBundle:
    encoder: torch.nn.Module
    predictor: torch.nn.Module
    target_encoder: torch.nn.Module
    embed_dim: int
    epoch: int
    # Set only for a goal-conditioned rollout; None means the goal is withheld.
    goal_schedule: Optional[GoalSchedule] = None


@dataclass
class ClipBatch:
    """A batch of equally-shaped long clips, plus where each came from."""

    frames: torch.Tensor  # (B, C, T, H, W)
    video_paths: List[str]
    start_frames: List[int]
    clip_ids: List[int]


@dataclass
class RolloutResult:
    rows: List[dict] = field(default_factory=list)
    pred_pooled: List[np.ndarray] = field(default_factory=list)  # one (num_steps, D) per clip
    gt_pooled: List[np.ndarray] = field(default_factory=list)
    clip_ids: List[int] = field(default_factory=list)
    num_clips_attempted: int = 0
    num_clips_failed: int = 0
    elapsed_s: float = 0.0


def configure_logging(log_path: str) -> None:
    """Bound to *this* module's logger: the sibling report generators configure
    their own, so reusing theirs would leave this report's log file empty."""
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


# --------------------------------------------------------------------------- #
# Geometry & model loading
# --------------------------------------------------------------------------- #


def build_geometry(config: dict, horizon_seconds: float) -> RolloutGeometry:
    args_data = config["data"]
    args_model = config["model"]

    fps = float(args_data["fps"])
    tubelet_size = int(args_data["tubelet_size"])
    patch_size = int(args_data["patch_size"])
    crop_size = int(args_data.get("crop_size", 224))
    frames_per_clip = int(max(args_data["dataset_fpcs"]))

    if frames_per_clip % tubelet_size != 0:
        raise ValueError(
            f"dataset_fpcs ({frames_per_clip}) is not a multiple of tubelet_size ({tubelet_size}); "
            "the clip cannot be split into whole temporal tokens."
        )
    if crop_size % patch_size != 0:
        raise ValueError(f"crop_size ({crop_size}) is not a multiple of patch_size ({patch_size}).")

    grid = crop_size // patch_size
    window_tokens = frames_per_clip // tubelet_size
    if window_tokens < 2:
        raise ValueError(
            f"A trained clip holds only {window_tokens} temporal token(s), so there is no past to "
            "predict a future token from."
        )

    # A model post-trained from a higher-frame-rate backbone samples at
    # `previous_fps` and keeps only the leading frames of every chunk, which makes
    # frame spacing non-uniform: dense inside a tubelet, sparse between tubelets.
    # Mirrors init_data + SimpleCollator in app/vjepa/train.py.
    use_pretrained_cfg = config.get("meta", {}).get("use_pretrained_model") or {}
    uses_pretrained_backbone = bool(use_pretrained_cfg.get("enabled", False))
    if uses_pretrained_backbone:
        previous_fps = float(use_pretrained_cfg["previous_fps"])
        frames_to_skip = int(previous_fps // fps)
        if frames_to_skip < 1:
            raise ValueError(
                f"use_pretrained_model.previous_fps ({previous_fps:g}) is below data.fps ({fps:g}); "
                "there is nothing to skip."
            )
        # train.py builds SimpleCollator with `previous_tubulet_size=tubelet_size`,
        # ignoring the config key of the same name, so the two must agree or the
        # clips this report builds would not be the clips the model was trained
        # on. Every config in the repo satisfies this; fail loudly if one stops.
        declared = int(use_pretrained_cfg.get("previous_tubulet_size", tubelet_size))
        if declared != tubelet_size:
            raise ValueError(
                f"use_pretrained_model.previous_tubulet_size ({declared}) != data.tubelet_size "
                f"({tubelet_size}). app/vjepa/train.py passes tubelet_size to SimpleCollator and "
                "ignores the config key, so training kept a different number of frames per chunk than "
                "this config declares; the intended clip layout is ambiguous."
            )
        sampling_fps = previous_fps
        frames_per_token = frames_to_skip
    else:
        frames_to_skip = 1
        sampling_fps = fps
        frames_per_token = tubelet_size

    if frames_per_token < tubelet_size:
        raise ValueError(
            f"a temporal token spans {tubelet_size} frames but only {frames_per_token} are sampled per "
            "token; the tubelet cannot be filled."
        )

    step_seconds = frames_per_token / sampling_fps
    num_steps = max(1, int(round(horizon_seconds / step_seconds)))

    return RolloutGeometry(
        fps=fps,
        sampling_fps=sampling_fps,
        tubelet_size=tubelet_size,
        frames_per_token=frames_per_token,
        frames_to_skip=frames_to_skip,
        patch_size=patch_size,
        crop_size=crop_size,
        frames_per_clip=frames_per_clip,
        spatial_tokens=grid * grid,
        window_tokens=window_tokens,
        # One temporal token is spent on the prediction slot, so the predictor
        # conditions on the remaining N_t - 1. This keeps the window - and hence
        # every RoPE position the predictor sees - identical to training.
        context_tokens=window_tokens - 1,
        num_steps=num_steps,
        is_causal=bool(args_model.get("is_causal", False)),
        uses_pretrained_backbone=uses_pretrained_backbone,
    )


def build_goal_schedule(config: dict, goal_lead_seconds: Optional[float]) -> Optional[GoalSchedule]:
    """Lay out the goal chunks for a rollout that keeps its goal `goal_lead_seconds`
    ahead of the context window. None when the goal is withheld.

    The lead is checked against the range the model was trained to expect
    (`world_model.goal_min_seconds` .. `goal_max_seconds`): a goal further out than the
    training distribution ever placed one would be measuring extrapolation, not the
    model. It does not have to be a whole number of chunks - training samples the goal
    at frame resolution and `goal_pos` is a float - so any lead in range works.
    """
    if goal_lead_seconds is None:
        return None

    cfgs_wm = config["world_model"]
    fps = float(config["data"]["fps"])
    frames_per_chunk = int(cfgs_wm.get("tokens_per_chunk", 4)) * int(
        config["vjepa"].get("tubelet_size", 2)
    )
    context_chunks = int(cfgs_wm.get("context_chunks", 8))
    goal_min = float(cfgs_wm.get("goal_min_seconds", 4.0))
    goal_max = float(cfgs_wm.get("goal_max_seconds", 16.0))
    if not goal_min - 1e-6 <= goal_lead_seconds <= goal_max + 1e-6:
        raise ValueError(
            f"goal_lead_seconds ({goal_lead_seconds:g}s) is outside the {goal_min:g}-{goal_max:g}s "
            "range this model was trained to place goals in, so the rollout would be measuring "
            "extrapolation rather than the model."
        )

    lead_frames = int(round(goal_lead_seconds * fps))
    if lead_frames < 1:
        raise ValueError(f"a goal {goal_lead_seconds:g}s ahead is less than one frame at {fps:g} fps")

    # The context of step 0 is clip tokens [1, 1 + context_chunks), so it ends at frame
    # (1 + context_chunks) * frames_per_chunk and its goal starts `lead_frames` later.
    window_tokens = context_chunks + 1
    return GoalSchedule(
        lead_seconds=float(goal_lead_seconds),
        lead_frames=lead_frames,
        first_frame=window_tokens * frames_per_chunk + lead_frames,
        frames_per_chunk=frames_per_chunk,
        position=context_chunks + lead_frames / frames_per_chunk,
    )


def build_goal_world_model_geometry(
    config: dict, horizon_seconds: float, goal_lead_seconds: Optional[float] = None
) -> RolloutGeometry:
    """`build_geometry` for the goal-guided chunk world model.

    Its autoregressive unit is a *chunk* -- `world_model.tokens_per_chunk` V-JEPA
    temporal tokens, so 4 x 2 = 8 frames = 2s at 4 fps -- not a single tubelet. That is
    the only structural difference: a chunk latent occupies one temporal slot of `P`
    patch tokens exactly as a tubelet embedding does for the other models, so the whole
    rollout, including `temporal_slice` and the persistence baseline, carries over
    unchanged with `tubelet_size` read as "frames per temporal unit".

    A step therefore advances 2s instead of 0.5s, and a fixed horizon takes four times
    fewer steps. That is a real property of the architecture, not a handicap given to it
    here: the report indexes every curve by lead time in seconds for this reason.
    """
    cfgs_data = config["data"]
    cfgs_wm = config["world_model"]
    cfgs_vjepa = config["vjepa"]

    fps = float(cfgs_data["fps"])
    crop_size = int(cfgs_data.get("crop_size", 256))
    patch_size = int(cfgs_vjepa.get("patch_size", 16))
    tubelet_size = int(cfgs_vjepa.get("tubelet_size", 2))
    tokens_per_chunk = int(cfgs_wm.get("tokens_per_chunk", 4))
    context_chunks = int(cfgs_wm.get("context_chunks", 8))

    if crop_size % patch_size != 0:
        raise ValueError(f"crop_size ({crop_size}) is not a multiple of patch_size ({patch_size}).")
    if context_chunks < 1:
        raise ValueError(f"context_chunks ({context_chunks}) leaves no past to predict from.")

    grid = crop_size // patch_size
    frames_per_chunk = tokens_per_chunk * tubelet_size
    # One chunk beyond the context is the prediction slot, matching the convention the
    # V-JEPA geometry uses (context_tokens = window_tokens - 1).
    window_tokens = context_chunks + 1
    step_seconds = frames_per_chunk / fps
    num_steps = max(1, int(round(horizon_seconds / step_seconds)))
    schedule = build_goal_schedule(config, goal_lead_seconds)

    return RolloutGeometry(
        fps=fps,
        sampling_fps=fps,
        # The temporal unit is a chunk, so "frames grouped into one temporal token" is
        # the whole chunk. Every derived duration on RolloutGeometry then describes
        # chunks, which is what the rollout advances by.
        tubelet_size=frames_per_chunk,
        frames_per_token=frames_per_chunk,
        frames_to_skip=1,
        patch_size=patch_size,
        crop_size=crop_size,
        frames_per_clip=window_tokens * frames_per_chunk,
        spatial_tokens=grid * grid,
        window_tokens=window_tokens,
        context_tokens=context_chunks,
        num_steps=num_steps,
        is_causal=True,
        uses_pretrained_backbone=False,
        # A goal held ahead of the frontier needs footage past the end of the rollout.
        extra_tokens=0 if schedule is None else schedule.extra_tokens,
    )


def build_geometry_for(model_cfg: dict, config: dict, horizon_seconds: float) -> RolloutGeometry:
    kind = model_cfg.get("kind", "vjepa")
    if kind == "goal_world_model":
        return build_goal_world_model_geometry(
            config, horizon_seconds, model_cfg.get("goal_lead_seconds")
        )
    if kind != "vjepa":
        raise ValueError(f"unknown model kind {kind!r}")
    return build_geometry(config, horizon_seconds)


def resolve_checkpoint(model_cfg: dict, config: dict) -> str:
    checkpoint = model_cfg.get("checkpoint") or os.path.join(config["folder"], "latest.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return checkpoint


def strip_ddp_prefix(state_dict: dict) -> dict:
    """Training saves DDP-wrapped modules, so every key carries a `module.`
    prefix that the bare modules built here do not have."""
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}


def prepare_world_model(config: dict, checkpoint_path: str, device: str) -> ModelBundle:
    """Rebuild encoder/predictor/target_encoder exactly as `app/vjepa/train.py`
    does, then load all three from the training checkpoint.

    Unlike the sibling reports - which only need a frozen encoder and can go
    through the classification eval's wrapper - a world-model rollout needs the
    predictor too, and it must be wired to the encoder with the same
    `MultiSeqWrapper` / `PredictorMultiSeqWrapper` plumbing the training loop
    used, otherwise the mask bookkeeping does not line up.
    """
    args_data = config["data"]
    args_model = config["model"]
    args_meta = config.get("meta", {})
    cfgs_mask = config.get("mask", []) or []

    encoder, predictor = init_video_model(
        uniform_power=args_model.get("uniform_power", False),
        use_mask_tokens=args_model.get("use_mask_tokens", False),
        num_mask_tokens=int(len(cfgs_mask) * len(args_data["dataset_fpcs"])),
        zero_init_mask_tokens=args_model.get("zero_init_mask_tokens", True),
        device=device,
        patch_size=int(args_data["patch_size"]),
        max_num_frames=int(max(args_data["dataset_fpcs"])),
        tubelet_size=int(args_data["tubelet_size"]),
        model_name=args_model["model_name"],
        crop_size=int(args_data.get("crop_size", 224)),
        pred_depth=args_model["pred_depth"],
        pred_num_heads=args_model.get("pred_num_heads"),
        pred_embed_dim=args_model["pred_embed_dim"],
        use_sdpa=args_meta.get("use_sdpa", False),
        use_silu=args_model.get("use_silu", False),
        use_pred_silu=args_model.get("use_pred_silu", False),
        wide_silu=args_model.get("wide_silu", True),
        use_rope=args_model.get("use_rope", False),
        # Activation checkpointing only trades compute for training memory; under
        # torch.no_grad() it buys nothing.
        use_activation_checkpointing=False,
        is_causal=bool(args_model.get("is_causal", False)),
    )
    target_encoder = copy.deepcopy(encoder)

    checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))
    epoch = int(checkpoint.get("epoch", 0))
    load_module_state_dict(encoder, strip_ddp_prefix(checkpoint["encoder"]), "encoder", epoch)
    load_module_state_dict(predictor, strip_ddp_prefix(checkpoint["predictor"]), "predictor", epoch)
    load_module_state_dict(
        target_encoder, strip_ddp_prefix(checkpoint["target_encoder"]), "target encoder", epoch
    )
    del checkpoint

    for module in (encoder, predictor, target_encoder):
        module.to(device)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    return ModelBundle(
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        embed_dim=int(encoder.embed_dim),
        epoch=epoch,
    )


def prepare_goal_world_model(
    config: dict, checkpoint_path: str, device: str, goal_lead_seconds: Optional[float] = None
) -> ModelBundle:
    """Rebuild the goal-guided chunk world model and dress it in the same interface
    `rollout_clips` uses for a V-JEPA world model.

    Two shims do the work, both from `app/world_model/utils.py`:
      * `ChunkWindowEncoder` fuses the frozen V-JEPA 2 backbone with the trained chunk
        encoder, so a window of pixels maps straight to a window of chunk latents - one
        temporal slot of `P` tokens per chunk, the same layout a tubelet embedding has.
      * `ChunkRolloutPredictor` drives the predictor one step at a time. With no
        `goal_lead_seconds` it runs the null-goal path, so the rollout gets exactly the
        information every other model gets: the past, and nothing else. With one, the
        goal is supplied at that fixed lead - a privileged input, see `GoalSchedule`.

    Both are then wrapped in the repo's `MultiSeqWrapper`/`PredictorMultiSeqWrapper`, so
    the returned bundle is structurally identical to `prepare_world_model`'s and the
    rollout does not have to know which kind of model it is driving.

    The frozen backbone is shared between the online and target encoders rather than
    duplicated - it is frozen and identical in both, and a second ViT-L would cost a
    gigabyte for nothing.
    """
    cfgs_data = config["data"]
    cfgs_model = config["model"]
    cfgs_meta = config.get("meta", {})
    cfgs_wm = config["world_model"]
    cfgs_vjepa = config["vjepa"]

    # Rolling out with the goal withheld exercises the null-goal path. If the run never
    # dropped its goal, those parameters are still at their zero initialization and the
    # model has never once been asked to predict without an intention - the numbers
    # would be measuring an untrained code path, not a world model.
    schedule = build_goal_schedule(config, goal_lead_seconds)
    goal_drop_prob = float(cfgs_wm.get("goal_drop_prob", 0.0))
    if schedule is None and goal_drop_prob <= 0.0:
        raise ValueError(
            "this model was trained with world_model.goal_drop_prob = 0, so its null goal was never "
            "trained and it has never predicted without a goal; rolling it out unconditionally would "
            "measure an untrained path. Retrain with goal_drop_prob > 0 to make it comparable."
        )

    crop_size = int(cfgs_data.get("crop_size", 256))
    patch_size = int(cfgs_vjepa.get("patch_size", 16))
    tubelet_size = int(cfgs_vjepa.get("tubelet_size", 2))
    tokens_per_chunk = int(cfgs_wm.get("tokens_per_chunk", 4))
    frames_per_chunk = tokens_per_chunk * tubelet_size
    grid = crop_size // patch_size

    vjepa = init_frozen_backbone(
        device=device,
        checkpoint=cfgs_vjepa["checkpoint"],
        checkpoint_key=cfgs_vjepa.get("checkpoint_key", "target_encoder"),
        model_name=cfgs_vjepa.get("model_name", "vit_large"),
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_chunk=frames_per_chunk,
        uniform_power=cfgs_vjepa.get("uniform_power", False),
        use_rope=cfgs_vjepa.get("use_rope", True),
        use_sdpa=cfgs_meta.get("use_sdpa", False),
        use_silu=cfgs_vjepa.get("use_silu", False),
        wide_silu=cfgs_vjepa.get("wide_silu", True),
    )

    encoder, predictor = init_world_model(
        device=device,
        frozen_dim=vjepa.embed_dim,
        grid_height=grid,
        grid_width=grid,
        tokens_per_chunk=tokens_per_chunk,
        context_chunks=int(cfgs_wm.get("context_chunks", 8)),
        embed_dim=cfgs_model.get("embed_dim", 768),
        enc_depth=cfgs_model.get("enc_depth", 6),
        enc_num_heads=cfgs_model.get("enc_num_heads", 12),
        pred_depth=cfgs_model.get("pred_depth", 12),
        pred_embed_dim=cfgs_model.get("pred_embed_dim", 384),
        pred_num_heads=cfgs_model.get("pred_num_heads", 12),
        goal_gate_init=cfgs_model.get("goal_gate_init", 1.0),
        horizon_embed_dim=cfgs_model.get("horizon_embed_dim", 128),
        use_sdpa=cfgs_meta.get("use_sdpa", False),
        use_silu=cfgs_model.get("use_silu", False),
        use_pred_silu=cfgs_model.get("use_pred_silu", False),
        wide_silu=cfgs_model.get("wide_silu", True),
        # Only trades compute for training memory; buys nothing under torch.no_grad().
        use_activation_checkpointing=False,
    )
    target_encoder = copy.deepcopy(encoder)

    # Load into the bare modules, before any wrapping, so the checkpoint's keys line up
    # with what training saved.
    checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))
    epoch = int(checkpoint.get("epoch", 0))
    load_module_state_dict(encoder, strip_ddp_prefix(checkpoint["encoder"]), "chunk encoder", epoch)
    load_module_state_dict(predictor, strip_ddp_prefix(checkpoint["predictor"]), "predictor", epoch)
    load_module_state_dict(
        target_encoder, strip_ddp_prefix(checkpoint["target_encoder"]), "target chunk encoder", epoch
    )
    del checkpoint

    backbone_batch_size = int(cfgs_vjepa.get("batch_size", -1))
    bundled = ModelBundle(
        encoder=MultiSeqWrapper(
            ChunkWindowEncoder(vjepa, encoder, frames_per_chunk, backbone_batch_size)
        ),
        predictor=PredictorMultiSeqWrapper(ChunkRolloutPredictor(predictor)),
        target_encoder=MultiSeqWrapper(
            ChunkWindowEncoder(vjepa, target_encoder, frames_per_chunk, backbone_batch_size)
        ),
        embed_dim=int(encoder.embed_dim),
        epoch=epoch,
        goal_schedule=schedule,
    )

    for module in (bundled.encoder, bundled.predictor, bundled.target_encoder):
        module.to(device)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
    return bundled


def goal_conditioning_label(model_cfg: dict) -> str:
    if model_cfg.get("kind") != "goal_world_model":
        return "n/a (model has no goal input)"
    lead = model_cfg.get("goal_lead_seconds")
    if lead is None:
        return "withheld (null goal)"
    return f"supplied, held {lead:g}s ahead of the context (PRIVILEGED: reads past the frontier)"


def prepare_bundle(model_cfg: dict, config: dict, checkpoint_path: str, device: str) -> ModelBundle:
    kind = model_cfg.get("kind", "vjepa")
    if kind == "goal_world_model":
        return prepare_goal_world_model(
            config, checkpoint_path, device, model_cfg.get("goal_lead_seconds")
        )
    if kind != "vjepa":
        raise ValueError(f"unknown model kind {kind!r}")
    return prepare_world_model(config, checkpoint_path, device)


# --------------------------------------------------------------------------- #
# Clip loading
# --------------------------------------------------------------------------- #


def frame_stride(geom: RolloutGeometry, video_fps: int) -> Optional[int]:
    """Frames of the source video per sampled frame, as `VideoDataset` computes it
    (`video_fps // fps`). None if the video is slower than the rate the model
    samples at, in which case its temporal spacing cannot be reproduced at all.

    Note this uses `sampling_fps`, not `fps`: a post-trained model decodes at its
    backbone's original frame rate and thins the result afterwards, so it needs a
    *finer* stride - and more source footage - than its nominal `fps` suggests."""
    stride = int(video_fps // geom.sampling_fps)
    return stride if stride >= 1 else None


def required_raw_frames(geom: RolloutGeometry, video_fps: int) -> Optional[int]:
    """Source frames a video must hold for one full rollout at this geometry."""
    stride = frame_stride(geom, video_fps)
    return None if stride is None else geom.total_sampled_frames * stride


def select_videos(
    manifest: pd.DataFrame,
    geometries: Dict[str, RolloutGeometry],
    num_clips: int,
) -> List[Tuple[int, str]]:
    """Choose videos that are long enough for *every* model, so all models are
    scored on the identical clip list.

    A model with a lower `fps` needs more source footage to cover the same
    horizon (60s at 0.5 fps spans 368 frames of a 4 fps video; at 4 fps it spans
    256), so filtering per model would silently score the models on different
    subsets of the manifest. Requiring every geometry to fit keeps the comparison
    honest, at the cost of dropping videos only the cheapest model could have
    used.

    Returns (manifest row index, video path) pairs; the row index seeds the clip's
    start offset, so a given video is entered at the same point for every model.
    """
    selected: List[Tuple[int, str]] = []
    num_missing = 0
    num_unreadable = 0
    num_too_short = 0

    for row_idx, row in enumerate(manifest.itertuples()):
        if len(selected) >= num_clips:
            break
        video_path = str(row.video_path).strip().strip('"')
        if not os.path.exists(video_path):
            logger.warning("video path not found: %s", video_path)
            num_missing += 1
            continue
        try:
            reader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            video_fps = math.ceil(reader.get_avg_fps())
            num_frames = len(reader)
            del reader
        except Exception as exc:
            logger.warning("could not probe %s: %s", video_path, exc)
            num_unreadable += 1
            continue

        shortfall = None
        for model_name, geom in geometries.items():
            needed = required_raw_frames(geom, video_fps)
            if needed is None:
                shortfall = f"{model_name} samples at {geom.fps:g} fps but the video is only {video_fps} fps"
                break
            if num_frames < needed:
                shortfall = (
                    f"{model_name} needs {needed} of the video's {num_frames} frames "
                    f"({geom.horizon_seconds:.0f}s horizon at {geom.fps:g} fps)"
                )
                break
        if shortfall is not None:
            logger.info("skipping %s: %s", video_path, shortfall)
            num_too_short += 1
            continue
        selected.append((row_idx, video_path))

    logger.info(
        "Selected %d/%d video(s) long enough for all %d model(s) (%d too short, %d missing, %d unreadable)",
        len(selected),
        num_clips,
        len(geometries),
        num_too_short,
        num_missing,
        num_unreadable,
    )
    return selected


def load_long_clip(
    video_path: str,
    geom: RolloutGeometry,
    transform,
    rng: np.random.Generator,
) -> Optional[Tuple[torch.Tensor, int]]:
    """Decode one clip laid out exactly as this model's training clips were.

    Two stages, mirroring the training data pipeline:
      1. Decode `geom.total_sampled_frames` frames at `geom.sampling_fps`, with the
         stride derived from the video's own frame rate as
         `src/datasets/video_dataset.py` does (`video_fps // fps`).
      2. For a model post-trained from a higher-frame-rate backbone, thin those
         frames per chunk the way `SimpleCollator` does, keeping the leading
         `tubelet_size` of every `frames_to_skip`. This is what makes frame
         spacing dense inside a tubelet and sparse between tubelets.
    Without post-training stage 2 is a no-op and the spacing is uniform.

    Returns (clip, start_frame) or None if the video is unusable/too short.
    """
    if not os.path.exists(video_path):
        logger.warning("video path not found: %s", video_path)
        return None

    try:
        reader = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
    except Exception as exc:
        logger.warning("could not open %s: %s", video_path, exc)
        return None

    try:
        video_fps = math.ceil(reader.get_avg_fps())
    except Exception as exc:
        logger.warning("could not read fps of %s: %s", video_path, exc)
        return None

    frame_step = frame_stride(geom, video_fps)
    if frame_step is None:
        logger.error(
            "%s is only %d fps but the model samples at %.3f fps; skipping",
            video_path,
            video_fps,
            geom.fps,
        )
        return None

    # `select_videos` already guaranteed this, so reaching here means the video
    # changed underneath the run - worth an error rather than a quiet skip,
    # because it makes this model's clip list diverge from the others'.
    span = geom.total_sampled_frames * frame_step
    if len(reader) < span:
        logger.error(
            "%s holds %d frames but %d are needed for a %.0fs rollout; skipping",
            video_path,
            len(reader),
            span,
            geom.horizon_seconds,
        )
        return None

    # Draw the start as a *fraction* of the usable range rather than an absolute
    # frame, so that a given video is entered at the same point for every model.
    # Models sampling at different rates need different amounts of footage for the
    # same horizon, so the absolute frame still differs slightly, but the clips
    # remain aligned enough for a fair cross-model comparison.
    start = int(round(float(rng.random()) * (len(reader) - span)))
    indices = start + np.arange(0, span, frame_step, dtype=np.int64)[: geom.total_sampled_frames]

    try:
        frames = reader.get_batch(indices).asnumpy()
    except Exception as exc:
        logger.warning("could not decode %s: %s", video_path, exc)
        return None

    views = transform(frames)
    if not views:
        logger.warning("transform produced no view for %s", video_path)
        return None

    clip = views[0]  # (C, T, H, W)
    if clip.shape[1] != geom.total_sampled_frames:
        logger.warning(
            "%s produced %d frames, expected %d; skipping",
            video_path,
            clip.shape[1],
            geom.total_sampled_frames,
        )
        return None

    if geom.uses_pretrained_backbone:
        # Thin per chunk after the transform, matching SimpleCollator's ordering
        # (the dataset transforms the full densely-sampled clip, the collator
        # thins it). The eval transform is per-frame, so the order is immaterial
        # to the pixels - but matching it keeps this path auditable against
        # training.
        clip = apply_pretrained_frame_skip(
            clip.unsqueeze(0), geom.frames_to_skip, geom.tubelet_size
        ).squeeze(0)
        if clip.shape[1] != geom.total_frames:
            logger.error(
                "%s: frame skip produced %d frames, expected %d; skipping",
                video_path,
                clip.shape[1],
                geom.total_frames,
            )
            return None
    return clip, start


def iter_clip_batches(
    videos: Sequence[Tuple[int, str]],
    geom: RolloutGeometry,
    transform,
    batch_size: int,
    seed: int,
) -> List[ClipBatch]:
    """Decode the pre-selected `videos` and group them into equally-shaped batches.

    Every clip of a model has the same shape by construction, so batching is
    purely an efficiency choice; the rollout is independent per clip. The clip id
    is the position in `videos`, which is the same list for every model, so
    `clip_id` refers to the same video across models.
    """
    batches: List[ClipBatch] = []
    pending: List[torch.Tensor] = []
    paths: List[str] = []
    starts: List[int] = []
    ids: List[int] = []

    for clip_id, (row_idx, video_path) in enumerate(
        tqdm(videos, desc="decoding clips", leave=False)
    ):
        # Seeded from the manifest row, not the position in `videos`, so a video
        # is entered at the same point however the selection turned out.
        loaded = load_long_clip(video_path, geom, transform, np.random.default_rng(seed + row_idx))
        if loaded is None:
            continue
        clip, start = loaded
        pending.append(clip)
        paths.append(video_path)
        starts.append(start)
        ids.append(clip_id)

        if len(pending) == batch_size:
            batches.append(
                ClipBatch(torch.stack(pending, dim=0), list(paths), list(starts), list(ids))
            )
            pending, paths, starts, ids = [], [], [], []

    if pending:
        batches.append(ClipBatch(torch.stack(pending, dim=0), paths, starts, ids))
    return batches


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #


def temporal_slice(clip: torch.Tensor, first_token: int, num_tokens: int, tubelet_size: int) -> torch.Tensor:
    """Frames of temporal tokens [first_token, first_token + num_tokens)."""
    start = first_token * tubelet_size
    return clip[:, :, start : start + num_tokens * tubelet_size]


# F.layer_norm's default. Kept explicit because `oracle_rescale` inverts the
# normalization and has to use the same constant to be an exact inverse.
LAYER_NORM_EPS = 1e-5


def layer_norm_last(x: torch.Tensor) -> torch.Tensor:
    """The exact target normalization used by the pretraining loss."""
    return F.layer_norm(x, (x.size(-1),), eps=LAYER_NORM_EPS)


def layer_norm_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The per-token shift/scale that `layer_norm_last` divides out.

    `F.layer_norm` with no affine is `y = (x - mu) / sqrt(var + eps)` over the
    channel axis, so `x = y * sigma + mu` with these two is its *exact* inverse -
    which is what makes a per-token scalar affine the right functional form for
    every `rescale` variant, and what `oracle_rescale` uses to undo the
    normalization exactly. Note the biased variance: that is what layer norm
    uses, unlike `torch.std`'s unbiased default.

    Returns two (..., ) tensors, one rank lower than `x`.
    """
    mean = x.mean(dim=-1)
    sigma = (x.var(dim=-1, unbiased=False) + LAYER_NORM_EPS).sqrt()
    return mean, sigma


@dataclass
class GroundTruth:
    """Target-encoder embeddings of the frames a rollout has to predict.

    `normalized` is what the predictions are *scored* against - the training
    loss's target. `mean`/`sigma` are the per-token statistics layer norm
    removed, kept only so `oracle_rescale` can put them back; nothing else may
    read them, because doing so would leak the frame being predicted.
    """

    normalized: torch.Tensor  # (B, num_steps + 1, S, D), layer-normalized
    mean: torch.Tensor  # (B, num_steps + 1, S), pre-normalization
    sigma: torch.Tensor  # (B, num_steps + 1, S), pre-normalization


@torch.no_grad()
def encode_ground_truth(
    bundle: ModelBundle,
    geom: RolloutGeometry,
    clip: torch.Tensor,
    autocast_kwargs: dict,
) -> GroundTruth:
    """Layer-normalized target-encoder embedding of every frame the rollout has
    to predict, plus the last observed frame.

    Each frame is embedded *in its own full-length ground-truth window*, which is
    how the training loop produces its targets: a target token is never seen in
    isolation, it is the target encoder's output for a clip of
    `frames_per_clip` frames. Window `w` spans temporal tokens
    [w, w + N_t - 1], so its last token is `w + N_t - 1`:

        w = 0            -> token N_t - 1  = the last *observed* frame
                            (the persistence-baseline reference)
        w = k + 1        -> token N_t + k  = the frame predicted at step k

    Index 0 is the reference, index k+1 is the ground truth for step k.
    """
    normalized, means, sigmas = [], [], []
    for window in range(geom.num_steps + 1):
        frames = temporal_slice(clip, window, geom.window_tokens, geom.tubelet_size)
        with torch.autocast(**autocast_kwargs):
            tokens = bundle.target_encoder([frames])[0]
        tokens = tokens.float()[:, -geom.spatial_tokens :]
        mean, sigma = layer_norm_stats(tokens)
        normalized.append(layer_norm_last(tokens))
        means.append(mean)
        sigmas.append(sigma)
    return GroundTruth(
        normalized=torch.stack(normalized, dim=1),
        mean=torch.stack(means, dim=1),
        sigma=torch.stack(sigmas, dim=1),
    )


@torch.no_grad()
def encode_context(
    bundle: ModelBundle,
    geom: RolloutGeometry,
    clip: torch.Tensor,
    first_token: int,
    autocast_kwargs: dict,
) -> torch.Tensor:
    """Context-encoder embedding of `geom.context_tokens` temporal tokens
    starting at `first_token`.

    No mask is passed: the encoder tokenizes exactly the frames it is given, so
    the returned tokens occupy window positions 0 .. n_ctx*S-1 - which is the
    positional frame the predictor's masks are expressed in.
    """
    frames = temporal_slice(clip, first_token, geom.context_tokens, geom.tubelet_size)
    with torch.autocast(**autocast_kwargs):
        return bundle.encoder([frames])[0]


@torch.no_grad()
def encode_goals(
    bundle: ModelBundle,
    geom: RolloutGeometry,
    clip: torch.Tensor,
    schedule: GoalSchedule,
    autocast_kwargs: dict,
) -> torch.Tensor:
    """Goal latents for every step of a goal-conditioned rollout, (B, num_steps, S, D).

    Step `k`'s goal is the chunk `schedule.lead_seconds` past the end of the context
    that step conditions on. Both slide forward one chunk per step, so the goals are the
    contiguous run of chunks starting at `schedule.first_frame` and can be embedded in a
    single call.

    Produced by the *target* encoder and layer-normalized, which is how training builds
    the goal (`forward_target` in app/world_model/train.py) - the same space the
    prediction targets live in.

    PRIVILEGED: these frames sit ahead of the rollout's frontier, so a model given them
    is seeing real future video that no other model in this report gets.
    """
    start = schedule.first_frame
    frames = clip[:, :, start : start + geom.num_steps * schedule.frames_per_chunk]
    with torch.autocast(**autocast_kwargs):
        tokens = bundle.target_encoder([frames])[0]
    return layer_norm_last(tokens.float()).unflatten(1, (geom.num_steps, geom.spatial_tokens))


def predictor_masks(geom: RolloutGeometry, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Context/prediction masks for a non-causal predictor.

    Patch tokens are laid out as `t * S + s` (see `_MaskGenerator`, which flattens
    a (duration, height, width) grid), so "every patch of the last temporal
    token" is the contiguous index block [n_ctx*S, N_t*S).
    """
    split = geom.context_tokens * geom.spatial_tokens
    total = geom.window_tokens * geom.spatial_tokens
    context = torch.arange(0, split, device=device).unsqueeze(0).expand(batch_size, -1)
    predict = torch.arange(split, total, device=device).unsqueeze(0).expand(batch_size, -1)
    return context.contiguous(), predict.contiguous()


@torch.no_grad()
def predict_next_token(
    bundle: ModelBundle,
    geom: RolloutGeometry,
    context: torch.Tensor,
    masks: Optional[Tuple[torch.Tensor, torch.Tensor]],
    mask_token_index: int,
    autocast_kwargs: dict,
    goal: Optional[torch.Tensor] = None,
    goal_pos: Optional[float] = None,
) -> torch.Tensor:
    """One step of the world model: (B, n_ctx*S, D) context -> (B, S, D) next frame.

    Causal and non-causal models reach the same result through the two different
    interfaces they were trained with:
      * causal - no mask at all. Like an LLM the predictor emits a next-token
        prediction at every input position, and the last S of them are the next
        frame (training reads exactly these, having chopped the first frame off
        the targets).
      * non-causal - the mask pair from `predictor_masks`, which asks for the
        patches of the last temporal token and nothing else. Mask tokens carrying
        the target positions are inserted by the predictor itself.

    Both calls go straight to `predictor.backbone`, which is exactly what
    `PredictorMultiSeqWrapper` does for a single (clip-length, mask) group - the
    wrapper is only a loop over groups. Going through the backbone directly is
    what makes `mask_token_index` selectable: the wrapper hard-codes it to the
    index of the dataset a clip came from, which is information a rollout over an
    arbitrary manifest does not have.

    `goal` is only ever set for the goal-conditioned chunk model, and is forwarded only
    when present - a V-JEPA predictor has no such argument.
    """
    goal_kwargs = {} if goal is None else {"goal": goal, "goal_pos": goal_pos}
    with torch.autocast(**autocast_kwargs):
        if geom.is_causal:
            out = bundle.predictor.backbone(context, has_cls=False, **goal_kwargs)
            out = out[:, -geom.spatial_tokens :]
        else:
            context_mask, predict_mask = masks
            out = bundle.predictor.backbone(
                context, context_mask, predict_mask, mask_index=mask_token_index, has_cls=False
            )
    return out.float()


def context_scale(context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-clip mean/std across channels of context tokens, averaged over tokens.

    Predictions come out of `predictor_proj` in the layer-normalized target space
    (roughly zero-mean, unit-variance per token), whereas the predictor was fed
    raw context-encoder tokens. These two statistics are what the `rescale` and
    `rescale_running` feedback modes use to put a prediction back on the context
    scale - the former from the first real window only, the latter from whatever
    the window currently holds.

    A single scalar pair per clip, because the scale of the *next* token is not
    knowable; averaging the observed tokens is the estimate. `oracle_rescale`
    replaces this estimate with the true per-token values.
    """
    tokens = context.float()
    mean = tokens.mean(dim=-1).mean(dim=-1)  # (B,)
    std = tokens.std(dim=-1).mean(dim=-1)  # (B,)
    return mean, std


def feedback_scale(
    mode: str,
    context: torch.Tensor,
    frozen: Tuple[torch.Tensor, torch.Tensor],
    ground_truth: GroundTruth,
    step: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The (mean, sigma) a rescaling mode feeds back with, broadcast to (B, *, 1).

    Returns None for the modes that do not rescale. The three that do differ only
    in where the two numbers come from:
      rescale         - frozen at the first real context window
      rescale_running - the current window, real or self-generated
      oracle_rescale  - the true pre-normalization statistics of the frame being
                        predicted. ORACLE: reads `ground_truth` at the step's own
                        index, which is the future.
    """
    if mode == "rescale":
        mean, std = frozen
        return mean.view(-1, 1, 1), std.view(-1, 1, 1)
    if mode == "rescale_running":
        mean, std = context_scale(context)
        return mean.view(-1, 1, 1), std.view(-1, 1, 1)
    if mode == "oracle_rescale":
        # Index step + 1: index 0 is the last *observed* frame, so this is the
        # frame the model is being asked to predict and has not seen.
        return (
            ground_truth.mean[:, step + 1].unsqueeze(-1),
            ground_truth.sigma[:, step + 1].unsqueeze(-1),
        )
    return None


def apply_feedback(
    prediction: torch.Tensor,
    mode: str,
    scale: Optional[Tuple[torch.Tensor, torch.Tensor]],
    dtype: torch.dtype,
) -> torch.Tensor:
    if mode == "raw":
        out = prediction
    elif mode == "layer_norm":
        # The observed context was layer-normalized too, so normalize the
        # prediction as well: predictions come out of `predictor_proj` only
        # approximately normalized, and letting that drift accumulate over a
        # hundred steps would defeat the point of this mode.
        out = layer_norm_last(prediction)
    elif mode in ("rescale", "rescale_running", "oracle_rescale"):
        if scale is None:
            raise ValueError(f"feedback mode {mode!r} needs a scale but none was supplied")
        mean, sigma = scale
        out = prediction * sigma + mean
    else:
        raise ValueError(f"Unknown feedback mode: {mode}")
    return out.to(dtype)


def frame_distances(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-clip distances between two (B, S, D) frame embeddings."""
    return {
        "l1": (pred - target).abs().mean(dim=(1, 2)),
        "l2": (pred - target).pow(2).sum(dim=-1).sqrt().mean(dim=1),
        "cosine": (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean(dim=1),
    }


@torch.no_grad()
def rollout_clips(
    bundle: ModelBundle,
    geom: RolloutGeometry,
    batches: Sequence[ClipBatch],
    device: str,
    feedback_mode: str,
    mask_token_index: int,
    autocast_kwargs: dict,
    teacher_forcing: bool,
    model_name: str,
) -> RolloutResult:
    """Roll every clip forward `geom.num_steps` steps and score each step."""
    result = RolloutResult()
    start_time = time.time()
    masks_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    total_steps = sum(len(b.clip_ids) for b in batches) * geom.num_steps
    progress = tqdm(total=total_steps, desc=f"{model_name} rollout", leave=False)

    for batch in batches:
        result.num_clips_attempted += len(batch.clip_ids)
        try:
            clip = batch.frames.to(device, non_blocking=True)
            batch_size = clip.shape[0]
            if batch_size not in masks_cache:
                masks_cache[batch_size] = predictor_masks(geom, batch_size, device)
            masks = None if geom.is_causal else masks_cache[batch_size]

            ground_truth = encode_ground_truth(bundle, geom, clip, autocast_kwargs)
            reference = ground_truth.normalized[:, 0]  # last observed frame, in GT space
            goals = (
                None
                if bundle.goal_schedule is None
                else encode_goals(bundle, geom, clip, bundle.goal_schedule, autocast_kwargs)
            )

            # Observed context starts at temporal token 1: token 0 exists only so
            # the last observed frame has a full ground-truth window behind it.
            context = encode_context(bundle, geom, clip, 1, autocast_kwargs)
            # The frozen `rescale` anchor: computed once, from the only window
            # that is entirely real. `rescale_running` recomputes per step
            # instead; both still record this for comparison.
            frozen_scale = context_scale(context)
            mean, std = frozen_scale
            if feedback_mode == "layer_norm":
                context = layer_norm_last(context.float()).to(context.dtype)

            pred_pooled = np.zeros((batch_size, geom.num_steps, bundle.embed_dim), dtype=np.float32)
            gt_pooled = np.zeros((batch_size, geom.num_steps, bundle.embed_dim), dtype=np.float32)
            batch_rows: List[dict] = []

            for step in range(geom.num_steps):
                prediction = predict_next_token(
                    bundle,
                    geom,
                    context,
                    masks,
                    mask_token_index,
                    autocast_kwargs,
                    goal=None if goals is None else goals[:, step],
                    goal_pos=None if goals is None else bundle.goal_schedule.position,
                )
                target = ground_truth.normalized[:, step + 1]

                pred_dist = frame_distances(prediction, target)
                ref_dist = frame_distances(reference, target)

                # Spatial spread within one frame: a predictor that has collapsed
                # onto a single "average patch" scores small distances while
                # carrying no spatial information at all, and only this ratio
                # makes that visible. Undefined for a model with one patch token
                # per frame (crop_size == patch_size), which has no spatial
                # structure to lose in the first place.
                if geom.spatial_tokens > 1:
                    pred_spatial_std = prediction.std(dim=1).mean(dim=-1)
                    gt_spatial_std = target.std(dim=1).mean(dim=-1)
                else:
                    pred_spatial_std = gt_spatial_std = torch.full(
                        (batch_size,), float("nan"), device=prediction.device
                    )

                # Layer-norm scale of the prediction: std *across channels*, per
                # token - the same reduction `context_scale` applies to real
                # context tokens, and a different quantity from the spatial std
                # above (which is spread across patches). The predictor is
                # trained to emit layer-normalized targets, so this should sit
                # at 1.0.
                #
                # Two things depend on it. It is a collapse signal in its own
                # right - drifting toward 0 means the prediction is flattening
                # toward a constant token. And it decides whether the frozen
                # `rescale` anchor matters: `mean`/`std` are computed once from
                # the first real context window, and the alternative of
                # recomputing them from the current window each step is only
                # equivalent if this value is 1.0. The real context is fully
                # evicted after `context_tokens` steps (7 of 120 for a
                # 16-frame/tubelet-2 model at 4 fps), so past that point a
                # recomputed anchor would compound any offset geometrically,
                # while the frozen one re-applies a constant and cannot drift.
                pred_token_std = prediction.std(dim=-1).mean(dim=-1)

                # The scale actually fed back this step. Constant for `rescale`,
                # a trajectory for `rescale_running` (this is the quantity that
                # compounds once the real context is evicted) and the true
                # per-token value for `oracle_rescale`. Recorded so the three can
                # be compared directly instead of inferred.
                scale = feedback_scale(feedback_mode, context, frozen_scale, ground_truth, step)
                if scale is None:
                    feedback_mean = feedback_std = torch.full(
                        (batch_size,), float("nan"), device=prediction.device
                    )
                else:
                    feedback_mean = scale[0].reshape(batch_size, -1).mean(dim=1)
                    feedback_std = scale[1].reshape(batch_size, -1).mean(dim=1)

                for i in range(batch_size):
                    row = {
                        "model": model_name,
                        "clip_id": batch.clip_ids[i],
                        "video_path": batch.video_paths[i],
                        "start_frame": batch.start_frames[i],
                        "step": step,
                        "lead_seconds": geom.lead_seconds(step),
                        "pred_spatial_std": float(pred_spatial_std[i]),
                        "gt_spatial_std": float(gt_spatial_std[i]),
                        "pred_token_std": float(pred_token_std[i]),
                        # Constant across steps (the frozen anchor), recorded per
                        # row so the recomputed-anchor trajectory can be simulated
                        # offline as sigma_{k+1} = mean(sigma_j * s_j) without a
                        # second GPU run.
                        "context_token_std": float(std[i]),
                        "feedback_std": float(feedback_std[i]),
                        "feedback_mean": float(feedback_mean[i]),
                    }
                    for metric in METRICS:
                        row[f"{metric}_pred"] = float(pred_dist[metric][i])
                        row[f"{metric}_ref"] = float(ref_dist[metric][i])
                    batch_rows.append(row)

                pred_pooled[:, step] = prediction.mean(dim=1).cpu().numpy()
                gt_pooled[:, step] = target.mean(dim=1).cpu().numpy()

                context = torch.cat(
                    [
                        context[:, geom.spatial_tokens :],
                        apply_feedback(prediction, feedback_mode, scale, context.dtype),
                    ],
                    dim=1,
                )
                progress.update(batch_size)

            if teacher_forcing:
                add_teacher_forced_metrics(
                    bundle,
                    geom,
                    clip,
                    ground_truth,
                    batch_rows,
                    masks,
                    mask_token_index,
                    autocast_kwargs,
                    feedback_mode,
                    goals,
                )

            result.rows.extend(batch_rows)
            for i in range(batch_size):
                result.pred_pooled.append(pred_pooled[i])
                result.gt_pooled.append(gt_pooled[i])
                result.clip_ids.append(batch.clip_ids[i])
        except Exception as exc:
            result.num_clips_failed += len(batch.clip_ids)
            logger.error("rollout failed for %s on clips %s: %s", model_name, batch.clip_ids, exc)
        finally:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    progress.close()
    result.elapsed_s = time.time() - start_time
    return result


@torch.no_grad()
def add_teacher_forced_metrics(
    bundle: ModelBundle,
    geom: RolloutGeometry,
    clip: torch.Tensor,
    ground_truth: GroundTruth,
    rows: List[dict],
    masks: Optional[Tuple[torch.Tensor, torch.Tensor]],
    mask_token_index: int,
    autocast_kwargs: dict,
    feedback_mode: str,
    goals: Optional[torch.Tensor] = None,
) -> None:
    """Attach the single-step (teacher-forced) distance for every step, in place.

    Same predictor, same target, but the context is the *real* video window
    instead of the model's own output. The gap between the two curves separates
    "this frame is intrinsically hard to predict" from "the rollout has drifted",
    which a rollout-only number cannot distinguish.
    """
    batch_size = clip.shape[0]
    by_step: Dict[int, List[dict]] = {}
    for row in rows:
        by_step.setdefault(row["step"], []).append(row)

    for step in range(geom.num_steps):
        context = encode_context(bundle, geom, clip, step + 1, autocast_kwargs)
        if feedback_mode == "layer_norm":
            context = layer_norm_last(context.float()).to(context.dtype)
        prediction = predict_next_token(
            bundle,
            geom,
            context,
            masks,
            mask_token_index,
            autocast_kwargs,
            goal=None if goals is None else goals[:, step],
            goal_pos=None if goals is None else bundle.goal_schedule.position,
        )
        target = ground_truth.normalized[:, step + 1]
        distances = frame_distances(prediction, target)

        step_rows = sorted(by_step.get(step, []), key=lambda r: r["clip_id"])
        if len(step_rows) != batch_size:
            continue
        for i, row in enumerate(step_rows):
            for metric in METRICS:
                row[f"{metric}_tf"] = float(distances[metric][i])


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """`_norm` columns: distance divided by the persistence baseline.

    The ratio is formed *per clip and per step* before any averaging, so a single
    clip whose baseline happens to be large cannot flatter the average.
    """
    if df.empty:
        return df
    df = df.copy()
    for metric in METRICS:
        reference = df[f"{metric}_ref"]
        usable = reference > MIN_REFERENCE_DISTANCE
        df[f"{metric}_norm"] = np.where(usable, df[f"{metric}_pred"] / reference.where(usable, np.nan), np.nan)
        if f"{metric}_tf" in df.columns:
            df[f"{metric}_tf_norm"] = np.where(
                usable, df[f"{metric}_tf"] / reference.where(usable, np.nan), np.nan
            )
    df["spatial_std_ratio"] = np.where(
        df["gt_spatial_std"] > 1e-8, df["pred_spatial_std"] / df["gt_spatial_std"].replace(0.0, np.nan), np.nan
    )
    return df


def cross_clip_dispersion(pooled: np.ndarray) -> np.ndarray:
    """Mean pairwise cosine distance between clips, per step.

    `pooled` is (num_clips, num_steps, D). If a model answers the same thing
    whatever it was shown - the failure mode raw distances hide - this collapses
    toward 0 while the ground truth's stays put.
    """
    num_clips, num_steps, _ = pooled.shape
    out = np.full(num_steps, np.nan, dtype=np.float64)
    if num_clips < 2:
        return out
    triu_i, triu_j = np.triu_indices(num_clips, k=1)
    for step in range(num_steps):
        vectors = pooled[:, step].astype(np.float64)
        norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        vectors = vectors / norms
        sims = vectors @ vectors.T
        out[step] = float(np.mean(1.0 - sims[triu_i, triu_j]))
    return out


def per_step_frame(df: pd.DataFrame, result: RolloutResult, geom: RolloutGeometry) -> pd.DataFrame:
    """One row per predicted frame: the evolution of every metric over
    prediction time, averaged across clips."""
    if df.empty:
        return pd.DataFrame()

    aggregations = {"clips": ("clip_id", "count"), "lead_seconds": ("lead_seconds", "first")}
    for metric in METRICS:
        aggregations[f"{metric}_pred"] = (f"{metric}_pred", "mean")
        aggregations[f"{metric}_ref"] = (f"{metric}_ref", "mean")
        aggregations[f"{metric}_norm"] = (f"{metric}_norm", "mean")
        aggregations[f"{metric}_norm_std"] = (f"{metric}_norm", "std")
        if f"{metric}_tf" in df.columns:
            aggregations[f"{metric}_tf"] = (f"{metric}_tf", "mean")
            aggregations[f"{metric}_tf_norm"] = (f"{metric}_tf_norm", "mean")
    aggregations["spatial_std_ratio"] = ("spatial_std_ratio", "mean")
    aggregations["pred_token_std"] = ("pred_token_std", "mean")
    aggregations["context_token_std"] = ("context_token_std", "mean")
    aggregations["feedback_std"] = ("feedback_std", "mean")
    aggregations["feedback_mean"] = ("feedback_mean", "mean")

    frame = df.groupby("step", as_index=False).agg(**aggregations).sort_values("step").reset_index(drop=True)

    if result.pred_pooled:
        pred_pooled = np.stack(result.pred_pooled, axis=0)
        gt_pooled = np.stack(result.gt_pooled, axis=0)
        pred_dispersion = cross_clip_dispersion(pred_pooled)
        gt_dispersion = cross_clip_dispersion(gt_pooled)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = pred_dispersion / np.where(gt_dispersion > 1e-8, gt_dispersion, np.nan)
        frame["pred_dispersion"] = pred_dispersion[frame["step"].to_numpy()]
        frame["gt_dispersion"] = gt_dispersion[frame["step"].to_numpy()]
        frame["dispersion_ratio"] = ratio[frame["step"].to_numpy()]
    return frame


def horizon_rows(step_frame: pd.DataFrame, geom: RolloutGeometry, primary: str) -> List[dict]:
    """The per-step curve sampled at the summary horizons, so models with
    different step durations can be lined up in one table."""
    if step_frame.empty:
        return []
    rows = []
    leads = step_frame["lead_seconds"].to_numpy()
    for horizon in SUMMARY_HORIZONS_S:
        if horizon > geom.horizon_seconds + 0.5 * geom.step_seconds:
            continue
        idx = int(np.argmin(np.abs(leads - horizon)))
        row = step_frame.iloc[idx]
        rows.append(
            {
                "horizon_s": horizon,
                "actual_lead_s": float(row["lead_seconds"]),
                "step": int(row["step"]),
                "norm": float(row[f"{primary}_norm"]),
                "pred": float(row[f"{primary}_pred"]),
                "ref": float(row[f"{primary}_ref"]),
                "dispersion_ratio": float(row.get("dispersion_ratio", np.nan)),
                "spatial_std_ratio": float(row.get("spatial_std_ratio", np.nan)),
            }
        )
    return rows


def baseline_comparison(
    step_frame: pd.DataFrame, primary: str, threshold: float = 1.0
) -> Tuple[Optional[float], float]:
    """How the rollout stands against the persistence baseline over the horizon.

    Returns (durable crossing time, share of steps below the baseline). The
    crossing is *durable*: the first lead time from which the curve never dips
    back below `threshold`. A plain "first time it reached 1.0" would be
    misleading, because a curve can start above the baseline - a model that is
    worse than persistence at one step but better at ten has not "crossed" at its
    very first step - and can wobble across the line either way.
    """
    if step_frame.empty:
        return None, float("nan")
    column = step_frame[f"{primary}_norm"].to_numpy(dtype=np.float64)
    leads = step_frame["lead_seconds"].to_numpy(dtype=np.float64)
    finite = np.isfinite(column)
    share_below = float(np.mean(column[finite] < threshold)) if finite.any() else float("nan")

    below = np.where(finite & (column < threshold))[0]
    if below.size == 0:
        # Never beat the baseline at all: there is no crossing to report.
        return None, share_below
    last_below = int(below[-1])
    if last_below == len(column) - 1:
        return None, share_below
    return float(leads[last_below + 1]), share_below


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def save_normalized_chart(step_frames: Dict[str, pd.DataFrame], output_path: Path) -> None:
    if not step_frames:
        return
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.0 * len(METRICS), 4.8), dpi=150)
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, METRICS):
        for model_name, frame in step_frames.items():
            if frame.empty:
                continue
            ax.plot(frame["lead_seconds"], frame[f"{metric}_norm"], linewidth=1.8, label=model_name)
            if f"{metric}_tf_norm" in frame.columns:
                ax.plot(
                    frame["lead_seconds"],
                    frame[f"{metric}_tf_norm"],
                    linewidth=1.2,
                    linestyle=":",
                    label=f"{model_name} (teacher-forced)",
                )
        ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Prediction lead time (s)")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Distance / persistence baseline")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(3, max(1, len(labels))),
            bbox_to_anchor=(0.5, 1.10),
            fontsize=9,
        )
    fig.suptitle("Normalized prediction error over rollout time (1.0 = no better than freezing the last frame)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_raw_chart(step_frames: Dict[str, pd.DataFrame], primary: str, output_path: Path) -> None:
    if not step_frames:
        return
    fig, axes = plt.subplots(
        1, len(step_frames), figsize=(5.6 * len(step_frames), 4.4), dpi=150, squeeze=False
    )
    for ax, (model_name, frame) in zip(axes[0], step_frames.items()):
        if frame.empty:
            continue
        ax.plot(frame["lead_seconds"], frame[f"{primary}_pred"], linewidth=1.8, label="autoregressive rollout")
        if f"{primary}_tf" in frame.columns:
            ax.plot(
                frame["lead_seconds"],
                frame[f"{primary}_tf"],
                linewidth=1.2,
                linestyle=":",
                label="teacher-forced (1 step)",
            )
        ax.plot(
            frame["lead_seconds"],
            frame[f"{primary}_ref"],
            linewidth=1.5,
            linestyle="--",
            color="#64748b",
            label="persistence baseline",
        )
        ax.set_xlabel("Prediction lead time (s)")
        ax.set_title(model_name, fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel(f"{METRIC_LABELS[primary]} (raw)")
    fig.suptitle("Raw embedding distance vs. the persistence baseline", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_collapse_chart(step_frames: Dict[str, pd.DataFrame], output_path: Path) -> None:
    if not step_frames:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)
    for model_name, frame in step_frames.items():
        if frame.empty:
            continue
        axes[0].plot(frame["lead_seconds"], frame["spatial_std_ratio"], linewidth=1.8, label=model_name)
        if "dispersion_ratio" in frame.columns:
            axes[1].plot(frame["lead_seconds"], frame["dispersion_ratio"], linewidth=1.8, label=model_name)

    for ax, title, ylabel in (
        (
            axes[0],
            "Spatial detail retained within a predicted frame",
            "std across patches (pred / ground truth)",
        ),
        (axes[1], "Distinctness across different clips", "cross-clip dispersion (pred / ground truth)"),
    ):
        ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Prediction lead time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Collapse diagnostics: does the prediction still carry information?", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def goal_pairs(
    model_meta: Dict[str, dict], step_frames: Dict[str, pd.DataFrame]
) -> List[Tuple[str, str]]:
    """(goal-withheld, goal-supplied) pairs that share a checkpoint.

    Two entries built from the same checkpoint, one privileged and one not, differ by
    exactly one thing - whether the goal was supplied - which is the only controlled
    comparison the report can make about the goal itself.
    """
    usable = [n for n in model_meta if n in step_frames and not step_frames[n].empty]
    pairs = []
    for supplied in usable:
        if not model_meta[supplied].get("privileged"):
            continue
        for withheld in usable:
            if (
                not model_meta[withheld].get("privileged")
                and model_meta[withheld].get("kind") == "goal_world_model"
                and model_meta[withheld]["checkpoint"] == model_meta[supplied]["checkpoint"]
            ):
                pairs.append((withheld, supplied))
                break
    return pairs


def fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def per_step_table(step_frame: pd.DataFrame, primary: str) -> List[str]:
    has_tf = f"{primary}_tf" in step_frame.columns
    header = (
        "| Step | Lead (s) | Rollout | Persistence | **Normalized** | "
        + ("Teacher-forced | TF normalized | " if has_tf else "")
        + "Spatial detail | Cross-clip | Token std |"
    )
    divider = (
        "| --- | --- | --- | --- | --- | " + ("--- | --- | " if has_tf else "") + "--- | --- | --- |"
    )
    lines = [header, divider]
    for row in step_frame.itertuples():
        cells = [
            str(int(row.step) + 1),
            fmt(row.lead_seconds, 2),
            fmt(getattr(row, f"{primary}_pred"), 4),
            fmt(getattr(row, f"{primary}_ref"), 4),
            f"**{fmt(getattr(row, f'{primary}_norm'))}**",
        ]
        if has_tf:
            cells.append(fmt(getattr(row, f"{primary}_tf"), 4))
            cells.append(fmt(getattr(row, f"{primary}_tf_norm")))
        cells.append(fmt(getattr(row, "spatial_std_ratio", np.nan)))
        cells.append(fmt(getattr(row, "dispersion_ratio", np.nan)))
        cells.append(fmt(getattr(row, "pred_token_std", np.nan)))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def generate_markdown_report(
    report_path: Path,
    output_dir: Path,
    model_meta: Dict[str, dict],
    step_frames: Dict[str, pd.DataFrame],
    charts: Dict[str, Optional[Path]],
    skipped_models: List[Tuple[str, str]],
    manifest_info: dict,
    args: argparse.Namespace,
) -> None:
    primary = args.primary_metric
    lines: List[str] = []

    ranked = sorted(
        (name for name in model_meta if name in step_frames and not step_frames[name].empty),
        key=lambda name: model_meta[name]["mean_norm"],
    )
    # A run that is handed real video from beyond its own frontier cannot be allowed to
    # win the headline claim, so the leader is picked from the like-for-like entries only.
    fair_ranked = [name for name in ranked if not model_meta[name].get("privileged")]

    lines.append("# Autoregressive World-Model Report")
    lines.append("")
    if args.feedback in ORACLE_FEEDBACK_MODES:
        lines.append(
            f"> **⚠ ORACLE RUN - NOT A MEASUREMENT.** This report was generated with "
            f"`--feedback {args.feedback}`, which rescales every prediction using the true "
            "pre-normalization statistics of *the frame being predicted*. The rollout therefore sees "
            "the future it is scored against, and every number below is a **diagnostic ceiling**. Its "
            "only legitimate use is comparison against a `rescale` run: if the two agree, the feedback "
            "bridge is not what limits the horizon. Do not quote these figures as model performance."
        )
        lines.append("")
    lines.append(
        f"How long can each model keep predicting its own representation of a video before the "
        f"prediction stops telling us anything about what actually happens? Every model is rolled "
        f"forward **one temporal token at a time** for **~{args.horizon_seconds:.0f} seconds** of video, "
        f"feeding each prediction back in as context, and every predicted frame is scored against the "
        f"target encoder's embedding of the real frame."
    )
    lines.append("")
    lines.append(
        "Absolute distances in an embedding space cannot be compared across models - a representation "
        "that has partially collapsed makes *every* distance small. So the headline number is the "
        "**normalized** distance: the rollout's error divided by the error of simply freezing the last "
        "observed frame (the *persistence baseline*). **Below 1.0** the model anticipates change; "
        "**1.0** means it is worth no more than doing nothing; **above 1.0** means predicting actively "
        "hurts."
    )
    lines.append("")

    # -- TL;DR
    lines.append("## TL;DR")
    lines.append("")
    if not ranked:
        lines.append("No model produced a usable rollout - check the log for errors.")
        lines.append("")
    else:
        leader = (fair_ranked or ranked)[0]
        meta = model_meta[leader]
        lines.append(
            f"- **{leader}** has the best horizon-averaged normalized {METRIC_LABELS[primary]}: "
            f"**{fmt(meta['mean_norm'])}** over {meta['geometry'].num_steps} steps "
            f"({meta['geometry'].horizon_seconds:.0f}s)"
            + (
                " (of the entries that see only the past; the privileged goal-conditioned run below is "
                "excluded from this claim)."
                if len(fair_ranked) < len(ranked)
                else "."
            )
        )
        for withheld, supplied in goal_pairs(model_meta, step_frames):
            off, on = model_meta[withheld], model_meta[supplied]
            delta = off["mean_norm"] - on["mean_norm"]
            lines.append(
                f"- **What the goal is worth**: the same weights score **{fmt(off['mean_norm'])}** with "
                f"the goal withheld and **{fmt(on['mean_norm'])}** with it supplied "
                f"{on['goal_lead_seconds']:g}s ahead - a difference of **{fmt(delta)}** "
                f"({'better' if delta > 0 else 'worse'} with the goal). The goal-supplied run reads real "
                "video past its own frontier, so this is the value of the conditioning signal, not a "
                "score comparable with the other models."
            )
        for name in ranked:
            meta = model_meta[name]
            crossing = meta.get("crossing_s")
            share = meta.get("share_below_baseline", float("nan"))
            if crossing is not None:
                crossing_text = f"never recovers below it after **{fmt(crossing, 1)}s**"
            elif np.isfinite(share) and share == 0.0:
                crossing_text = "never beats it at any horizon"
            else:
                crossing_text = (
                    f"is still below it at the end of the {meta['geometry'].horizon_seconds:.0f}s horizon"
                )
            lines.append(
                f"- **{name}**: first step (+{meta['geometry'].step_seconds:.2f}s) normalized "
                f"**{fmt(meta.get('first_step_norm'))}**, horizon average **{fmt(meta['mean_norm'])}**; "
                f"beats the persistence baseline on **{100.0 * share:.0f}%** of predicted frames and "
                f"{crossing_text}."
            )
        collapse_notes = []
        for name in ranked:
            dispersion = model_meta[name].get("mean_dispersion_ratio", np.nan)
            if not np.isfinite(dispersion):
                continue
            note = f"**{name}** retains {fmt(dispersion)}x the ground truth's cross-clip spread"
            spatial = model_meta[name].get("mean_spatial_ratio", np.nan)
            if np.isfinite(spatial):
                # Undefined for a one-patch-per-frame model, which has no
                # within-frame spatial structure to begin with.
                note += f" and {fmt(spatial)}x its within-frame spatial spread"
            collapse_notes.append(note)
        if collapse_notes:
            lines.append(
                "- Collapse check (averaged over the horizon): "
                + "; ".join(collapse_notes)
                + ". A value near 0 means the prediction has become the same vector regardless of input, "
                "which would make the raw distances above meaningless on their own."
            )
        token_std_notes = []
        for name in ranked:
            meta = model_meta[name]
            first = meta.get("first_pred_token_std", np.nan)
            last = meta.get("last_pred_token_std", np.nan)
            if not np.isfinite(first) or not np.isfinite(last):
                continue
            token_std_notes.append(
                f"**{name}** {fmt(first, 4)} at step 1 -> {fmt(last, 4)} at step "
                f"{meta['geometry'].num_steps}"
            )
        if token_std_notes:
            lines.append(
                "- Predictor output scale (std across channels per token): "
                + "; ".join(token_std_notes)
                + ". The predictor is trained to emit layer-normalized targets, so **1.0 is the "
                "expected value**. Drift toward 0 is the prediction flattening into a constant token. "
                "It also decides whether the frozen `rescale` anchor is a live choice: the anchor is "
                "computed once from the first real context window, and recomputing it from the current "
                "window each step would be equivalent *only* if this stays at 1.0 - the real context is "
                f"evicted after {model_meta[ranked[0]]['geometry'].context_tokens} step(s), past which a "
                "recomputed anchor compounds any offset geometrically while a frozen one cannot."
            )
        lines.append("")

    # -- Headline
    if ranked:
        if charts.get("normalized") is not None:
            rel = Path(charts["normalized"]).relative_to(output_dir).as_posix()
            lines.append(f"![Normalized error over rollout time]({rel})")
            lines.append("")

        lines.append("## Headline: horizon-averaged metrics")
        lines.append("")
        lines.append(
            "Averaged over every predicted frame of the rollout. `Normalized` is the headline; the raw "
            "columns are shown so it is clear the normalization is not hiding anything."
        )
        lines.append("")
        lines.append(
            "| Model | Steps | Step size (s) | Horizon (s) | **Normalized "
            + METRIC_LABELS[primary]
            + "** | Raw rollout | Persistence | Cross-clip ratio | Spatial ratio | Clips |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name in ranked:
            meta = model_meta[name]
            geom = meta["geometry"]
            lines.append(
                f"| {name} | {geom.num_steps} | {geom.step_seconds:.2f} | {geom.horizon_seconds:.0f} | "
                f"**{fmt(meta['mean_norm'])}** | {fmt(meta['mean_pred'], 4)} | {fmt(meta['mean_ref'], 4)} | "
                f"{fmt(meta['mean_dispersion_ratio'])} | {fmt(meta['mean_spatial_ratio'])} | "
                f"{meta['num_clips']} |"
            )
        lines.append("")

        lines.append("### The same metric at fixed horizons")
        lines.append("")
        lines.append(
            "Models advance by different amounts per step (`tubelet_size / fps`), so each column shows "
            "the step whose lead time is closest to the requested horizon."
        )
        lines.append("")
        horizons = sorted(
            {row["horizon_s"] for meta in model_meta.values() for row in meta.get("horizon_rows", [])}
        )
        if horizons:
            lines.append("| Model | " + " | ".join(f"+{h:.0f}s" for h in horizons) + " |")
            lines.append("| --- | " + " | ".join(["---"] * len(horizons)) + " |")
            for name in ranked:
                by_horizon = {row["horizon_s"]: row for row in model_meta[name].get("horizon_rows", [])}
                cells = []
                for horizon in horizons:
                    row = by_horizon.get(horizon)
                    cells.append("-" if row is None else fmt(row["norm"]))
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            lines.append("")

        if charts.get("raw") is not None:
            rel = Path(charts["raw"]).relative_to(output_dir).as_posix()
            lines.append(f"![Raw distance vs persistence baseline]({rel})")
            lines.append("")
        if charts.get("collapse") is not None:
            rel = Path(charts["collapse"]).relative_to(output_dir).as_posix()
            lines.append("### Collapse diagnostics")
            lines.append("")
            lines.append(
                "Both panels are ratios against the ground truth, so 1.0 means the prediction is as "
                "structured as reality and 0.0 means it has degenerated to a constant. The left panel "
                "asks whether a predicted frame still distinguishes its own patches; the right asks "
                "whether predictions for *different clips* are still different from each other."
            )
            lines.append("")
            lines.append(f"![Collapse diagnostics]({rel})")
            lines.append("")

    # -- Per-frame evolution
    if ranked:
        lines.append("## Metric per predicted frame")
        lines.append("")
        lines.append(
            "The full degradation curve, one row per autoregressive step, averaged over clips. "
            "`Spatial detail` and `Cross-clip` are the two collapse ratios described above. "
            "`Token std` is the std across channels of a predicted token, which should be 1.0 for a "
            "predictor emitting layer-normalized targets."
        )
        lines.append("")
        for name in ranked:
            frame = step_frames[name]
            geom = model_meta[name]["geometry"]
            lines.append(
                f"### {name} - {geom.num_steps} steps of {geom.step_seconds:.2f}s "
                f"({METRIC_LABELS[primary]})"
            )
            lines.append("")
            lines.append(
                f"<details><summary>All {len(frame)} predicted frames</summary>"
            )
            lines.append("")
            lines.extend(per_step_table(frame, primary))
            lines.append("")
            lines.append("</details>")
            lines.append("")

        lines.append("### All three distance metrics, horizon-averaged")
        lines.append("")
        lines.append("| Model | " + " | ".join(f"{METRIC_LABELS[m]} (norm.)" for m in METRICS) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(METRICS)) + " |")
        for name in ranked:
            frame = step_frames[name]
            cells = [fmt(float(frame[f"{m}_norm"].mean())) for m in METRICS]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")

    # -- Setup
    lines.append("## Setup")
    lines.append("")
    lines.append("### Protocol")
    lines.append("")
    lines.append(
        "1. **All three modules come from the training checkpoint.** The context `encoder`, the "
        "`predictor` and the EMA `target_encoder` are rebuilt by the same `init_video_model` call "
        "`app/vjepa/train.py` uses, with every switch (`is_causal`, `use_rope`, `tubelet_size`, "
        "`patch_size`, `crop_size`, `pred_depth`, ...) read from the model's own pretraining config. "
        "Nothing is trained or fine-tuned here."
    )
    lines.append(
        "2. **One temporal token per step, never more.** A step predicts exactly one tubelet "
        "(`tubelet_size` frames) - the single-frame-ahead problem the model was trained on. The "
        "predictor always sees a window of `dataset_fpcs / tubelet_size - 1` temporal tokens: the "
        "prediction is appended and the oldest token dropped, so every RoPE position stays inside the "
        "range seen during pretraining."
    )
    if any(meta.get("kind") == "goal_world_model" for meta in model_meta.values()):
        lines.append(
            "2b. **The goal-guided world model steps by a *chunk*, and is rolled out with its goal "
            "withheld.** Its encoder summarizes `tokens_per_chunk` V-JEPA temporal tokens (4 x 2 = 8 "
            "frames = 2s at 4 fps) into one latent, so a chunk occupies one temporal slot of `S` tokens "
            "exactly as a tubelet embedding does and everything above carries over with `tubelet_size` "
            "read as *frames per temporal unit*. A step advances 2s rather than 0.5s, so the same "
            "horizon takes four times fewer steps - a property of the architecture, which is why every "
            "curve here is indexed by lead time in seconds rather than by step. During training its "
            "predictor is also conditioned on the latent of a chunk 4-16s further ahead (the goal); "
            "**that input is withheld here**, using the learned null goal that "
            "`world_model.goal_drop_prob` trains, so it rolls out on the past alone like every other "
            "model. Telling it where the video ends up would make its problem strictly easier and the "
            "comparison meaningless. What this measures is whether training *with* a goal produced a "
            "better world model - not whether a goal helps at inference, which this report does not ask."
        )
    if any(meta.get("privileged") for meta in model_meta.values()):
        lines.append(
            "2c. **A second, PRIVILEGED entry hands that goal back.** The `goal Ns ahead` variant is the "
            "same weights and the same clips, but at every step it is given the target-encoder latent of "
            "the chunk `N` seconds past the end of the context it is conditioning on - so the goal slides "
            "forward with the rollout and stays at the fixed lead the model was trained on. Being a real "
            "chunk of the video ahead of the rollout's frontier, it is **information no other model here "
            "has**, which is why this entry is a ceiling rather than a measurement. Read it only against "
            "its own goal-withheld twin: the gap between the two is what the goal is worth to this model, "
            "and it is the only pair in the report that differs by exactly one thing."
        )
    lines.append(
        "3. **Masks follow the model's own interface.** For a non-causal model the context mask covers "
        "every patch of every temporal token but the last and the prediction mask covers exactly the "
        "patches of the last temporal token (patch indices are `t * S + s`, so that is the contiguous "
        "block `[n_ctx*S, N_t*S)`). A causal model gets no mask at all: like an LLM it emits a "
        "next-token prediction at every position and the last `S` outputs are the next frame - the same "
        "outputs the training loss reads."
    )
    lines.append(
        "4. **Ground truth is the training target.** Each real frame is embedded by the target encoder "
        "*inside its own full-length window* and layer-normalized, exactly as `forward_target` does in "
        "the training loop. The persistence reference is the same quantity for the last frame the model "
        "actually observed."
    )
    lines.append(
        f"5. **Prediction feedback: `{args.feedback}`.** The predictor is trained to *output* "
        "layer-normalized target features but to *consume* raw context-encoder features, so an "
        "autoregressive rollout has to bridge the two spaces. Layer norm without an affine is "
        "`y = (x - mu) / sigma` over channels, so its exact inverse is a *per-token scalar affine* - "
        "which means every `rescale` variant has the right functional form and differs only in where "
        "it gets `mu` and `sigma`:"
    )
    lines.append("")
    lines.append(
        "   - `rescale` (default) - the average per-token `mu`/`sigma` of the **first real context "
        "window**, frozen for the whole rollout. Keeps the predictor's *input* distribution correct "
        "and cannot drift, because the same constant is re-applied every step."
    )
    lines.append(
        "   - `rescale_running` - the same statistics, but recomputed from the **current window** at "
        "every step. Leak-free, and identical to `rescale` while the window still holds real tokens. "
        "Past that - the real context is fully evicted after `context_tokens` steps - the statistics "
        "come entirely from previously rescaled predictions, so any systematic offset in the "
        "predictor's output scale compounds geometrically. The `Token std` column is what decides "
        "whether that matters: at exactly 1.0 the two modes coincide."
    )
    lines.append(
        "   - `oracle_rescale` - **sees the future.** Uses the true per-token `mu`/`sigma` of the "
        "un-normalized target-encoder embedding of the frame being predicted, i.e. the exact inverse "
        "of the training loss's layer norm. With a perfect predictor this reproduces the real "
        "embedding and the rollout degenerates to teacher forcing, which is what makes it an upper "
        "bound rather than a measurement."
    )
    lines.append(
        "   - `raw` - predictions fed back untouched. `layer_norm` - the *observed* context is "
        "normalized too, moving the whole rollout into the predictor's output space."
    )
    lines.append("")
    lines.append(
        "   Re-run with `--feedback` to check that a conclusion does not hinge on this choice."
    )
    lines.append("")

    lines.append("### Per-model geometry")
    lines.append("")
    lines.append(
        "Frame spacing is not uniform for every model. A model post-trained from a checkpoint that was "
        "pretrained at a higher frame rate (`meta.use_pretrained_model`) sees **two timescales at once**: "
        "the frames inside one tubelet stay at the backbone's original `previous_fps`, while consecutive "
        "tubelets are spaced at the new, coarser `fps`. The data pipeline builds this by decoding at "
        "`previous_fps` and keeping only the leading `tubelet_size` frames of every `frames_to_skip` "
        "(`SimpleCollator`), and this report reproduces it frame for frame. An autoregressive step "
        "advances by one *tubelet start*, so it is the between-tubelet column that sets the step size."
    )
    lines.append("")
    lines.append(
        "| Model | Config | Causal | Post-trained | Decode fps | Tubelet | Within-tubelet gap (s) | "
        "Between-tubelet step (s) | Spatial tokens | Context window | Trained window | Steps | "
        "Source frames |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for name, meta in model_meta.items():
        geom = meta.get("geometry")
        if geom is None:
            continue
        intra = f"{geom.intra_tubelet_seconds:.2f}" if geom.tubelet_size > 1 else "n/a (1 frame)"
        if meta.get("kind") == "goal_world_model":
            name = f"{name} \u2020"
        if meta.get("privileged"):
            name = f"{name} \u2021"
        lines.append(
            f"| {name} | `{meta['config']}` | {'yes' if geom.is_causal else 'no'} | "
            f"{'yes' if geom.uses_pretrained_backbone else 'no'} | {geom.sampling_fps:g} | "
            f"{geom.tubelet_size} frame(s) | {intra} | {geom.step_seconds:.2f} | "
            f"{geom.spatial_tokens} | {geom.context_tokens} tok / {geom.context_seconds:.2f}s | "
            f"{geom.window_tokens} tok / {geom.window_seconds:.2f}s | {geom.num_steps} | "
            f"{geom.total_sampled_frames} @ {geom.sampling_fps:g} fps -> {geom.total_frames} kept |"
        )
    lines.append("")
    lines.append("| Model | Checkpoint | Epoch | Patch | Crop |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, meta in model_meta.items():
        geom = meta.get("geometry")
        if geom is None:
            continue
        lines.append(
            f"| {name} | `{meta['checkpoint']}` | {meta.get('epoch', 'n/a')} | {geom.patch_size} | "
            f"{geom.crop_size} |"
        )
    lines.append("")
    if any(meta.get("kind") == "goal_world_model" for meta in model_meta.values()):
        lines.append(
            "\u2020 Its temporal unit is a **chunk** of 4 V-JEPA tokens, not a single tubelet, so the "
            "`Tubelet` column reads 8 frames and the within-tubelet gap is the spacing inside a whole "
            "chunk. Whether its goal is withheld or supplied is in the entry's own name; only a row "
            "also marked \u2021 was given one."
        )
        lines.append("")
    privileged = [
        (name, meta) for name, meta in model_meta.items() if meta.get("privileged")
    ]
    if privileged:
        for name, meta in privileged:
            geom = meta["geometry"]
            lines.append(
                f"\u2021 **{name} is privileged and is not a like-for-like measurement.** Its goal is "
                f"supplied at every step, held {meta['goal_lead_seconds']:g}s ahead of the context, which "
                f"means {geom.extra_tokens * geom.frames_per_token} extra frame(s) of real video are read "
                f"past the end of the rollout - {geom.total_frames} frames decoded in total against "
                f"{geom.window_tokens + geom.num_steps} tokens' worth of rollout. Compare it with its "
                "goal-withheld twin, not with the other models."
            )
        lines.append("")
    lines.append(
        "> The step size fixes how many steps a fixed ~1-minute horizon takes, and it varies a lot "
        "between models (0.5s/step needs 120 steps, 2s/step needs 30). The per-frame tables are indexed "
        "by *lead time in seconds* for that reason, and the step count is reported next to every average."
    )
    lines.append("")

    lines.append("### Data")
    lines.append("")
    lines.append(f"- Manifest: `{manifest_info['csv']}` ({manifest_info['num_videos_available']} videos)")
    lines.append(
        f"- {manifest_info['num_videos_selected']} video(s) selected for {args.num_clips} requested clip "
        f"slot(s): the manifest is shuffled with seed {args.seed} and walked until enough videos are long "
        f"enough for **every** model's rollout. One clip per video, starting at a seeded random fraction "
        f"into the video."
    )
    lines.append(
        "- Selection deliberately requires *all* models to fit, rather than filtering per model. A model "
        "with a lower `fps` needs more footage for the same horizon (60s at 0.5 fps spans 368 frames of a "
        "4 fps video; at 4 fps it spans 256), so per-model filtering would quietly score the models on "
        "different subsets. The cost is that videos only the cheapest model could have used are dropped."
    )
    lines.append(
        "- Every model therefore sees the *same* videos, in the same order, entered at the *same* "
        "fraction of their length. The absolute start frame still differs slightly between models, "
        "because they consume different amounts of footage from that point on."
    )
    for name, meta in model_meta.items():
        if "num_clips" not in meta:
            continue
        lines.append(
            f"- **{name}**: {meta['num_clips']} clip(s) rolled out"
            + (f", {meta['num_clips_failed']} failed" if meta.get("num_clips_failed") else "")
            + f", {meta.get('elapsed_minutes', float('nan')):.1f} min"
        )
    lines.append("")
    if skipped_models:
        lines.append("Models declared in `MODELS_CONFIG` but skipped:")
        lines.append("")
        for name, reason in skipped_models:
            lines.append(f"- **{name}** - {reason}")
        lines.append("")

    # -- Caveats
    lines.append("## How to read these numbers")
    lines.append("")
    lines.append(
        "- **1.0 is the bar, not 0.** The persistence baseline is strong on egocentric kitchen video: "
        "over half a second little changes. A normalized value of 0.9 is a real but modest win; a value "
        "that climbs past 1.0 means the rollout has drifted somewhere worse than standing still."
    )
    lines.append(
        "- **Normalization is what makes the comparison legitimate.** Raw distances depend on the scale "
        "and effective dimensionality of each model's embedding space, which differ between "
        "checkpoints. Dividing by the persistence baseline - measured in the *same* space, against the "
        "*same* ground-truth frame - cancels that scale. It does not, on its own, rule out collapse: "
        "that is what the two dispersion ratios are for."
    )
    lines.append(
        "- **The mask pattern the non-causal model was trained on is not future prediction.** In the "
        "V-JEPA mask configs used here every prediction block has `temporal_scale: [1.0, 1.0]`, so the "
        "masked tubes span the *whole* temporal extent of the clip: the model was trained to inpaint "
        "space across all frames, never to extrapolate forward in time. Asking it for the last temporal "
        "token given the earlier ones is therefore out of distribution, and a weak result here is "
        "evidence about the pretraining task, not only about the encoder."
    )
    lines.append(
        "- **The feedback bridge is a real assumption.** No rollout of a JEPA predictor can be exactly "
        "faithful, because the predictor's output space (layer-normalized target features) is not its "
        "input space (raw context features). `--feedback` exposes the reasonable choices; conclusions "
        "that flip between them are conclusions about the bridge, not about the model. The bridge can "
        "be bounded rather than argued about: `--feedback oracle_rescale` supplies the exact scale of "
        "the frame being predicted (an oracle - it reads the future), so the gap between it and "
        "`rescale` is the entire cost of not knowing that scale. If the gap is small, the bridge is "
        "not the bottleneck and these numbers are about the model."
    )
    lines.append(
        "- **Only the first step is free of drift.** Every later step conditions on the model's own "
        "output, so a rising curve mixes two different things: the frame genuinely got harder to "
        "predict, and the rollout has drifted away from the real video. Run with `--teacher-forcing` to "
        "get the single-step curve alongside and separate the two."
        if not args.teacher_forcing
        else "- **Rollout vs. teacher-forced.** The teacher-forced curve re-predicts every step from the "
        "*real* video window, so the gap between the two curves is exactly the cost of compounding "
        "error, while the teacher-forced curve alone shows how hard each frame is one step out."
    )
    lines.append(
        "- **Held-out video, but the same domain.** The default manifest is the EK55 test split, which "
        "appears in none of the pretraining manifests, so no rollout is scored on a video the model was "
        "trained on. It is still egocentric kitchen footage from the same corpus, so this measures "
        "generalization to unseen video, not to an unseen domain."
    )
    lines.append(
        "- **Long videos only.** Requiring room for a full rollout selects against short recordings, so "
        "the clip set skews toward longer, and therefore possibly less eventful, footage. This biases the "
        "persistence baseline and the model identically, so the normalized numbers are unaffected, but it "
        "is worth remembering when reading the raw columns."
    )
    lines.append("")

    # -- Artifacts
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Per-clip, per-step measurements: `{(output_dir / 'world_model_steps.csv').name}`")
    lines.append(f"- Per-step averages used by the tables/charts: `{(output_dir / 'per_step_metrics.csv').name}`")
    lines.append("- Per-model pooled frame embeddings: `<model-slug>/{pred,gt}_pooled.npy` (clips x steps x dim)")
    lines.append("- Sampled clip list: `sampled_manifest.csv`")
    lines.append(f"- Run log: `{(output_dir / 'world_model_report.log').name}`")
    lines.append("")
    lines.append(
        f"Reproduce with `python evals/generate_world_model_report.py --dataset-csv {args.dataset_csv} "
        f"--num-clips {args.num_clips} --horizon-seconds {args.horizon_seconds:g} "
        f"--feedback {args.feedback} --seed {args.seed}`."
    )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an autoregressive world-model report for every model in MODELS_CONFIG"
    )
    parser.add_argument(
        "--dataset-csv",
        default="data/ek55_4fps_test.csv",
        help=(
            "Whitespace-separated manifest of *long* videos (video path in the first column). Videos "
            "shorter than the context window plus --horizon-seconds at a model's fps are skipped, so "
            "the manifest does not have to be pre-filtered. The default is the held-out EK55 test split "
            "(data/ek55_processed/test), which appears in no pretraining manifest."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="preliminary_experiments/evals/vitl/vjepa_ek100_world_model",
        help="Directory where the report artifacts will be saved.",
    )
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=60.0,
        help="How far ahead to roll out, in seconds of video time.",
    )
    parser.add_argument("--num-clips", type=int, default=16, help="Number of clips to roll out per model.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help=(
            "Clips rolled out simultaneously. Raise for speed if GPU memory allows: the ground truth of "
            "a whole batch is held on the device, costing batch_size * (steps+1) * spatial_tokens * "
            "embed_dim floats (~125 MB per clip for a 60s rollout of a 256-token/frame model)."
        ),
    )
    parser.add_argument(
        "--feedback",
        choices=FEEDBACK_MODES,
        default="rescale",
        help=(
            "How a prediction is mapped back into the predictor's input space. `rescale` (default) "
            "restores the scale of the first real context window, `raw` feeds predictions untouched, "
            "`layer_norm` moves the whole rollout into the predictor's output space, `rescale_running` "
            "recomputes the scale from the current window every step (leak-free; diverges from "
            "`rescale` only once the real context has been evicted). `oracle_rescale` restores the true "
            "per-token scale of the frame being predicted - it SEES THE FUTURE and is a diagnostic "
            "ceiling, not a measurement; use it to test whether the bridge is what limits the horizon."
        ),
    )
    parser.add_argument(
        "--primary-metric",
        choices=METRICS,
        default="l1",
        help="Metric used for the headline tables and per-frame curves (all three are always recorded).",
    )
    parser.add_argument(
        "--teacher-forcing",
        action="store_true",
        help=(
            "Also predict every step from the *real* video window, to separate compounding rollout "
            "drift from intrinsic per-frame difficulty. Roughly doubles the runtime."
        ),
    )
    parser.add_argument(
        "--mask-token-index",
        type=int,
        default=0,
        help="Which of the predictor's mask tokens to use (training uses the dataset index; 0 is safe).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for video and start-offset sampling.")
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        default=True,
        help=(
            "Run the rollout in fp32 instead of the config's `meta.dtype`. Slower, but a useful check "
            "that a long rollout is not being shaped by reduced precision."
        ),
    )
    return parser.parse_args()


def autocast_config(config: dict, device: str, use_amp: bool) -> dict:
    """Match the dtype the model was trained in, unless AMP is disabled."""
    if not use_amp or not device.startswith("cuda"):
        return {"device_type": "cuda" if device.startswith("cuda") else "cpu", "enabled": False}
    which = str(config.get("meta", {}).get("dtype", "float32")).lower()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(which)
    if dtype is None:
        return {"device_type": "cuda", "enabled": False}
    return {"device_type": "cuda", "dtype": dtype, "enabled": True}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(str(output_dir / "world_model_report.log"))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    logger.info("Feedback mode: %s", args.feedback)
    if args.feedback in ORACLE_FEEDBACK_MODES:
        logger.warning(
            "ORACLE RUN: --feedback %s rescales each prediction with the true statistics of the frame "
            "being predicted, so the rollout sees the future it is scored against. Results are a "
            "diagnostic ceiling to compare against a `rescale` run, not model performance.",
            args.feedback,
        )
    if args.feedback == "rescale_running":
        logger.info(
            "rescale_running: the feedback scale is recomputed from the current window each step, so "
            "it is identical to `rescale` until the real context is evicted and self-referential "
            "afterwards. Watch the `feedback_std` column for geometric drift."
        )

    all_rows: List[dict] = []
    model_meta: Dict[str, dict] = {}
    step_frames: Dict[str, pd.DataFrame] = {}
    skipped_models: List[Tuple[str, str]] = []

    # -- Plan every model before touching any video: clip selection has to satisfy
    # all of them at once (see select_videos), which needs every geometry up front.
    # Only configs are read here, no checkpoints, so this is cheap.
    planned: List[dict] = []
    geometries: Dict[str, RolloutGeometry] = {}
    for model_cfg in ALL_MODELS_CONFIG:
        model_name = model_cfg["name"]
        try:
            config = load_config(model_cfg["config"])
            geom = build_geometry_for(model_cfg, config, args.horizon_seconds)
            checkpoint_path = resolve_checkpoint(model_cfg, config)
        except Exception as exc:
            logger.error("Failed to plan %s: %s", model_name, exc)
            skipped_models.append((model_name, f"could not be loaded ({exc})"))
            continue
        planned.append(
            {"cfg": model_cfg, "config": config, "geometry": geom, "checkpoint": checkpoint_path}
        )
        geometries[model_name] = geom

    if not planned:
        logger.error("No model could be planned - nothing to evaluate")
        return

    manifest = load_manifest(args.dataset_csv)
    # Shuffle the whole manifest once with the run seed, then walk it until enough
    # videos clear every model's length requirement. Every model then gets the
    # same videos in the same order, entered at the same fraction of their length.
    shuffled = sample_rows(manifest, len(manifest), args.seed)
    videos = select_videos(shuffled, geometries, args.num_clips)
    if not videos:
        strictest = max(planned, key=lambda p: p["geometry"].total_tokens)["geometry"]
        logger.error(
            "No video in %s is long enough for a %.0fs rollout by every model (the strictest needs "
            "%d sampled frames at %g fps)",
            args.dataset_csv,
            args.horizon_seconds,
            strictest.total_frames,
            strictest.fps,
        )
        return

    pd.DataFrame({"video_path": [path for _, path in videos], "label": "no-label"}).to_csv(
        output_dir / "sampled_manifest.csv", sep=" ", header=False, index=False
    )
    manifest_info = {
        "csv": args.dataset_csv,
        "num_videos_available": int(len(manifest)),
        "num_videos_selected": len(videos),
    }

    for entry in planned:
        model_cfg = entry["cfg"]
        config = entry["config"]
        geom = entry["geometry"]
        checkpoint_path = entry["checkpoint"]
        model_name = model_cfg["name"]
        model_slug = slugify(model_name)
        logger.info("%s", "=" * 80)
        logger.info("Processing model: %s", model_name)
        logger.info("%s", "=" * 80)

        try:
            bundle = prepare_bundle(model_cfg, config, checkpoint_path, device)
        except Exception as exc:
            logger.error("Failed to load %s: %s", model_name, exc)
            skipped_models.append((model_name, f"could not be loaded ({exc})"))
            continue

        if model_cfg.get("goal_lead_seconds") is not None:
            logger.warning(
                "PRIVILEGED RUN: %s is rolled out with its goal supplied, held %.1fs ahead of the "
                "context at every step. That reads real video %d frame(s) past the rollout's "
                "frontier, which no other model here gets, so its numbers are a ceiling to compare "
                "against the goal-withheld entry - not a like-for-like measurement.",
                model_name,
                float(model_cfg["goal_lead_seconds"]),
                geom.extra_tokens * geom.frames_per_token,
            )
        logger.info(
            "%s: %s, %.3f fps, tubelet %d, %d spatial tokens, %d context tokens (%.1fs), "
            "%d steps of %.2fs = %.0fs horizon",
            model_name,
            "causal" if geom.is_causal else "non-causal",
            geom.fps,
            geom.tubelet_size,
            geom.spatial_tokens,
            geom.context_tokens,
            geom.context_seconds,
            geom.num_steps,
            geom.step_seconds,
            geom.horizon_seconds,
        )

        model_dir = output_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)
        model_meta[model_name] = {
            "config": model_cfg["config"],
            "checkpoint": checkpoint_path,
            "epoch": bundle.epoch,
            "geometry": geom,
            "kind": model_cfg.get("kind", "vjepa"),
            "goal_lead_seconds": model_cfg.get("goal_lead_seconds"),
            "privileged": model_cfg.get("goal_lead_seconds") is not None,
        }

        try:
            transform = make_transforms(
                training=False,
                num_views_per_clip=1,
                crop_size=geom.crop_size,
                normalize=config["data"].get("normalization", DEFAULT_NORMALIZATION),
            )
            batches = iter_clip_batches(videos, geom, transform, args.batch_size, args.seed)
            if not batches:
                raise RuntimeError(
                    f"none of the {len(videos)} selected videos could be decoded at {geom.fps:g} fps"
                )
            logger.info(
                "%s: decoded %d clip(s) into %d batch(es)",
                model_name,
                sum(len(b.clip_ids) for b in batches),
                len(batches),
            )

            result = rollout_clips(
                bundle,
                geom,
                batches,
                device=device,
                feedback_mode=args.feedback,
                mask_token_index=args.mask_token_index,
                autocast_kwargs=autocast_config(config, device, args.use_amp),
                teacher_forcing=args.teacher_forcing,
                model_name=model_name,
            )
        except Exception as exc:
            logger.error("Rollout failed for %s: %s", model_name, exc)
            skipped_models.append((model_name, f"rollout failed ({exc})"))
            model_meta.pop(model_name, None)
            continue
        finally:
            del bundle
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        if not result.rows:
            logger.error("No usable rollout steps for %s", model_name)
            skipped_models.append((model_name, "produced no usable rollout steps"))
            model_meta.pop(model_name, None)
            continue

        model_df = add_normalized_columns(pd.DataFrame(result.rows))
        all_rows.extend(model_df.to_dict("records"))
        frame = per_step_frame(model_df, result, geom)
        step_frames[model_name] = frame

        if result.pred_pooled:
            np.save(model_dir / "pred_pooled.npy", np.stack(result.pred_pooled, axis=0))
            np.save(model_dir / "gt_pooled.npy", np.stack(result.gt_pooled, axis=0))
        frame.to_csv(model_dir / "per_step_metrics.csv", index=False)
        (model_dir / "geometry.json").write_text(
            json.dumps(
                {
                    "fps": geom.fps,
                    "tubelet_size": geom.tubelet_size,
                    "patch_size": geom.patch_size,
                    "crop_size": geom.crop_size,
                    "spatial_tokens": geom.spatial_tokens,
                    "window_tokens": geom.window_tokens,
                    "context_tokens": geom.context_tokens,
                    "num_steps": geom.num_steps,
                    "step_seconds": geom.step_seconds,
                    "horizon_seconds": geom.horizon_seconds,
                    "is_causal": geom.is_causal,
                    "kind": model_cfg.get("kind", "vjepa"),
                    "goal_conditioning": goal_conditioning_label(model_cfg),
                    "goal_lead_seconds": model_cfg.get("goal_lead_seconds"),
                    "reads_future_beyond_rollout": model_cfg.get("goal_lead_seconds") is not None,
                    "extra_tokens": geom.extra_tokens,
                    "feedback": args.feedback,
                    "feedback_is_oracle": args.feedback in ORACLE_FEEDBACK_MODES,
                    "checkpoint": checkpoint_path,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        primary = args.primary_metric
        model_meta[model_name].update(
            {
                "num_clips": int(model_df["clip_id"].nunique()),
                "num_clips_failed": result.num_clips_failed,
                "elapsed_minutes": result.elapsed_s / 60.0,
                "mean_norm": float(frame[f"{primary}_norm"].mean()),
                "mean_pred": float(frame[f"{primary}_pred"].mean()),
                "mean_ref": float(frame[f"{primary}_ref"].mean()),
                "first_step_norm": float(frame.iloc[0][f"{primary}_norm"]),
                "mean_dispersion_ratio": float(frame["dispersion_ratio"].mean())
                if "dispersion_ratio" in frame.columns
                else float("nan"),
                "mean_spatial_ratio": float(frame["spatial_std_ratio"].mean()),
                "mean_pred_token_std": float(frame["pred_token_std"].mean()),
                "first_pred_token_std": float(frame.iloc[0]["pred_token_std"]),
                "last_pred_token_std": float(frame.iloc[-1]["pred_token_std"]),
                "first_feedback_std": float(frame.iloc[0]["feedback_std"]),
                "last_feedback_std": float(frame.iloc[-1]["feedback_std"]),
                "horizon_rows": horizon_rows(frame, geom, primary),
            }
        )
        crossing_s, share_below = baseline_comparison(frame, primary)
        model_meta[model_name]["crossing_s"] = crossing_s
        model_meta[model_name]["share_below_baseline"] = share_below
        logger.info(
            "%s: normalized %s = %.3f (first step %.3f, horizon average over %d steps), "
            "cross-clip dispersion ratio %.3f",
            model_name,
            primary,
            model_meta[model_name]["mean_norm"],
            model_meta[model_name]["first_step_norm"],
            geom.num_steps,
            model_meta[model_name]["mean_dispersion_ratio"],
        )
        logger.info(
            "%s: predictor output scale (std across channels, 1.0 = exactly layer-normalized) "
            "%.4f at step 1 -> %.4f at step %d, mean %.4f. The real context is evicted after %d "
            "step(s), so a per-step-recomputed rescale anchor would compound this offset from there on; "
            "the frozen anchor does not.",
            model_name,
            model_meta[model_name]["first_pred_token_std"],
            model_meta[model_name]["last_pred_token_std"],
            geom.num_steps,
            model_meta[model_name]["mean_pred_token_std"],
            geom.context_tokens,
        )
        logger.info(
            "%s: feedback scale (%s) %.4f at step 1 -> %.4f at step %d",
            model_name,
            args.feedback,
            model_meta[model_name]["first_feedback_std"],
            model_meta[model_name]["last_feedback_std"],
            geom.num_steps,
        )

    steps_df = pd.DataFrame(all_rows)
    steps_df.to_csv(output_dir / "world_model_steps.csv", index=False)
    if step_frames:
        pd.concat(
            [frame.assign(model=name) for name, frame in step_frames.items()], ignore_index=True
        ).to_csv(output_dir / "per_step_metrics.csv", index=False)

    charts: Dict[str, Optional[Path]] = {"normalized": None, "raw": None, "collapse": None}
    usable = {name: frame for name, frame in step_frames.items() if not frame.empty}
    if usable:
        charts["normalized"] = output_dir / "normalized_error_vs_time.png"
        save_normalized_chart(usable, charts["normalized"])
        charts["raw"] = output_dir / "raw_error_vs_time.png"
        save_raw_chart(usable, args.primary_metric, charts["raw"])
        charts["collapse"] = output_dir / "collapse_diagnostics.png"
        save_collapse_chart(usable, charts["collapse"])

    report_path = output_dir / "world_model_report.md"
    generate_markdown_report(
        report_path=report_path,
        output_dir=output_dir,
        model_meta=model_meta,
        step_frames=step_frames,
        charts=charts,
        skipped_models=skipped_models,
        manifest_info=manifest_info,
        args=args,
    )
    logger.info("Markdown report saved to %s", report_path)


if __name__ == "__main__":
    main()
