# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import src.models.predictor as vit_pred
import src.models.vision_transformer as video_vit
from app.vjepa.utils import load_module_state_dict
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# Parameters of the sub-goal cross-attention. They do not exist in a checkpoint of an
# un-guided predictor, so when the predictor is warm-started from a plain V-JEPA 2 one
# (the usual case) they are legitimately absent and get freshly initialized. Resuming
# this run's own checkpoint has all of them, so nothing is ever silently dropped.
GUIDANCE_KEY = re.compile(r"(.*\.)?(xattn\..*|norm_xattn\..*|gamma_xattn|guidance_norm\..*|guidance_proj\..*)$")


def _strip_ddp_prefix(state_dict):
    return {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def _clean_encoder_keys(state_dict, drop_pos_embed):
    """Strip the DDP / `MultiSeqWrapper` prefixes a released V-JEPA 2 checkpoint carries,
    so it loads into a bare `VisionTransformer`."""
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        if drop_pos_embed and k.endswith("pos_embed"):
            # A RoPE encoder has no learned position embedding.
            continue
        cleaned[k] = v
    return cleaned


def init_low_level_model(
    device,
    encoder_checkpoint,
    guidance_dim,
    num_frames,
    encoder_checkpoint_key="target_encoder",
    model_name="vit_large",
    crop_size=256,
    patch_size=16,
    tubelet_size=2,
    pred_depth=12,
    pred_num_heads=12,
    pred_embed_dim=384,
    num_mask_tokens=4,
    guidance_step_ratio=4,
    guidance_context_offset=3,
    guidance_gate_init=0.0,
    guidance_window=None,
    uniform_power=True,
    use_sdpa=True,
    use_rope=True,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=True,
    use_activation_checkpointing=False,
):
    """Build the low-level model: a *frozen* causal V-JEPA 2 encoder and a trainable
    predictor that cross-attends to the high-level model's sub-goals.

    The encoder runs with `is_causal=True`, which is what keeps the representation of
    step `s` free of anything after `s`. Note that the released V-JEPA 2 weights were
    pretrained with full bidirectional attention, so running them causally is a
    distribution shift; point `encoder_checkpoint` at a causally post-trained checkpoint
    (see configs/train/vitl16-EK100/pretrain-causal-4fps-original-vjepa2-ckpt.yaml) when
    one is available.

    Both modules keep the `MultiSeqWrapper` / `PredictorMultiSeqWrapper` plumbing every
    other app uses, so this run's checkpoints stay loadable by the shared tooling.
    """
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=False,
        use_rope=use_rope,
        is_causal=True,
    )
    logger.info(f"Loading frozen low-level encoder from {encoder_checkpoint} (key '{encoder_checkpoint_key}')")
    ckpt = robust_checkpoint_loader(encoder_checkpoint, map_location=torch.device("cpu"))
    state_dict = ckpt[encoder_checkpoint_key] if encoder_checkpoint_key in ckpt else ckpt["encoder"]
    load_module_state_dict(
        encoder,
        _clean_encoder_keys(state_dict, drop_pos_embed=use_rope),
        "low-level encoder",
        ckpt.get("epoch", -1),
    )
    del ckpt
    encoder = MultiSeqWrapper(encoder)

    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=True,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=True,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=True,
        use_guidance=True,
        guidance_dim=guidance_dim,
        guidance_gate_init=guidance_gate_init,
        guidance_step_ratio=guidance_step_ratio,
        guidance_context_offset=guidance_context_offset,
        guidance_window=guidance_window,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    predictor.to(device)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Low-level encoder is frozen ({sum(p.numel() for p in encoder.parameters())} parameters)")
    logger.info(f"Low-level predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def warm_start_predictor(predictor, checkpoint, checkpoint_key="predictor"):
    """Initialize the trainable predictor from an existing V-JEPA 2 predictor.

    The sub-goal cross-attention does not exist in such a checkpoint, so those
    parameters stay freshly initialized; with `guidance_gate_init: 0.0` the model
    therefore starts out as exactly the predictor it was loaded from and has to learn to
    open the gate.
    """
    logger.info(f"Warm-starting the predictor from {checkpoint} (key '{checkpoint_key}')")
    ckpt = robust_checkpoint_loader(checkpoint, map_location=torch.device("cpu"))
    if checkpoint_key not in ckpt:
        raise KeyError(f"{checkpoint} has no '{checkpoint_key}' entry (keys: {sorted(ckpt)})")
    load_module_state_dict(
        predictor,
        _strip_ddp_prefix(ckpt[checkpoint_key]),
        "predictor (warm start)",
        ckpt.get("epoch", -1),
        allow_missing=GUIDANCE_KEY,
    )
    del ckpt
    return predictor


def load_checkpoint(r_path, predictor, opt, scaler):
    """Resume this run. Only the predictor is trained, so the frozen encoder is rebuilt
    from its own checkpoint rather than read back from here."""
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]

    load_module_state_dict(predictor, checkpoint["predictor"], "predictor", epoch, allow_missing=GUIDANCE_KEY)
    try:
        opt.load_state_dict(checkpoint["opt"])
        logger.info(f"loaded optimizers from epoch {epoch}")
    except ValueError as e:
        logger.warning(f"could not load optimizer state from {r_path} ({e}); starting from a fresh optimizer state")
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return predictor, opt, scaler, epoch


@dataclass
class HierarchyLayout:
    """How the two levels line up around a single split time T.

    T is the instant both models start predicting from: the end of the high-level
    context, at frame `context_chunks * frames_per_chunk` of the clip. Everything is
    anchored to it -- the goal sits `goal_seconds` past T, the high-level model rolls
    `guidance_chunks` chunks forward from T, and the low-level window straddles it.

    In low-level steps (one tubelet each, `steps_per_chunk` of them to a chunk):

        chunk C-1        |  chunk C       |  chunk C+1     | ...
        lead-in          |  sub-goal 0    |  sub-goal 1    |
        steps 0 1 2      |  steps 3 4 5 6 |  steps 7 8 9 10|
                         T

    The lead-in chunk is there so the step predicting the first tubelet after T has real
    context behind it; its own steps predict frames before T, which no sub-goal covers,
    so they run unconditioned.
    """

    frames_per_chunk: int
    steps_per_chunk: int
    tubelet_size: int
    context_chunks: int
    guidance_chunks: int
    patches_per_chunk: int
    goal_start_frame: int
    goal_position: float

    @property
    def split_frame(self):
        """T: the frame both models start predicting from."""
        return self.context_chunks * self.frames_per_chunk

    @property
    def low_level_chunks(self):
        return self.guidance_chunks + 1

    @property
    def low_level_frames(self):
        return self.low_level_chunks * self.frames_per_chunk

    @property
    def low_level_start_frame(self):
        """First frame of the low-level window: one chunk of lead-in before T."""
        return self.split_frame - self.frames_per_chunk

    @property
    def clip_frames(self):
        return max(
            self.goal_start_frame + self.frames_per_chunk,
            self.low_level_start_frame + self.low_level_frames,
        )

    @property
    def guidance_context_offset(self):
        """Low-level steps that must pass before sub-goal `c` is safe to read.

        Every sub-goal is rolled out from T, so they all rest on the same observations:
        the frames up to T, which the low-level model reaches at step
        `steps_per_chunk - 1`. Beyond that the offset is not a causality constraint but
        the schedule itself -- combined with `guidance_window = 1` it hands each step
        exactly the sub-goal covering the chunk that step is predicting into, and
        switches to the next one four steps later.
        """
        return self.steps_per_chunk - 1


@torch.no_grad()
def plan_sub_goals(high_level, clip, layout):
    """Roll the frozen high-level model forward from T, and return its plan.

    At T the model has observed `context_chunks` chunks and is given the latent of the
    chunk `goal_seconds` further on -- a fixed goal, as if holding a photograph of where
    this is heading. It then predicts the next chunk, feeds that prediction back as
    though it had been observed, predicts the one after, and so on for
    `guidance_chunks` steps. Every sub-goal therefore rests on the same observations,
    the ones up to T: this is a plan made once, not a running commentary.

    Two details keep the rollout inside what the model was trained on. The context slides
    rather than grows, so the sequence stays `context_chunks` long and no RoPE position
    runs past the trained range. And the goal's position is given relative to the start
    of that sliding window, so it drops by one chunk per step -- the goal does not move,
    but each step of the plan stands one chunk closer to it.

    The predictor emits layer-normalized target-space latents while consuming raw
    chunk-encoder ones, so a prediction fed back has its per-token scale restored from
    the real context first -- the `rescale` bridge of evals/generate_world_model_report.py.

    :return: [B, guidance_chunks * P, high-level embed dim]
    """
    fpc, patches = layout.frames_per_chunk, layout.patches_per_chunk
    context = clip[:, :, : layout.split_frame]
    goal_clip = clip[:, :, layout.goal_start_frame : layout.goal_start_frame + fpc]

    z = high_level.encoder(context)
    goal = high_level.target_encoder(goal_clip)
    goal = F.layer_norm(goal, (goal.size(-1),))

    # Scale of the real chunk latents, frozen at the only window that is entirely real.
    real = z.float()
    mean = real.mean(dim=-1).mean(dim=-1).view(-1, 1, 1)
    std = real.std(dim=-1).mean(dim=-1).view(-1, 1, 1)

    sub_goals = []
    for step in range(layout.guidance_chunks):
        goal_pos = torch.full((clip.size(0),), layout.goal_position - step, device=clip.device)
        prediction = high_level.predictor(z, goal=goal, goal_pos=goal_pos)[:, -patches:]
        prediction = F.layer_norm(prediction, (prediction.size(-1),))
        sub_goals.append(prediction)
        if step + 1 < layout.guidance_chunks:
            fed_back = (prediction.float() * std + mean).to(z.dtype)
            z = torch.cat([z[:, patches:], fed_back], dim=1)
    return torch.cat(sub_goals, dim=1)


@torch.no_grad()
def encode_low_level_window(encoder, clip, layout):
    """Frozen-encoder context and targets for the low-level window.

    One forward serves both roles. The encoder is causal, so a token depends only on the
    tokens before it: the first N-1 tokens of this window are bit-identical to what
    encoding the window-minus-its-last-tubelet would give, which is the context the
    predictor consumes. Targets are the same tokens layer-normalized, as in the loss.

    :return: (context, targets), each [B, (steps - 1) * P, encoder dim]
    """
    patches = layout.patches_per_chunk
    start = layout.low_level_start_frame
    window = clip[:, :, start : start + layout.low_level_frames]
    feats = encoder([window])[0]
    targets = F.layer_norm(feats, (feats.size(-1),))
    # Step s predicts tubelet s+1, so contexts drop the last tubelet and targets the first.
    return feats[:, :-patches], targets[:, patches:]
