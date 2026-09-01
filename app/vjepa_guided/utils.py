# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import sys

import torch

import src.models.predictor as vit_pred
import src.models.vision_transformer as video_vit
from app.vjepa.utils import RECOMPUTED_BUFFERS, MASK_TOKEN_KEY
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# Parameters of the guidance cross-attention. They do not exist in a checkpoint of an
# un-guided predictor, so when this run is initialized from one (the usual case: start
# from the same-fps causal model and only add guidance) they are legitimately absent and
# get freshly initialized. When resuming from a guided checkpoint they are all present,
# so nothing is ever silently dropped.
GUIDANCE_KEY = re.compile(r"(.*\.)?(xattn\..*|norm_xattn\..*|gamma_xattn|guidance_norm\..*|guidance_proj\..*)$")


def _strip_ddp_prefix(state_dict):
    return {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def load_module_state_dict(module, pretrained_dict, tag, epoch, allow_missing=None):
    """Same contract as `app.vjepa.utils.load_module_state_dict`, plus tolerance for keys
    matching `allow_missing` (a compiled regex) being absent from the checkpoint."""
    missing, unexpected = module.load_state_dict(pretrained_dict, strict=False)
    recomputed = [k for k in missing + unexpected if k.endswith(RECOMPUTED_BUFFERS)]
    missing = [k for k in missing if not k.endswith(RECOMPUTED_BUFFERS)]
    unexpected = [k for k in unexpected if not k.endswith(RECOMPUTED_BUFFERS)]

    newly_initialized = []
    if allow_missing is not None:
        newly_initialized = [k for k in missing if allow_missing.match(k)]
        missing = [k for k in missing if k not in newly_initialized]

    surplus_mask_tokens = [k for k in unexpected if MASK_TOKEN_KEY.match(k)]
    unexpected = [k for k in unexpected if k not in surplus_mask_tokens]
    if missing or unexpected:
        raise RuntimeError(
            f"Error(s) in loading state_dict for {tag}: missing keys {missing}, unexpected keys {unexpected}"
        )
    if surplus_mask_tokens:
        logger.warning(f"{tag}: dropped {len(surplus_mask_tokens)} surplus mask token(s) from the checkpoint")
    if newly_initialized:
        logger.warning(
            f"{tag}: {len(newly_initialized)} guidance cross-attention parameter(s) were not in the checkpoint "
            "and are freshly initialized. This is expected when starting from an un-guided causal checkpoint; "
            "it is NOT expected when resuming a guided run."
        )
    logger.info(f"loaded pretrained {tag} from epoch {epoch} ({len(recomputed)} recomputed buffer(s) ignored)")


def load_checkpoint(r_path, encoder, predictor, target_encoder, opt, scaler, is_anneal=False):
    """Like `app.vjepa.utils.load_checkpoint`, but lets the predictor's guidance
    cross-attention parameters be absent from the checkpoint."""
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0 if is_anneal else checkpoint["epoch"]

    load_module_state_dict(encoder, checkpoint["encoder"], "encoder", epoch)
    load_module_state_dict(predictor, checkpoint["predictor"], "predictor", epoch, allow_missing=GUIDANCE_KEY)
    if target_encoder is not None:
        load_module_state_dict(target_encoder, checkpoint["target_encoder"], "target encoder", epoch)

    try:
        opt.load_state_dict(checkpoint["opt"])
        logger.info(f"loaded optimizers from epoch {epoch}")
    except ValueError as e:
        logger.warning(f"could not load optimizer state from {r_path} ({e}); starting from a fresh optimizer state")
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_large",
    crop_size=224,
    pred_depth=12,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    use_activation_checkpointing=False,
    is_causal=True,
    use_guidance=True,
    guidance_dim=None,
    guidance_gate_init=0.0,
    guidance_step_ratio=4,
    guidance_window=None,
):
    """Build the short-horizon (student) encoder + predictor. Identical to
    `app.vjepa.utils.init_video_model` except that the predictor is given a guidance
    cross-attention branch in every block."""
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
        is_causal=is_causal,
    )
    encoder = MultiSeqWrapper(encoder)
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=is_causal,
        use_guidance=use_guidance,
        guidance_dim=guidance_dim,
        guidance_gate_init=guidance_gate_init,
        guidance_step_ratio=guidance_step_ratio,
        guidance_window=guidance_window,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_guidance_model(
    device,
    checkpoint,
    num_frames,
    patch_size=16,
    tubelet_size=2,
    model_name="vit_large",
    crop_size=256,
    pred_depth=12,
    pred_num_heads=None,
    pred_embed_dim=384,
    num_mask_tokens=4,
    uniform_power=True,
    use_sdpa=True,
    use_rope=True,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=True,
):
    """Build the frozen long-horizon world model (encoder + causal predictor) and load
    it from `checkpoint`.

    The encoder is the *online* one rather than the EMA target encoder, because that is
    the one the predictor was trained to read from.
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
        use_activation_checkpointing=False,
        is_causal=True,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    logger.info(f"Loading guidance (long-horizon) model from {checkpoint}")
    ckpt = robust_checkpoint_loader(checkpoint, map_location=torch.device("cpu"))
    epoch = ckpt.get("epoch", -1)
    load_module_state_dict(encoder, _strip_ddp_prefix(ckpt["encoder"]), "guidance encoder", epoch)
    load_module_state_dict(predictor, _strip_ddp_prefix(ckpt["predictor"]), "guidance predictor", epoch)
    del ckpt

    encoder.to(device).eval()
    predictor.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in predictor.parameters():
        p.requires_grad = False

    return encoder, predictor


def subsample_guidance_clip(clip, frames_to_skip, guidance_tubelet_size):
    """Build the long-horizon model's view of `clip`.

    The guidance model runs at a lower frame rate but was trained on tubelets of
    `guidance_tubelet_size` *adjacent* frames: it keeps the first
    `guidance_tubelet_size` frames of every `frames_to_skip`-frame group and drops the
    rest, exactly like `SimpleCollator`'s `use_pretrained_model` branch. E.g. a 4fps clip
    guided by a 0.5fps model (frames_to_skip=8, tubelet 2) keeps frames 0,1, 8,9, 16,17...

    :param clip: [B, C, T, H, W] at the student's frame rate
    :return: [B, C, L * guidance_tubelet_size, H, W] with L = T // frames_to_skip
    """
    B, C, T, H, W = clip.shape
    assert frames_to_skip >= guidance_tubelet_size, (
        f"cannot build a {guidance_tubelet_size}-frame guidance tubelet out of groups of {frames_to_skip} frames"
    )
    num_steps = T // frames_to_skip
    assert num_steps >= 1, f"clip of {T} frames is shorter than one guidance step ({frames_to_skip} frames)"
    clip = clip[:, :, : num_steps * frames_to_skip]
    clip = clip.view(B, C, num_steps, frames_to_skip, H, W)[:, :, :, :guidance_tubelet_size]
    return clip.reshape(B, C, num_steps * guidance_tubelet_size, H, W)
