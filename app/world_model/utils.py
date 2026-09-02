# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import torch

import src.models.vision_transformer as video_vit
from app.vjepa.utils import load_module_state_dict
from app.world_model.models import chunk_encoder, goal_conditioned_predictor
from src.utils.checkpoint_loader import robust_checkpoint_loader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _clean_backbone_keys(state_dict, drop_pos_embed):
    """Strip the DDP/`MultiSeqWrapper` prefixes a V-JEPA 2 checkpoint carries."""
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        if drop_pos_embed and k.endswith("pos_embed"):
            # A RoPE backbone has no learned position embedding; released checkpoints
            # of non-RoPE variants carry one.
            continue
        cleaned[k] = v
    return cleaned


def init_frozen_backbone(
    device,
    checkpoint,
    checkpoint_key="target_encoder",
    model_name="vit_large",
    crop_size=256,
    patch_size=16,
    tubelet_size=2,
    frames_per_chunk=8,
    uniform_power=False,
    use_rope=True,
    use_sdpa=True,
    use_silu=False,
    wide_silu=True,
):
    """Build the frozen V-JEPA 2 encoder and load it from `checkpoint`.

    It is only ever run on a single chunk at a time (`frames_per_chunk` frames), which
    is what makes its bidirectional attention safe for a causal world model: a chunk's
    features never see anything outside that chunk.
    """
    backbone = video_vit.__dict__[model_name](
        img_size=(crop_size, crop_size),
        patch_size=patch_size,
        num_frames=frames_per_chunk,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=False,
        use_rope=use_rope,
        is_causal=False,
    )

    logger.info(f"Loading frozen V-JEPA 2 backbone from {checkpoint} (key '{checkpoint_key}')")
    ckpt = robust_checkpoint_loader(checkpoint, map_location=torch.device("cpu"))
    state_dict = ckpt[checkpoint_key] if checkpoint_key in ckpt else ckpt["encoder"]
    load_module_state_dict(
        backbone,
        _clean_backbone_keys(state_dict, drop_pos_embed=use_rope),
        "frozen V-JEPA 2 backbone",
        ckpt.get("epoch", -1),
    )
    del ckpt

    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    return backbone


def init_world_model(
    device,
    frozen_dim,
    grid_height,
    grid_width,
    tokens_per_chunk=4,
    context_chunks=8,
    embed_dim=768,
    enc_depth=6,
    enc_num_heads=12,
    pred_depth=12,
    pred_embed_dim=384,
    pred_num_heads=12,
    goal_gate_init=1.0,
    horizon_embed_dim=128,
    drop_path_rate=0.0,
    use_sdpa=True,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=True,
    use_activation_checkpointing=False,
):
    encoder = chunk_encoder(
        in_dim=frozen_dim,
        embed_dim=embed_dim,
        depth=enc_depth,
        num_heads=enc_num_heads,
        grid_height=grid_height,
        grid_width=grid_width,
        tokens_per_chunk=tokens_per_chunk,
        drop_path_rate=drop_path_rate,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    predictor = goal_conditioned_predictor(
        embed_dim=embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=pred_num_heads,
        grid_height=grid_height,
        grid_width=grid_width,
        max_chunks=context_chunks,
        goal_gate_init=goal_gate_init,
        horizon_embed_dim=horizon_embed_dim,
        drop_path_rate=drop_path_rate,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Chunk-encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def load_checkpoint(r_path, encoder, predictor, target_encoder, opt, scaler):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]

    load_module_state_dict(encoder, checkpoint["encoder"], "chunk encoder", epoch)
    load_module_state_dict(predictor, checkpoint["predictor"], "predictor", epoch)
    if target_encoder is not None:
        load_module_state_dict(target_encoder, checkpoint["target_encoder"], "target chunk encoder", epoch)

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


def split_into_chunks(clip, frames_per_chunk, num_chunks):
    """Cut the leading `num_chunks * frames_per_chunk` frames of a clip into chunks.

    :param clip: [B, C, T, H, W]
    :return: [B * num_chunks, C, frames_per_chunk, H, W], chunk-major within each sample
        (sample 0's chunks first), which is the layout `[B, num_chunks, ...]` unflattens to
    """
    B, C, T, H, W = clip.shape
    needed = num_chunks * frames_per_chunk
    assert T >= needed, f"clip of {T} frames is too short for {num_chunks} chunks of {frames_per_chunk}"
    clip = clip[:, :, :needed]
    clip = clip.view(B, C, num_chunks, frames_per_chunk, H, W)
    return clip.permute(0, 2, 1, 3, 4, 5).reshape(B * num_chunks, C, frames_per_chunk, H, W)


def gather_window(clip, starts, num_frames):
    """Slice a per-sample window of `num_frames` frames out of a batch of clips.

    Each sample takes its own start index, which is what lets the goal be sampled at a
    different distance for every clip in the batch.

    :param clip: [B, C, T, H, W]
    :param starts: [B] long tensor of first frames
    :return: [B, C, num_frames, H, W]
    """
    B, C, T, H, W = clip.shape
    idx = starts.view(B, 1) + torch.arange(num_frames, device=clip.device).view(1, num_frames)
    assert int(idx.max()) < T, f"goal window runs past the end of a {T}-frame clip"
    idx = idx.view(B, 1, num_frames, 1, 1).expand(B, C, num_frames, H, W)
    return torch.gather(clip, 2, idx)
