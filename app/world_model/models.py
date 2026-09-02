# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Building blocks of a goal-conditioned latent world model that lives on top of a
# *frozen* V-JEPA 2 backbone.
#
# The frozen backbone turns `tubelet_size` frames into one temporal token. This model
# works at a coarser granularity: a "chunk" is `tokens_per_chunk` consecutive temporal
# tokens (e.g. 4 tokens x 2 frames = 8 frames = 2s at 4fps). `ChunkEncoder` collapses
# one chunk into a single latent (one token per spatial patch) and
# `GoalConditionedPredictor` reads a causal sequence of those latents and predicts the
# next one, cross-attending to the latent of a chunk further in the future -- the goal.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.modules import Block, GuidedBlock
from src.utils.tensors import trunc_normal_


class ChunkEncoder(nn.Module):
    """Summarize one chunk of frozen V-JEPA 2 tokens into a chunk latent.

    :param in_dim: width of the frozen backbone's output (1024 for a ViT-L)
    :param tokens_per_chunk: temporal tokens of the frozen backbone in one chunk

    Input is [B, tokens_per_chunk * P, in_dim], the backbone's output for one chunk,
    laid out time-major (P = grid_height * grid_width patches per temporal token). A few
    bidirectional RoPE blocks mix the chunk's own space-time grid -- the chunk is the
    unit of observation here, so seeing all of it at once leaks nothing -- and the
    temporal axis is then pooled away. The chunk latent is [B, P, embed_dim]: one token
    per spatial position, which keeps the spatial layout the predictor's 3D RoPE needs.
    """

    def __init__(
        self,
        in_dim,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        grid_height=16,
        grid_width=16,
        tokens_per_chunk=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        assert grid_height == grid_width, "RoPEAttention assumes a square patch grid"
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.patches_per_token = grid_height * grid_width
        self.tokens_per_chunk = tokens_per_chunk
        self.use_activation_checkpointing = use_activation_checkpointing

        self.in_proj = nn.Linear(in_dim, embed_dim, bias=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=True,
                    grid_size=grid_height,
                    grid_depth=tokens_per_chunk,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    norm_layer=norm_layer,
                    use_sdpa=use_sdpa,
                    is_causal=False,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        _rescale_blocks(self.blocks)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        :param x: [B, tokens_per_chunk * P, in_dim] frozen backbone tokens of one chunk
        :return: [B, P, embed_dim] chunk latent
        """
        T, P = self.tokens_per_chunk, self.patches_per_token
        assert x.size(1) == T * P, f"expected {T * P} frozen tokens per chunk, got {x.size(1)}"

        # The frozen features are not in any particular scale; normalize them the same
        # way V-JEPA normalizes its targets before projecting into this model's width.
        x = F.layer_norm(x, (x.size(-1),))
        x = self.in_proj(x)

        for blk in self.blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, T=T, H_patches=self.grid_height, W_patches=self.grid_width, use_reentrant=False
                )
            else:
                x = blk(x, T=T, H_patches=self.grid_height, W_patches=self.grid_width)

        # -- pool the chunk's temporal axis away: [B, T, P, D] -> [B, P, D]
        x = x.unflatten(1, (T, P)).mean(dim=1)
        return self.norm(x)


class GoalConditionedPredictor(nn.Module):
    """Causal next-chunk predictor, conditioned on the latent of a goal chunk.

    The input is a sequence of `S` chunk latents flattened to [B, S * P, embed_dim].
    Self-attention is causal at chunk granularity (`RoPEAttention`'s `temp_attn_mask`
    blocks every position of a later chunk), so the output at chunk `s` is a function of
    chunks `0..s` only and is trained to match the latent of chunk `s + 1`.

    Every block additionally cross-attends to the goal latent [B, P, embed_dim] through
    a gated RoPE cross-attention branch. Queries and the goal share one (time, height,
    width) coordinate system measured in chunks: query `s` predicts the chunk starting
    at `s + 1` and the goal sits at `goal_pos` (a float, since goals are sampled at
    frame resolution). That distance varies per sample, which is why `goal_pos` is a
    per-sample tensor.

    The rotation alone is a weak channel for *how far away* the goal is, though: there
    is only one goal time-step, so a shift in `goal_pos` can only reshuffle the
    attention weights over the goal's spatial tokens -- the values it averages carry no
    time at all. How much time is left to reach the goal is central to what the agent
    would do next, so it is also injected directly: each step's remaining horizon
    `goal_pos - (s + 1)` gets a sinusoidal embedding that is added to its tokens before
    the first block.

    Goal conditioning can be dropped per sample (`keep_goal`). A dropped sample reads a
    learned null goal and a learned null horizon instead, so the same weights also learn
    to roll the world forward with no intention supplied -- which keeps the predictor
    from leaning on the goal for everything, and gives an unconditional mode at
    inference. The null goal is a single token broadcast over the spatial grid: every
    key the cross-attention sees is then identical, so its output is that token whatever
    `goal_pos` says, and nothing about where the dropped goal was leaks through.
    """

    def __init__(
        self,
        embed_dim,
        predictor_embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        grid_height=16,
        grid_width=16,
        max_chunks=8,
        goal_gate_init=1.0,
        horizon_embed_dim=128,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        assert grid_height == grid_width, "RoPEAttention assumes a square patch grid"
        self.embed_dim = embed_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.patches_per_chunk = grid_height * grid_width
        self.max_chunks = max_chunks
        self.use_activation_checkpointing = use_activation_checkpointing

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # The goal is projected once per step and shared by every block, as the guidance
        # latents are in `VisionTransformerPredictor`.
        self.goal_norm = norm_layer(embed_dim)
        self.goal_proj = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Time left to reach the goal, in chunks -> a vector added to that step's tokens.
        assert horizon_embed_dim % 2 == 0, "horizon_embed_dim must be even"
        self.horizon_embed_dim = horizon_embed_dim
        self.horizon_mlp = nn.Sequential(
            nn.Linear(horizon_embed_dim, predictor_embed_dim),
            nn.SiLU(),
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
        )

        # What a sample reads when its goal is dropped. Zero-initialized, so an
        # unconditioned step starts out as exactly the model with no goal branch at all
        # and has to learn what "no intention given" should mean.
        self.null_goal = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.null_horizon = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList(
            [
                GuidedBlock(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    guidance_gate_init=goal_gate_init,
                    use_rope=True,
                    grid_size=grid_height,
                    grid_depth=max_chunks,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    norm_layer=norm_layer,
                    use_sdpa=use_sdpa,
                    is_causal=True,
                )
                for i in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.init_std = init_std
        self.apply(self._init_weights)
        _rescale_blocks(self.predictor_blocks)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_positions(self, num_chunks, goal_pos):
        """Place the queries and the goal on one shared (time, height, width) grid.

        Time is counted in chunks, and a token's coordinate is the *start* of the chunk
        it stands for: query `s` predicts chunk `s + 1`, so it sits at `s + 1`, and the
        goal sits at `goal_pos` (in the same units, measured from the start of the clip).
        The relative rotation between them is therefore exactly how many chunks ahead
        the goal is from what this step is predicting.

        :param goal_pos: [B] float tensor, the goal chunk's start in chunk units
        :return: (q_pos, k_pos), each a (t, h, w) triplet of position tensors
        """
        P = self.patches_per_chunk
        device = goal_pos.device
        B = goal_pos.size(0)

        patch_ids = torch.arange(P, device=device)
        patch_h = (patch_ids // self.grid_width).float()
        patch_w = (patch_ids % self.grid_width).float()

        chunk_ids = torch.arange(num_chunks, device=device)
        q_pos = (
            (chunk_ids + 1).float().repeat_interleave(P),
            patch_h.repeat(num_chunks),
            patch_w.repeat(num_chunks),
        )
        # The goal's time coordinate differs per sample, so it is broadcast over heads
        # as [B, 1, P] while the (shared) spatial coordinates stay 1D.
        k_pos = (goal_pos.to(torch.float32).view(B, 1, 1).expand(B, 1, P), patch_h, patch_w)
        return q_pos, k_pos

    def horizon_conditioning(self, num_chunks, goal_pos):
        """Embedding of how far each step still is from the goal.

        Step `s` predicts the chunk starting at `s + 1`, so its horizon is
        `goal_pos - (s + 1)` chunks -- positive, and shrinking along the sequence.

        :return: [B, num_chunks, predictor_embed_dim], one vector per (sample, step)
        """
        chunk_ids = torch.arange(num_chunks, device=goal_pos.device, dtype=torch.float32)
        horizon = goal_pos.to(torch.float32).unsqueeze(1) - (chunk_ids + 1).unsqueeze(0)  # [B, S]
        return self.horizon_mlp(sincos_embedding(horizon, self.horizon_embed_dim))

    def forward(self, x, goal, goal_pos, keep_goal=None):
        """
        :param x: [B, S * P, embed_dim] latents of the S observed chunks
        :param goal: [B, P, embed_dim] latent of the goal chunk
        :param goal_pos: [B] start of the goal chunk, in chunk units
        :param keep_goal: [B] bool, False where the goal is dropped for that sample
            (default: keep every goal). Pass all-False to roll out unconditionally.
        :return: [B, S * P, embed_dim] prediction of chunks 1..S
        """
        P = self.patches_per_chunk
        assert x.size(1) % P == 0, f"{x.size(1)} predictor tokens is not a whole number of chunks"
        num_chunks = x.size(1) // P
        assert num_chunks <= self.max_chunks, (
            f"predictor was built for at most {self.max_chunks} chunks, got {num_chunks}"
        )
        assert goal.size(1) == P, f"expected a goal latent of {P} tokens, got {goal.size(1)}"

        B = x.size(0)
        x = self.predictor_embed(x)
        goal = self.goal_proj(self.goal_norm(goal))
        horizon = self.horizon_conditioning(num_chunks, goal_pos)
        if keep_goal is not None:
            # `torch.where` rather than indexing so the null parameters stay in the
            # graph on every step, whatever the draw -- DDP would otherwise see them
            # as unused in the iterations where nothing is dropped.
            keep = keep_goal.view(B, 1, 1)
            goal = torch.where(keep, goal, self.null_goal.to(goal.dtype))
            horizon = torch.where(keep, horizon, self.null_horizon.to(horizon.dtype))
        x = x + horizon.repeat_interleave(P, dim=1)
        q_pos, k_pos = self.build_positions(num_chunks, goal_pos)

        blk_kwargs = dict(
            guidance=goal,
            q_pos=q_pos,
            k_pos=k_pos,
            xattn_mask=None,  # the goal is always in the future, so it is always readable
            T=num_chunks,
            H_patches=self.grid_height,
            W_patches=self.grid_width,
        )
        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False, **blk_kwargs)
            else:
                x = blk(x, **blk_kwargs)

        x = self.predictor_norm(x)
        return self.predictor_proj(x)

    def goal_gate(self):
        """Mean magnitude of the per-block goal cross-attention gates -- how much the
        predictor is actually leaning on the goal."""
        gates = [blk.gamma_xattn.detach() for blk in self.predictor_blocks]
        return float(torch.stack([g.abs().mean() for g in gates]).mean())


def sincos_embedding(t, dim, max_period=10000.0):
    """Sinusoidal embedding of a continuous scalar, [...] -> [..., dim]."""
    half = dim // 2
    omega = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = t.to(torch.float32).unsqueeze(-1) * omega
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def _rescale_blocks(blocks):
    """Shrink each block's residual output projections by 1/sqrt(2 * depth), as every
    other transformer in this repo does. `SwiGLUFFN` projects out of `fc3` rather than
    `fc2`, so the MLP's actual output layer is picked by name."""

    def rescale(param, layer_id):
        param.div_(math.sqrt(2.0 * layer_id))

    for layer_id, layer in enumerate(blocks):
        rescale(layer.attn.proj.weight.data, layer_id + 1)
        mlp_out = getattr(layer.mlp, "fc3", None) or layer.mlp.fc2
        rescale(mlp_out.weight.data, layer_id + 1)


def chunk_encoder(**kwargs):
    return ChunkEncoder(norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def goal_conditioned_predictor(**kwargs):
    return GoalConditionedPredictor(norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
