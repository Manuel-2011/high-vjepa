# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Building blocks for the flow-matching decoder: timestep/scale embeddings,
masked attention, AdaLN-Zero DiT blocks, and the Perceiver resampler.

Attention is written here rather than reused from `src/models/utils/modules.py`
for two reasons. First, every attention in this decoder needs a key-padding mask
- latent sequences have varying length `L`, so a batch mixing two lengths has
padding that must not be attended to, and the repo's `Attention` takes no such
mask. Second, AdaLN-Zero needs the residual branches split out so a per-branch
gate can be applied, which is a different forward structure from `Block`.

AdaLN-Zero, briefly: instead of adding the conditioning vector to the tokens, it
uses the conditioning to *modulate* each residual branch - a shift and scale on
the branch input, and a gate on the branch output - with the modulation
projection zero-initialized. At step 0 every gate is 0, so the whole network is
the identity on its input and training starts from a well-conditioned place
rather than from a random perturbation of the flow field. That matters a lot here
because the decoder is trained on a fixed step budget with no early stopping: a
bad first thousand steps cannot be recovered by training longer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation. `shift`/`scale` are (B, D); `x` is (B, N, D)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalEmbedding(nn.Module):
    """Continuous scalar -> `dim` sinusoidal features -> MLP.

    Used for both the flow time `tau` and the latent-noise level `sigma`, which
    are both continuous scalars in roughly [0, 1] and want the same treatment.
    `max_period` follows the DDPM/DiT convention.
    """

    def __init__(self, dim: int, max_period: float = 10000.0, input_scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.input_scale = input_scale
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=values.device, dtype=torch.float32) / half
        )
        args = values.float().reshape(-1, 1) * self.input_scale * freqs.reshape(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb.to(next(self.mlp.parameters()).dtype))


def _key_mask_to_attn_mask(key_mask: Optional[torch.Tensor], num_queries: int) -> Optional[torch.Tensor]:
    """(B, L) bool keep-mask -> (B, 1, 1, L) bool mask for `scaled_dot_product_attention`.

    A row with no valid key would make softmax produce NaN, so an all-padding
    row is forced to attend to its first key instead. That situation should never
    arise - the decoder substitutes a learned null token rather than an empty
    sequence when conditioning is dropped - but a NaN here would silently poison
    a whole training run, and the guard is free.
    """
    if key_mask is None:
        return None
    if key_mask.dtype != torch.bool:
        key_mask = key_mask.bool()
    empty = ~key_mask.any(dim=-1, keepdim=True)
    if empty.any():
        key_mask = key_mask.clone()
        key_mask[:, 0] |= empty.squeeze(-1)
    return key_mask[:, None, None, :]


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=_key_mask_to_attn_mask(key_mask, n))
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class CrossAttention(nn.Module):
    """Queries from `x`, keys/values from `memory`, with a memory padding mask.

    `memory` may be a different width from `x` (`kv_dim`), which is what lets
    Config B point the DiT straight at adapter output of width `cond_dim`
    without forcing `cond_dim == dim`.
    """

    def __init__(self, dim: int, num_heads: int, kv_dim: Optional[int] = None, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        kv_dim = kv_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, n, c = x.shape
        _, m, _ = memory.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(memory).reshape(b, m, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=_key_mask_to_attn_mask(memory_mask, n))
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """Self-attention + cross-attention + MLP, each AdaLN-Zero conditioned.

    Nine modulation parameters per block (shift/scale/gate x 3 branches) come
    from one zero-initialized projection of the conditioning vector `c`, so all
    three branches start gated off.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        kv_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.xattn = CrossAttention(dim, num_heads, kv_dim=kv_dim)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(dim, mlp_ratio=mlp_ratio)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 9 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (sa_shift, sa_scale, sa_gate, ca_shift, ca_scale, ca_gate, mlp_shift, mlp_scale, mlp_gate) = (
            self.modulation(c).chunk(9, dim=-1)
        )
        x = x + sa_gate.unsqueeze(1) * self.attn(modulate(self.norm1(x), sa_shift, sa_scale))
        x = x + ca_gate.unsqueeze(1) * self.xattn(
            modulate(self.norm2(x), ca_shift, ca_scale), memory, memory_mask=memory_mask
        )
        x = x + mlp_gate.unsqueeze(1) * self.mlp(modulate(self.norm3(x), mlp_shift, mlp_scale))
        return x


class FinalLayer(nn.Module):
    """AdaLN-modulated projection back to `patch_size**2 * out_channels`.

    Zero-initialized, so the decoder's initial velocity prediction is exactly
    zero everywhere. Under rectified flow that is the field of a constant, which
    is a far better starting point than noise.
    """

    def __init__(self, dim: int, cond_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


class PerceiverResampler(nn.Module):
    """Config A: variable-length latents -> a fixed-length, fixed-width memory.

    A fixed bank of learned queries cross-attends to the adapted latent tokens,
    then self-attends among themselves, for `depth` layers. The output is always
    (B, num_queries, cond_dim) whatever `L` was, which has two consequences worth
    being explicit about:

      * The DiT's cross-attention cost stops depending on the world model's token
        count. Comparing a 256-token model against a 1024-token one then costs
        the same and, more importantly, gives the decoder the same capacity budget
        for both - so a difference in the panels is a difference in the latents,
        not a difference in how much compute the decoder got to spend.
      * It is an information bottleneck. `num_queries` smaller than `L` forces a
        summary, and a summary can hide detail the latent actually contains.
        That is what Config B exists to check.

    The queries are *not* positional embeddings for the latent tokens - they carry
    no index - so the §3.1 prohibition is respected: all positional information
    reaches them through the adapter's continuous coordinates on the keys.
    """

    def __init__(
        self,
        cond_dim: int,
        num_queries: int = 128,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, cond_dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(cond_dim, eps=1e-6),
                        CrossAttention(cond_dim, num_heads),
                        nn.LayerNorm(cond_dim, eps=1e-6),
                        SelfAttention(cond_dim, num_heads),
                        nn.LayerNorm(cond_dim, eps=1e-6),
                        FeedForward(cond_dim, mlp_ratio=mlp_ratio),
                    ]
                )
            )
        self.out_norm = nn.LayerNorm(cond_dim, eps=1e-6)

    def forward(self, tokens: torch.Tensor, key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.queries.expand(tokens.size(0), -1, -1).to(tokens.dtype)
        for norm_x, xattn, norm_s, sattn, norm_m, mlp in self.layers:
            q = q + xattn(norm_x(q), tokens, memory_mask=key_mask)
            q = q + sattn(norm_s(q))
            q = q + mlp(norm_m(q))
        return self.out_norm(q)


def masked_mean(tokens: torch.Tensor, key_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over the sequence axis, ignoring padded positions.

    The pooled summary feeds AdaLN, so letting padding into it would make the
    global conditioning depend on how a batch happened to be padded - which would
    show up as a mysterious batch-composition effect in the panels.
    """
    if key_mask is None:
        return tokens.mean(dim=1)
    weights = key_mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
