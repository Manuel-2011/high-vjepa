# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""D(x_t, z_{t+1}) -> x_hat_{t+1}: the flow-matching decoder.

What it is. A DiT that predicts the rectified-flow velocity field over the
frozen codec's latent of the *next* frame, conditioned on two things:

  * `x_t`, the current frame, entering as extra channels concatenated to the
    noisy target latent. Channel concatenation rather than cross-attention
    because `x_t` is spatially aligned with the target pixel-for-pixel - it is
    the same scene one step earlier - so the alignment is free information that
    a convolutional patchify preserves and attention would have to rediscover.
  * `z_{t+1}`, the world model's latent tokens for the target step, entering
    through cross-attention. Not aligned with the target in any fixed way (its
    grid may be coarser, finer, or temporally pooled), and of varying length and
    width - hence attention.

Why the asymmetry matters for the deliverable. `x_t` is always present and is
never dropped; only `z` is dropped for classifier-free guidance. So the guidance
scale sweeps *the latent's contribution alone*, against a baseline that already
knows the current frame. Panel 2's guidance ladder is therefore a direct read-out
of how much the latent adds over persistence - which is exactly the question a
microscope on latent spaces should be answering, and it would be muddied if the
unconditional branch had to hallucinate the scene from scratch.

Two conditioning routes, selected by `conditioning`:

  Config A (`perceiver`) - the adapted latents are resampled to a fixed bank of
  `num_queries` tokens, then cross-attended. Constant cost and constant capacity
  across world models; an information bottleneck by construction.

  Config B (`direct`)   - the DiT cross-attends straight to the adapted latent
  tokens, with a padding mask. No bottleneck, but cost grows with `L`.

Both hand the DiT body an identical (memory, memory_mask) pair, so the body -
and every hyperparameter in it - is literally the same module in both configs.
The only per-world-model state anywhere in the model is inside `LatentAdapter`
(`d_m`-shaped projection, grid-shaped nothing, normalization buffers), which is
what the spec's prohibition requires.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.models.flow_decoder.blocks import (
    DiTBlock,
    FinalLayer,
    PerceiverResampler,
    SinusoidalEmbedding,
    masked_mean,
)
from src.models.flow_decoder.latent_adapter import LatentAdapter, build_token_coords
from src.models.utils.pos_embs import get_2d_sincos_pos_embed

logger = logging.getLogger(__name__)

CONDITIONING_MODES = ("perceiver", "direct")


class FlowMatchingDecoder(nn.Module):
    """The decoder. See the module docstring for the design; shapes below.

    Args:
        latent_dim: `d_m`, the world model's token width. The ONLY width that
            may differ between world models.
        latent_grid: the world model's default (T, H, W) token grid, used to
            build coordinates when a caller does not supply them. Purely a
            convenience - nothing in the parameters depends on it.
        codec_channels: channels of the frozen codec's latent (4 for an SD VAE).
        codec_size: spatial size of the codec latent (32 for 256px at f=8).
        patch_size: DiT patchification of the codec latent. 2 gives 16x16=256
            tokens at codec_size 32.
    """

    def __init__(
        self,
        latent_dim: int,
        latent_grid: Sequence[int] = (1, 16, 16),
        codec_channels: int = 4,
        codec_size: int = 32,
        patch_size: int = 2,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        cond_dim: int = 768,
        mlp_ratio: float = 4.0,
        conditioning: str = "perceiver",
        num_queries: int = 128,
        resampler_depth: int = 2,
        resampler_heads: int = 8,
        latent_norm_mode: str = "channel",
        num_coord_bands: int = 8,
        use_activation_checkpointing: bool = False,
    ):
        super().__init__()
        if conditioning not in CONDITIONING_MODES:
            raise ValueError(f"conditioning must be one of {CONDITIONING_MODES}; got {conditioning!r}")
        if codec_size % patch_size:
            raise ValueError(f"codec_size {codec_size} is not divisible by patch_size {patch_size}")

        self.latent_dim = latent_dim
        self.latent_grid = tuple(int(v) for v in latent_grid)
        self.codec_channels = codec_channels
        self.codec_size = codec_size
        self.patch_size = patch_size
        self.dim = dim
        self.cond_dim = cond_dim
        self.conditioning = conditioning
        self.use_activation_checkpointing = use_activation_checkpointing
        self.grid_side = codec_size // patch_size
        self.num_tokens = self.grid_side**2

        # -- conditioning path -------------------------------------------------
        self.adapter = LatentAdapter(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            norm_mode=latent_norm_mode,
            num_coord_bands=num_coord_bands,
        )
        self.resampler = (
            PerceiverResampler(
                cond_dim=cond_dim,
                num_queries=num_queries,
                depth=resampler_depth,
                num_heads=resampler_heads,
                mlp_ratio=mlp_ratio,
            )
            if conditioning == "perceiver"
            else None
        )
        # Substituted for the whole memory when the latent is dropped, which is
        # what makes an unconditional branch possible without ever handing
        # attention an empty key sequence.
        self.null_memory = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)

        # -- x_tau path --------------------------------------------------------
        # 2 * codec_channels in: the noisy target latent, and x_t's latent.
        self.patchify = nn.Conv2d(
            2 * codec_channels, dim, kernel_size=patch_size, stride=patch_size, bias=True
        )
        pos = get_2d_sincos_pos_embed(dim, self.grid_side, cls_token=False)
        self.register_buffer("pos_embed", torch.from_numpy(pos).float().unsqueeze(0), persistent=False)

        # -- global conditioning vector ---------------------------------------
        self.tau_embed = SinusoidalEmbedding(dim)
        self.sigma_embed = SinusoidalEmbedding(dim)
        self.pool_proj = nn.Linear(cond_dim, dim)

        # -- body --------------------------------------------------------------
        self.blocks = nn.ModuleList(
            [
                DiTBlock(dim=dim, num_heads=num_heads, cond_dim=dim, mlp_ratio=mlp_ratio, kv_dim=cond_dim)
                for _ in range(depth)
            ]
        )
        self.final = FinalLayer(dim, cond_dim=dim, patch_size=patch_size, out_channels=codec_channels)

        self.apply(self._init_weights)
        # Re-zero what `_init_weights` walked over: AdaLN-Zero and the output
        # projection must stay exactly zero (see blocks.py).
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final.modulation[-1].weight)
        nn.init.zeros_(self.final.modulation[-1].bias)
        nn.init.zeros_(self.final.proj.weight)
        nn.init.zeros_(self.final.proj.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ #
    # shape helpers
    # ------------------------------------------------------------------ #

    def default_coords(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """(L, 6) coordinates for `self.latent_grid`."""
        return build_token_coords(self.latent_grid, device=device, dtype=dtype)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, p*p*C) -> (B, C, codec_size, codec_size)."""
        b = tokens.size(0)
        p, s, c = self.patch_size, self.grid_side, self.codec_channels
        x = tokens.reshape(b, s, s, p, p, c).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(b, c, s * p, s * p)

    # ------------------------------------------------------------------ #
    # conditioning
    # ------------------------------------------------------------------ #

    def encode_conditioning(
        self,
        z: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        z_mask: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        drop_latent: Optional[torch.Tensor] = None,
        aug_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Adapt (and optionally resample) the latents into a DiT memory.

        Returns `(memory, memory_mask, sigma_used)`. Split out from `forward` so
        an ODE sampler can compute it once and reuse it across every solver step
        - the conditioning does not depend on `tau`, and re-running a Perceiver
        50 times per sample would dominate sampling cost for no reason.
        """
        b = z.size(0)
        device = z.device
        if coords is None:
            coords = self.default_coords(device, dtype=torch.float32)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(b, -1, -1)

        if sigma is None:
            sigma = torch.zeros(b, device=device)
        sigma = sigma.to(device).reshape(-1).float()
        if sigma.numel() == 1 and b > 1:
            sigma = sigma.expand(b)

        if drop_latent is None:
            drop_latent = torch.zeros(b, dtype=torch.bool, device=device)
        drop_latent = drop_latent.to(device).reshape(-1).bool()
        # A dropped row's noise level is meaningless and would otherwise make the
        # unconditional branch depend on an augmentation it cannot see. Forcing 0
        # makes that branch canonical, so a CFG pair is reproducible.
        sigma = torch.where(drop_latent, torch.zeros_like(sigma), sigma)

        tokens = self.adapter(z, coords, sigma=sigma, noise=aug_noise)
        if self.resampler is not None:
            memory = self.resampler(tokens, key_mask=z_mask)
            memory_mask = None  # resampler output is dense by construction
        else:
            memory = tokens
            memory_mask = z_mask

        if drop_latent.any():
            null = self.null_memory.to(memory.dtype).expand(b, 1, -1)
            keep = (~drop_latent).view(b, 1, 1)
            # Broadcast the null token to the memory length so both branches keep
            # identical shapes; the mask below hides every position past the first.
            memory = torch.where(keep, memory, null.expand(b, memory.size(1), -1))
            first_only = torch.zeros(b, memory.size(1), dtype=torch.bool, device=device)
            first_only[:, 0] = True
            base_mask = (
                memory_mask.bool()
                if memory_mask is not None
                else torch.ones(b, memory.size(1), dtype=torch.bool, device=device)
            )
            memory_mask = torch.where(drop_latent.view(b, 1), first_only, base_mask)

        return memory, memory_mask, sigma

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward_with_memory(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        cond_latent: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """The velocity field, given an already-built conditioning memory."""
        b = x_tau.size(0)
        if x_tau.shape[1:] != (self.codec_channels, self.codec_size, self.codec_size):
            raise ValueError(
                f"x_tau must be (B, {self.codec_channels}, {self.codec_size}, {self.codec_size}); "
                f"got {tuple(x_tau.shape)}"
            )
        if cond_latent.shape != x_tau.shape:
            raise ValueError(
                f"cond_latent {tuple(cond_latent.shape)} must match x_tau {tuple(x_tau.shape)}"
            )

        h = self.patchify(torch.cat([x_tau, cond_latent], dim=1))
        h = h.flatten(2).transpose(1, 2) + self.pos_embed.to(h.dtype)

        tau = tau.to(x_tau.device).reshape(-1).float()
        if tau.numel() == 1 and b > 1:
            tau = tau.expand(b)
        c = self.tau_embed(tau) + self.sigma_embed(sigma) + self.pool_proj(masked_mean(memory, memory_mask))

        for block in self.blocks:
            if self.use_activation_checkpointing and self.training:
                h = checkpoint(block, h, c, memory, memory_mask, use_reentrant=False)
            else:
                h = block(h, c, memory, memory_mask)

        return self.unpatchify(self.final(h, c))

    def forward(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        cond_latent: torch.Tensor,
        z: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        z_mask: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        drop_latent: Optional[torch.Tensor] = None,
        aug_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predicted velocity at `(x_tau, tau)`, shaped like `x_tau`.

        Args:
            x_tau: (B, C, h, w) point on the flow path.
            tau: (B,) or scalar flow time in [0, 1].
            cond_latent: (B, C, h, w) codec latent of `x_t`.
            z: (B, L, d_m) world-model latents for the target step.
            coords: (B, L, 6) or (L, 6) continuous coordinates; defaults to
                `self.latent_grid`.
            z_mask: (B, L) bool, True where `z` is real rather than padding.
            sigma: (B,) latent-noise-augmentation level.
            drop_latent: (B,) bool, True to replace `z` with the null memory.
            aug_noise: (B, L, d_m) fixed augmentation noise, for reproducibility.
        """
        memory, memory_mask, sigma_used = self.encode_conditioning(
            z, coords=coords, z_mask=z_mask, sigma=sigma, drop_latent=drop_latent, aug_noise=aug_noise
        )
        return self.forward_with_memory(x_tau, tau, cond_latent, memory, memory_mask, sigma_used)

    # ------------------------------------------------------------------ #
    # bookkeeping
    # ------------------------------------------------------------------ #

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def describe(self) -> str:
        shared = sum(
            p.numel() for n, p in self.named_parameters() if not n.startswith("adapter.proj")
        )
        adapter = sum(p.numel() for n, p in self.named_parameters() if n.startswith("adapter.proj"))
        return (
            f"FlowMatchingDecoder[{self.conditioning}] d_m={self.latent_dim} grid={self.latent_grid} "
            f"codec={self.codec_channels}x{self.codec_size}^2 dim={self.dim} "
            f"tokens={self.num_tokens} params={self.num_parameters() / 1e6:.1f}M "
            f"(world-model-specific: {adapter / 1e6:.2f}M, shared: {shared / 1e6:.1f}M)"
        )
