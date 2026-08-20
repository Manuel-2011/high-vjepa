# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""D(z) -> x_hat: the flow-matching decoder.

What it is. A DiT that predicts the rectified-flow velocity field over the
frozen codec's latent of a target frame, conditioned on the latent tokens a
frozen world model emits for that frame. Those tokens enter through
cross-attention: they are not aligned with the target in any fixed way (the grid
may be coarser or finer), and they vary in length and width between world
models - hence attention rather than concatenation.

Two frame-conditioning modes, set by `frame_conditioning` and chosen globally for
a comparison (never per world model):

  `none` (default) - the latent is the ONLY input. The decoder must rebuild the
      frame from the world model's tokens and nothing else, so the sample is a
      direct read-out of what those tokens contain. This is the setting the
      deliverable is about: whatever appears in the image was in the latent,
      because there was no other source for it.

  `current_frame` - the codec latent of the preceding frame `x_t` is
      concatenated channel-wise to the noisy target, making the task
      `D(x_t, z) -> x_hat`. Channel concatenation rather than cross-attention
      because `x_t` is spatially aligned with the target pixel-for-pixel. Much
      easier, and much weaker as a measurement: a decoder given the previous
      frame can produce a plausible image while barely reading the latent, and
      distinguishing the two cases then needs the latent-swap control.

The mode changes what classifier-free guidance means, which matters for reading
the panels. Only `z` is ever dropped for the unconditional branch, so:

  * under `none`, `w = 0` is the decoder's unconditional prior over frames of
    this dataset, and the guidance ladder sweeps what the latent adds over
    "a generic frame".
  * under `current_frame`, `w = 0` still sees `x_t`, so it is a *persistence*
    prior and the ladder sweeps what the latent adds over "the scene did not
    change".

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
FRAME_CONDITIONING_MODES = ("none", "current_frame")


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
        frame_conditioning: str = "none",
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
        if frame_conditioning not in FRAME_CONDITIONING_MODES:
            raise ValueError(
                f"frame_conditioning must be one of {FRAME_CONDITIONING_MODES}; got {frame_conditioning!r}"
            )
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
        self.frame_conditioning = frame_conditioning
        self.uses_frame_conditioning = frame_conditioning == "current_frame"
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
        # One codec_channels block for the noisy target latent, plus a second one
        # for x_t's latent only under `current_frame`. This is the only place the
        # frame-conditioning mode changes a parameter shape, so a checkpoint from
        # one mode will refuse to load into the other rather than load wrongly.
        self.patchify_channels = codec_channels * (2 if self.uses_frame_conditioning else 1)
        self.patchify = nn.Conv2d(
            self.patchify_channels, dim, kernel_size=patch_size, stride=patch_size, bias=True
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

    def latent_shape(self, batch_size: int) -> Tuple[int, int, int, int]:
        """Shape of the codec latent this decoder produces.

        The sampler needs it to draw its initial noise. Under `none` there is no
        `cond_latent` to read a shape off, so it has to come from here.
        """
        return (int(batch_size), self.codec_channels, self.codec_size, self.codec_size)

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
        cond_latent: Optional[torch.Tensor],
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
        if self.uses_frame_conditioning:
            if cond_latent is None:
                raise ValueError(
                    "frame_conditioning='current_frame' needs cond_latent, the codec latent of x_t."
                )
            if cond_latent.shape != x_tau.shape:
                raise ValueError(
                    f"cond_latent {tuple(cond_latent.shape)} must match x_tau {tuple(x_tau.shape)}"
                )
            h = self.patchify(torch.cat([x_tau, cond_latent], dim=1))
        else:
            # Raise rather than ignore: silently dropping a frame the caller
            # believed was being used would make every panel unreadable, and the
            # two modes are otherwise indistinguishable from the outside.
            if cond_latent is not None:
                raise ValueError(
                    "frame_conditioning='none' reconstructs from the latent alone, but a cond_latent "
                    "was supplied. Pass None, or build the decoder with "
                    "frame_conditioning='current_frame'."
                )
            h = self.patchify(x_tau)
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
        z: torch.Tensor,
        cond_latent: Optional[torch.Tensor] = None,
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
            z: (B, L, d_m) world-model latents for the target step.
            cond_latent: (B, C, h, w) codec latent of `x_t`. Required under
                `frame_conditioning='current_frame'`, forbidden under `none`.
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
            f"FlowMatchingDecoder[{self.conditioning}, frame={self.frame_conditioning}] "
            f"d_m={self.latent_dim} grid={self.latent_grid} "
            f"codec={self.codec_channels}x{self.codec_size}^2 dim={self.dim} "
            f"tokens={self.num_tokens} params={self.num_parameters() / 1e6:.1f}M "
            f"(world-model-specific: {adapter / 1e6:.2f}M, shared: {shared / 1e6:.1f}M)"
        )
