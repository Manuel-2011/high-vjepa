# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Latent adapter and coordinate encoding: the layer that makes one decoder
architecture accept latents from *any* frozen world model.

The problem this solves. A world model hands the decoder a token sequence
`z` of shape (B, L, d_m). Across the models being compared, all three of those
numbers move:

  * `d_m` differs with the backbone (1024 for ViT-L, 1408 for ViT-g, 384 if the
    predictor's own width is used instead of the encoder's).
  * `L` differs with the patch size and the number of temporal tokens fed in -
    a 16x16 grid gives 256 tokens per frame, a 32x32 grid gives 1024, and a
    caller may pass one temporal token or several.
  * The *scale* of the tokens differs by construction: target-encoder outputs
    are raw ViT activations, predictor outputs come out of `predictor_proj`
    approximately layer-normalized (the world-model report measures the
    residual mismatch: std ~0.88 rather than 1.0).

The spec allows exactly three things to differ between world models - `d_m`, the
grid shape, and the normalization buffers - and forbids everything else. This
module is where all three live, and nothing outside it is allowed to know which
world model it is looking at:

  1. `LatentNormalization` holds the frozen per-channel statistics as buffers.
     They are *fitted once by the cache pipeline* and then never touched, which
     is what keeps them a property of the latent space rather than a tuned
     hyperparameter.
  2. `nn.Linear(d_m, cond_dim)` absorbs the width difference. It is the only
     shape-varying parameter in the whole decoder.
  3. `CoordinateEncoding` absorbs the length difference.

Why coordinates rather than positional embeddings (§3.1's prohibition). A
learned per-index embedding table is indexed by token *position*, so it silently
encodes the grid it was trained on: token 37 of a 16x16 grid and token 37 of a
32x32 grid are at completely different places in the image, and a table cannot
tell them apart. It also cannot be evaluated at all on a grid larger than the
table. Continuous coordinates fix both: every token carries where it *is*, as
`(t, y, x)` mapped onto the same normalized [-1, 1] cube regardless of how
finely that cube is diced, plus the token's own `(dt, dy, dx)` extent so density
is observable too. A 16x16 latent and a 32x32 latent then land in the same
coordinate frame, differing only in sampling density, and the decoder's spatial
reasoning transfers between them instead of being relearned.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


def build_token_coords(
    grid: Sequence[int],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Continuous coordinates for a full (T, H, W) token grid, flattened `t*H*W + y*W + x`.

    This is the same row-major flattening the V-JEPA mask generators and
    `predictor_masks` use, so a token's index here means what it means in the
    world model.

    Returns (L, 6): `(t, y, x)` cell centres mapped to (-1, 1), followed by the
    cell extents `(dt, dy, dx) = 2 / N`. Centres rather than left edges so the
    encoding is symmetric under flipping an axis, and extents alongside so two
    grids of different density are distinguishable without either one needing to
    have been seen in training.

    A singleton axis gets coordinate 0.0 and extent 2.0 - it covers the whole
    range, which is exactly what one temporal token spanning the clip means.
    """
    if len(grid) != 3:
        raise ValueError(f"grid must be (T, H, W); got {tuple(grid)}")
    axes = []
    extents = []
    for size in grid:
        size = int(size)
        if size < 1:
            raise ValueError(f"grid dimensions must be >= 1; got {tuple(grid)}")
        idx = torch.arange(size, device=device, dtype=dtype)
        axes.append((idx + 0.5) / size * 2.0 - 1.0 if size > 1 else torch.zeros_like(idx))
        extents.append(2.0 / size)

    t, y, x = torch.meshgrid(*axes, indexing="ij")
    centres = torch.stack([t.reshape(-1), y.reshape(-1), x.reshape(-1)], dim=-1)
    extent = torch.tensor(extents, device=device, dtype=dtype).expand_as(centres)
    return torch.cat([centres, extent], dim=-1)


class FourierFeatures(nn.Module):
    """Axis-wise sinusoidal expansion of continuous coordinates.

    Frequencies are a fixed octave bank (1, 2, 4, ... 2^(bands-1)) times pi, not
    learned, so two decoders trained against different world models expand the
    same coordinate to the same features - which is a precondition for the panels
    being a controlled comparison rather than two unrelated models side by side.
    """

    def __init__(self, num_inputs: int, num_bands: int = 8, include_input: bool = True):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_bands = num_bands
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(num_bands, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return self.num_inputs * (2 * self.num_bands + (1 if self.include_input else 0))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        scaled = coords.unsqueeze(-1) * self.freqs.to(coords.dtype)  # (..., n, bands)
        feats = [torch.sin(scaled), torch.cos(scaled)]
        out = torch.cat(feats, dim=-1).flatten(-2)
        if self.include_input:
            out = torch.cat([coords, out], dim=-1)
        return out


class CoordinateEncoding(nn.Module):
    """(t, y, x, dt, dy, dx) -> `cond_dim`, via fixed Fourier features and an MLP."""

    def __init__(self, cond_dim: int, num_bands: int = 8, hidden_mult: int = 2):
        super().__init__()
        self.fourier = FourierFeatures(num_inputs=6, num_bands=num_bands)
        hidden = cond_dim * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(self.fourier.out_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, cond_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """`coords` is (B, L, 6) or (L, 6); returns the matching (..., cond_dim)."""
        if coords.size(-1) != 6:
            raise ValueError(f"coords must have 6 channels (t,y,x,dt,dy,dx); got {coords.size(-1)}")
        return self.mlp(self.fourier(coords))


class LatentNormalization(nn.Module):
    """Frozen normalization that puts a world model's latents at unit scale.

    Three modes, chosen globally for a comparison (never per world model):

      `channel`   - subtract/divide the per-channel statistics held in the
                    buffers. Fitted by the cache pipeline over the training
                    shards. This is the default and the only mode that removes a
                    *channel-wise* scale difference, which is what actually
                    differs between a raw target-encoder output and a
                    `predictor_proj` output.
      `token`     - per-token layer norm, computed on the fly. Carries no fitted
                    state, so it is the mode to use when decoding latents whose
                    statistics were never measured (an arbitrary rollout latent
                    supplied by a caller); it also discards each token's norm,
                    which is real information about that token.
      `none`      - pass through. Only sensible if the latents are already
                    normalized upstream.

    The buffers exist in every mode so a checkpoint's `state_dict` is
    mode-independent and a run can be re-loaded under a different mode for a
    diagnostic without a key mismatch.
    """

    MODES = ("channel", "token", "none")

    def __init__(self, latent_dim: int, mode: str = "channel", eps: float = 1e-5):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown normalization mode {mode!r}; expected one of {self.MODES}")
        self.mode = mode
        self.eps = eps
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("std", torch.ones(latent_dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install statistics measured by the cache pipeline. Called once."""
        if mean.shape != self.mean.shape or std.shape != self.std.shape:
            raise ValueError(
                f"statistics shape mismatch: got {tuple(mean.shape)}/{tuple(std.shape)}, "
                f"expected {tuple(self.mean.shape)}"
            )
        self.mean.copy_(mean.to(self.mean.dtype))
        # clamp_min: a dead channel would otherwise divide by ~0 and blow the
        # normalized latent up to whatever fp16 noise the cache happened to hold.
        self.std.copy_(std.clamp_min(self.eps).to(self.std.dtype))
        self.fitted.fill_(True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "channel":
            return (z - self.mean.to(z.dtype)) / self.std.to(z.dtype)
        if self.mode == "token":
            variance = z.var(dim=-1, keepdim=True, unbiased=False)
            return (z - z.mean(dim=-1, keepdim=True)) / (variance + self.eps).sqrt()
        return z


class LatentAdapter(nn.Module):
    """Normalize -> optional noise augmentation -> project -> add coordinates.

    The noise augmentation happens *after* normalization and *before* the
    projection, so `sigma` is in units of the normalized latent and means the
    same thing for every world model. Its purpose is stated in the training
    spec: stop the decoder from keying on brittle high-frequency detail of the
    training latents, so that a rollout latent - which is off-distribution by
    construction after a few autoregressive steps - still decodes to something
    legible instead of collapsing.
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        norm_mode: str = "channel",
        num_coord_bands: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.norm = LatentNormalization(latent_dim, mode=norm_mode)
        self.proj = nn.Linear(latent_dim, cond_dim)
        self.coords = CoordinateEncoding(cond_dim, num_bands=num_coord_bands)
        self.out_norm = nn.LayerNorm(cond_dim)

    def forward(
        self,
        z: torch.Tensor,
        coords: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """`z` (B, L, d_m), `coords` (B, L, 6) or (L, 6), `sigma` (B,) or scalar.

        `noise` may be supplied to make an augmentation reproducible across two
        calls (the panels use it to hold the perturbation fixed while sweeping
        something else); otherwise it is drawn fresh.
        """
        if z.dim() != 3:
            raise ValueError(f"z must be (B, L, d_m); got {tuple(z.shape)}")
        if z.size(-1) != self.latent_dim:
            raise ValueError(f"z has width {z.size(-1)} but the adapter was built for {self.latent_dim}")

        h = self.norm(z)
        if sigma is not None:
            if not torch.is_tensor(sigma):
                sigma = torch.tensor(sigma, dtype=h.dtype, device=h.device)
            sigma = sigma.to(h.dtype)
            sigma = sigma.reshape(-1, 1, 1) if sigma.dim() else sigma.reshape(1, 1, 1)
            if noise is None:
                noise = torch.randn_like(h)
            h = h + sigma * noise.to(h.dtype)

        h = self.proj(h)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(h.size(0), -1, -1)
        if coords.shape[:2] != h.shape[:2]:
            raise ValueError(
                f"coords {tuple(coords.shape[:2])} do not match the token sequence {tuple(h.shape[:2])}"
            )
        return self.out_norm(h + self.coords(coords.to(h.dtype)))


def latent_grid_from_geometry(num_temporal_tokens: int, spatial_tokens: int) -> Tuple[int, int, int]:
    """(T, H, W) for `num_temporal_tokens` frames of a square `spatial_tokens` grid.

    Raises if the spatial token count is not a perfect square: every world model
    in this repo tokenizes a square crop, and silently guessing a non-square
    factorization would put coordinates in the wrong place.
    """
    side = int(round(math.sqrt(spatial_tokens)))
    if side * side != spatial_tokens:
        raise ValueError(
            f"{spatial_tokens} spatial tokens is not a perfect square, so the (H, W) grid is ambiguous; "
            "pass an explicit grid."
        )
    return int(num_temporal_tokens), side, side
