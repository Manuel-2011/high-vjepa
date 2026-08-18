# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Frozen pixel<->latent codec for the flow-matching decoder.

The decoder runs rectified flow in the latent space of a *frozen* autoencoder,
never in pixel space, for the usual reason: a 256x256 RGB frame is 196k
dimensions, an f=8 VAE latent is 4096, and the DiT that has to be trained on one
consumer GPU only fits in the latter. The VAE is never trained or fine-tuned
here - it is loaded, `eval()`-ed and `requires_grad_(False)`-ed exactly like the
world models are (see the hard prohibitions in the spec).

Two codecs are provided behind one interface:

  * `KLVAECodec` - a diffusers `AutoencoderKL` (default
    `stabilityai/sd-vae-ft-mse`). This is the codec to use for any real run. It
    downsamples by 8 and emits 4 channels, so a 256px frame becomes a
    (4, 32, 32) latent. Encoding uses the distribution *mode* rather than a
    sample: the decoder's own noise is supposed to come from the flow prior, not
    from the codec, and a deterministic encode is what makes the
    reconstruction ceiling panel reproducible.

  * `PatchifyCodec` - a weight-free, information-preserving space-to-depth
    codec (`f` x `f` pixel blocks folded into channels). It is *not* a
    perceptual compressor, so it needs many more channels and gives a
    correspondingly bigger DiT; its purpose is to let the whole pipeline -
    caching, training, sampling, panels - be exercised on a machine with no
    network access and no VAE weights, and to serve as an exact-roundtrip
    control when a panel looks wrong and the question is "is that the codec or
    the decoder?".

Both codecs expose the same contract:

    latents = codec.encode(frames)      # (B, 3, H, W) in [0, 1] -> (B, C, h, w)
    frames  = codec.decode(latents)     # (B, C, h, w) -> (B, 3, H, W) in [0, 1]

with `codec.latent_channels`, `codec.downsample_factor` and a
`codec.scaling_factor` that puts latents at roughly unit variance, which is what
the flow objective assumes. Frames are always [0, 1] RGB at the codec boundary;
the ImageNet-style normalization the world models want is applied separately
(see `app/flow_decoder/latent_cache.py`), because the two normalizations serve
different consumers and conflating them is how channel swaps happen.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_VAE_ID = "stabilityai/sd-vae-ft-mse"


class PixelCodec(nn.Module):
    """Interface every frozen codec implements. Subclasses must not train."""

    latent_channels: int
    downsample_factor: int
    scaling_factor: float

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def latent_size(self, frame_size: int) -> int:
        if frame_size % self.downsample_factor != 0:
            raise ValueError(
                f"frame size {frame_size} is not a multiple of the codec's downsample factor "
                f"{self.downsample_factor}; the latent grid would not be square-aligned."
            )
        return frame_size // self.downsample_factor

    def freeze(self) -> "PixelCodec":
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        return self


class KLVAECodec(PixelCodec):
    """Frozen diffusers `AutoencoderKL`.

    `scaling_factor` comes from the checkpoint's own config (0.18215 for the SD
    family) rather than being hard-coded, so swapping in a different KL VAE does
    not silently mis-scale the flow target.
    """

    def __init__(self, model_id: str = DEFAULT_VAE_ID, dtype: torch.dtype = torch.float32):
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "KLVAECodec needs `diffusers` (pip install diffusers). Use "
                "--codec patchify for a weight-free run that needs no download."
            ) from exc

        self.vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=dtype)
        self.model_id = model_id
        self.latent_channels = int(self.vae.config.latent_channels)
        # 2 ** (number of downsampling stages), which diffusers encodes as one
        # `block_out_channels` entry per stage.
        self.downsample_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.scaling_factor = float(self.vae.config.scaling_factor)
        self.freeze()

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        # diffusers expects [-1, 1]; the codec boundary is [0, 1].
        x = frames.to(dtype=next(self.vae.parameters()).dtype) * 2.0 - 1.0
        posterior = self.vae.encode(x).latent_dist
        # `.mode()`, not `.sample()`: see the module docstring.
        return posterior.mode() * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(dtype=next(self.vae.parameters()).dtype) / self.scaling_factor
        frames = self.vae.decode(z).sample
        return ((frames + 1.0) / 2.0).clamp(0.0, 1.0)


class PatchifyCodec(PixelCodec):
    """Weight-free space-to-depth codec: an exact, lossless pixel rearrangement.

    Encoding folds each `f` x `f` RGB block into `3 * f**2` channels and centres
    the result; decoding is the exact inverse. There is no compression and no
    perceptual prior, so `latent_channels` is large (192 at f=8) and any decoder
    trained on it is doing a much harder job than one trained on a KL VAE. Use
    it for plumbing tests and roundtrip controls, not for the comparison panels
    the deliverable is actually about.
    """

    def __init__(self, downsample_factor: int = 8):
        super().__init__()
        self.downsample_factor = int(downsample_factor)
        self.latent_channels = 3 * self.downsample_factor**2
        # Centred [0,1] pixels have std ~0.29; scale to roughly unit variance so
        # the flow objective sees the same target scale as with a KL VAE.
        self.scaling_factor = 3.0
        self.model_id = f"patchify-f{self.downsample_factor}"
        self.freeze()

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        f = self.downsample_factor
        b, c, h, w = frames.shape
        if h % f or w % f:
            raise ValueError(f"frame {h}x{w} is not divisible by the patchify factor {f}.")
        x = frames.reshape(b, c, h // f, f, w // f, f).permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(b, c * f * f, h // f, w // f)
        return (x - 0.5) * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        f = self.downsample_factor
        b, cff, h, w = latents.shape
        c = cff // (f * f)
        x = latents / self.scaling_factor + 0.5
        x = x.reshape(b, c, f, f, h, w).permute(0, 1, 4, 2, 5, 3)
        return x.reshape(b, c, h * f, w * f).clamp(0.0, 1.0)


def build_codec(
    kind: str = "kl",
    model_id: str = DEFAULT_VAE_ID,
    downsample_factor: int = 8,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> PixelCodec:
    """Build and freeze one of the codecs. `kind` is 'kl' or 'patchify'."""
    if kind == "kl":
        codec: PixelCodec = KLVAECodec(model_id=model_id, dtype=dtype)
    elif kind == "patchify":
        codec = PatchifyCodec(downsample_factor=downsample_factor)
    else:
        raise ValueError(f"unknown codec kind: {kind!r} (expected 'kl' or 'patchify')")
    if device is not None:
        codec.to(device)
    logger.info(
        "codec %s: %d latent channel(s), downsample x%d, scaling factor %.5f",
        getattr(codec, "model_id", kind),
        codec.latent_channels,
        codec.downsample_factor,
        codec.scaling_factor,
    )
    return codec.freeze()


@torch.no_grad()
def roundtrip_psnr(codec: PixelCodec, frames: torch.Tensor) -> float:
    """PSNR of `decode(encode(frames))`, the reconstruction ceiling the flow
    decoder is measured against. Panel 1 reports it so a blurry sample is never
    misattributed to the decoder when the codec itself is the limit."""
    recon = codec.decode(codec.encode(frames))
    mse = F.mse_loss(recon.float(), frames.float()).item()
    return float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()
