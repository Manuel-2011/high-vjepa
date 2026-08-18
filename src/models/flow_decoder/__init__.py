# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Flow-matching decoder: a qualitative microscope for frozen world-model latents.

`D(x_t, z_{t+1}) -> x_hat_{t+1}` - reconstruct the next video frame at pixel
level from the current frame plus the latent tokens a frozen world model emits
for the next step. See `src/models/flow_decoder/decoder.py` for the design and
`evals/generate_flow_decoder_panels.py` for the six diagnostic panels this exists
to produce.
"""

from src.models.flow_decoder.decoder import CONDITIONING_MODES, FlowMatchingDecoder
from src.models.flow_decoder.flow import (
    FlowConfig,
    RectifiedFlow,
    decode_frames,
    sample_ode,
    sample_timesteps,
    timestep_schedule,
)
from src.models.flow_decoder.latent_adapter import (
    CoordinateEncoding,
    LatentAdapter,
    LatentNormalization,
    build_token_coords,
    latent_grid_from_geometry,
)
from src.models.flow_decoder.vae import PixelCodec, build_codec, roundtrip_psnr

__all__ = [
    "CONDITIONING_MODES",
    "CoordinateEncoding",
    "FlowConfig",
    "FlowMatchingDecoder",
    "LatentAdapter",
    "LatentNormalization",
    "PixelCodec",
    "RectifiedFlow",
    "build_codec",
    "build_token_coords",
    "decode_frames",
    "latent_grid_from_geometry",
    "roundtrip_psnr",
    "sample_ode",
    "sample_timesteps",
    "timestep_schedule",
]
