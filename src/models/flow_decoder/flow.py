# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Rectified-flow training objective and the deterministic ODE sampler.

Convention. The path is the straight line between noise and data,

    x_tau = (1 - tau) * eps + tau * x1,      eps ~ N(0, I),  tau in [0, 1]

so `tau = 0` is pure noise, `tau = 1` is the codec latent of the target frame,
and the velocity the decoder regresses is the constant

    v* = dx_tau / dtau = x1 - eps.

The loss is a plain MSE on that velocity. No GAN, no perceptual loss, no
discriminator - the spec prohibits them, and for this deliverable that is the
right call for a reason worth recording: an adversarial term buys apparent
sharpness by letting the model invent detail, and invented detail is
indistinguishable from detail the latent actually carried. That would destroy the
one property the panels depend on.

Sampling integrates the same ODE from `tau = 0` to `tau = 1` with a fixed step
schedule and no injected noise, so a (seed, latent) pair maps to exactly one
image. Determinism is not a nicety here: panels 3 and 6 hold the seed fixed and
vary the latent or the world model, and any stochasticity in the solver would
show up as a difference the reader would attribute to the latent.

Classifier-free guidance drops only `z`. The extrapolated field

    v_w = v_uncond + w * (v_cond - v_uncond)

is exactly the conditional field at `w = 1`; above 1 it amplifies whatever the
latent contributes over the unconditional branch. What that branch *is* depends
on the decoder's `frame_conditioning` (see decoder.py): under the default `none`
it is the decoder's prior over frames of this dataset, so the ladder measures
what the latent adds over "a generic frame"; under `current_frame` it still sees
`x_t`, so the ladder measures what the latent adds over "nothing moved". Both are
legible measurements, but they are not the same measurement, and a panel caption
that names the wrong one is worse than no caption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

TIMESTEP_MODES = ("logit_normal", "uniform", "cosine_shift")
SOLVERS = ("euler", "midpoint", "heun")


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    mode: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw flow times in (0, 1).

    `logit_normal` (the default, as in SD3) concentrates samples near tau = 0.5,
    where the velocity field is hardest to predict and most of the perceptible
    structure is decided; uniform sampling spends a lot of the step budget near
    the endpoints, where the field is nearly trivial. On a *fixed* step budget -
    which the spec mandates, with no early stopping - how the budget is spent
    across tau is the single most consequential choice in this file, so it is a
    global setting and never tuned per world model.
    """
    if mode == "uniform":
        return torch.rand(batch_size, device=device, generator=generator)
    if mode == "logit_normal":
        normal = torch.randn(batch_size, device=device, generator=generator) * logit_std + logit_mean
        return torch.sigmoid(normal)
    if mode == "cosine_shift":
        u = torch.rand(batch_size, device=device, generator=generator)
        return 1.0 - torch.cos(u * torch.pi / 2.0)
    raise ValueError(f"unknown timestep mode {mode!r}; expected one of {TIMESTEP_MODES}")


@dataclass
class FlowConfig:
    """Objective-side settings. Global for a comparison, never per world model."""

    timestep_mode: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    # Probability of replacing the latent memory with the null token, which is
    # what makes classifier-free guidance available at sampling time. 0.1 is the
    # usual value; much higher starts costing conditional quality, much lower
    # leaves the unconditional branch undertrained and makes guided samples
    # unstable.
    latent_dropout: float = 0.1
    # Latent noise augmentation, sigma ~ U(0, sigma_max), applied to the
    # *normalized* latent and fed to the model as a conditioning scalar. Because
    # sigma is an input, the decoder learns the whole family of noise levels at
    # once and can be asked for sigma = 0 at sampling time; the augmentation
    # therefore buys robustness to off-distribution rollout latents without
    # blurring the clean-latent case.
    sigma_max: float = 0.5
    sigma_prob: float = 0.5
    # There is deliberately no dropout for x_t. It is always available at
    # sampling time, and keeping it in both CFG branches is what makes the
    # guidance scale a measurement of the latent alone (see decoder.py).


class RectifiedFlow(nn.Module):
    """Wraps a `FlowMatchingDecoder` with its training objective.

    Holds no parameters of its own, so a checkpoint of `decoder` is complete and
    the objective can be swapped without touching a saved model.
    """

    def __init__(self, decoder: nn.Module, config: Optional[FlowConfig] = None):
        super().__init__()
        self.decoder = decoder
        self.config = config or FlowConfig()

    def loss(
        self,
        target_latent: torch.Tensor,
        z: torch.Tensor,
        cond_latent: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        z_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """MSE between the predicted and true velocity, plus diagnostics.

        `target_latent` is the codec latent of the frame being reconstructed and
        `z` the world model's tokens for it. `cond_latent` is the codec latent of
        the preceding frame, required only when the decoder was built with
        `frame_conditioning='current_frame'`. Everything stochastic (tau, eps,
        the dropout mask, sigma) is drawn here so a training step is one call.
        """
        cfg = self.config
        b = target_latent.size(0)
        device = target_latent.device

        tau = sample_timesteps(
            b,
            device,
            mode=cfg.timestep_mode,
            logit_mean=cfg.logit_mean,
            logit_std=cfg.logit_std,
            generator=generator,
        )
        eps = torch.randn(target_latent.shape, device=device, generator=generator, dtype=target_latent.dtype)
        tau_b = tau.reshape(-1, 1, 1, 1).to(target_latent.dtype)
        x_tau = (1.0 - tau_b) * eps + tau_b * target_latent
        velocity_target = target_latent - eps

        drop_latent = (
            torch.rand(b, device=device, generator=generator) < cfg.latent_dropout
            if cfg.latent_dropout > 0
            else torch.zeros(b, dtype=torch.bool, device=device)
        )
        if cfg.sigma_max > 0 and cfg.sigma_prob > 0:
            active = torch.rand(b, device=device, generator=generator) < cfg.sigma_prob
            sigma = torch.rand(b, device=device, generator=generator) * cfg.sigma_max * active
        else:
            sigma = torch.zeros(b, device=device)

        prediction = self.decoder(
            x_tau=x_tau,
            tau=tau,
            z=z,
            cond_latent=cond_latent,
            coords=coords,
            z_mask=z_mask,
            sigma=sigma,
            drop_latent=drop_latent,
        )
        per_sample = (prediction.float() - velocity_target.float()).pow(2).mean(dim=(1, 2, 3))
        loss = per_sample.mean()

        with torch.no_grad():
            metrics = {
                "loss": loss.item(),
                "loss_cond": per_sample[~drop_latent].mean().item() if (~drop_latent).any() else float("nan"),
                "loss_uncond": per_sample[drop_latent].mean().item() if drop_latent.any() else float("nan"),
                "tau_mean": tau.mean().item(),
                "sigma_mean": sigma.mean().item(),
                "target_std": target_latent.float().std().item(),
            }
        return loss, metrics


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def timestep_schedule(num_steps: int, device: torch.device, shift: float = 1.0) -> torch.Tensor:
    """`num_steps + 1` knots from 0 to 1.

    `shift > 1` bunches knots near tau = 0, where the field turns fastest under a
    logit-normal training distribution; `shift = 1` is uniform. Same functional
    form as the SD3 timestep shift, kept as one scalar so a sampler setting is
    reproducible from a single number in a panel caption.
    """
    tau = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    if shift == 1.0:
        return tau
    return (shift * tau) / (1.0 + (shift - 1.0) * tau)


@torch.no_grad()
def sample_ode(
    decoder: nn.Module,
    z: torch.Tensor,
    cond_latent: Optional[torch.Tensor] = None,
    coords: Optional[torch.Tensor] = None,
    z_mask: Optional[torch.Tensor] = None,
    num_steps: int = 50,
    guidance: float = 1.5,
    solver: str = "heun",
    shift: float = 1.0,
    sigma: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    aug_noise: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """Integrate the velocity field from noise to a target codec latent.

    Deterministic given `(seed or noise, z, cond_latent, guidance, solver,
    num_steps, shift)`. Returns (B, C, h, w), or the stacked trajectory
    (num_steps + 1, B, C, h, w) if `return_trajectory`.

    `cond_latent` is required only under `frame_conditioning='current_frame'`;
    otherwise the output shape comes from `decoder.latent_shape`, since there is
    no conditioning image to read it off.

    The conditioning memory is built once, outside the solver loop: it does not
    depend on tau, and for Config B with a 1024-token latent the cross-attention
    keys are the bulk of the sampling cost.
    """
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; expected one of {SOLVERS}")
    device = z.device
    b = z.size(0)
    shape = decoder.latent_shape(b)
    compute_dtype = cond_latent.dtype if cond_latent is not None else z.dtype

    if noise is None:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        noise = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
    x = noise.to(device=device, dtype=torch.float32)

    memory, memory_mask, sigma_used = decoder.encode_conditioning(
        z, coords=coords, z_mask=z_mask, sigma=sigma, drop_latent=None, aug_noise=aug_noise
    )
    guided = guidance != 1.0
    if guided:
        null_memory, null_mask, null_sigma = decoder.encode_conditioning(
            z,
            coords=coords,
            z_mask=z_mask,
            sigma=None,
            drop_latent=torch.ones(b, dtype=torch.bool, device=device),
            aug_noise=aug_noise,
        )

    def velocity(state: torch.Tensor, tau_value: torch.Tensor) -> torch.Tensor:
        v_cond = decoder.forward_with_memory(
            state.to(compute_dtype), tau_value, cond_latent, memory, memory_mask, sigma_used
        ).float()
        if not guided:
            return v_cond
        v_uncond = decoder.forward_with_memory(
            state.to(compute_dtype), tau_value, cond_latent, null_memory, null_mask, null_sigma
        ).float()
        return v_uncond + guidance * (v_cond - v_uncond)

    knots = timestep_schedule(num_steps, device, shift=shift)
    trajectory = [x.clone()] if return_trajectory else None

    for i in range(num_steps):
        tau0, tau1 = knots[i], knots[i + 1]
        dt = (tau1 - tau0).item()
        t0 = tau0.expand(b)
        if solver == "euler":
            x = x + dt * velocity(x, t0)
        elif solver == "midpoint":
            v0 = velocity(x, t0)
            x_mid = x + 0.5 * dt * v0
            x = x + dt * velocity(x_mid, (tau0 + 0.5 * (tau1 - tau0)).expand(b))
        else:  # heun
            v0 = velocity(x, t0)
            x_euler = x + dt * v0
            # No correction on the final knot: tau = 1 is the data endpoint and
            # the field there is evaluated at a state that is already the answer,
            # so the extra call only adds numerical noise.
            if i == num_steps - 1:
                x = x_euler
            else:
                v1 = velocity(x_euler, tau1.expand(b))
                x = x + dt * 0.5 * (v0 + v1)
        if trajectory is not None:
            trajectory.append(x.clone())

    return torch.stack(trajectory) if trajectory is not None else x


@torch.no_grad()
def decode_frames(
    decoder: nn.Module,
    codec,
    z: torch.Tensor,
    cond_frames: Optional[torch.Tensor] = None,
    **sample_kwargs,
) -> torch.Tensor:
    """End-to-end D(z) -> x_hat in pixel space, in [0, 1].

    `cond_frames` (B, 3, H, W) in [0, 1] is needed only under
    `frame_conditioning='current_frame'`.
    """
    cond_latent = codec.encode(cond_frames) if cond_frames is not None else None
    latent = sample_ode(decoder, z=z, cond_latent=cond_latent, **sample_kwargs)
    return codec.decode(latent)
