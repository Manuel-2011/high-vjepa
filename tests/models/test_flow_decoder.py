# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Invariants of the flow-matching decoder.

These tests are deliberately about the *contract* rather than about output
quality: that one architecture accepts latents of varying length and width, that
the coordinate encoding is grid-relative and carries no per-index table, that
sampling is deterministic, that padding never leaks, and that dropping the latent
really removes it. Every one of them corresponds to a claim the comparison
harness relies on being true - if any fails, the panels stop being a controlled
comparison and start being decoration.
"""

import unittest

import torch

from src.models.flow_decoder.blocks import PerceiverResampler, masked_mean
from src.models.flow_decoder.decoder import FlowMatchingDecoder
from src.models.flow_decoder.flow import (
    FlowConfig,
    RectifiedFlow,
    sample_ode,
    sample_timesteps,
    timestep_schedule,
)
from src.models.flow_decoder.latent_adapter import (
    LatentNormalization,
    build_token_coords,
    latent_grid_from_geometry,
)
from src.models.flow_decoder.vae import PatchifyCodec

CODEC_CHANNELS = 4
CODEC_SIZE = 16


def make_decoder(latent_dim=64, grid=(1, 8, 8), conditioning="perceiver", **kwargs):
    return FlowMatchingDecoder(
        latent_dim=latent_dim,
        latent_grid=grid,
        codec_channels=CODEC_CHANNELS,
        codec_size=CODEC_SIZE,
        patch_size=2,
        dim=64,
        depth=2,
        num_heads=4,
        cond_dim=48,
        conditioning=conditioning,
        num_queries=16,
        resampler_depth=1,
        resampler_heads=4,
        **kwargs,
    )


def perturb_zero_init(decoder, std=0.02):
    """Give a freshly built decoder non-zero AdaLN gates.

    At initialization every AdaLN-Zero gate and the output projection are exactly
    zero, so the network is the identity on its input and its output does not
    depend on the conditioning memory *at all* - which is correct, and is what
    `TestFlowObjective.test_velocity_target_scale_at_init` checks. But it means a
    fresh decoder cannot be used to test that `z` reaches the output: nothing
    reaches the output. Tests that need a live conditioning path perturb the gates
    here, which stands in for "has been trained for a few steps".
    """
    with torch.no_grad():
        for block in decoder.blocks:
            block.modulation[-1].weight.normal_(0, std)
        decoder.final.modulation[-1].weight.normal_(0, std)
        decoder.final.proj.weight.normal_(0, std)
    return decoder


def make_inputs(batch=2, length=64, latent_dim=64, grid=(1, 8, 8)):
    return {
        "x_tau": torch.randn(batch, CODEC_CHANNELS, CODEC_SIZE, CODEC_SIZE),
        "tau": torch.rand(batch),
        "cond_latent": torch.randn(batch, CODEC_CHANNELS, CODEC_SIZE, CODEC_SIZE),
        "z": torch.randn(batch, length, latent_dim),
        "coords": build_token_coords(grid),
    }


class TestCoordinateEncoding(unittest.TestCase):
    def test_coords_are_grid_relative(self):
        """Two grids of different density must span the same coordinate range.

        This is the property that lets one decoder see a 16x16 and a 32x32 latent
        as the same image sampled at two densities, which a learned per-index
        table cannot express.
        """
        coarse = build_token_coords((1, 4, 4))
        fine = build_token_coords((1, 16, 16))
        for tensor in (coarse, fine):
            self.assertGreater(tensor[:, 1].min(), -1.0)
            self.assertLess(tensor[:, 1].max(), 1.0)
        # Cell centres, so the extremes sit half a cell inside the range and the
        # coarse grid's extremes sit further in than the fine grid's.
        self.assertLess(coarse[:, 1].max(), fine[:, 1].max())

    def test_extent_channels_report_density(self):
        self.assertAlmostEqual(build_token_coords((1, 4, 4))[0, 4].item(), 0.5, places=6)
        self.assertAlmostEqual(build_token_coords((1, 16, 16))[0, 4].item(), 0.125, places=6)

    def test_singleton_axis_covers_the_range(self):
        coords = build_token_coords((1, 8, 8))
        self.assertTrue(torch.all(coords[:, 0] == 0.0))
        self.assertAlmostEqual(coords[0, 3].item(), 2.0, places=6)

    def test_row_major_flattening_matches_the_mask_generators(self):
        coords = build_token_coords((2, 4, 4))
        # Token t*H*W + y*W + x; token 16 is the first of temporal index 1.
        self.assertLess(coords[0, 0], coords[16, 0])
        self.assertAlmostEqual(coords[0, 1].item(), coords[16, 1].item(), places=6)

    def test_no_learned_positional_table_exists(self):
        """The spec forbids learned per-index latent position embeddings."""
        decoder = make_decoder()
        for name, param in decoder.adapter.named_parameters():
            self.assertNotIn("pos_embed", name)
            # A per-index table would have to be at least as long as the grid.
            self.assertNotEqual(param.shape[:1], torch.Size([64]))

    def test_grid_from_geometry_rejects_non_square(self):
        self.assertEqual(latent_grid_from_geometry(1, 256), (1, 16, 16))
        with self.assertRaises(ValueError):
            latent_grid_from_geometry(1, 200)


class TestVaryingLatentShapes(unittest.TestCase):
    def test_same_module_accepts_varying_length(self):
        """One decoder instance, three latent lengths, no architectural change."""
        decoder = make_decoder(latent_dim=64, grid=(1, 8, 8))
        for grid in ((1, 4, 4), (1, 8, 8), (2, 8, 8), (1, 16, 16)):
            length = grid[0] * grid[1] * grid[2]
            inputs = make_inputs(length=length, grid=grid)
            out = decoder(**inputs)
            self.assertEqual(out.shape, (2, CODEC_CHANNELS, CODEC_SIZE, CODEC_SIZE))

    def test_varying_width_needs_only_the_adapter(self):
        """Changing d_m must change exactly one parameter tensor's shape."""
        narrow = make_decoder(latent_dim=64)
        wide = make_decoder(latent_dim=1408)
        differing = [
            name
            for (name, a), (_, b) in zip(narrow.named_parameters(), wide.named_parameters())
            if a.shape != b.shape
        ]
        self.assertEqual(differing, ["adapter.proj.weight"])

    def test_direct_conditioning_accepts_varying_length(self):
        decoder = make_decoder(conditioning="direct")
        for grid in ((1, 4, 4), (2, 8, 8)):
            length = grid[0] * grid[1] * grid[2]
            out = decoder(**make_inputs(length=length, grid=grid))
            self.assertEqual(out.shape[-1], CODEC_SIZE)


class TestMasking(unittest.TestCase):
    def test_padding_does_not_change_the_output(self):
        """Padded tokens must be invisible: the whole point of the mask.

        Config B attends to the latent tokens directly, so a batch mixing two
        lengths would otherwise let one sample's padding change the other's
        reconstruction.
        """
        torch.manual_seed(0)
        decoder = make_decoder(conditioning="direct").eval()
        inputs = make_inputs(batch=1, length=16, grid=(1, 4, 4))
        with torch.no_grad():
            reference = decoder(**inputs, z_mask=torch.ones(1, 16, dtype=torch.bool))

        padded = dict(inputs)
        padded["z"] = torch.cat([inputs["z"], torch.randn(1, 48, 64) * 100.0], dim=1)
        padded["coords"] = torch.cat([inputs["coords"], torch.randn(48, 6)], dim=0)
        mask = torch.zeros(1, 64, dtype=torch.bool)
        mask[:, :16] = True
        with torch.no_grad():
            masked = decoder(**padded, z_mask=mask)
        torch.testing.assert_close(reference, masked, rtol=1e-4, atol=1e-5)

    def test_masked_mean_ignores_padding(self):
        tokens = torch.stack([torch.ones(4, 3), torch.zeros(4, 3)], dim=0)
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
        pooled = masked_mean(tokens, mask)
        torch.testing.assert_close(pooled[0], torch.ones(3))
        torch.testing.assert_close(pooled[1], torch.zeros(3))

    def test_resampler_output_length_is_fixed(self):
        resampler = PerceiverResampler(cond_dim=32, num_queries=9, depth=1, num_heads=4)
        for length in (5, 50, 500):
            out = resampler(torch.randn(2, length, 32))
            self.assertEqual(out.shape, (2, 9, 32))


class TestLatentDropout(unittest.TestCase):
    def test_dropping_the_latent_removes_it(self):
        """A dropped row must not depend on z at all - CFG relies on this."""
        torch.manual_seed(0)
        decoder = make_decoder().eval()
        inputs = make_inputs(batch=1)
        drop = torch.ones(1, dtype=torch.bool)
        with torch.no_grad():
            a = decoder(**inputs, drop_latent=drop)
            other = dict(inputs)
            other["z"] = torch.randn_like(inputs["z"]) * 50.0
            b = decoder(**other, drop_latent=drop)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)

    def test_keeping_the_latent_uses_it(self):
        torch.manual_seed(0)
        decoder = perturb_zero_init(make_decoder().eval())
        inputs = make_inputs(batch=1)
        with torch.no_grad():
            a = decoder(**inputs)
            other = dict(inputs)
            other["z"] = torch.randn_like(inputs["z"])
            b = decoder(**other)
        self.assertGreater((a - b).abs().max().item(), 1e-6)

    def test_x_t_is_never_dropped(self):
        """The unconditional branch must still see the current frame."""
        torch.manual_seed(0)
        decoder = perturb_zero_init(make_decoder().eval())
        inputs = make_inputs(batch=1)
        drop = torch.ones(1, dtype=torch.bool)
        with torch.no_grad():
            a = decoder(**inputs, drop_latent=drop)
            other = dict(inputs)
            other["cond_latent"] = torch.randn_like(inputs["cond_latent"])
            b = decoder(**other, drop_latent=drop)
        self.assertGreater((a - b).abs().max().item(), 1e-6)


class TestNormalization(unittest.TestCase):
    def test_channel_mode_uses_fitted_buffers(self):
        norm = LatentNormalization(4, mode="channel")
        z = torch.randn(2, 5, 4) * 3.0 + 7.0
        mean = z.reshape(-1, 4).mean(0)
        std = z.reshape(-1, 4).std(0)
        norm.fit(mean, std)
        self.assertTrue(bool(norm.fitted))
        out = norm(z)
        torch.testing.assert_close(out.reshape(-1, 4).mean(0), torch.zeros(4), atol=1e-5, rtol=1e-4)

    def test_buffers_survive_a_state_dict_roundtrip(self):
        """The panels rebuild a decoder from a checkpoint; buffers must persist."""
        a = make_decoder()
        a.adapter.norm.fit(torch.full((64,), 2.0), torch.full((64,), 3.0))
        b = make_decoder()
        b.load_state_dict(a.state_dict())
        torch.testing.assert_close(b.adapter.norm.mean, torch.full((64,), 2.0))
        torch.testing.assert_close(b.adapter.norm.std, torch.full((64,), 3.0))
        self.assertTrue(bool(b.adapter.norm.fitted))

    def test_token_mode_needs_no_buffers(self):
        norm = LatentNormalization(8, mode="token")
        out = norm(torch.randn(2, 3, 8) * 5.0 + 1.0)
        torch.testing.assert_close(out.mean(dim=-1), torch.zeros(2, 3), atol=1e-5, rtol=1e-4)


class TestFlowObjective(unittest.TestCase):
    def test_velocity_target_scale_at_init(self):
        """A zero-init decoder predicts 0, so the loss must start at Var(x1 - eps).

        With unit-variance targets that is 2.0. A different number at step 0 means
        the AdaLN-Zero / output-projection zero-initialization has been broken,
        which silently costs a chunk of a fixed step budget.
        """
        torch.manual_seed(0)
        decoder = make_decoder()
        flow = RectifiedFlow(decoder, FlowConfig(latent_dropout=0.0, sigma_max=0.0))
        target = torch.randn(64, CODEC_CHANNELS, CODEC_SIZE, CODEC_SIZE)
        cond = torch.randn_like(target)
        z = torch.randn(64, 64, 64)
        loss, metrics = flow.loss(target, cond, z, coords=build_token_coords((1, 8, 8)))
        self.assertAlmostEqual(loss.item(), 2.0, delta=0.15)
        self.assertAlmostEqual(metrics["target_std"], 1.0, delta=0.05)

    def test_gradients_reach_every_trainable_tensor(self):
        """Both branches together must touch every parameter.

        `null_memory` is only on the graph when a row is actually dropped, and
        the resampler/adapter only when a row is kept, so a single batch at
        p = 0.1 reaches all parameters *usually* rather than always. Driving the
        two extremes makes the check deterministic and says which parameter
        belongs to which branch.
        """
        decoder = make_decoder()
        for dropout in (0.0, 1.0):
            flow = RectifiedFlow(decoder, FlowConfig(latent_dropout=dropout))
            loss, _ = flow.loss(
                torch.randn(4, CODEC_CHANNELS, CODEC_SIZE, CODEC_SIZE),
                torch.randn(4, CODEC_CHANNELS, CODEC_SIZE, CODEC_SIZE),
                torch.randn(4, 64, 64),
                coords=build_token_coords((1, 8, 8)),
            )
            loss.backward()
        missing = [n for n, p in decoder.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])

    def test_null_memory_is_trained_only_by_dropped_rows(self):
        """Makes the coupling above explicit: no dropout, no null-memory gradient.

        Which is also the reason `flow.latent_dropout` must stay well above zero -
        a decoder whose null branch never trained produces unstable guided
        samples, and guidance is what panels 2 and 6 are built on.
        """
        decoder = make_decoder()
        flow = RectifiedFlow(decoder, FlowConfig(latent_dropout=0.0))
        loss, _ = flow.loss(
            torch.randn(4, CODEC_CHANNELS, CODEC_SIZE, CODEC_SIZE),
            torch.randn(4, CODEC_CHANNELS, CODEC_SIZE, CODEC_SIZE),
            torch.randn(4, 64, 64),
            coords=build_token_coords((1, 8, 8)),
        )
        loss.backward()
        self.assertIsNone(decoder.null_memory.grad)

    def test_timestep_modes_stay_in_the_unit_interval(self):
        for mode in ("uniform", "logit_normal", "cosine_shift"):
            tau = sample_timesteps(512, torch.device("cpu"), mode=mode)
            self.assertGreaterEqual(tau.min().item(), 0.0)
            self.assertLessEqual(tau.max().item(), 1.0)

    def test_logit_normal_concentrates_mid_path(self):
        uniform = sample_timesteps(4096, torch.device("cpu"), mode="uniform")
        logit = sample_timesteps(4096, torch.device("cpu"), mode="logit_normal")
        self.assertLess(logit.std().item(), uniform.std().item())

    def test_schedule_spans_the_unit_interval(self):
        for shift in (1.0, 3.0):
            knots = timestep_schedule(8, torch.device("cpu"), shift=shift)
            self.assertEqual(knots.numel(), 9)
            self.assertAlmostEqual(knots[0].item(), 0.0, places=6)
            self.assertAlmostEqual(knots[-1].item(), 1.0, places=6)
            self.assertTrue(bool(torch.all(knots[1:] > knots[:-1])))


class TestSampler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.decoder = perturb_zero_init(make_decoder().eval())
        self.cond = torch.randn(2, CODEC_CHANNELS, CODEC_SIZE, CODEC_SIZE)
        self.z = torch.randn(2, 64, 64)
        self.coords = build_token_coords((1, 8, 8))

    def test_sampling_is_deterministic(self):
        kwargs = dict(coords=self.coords, num_steps=4, guidance=2.0, solver="heun", seed=3)
        a = sample_ode(self.decoder, self.cond, self.z, **kwargs)
        b = sample_ode(self.decoder, self.cond, self.z, **kwargs)
        torch.testing.assert_close(a, b)

    def test_all_solvers_run_and_stay_finite(self):
        for solver in ("euler", "midpoint", "heun"):
            out = sample_ode(
                self.decoder, self.cond, self.z, coords=self.coords, num_steps=3, solver=solver, seed=0
            )
            self.assertTrue(bool(torch.isfinite(out).all()))
            self.assertEqual(out.shape, self.cond.shape)

    def test_guidance_one_matches_the_unguided_field(self):
        a = sample_ode(self.decoder, self.cond, self.z, coords=self.coords, num_steps=3, guidance=1.0, seed=1)
        b = sample_ode(self.decoder, self.cond, self.z, coords=self.coords, num_steps=3, guidance=1.0, seed=1)
        torch.testing.assert_close(a, b)

    def test_guidance_changes_the_result(self):
        a = sample_ode(self.decoder, self.cond, self.z, coords=self.coords, num_steps=3, guidance=1.0, seed=1)
        b = sample_ode(self.decoder, self.cond, self.z, coords=self.coords, num_steps=3, guidance=4.0, seed=1)
        self.assertGreater((a - b).abs().max().item(), 1e-6)

    def test_trajectory_has_one_state_per_knot(self):
        traj = sample_ode(
            self.decoder, self.cond, self.z, coords=self.coords, num_steps=5, seed=0, return_trajectory=True
        )
        self.assertEqual(traj.shape[0], 6)

    def test_fresh_decoder_is_the_identity_on_its_conditioning(self):
        """A zero-init decoder ignores z entirely - that IS AdaLN-Zero working.

        Recorded as a test because it is a real trap: any test of "does z reach
        the output" written against a fresh decoder passes vacuously in the wrong
        direction. See `perturb_zero_init`.
        """
        fresh = make_decoder().eval()
        kwargs = dict(coords=self.coords, num_steps=3, guidance=3.0, seed=0)
        a = sample_ode(fresh, self.cond, self.z, **kwargs)
        b = sample_ode(fresh, self.cond, torch.randn_like(self.z), **kwargs)
        torch.testing.assert_close(a, b)

    def test_latent_changes_the_sample(self):
        """The sample must depend on z, or the whole microscope is a mirror."""
        a = sample_ode(self.decoder, self.cond, self.z, coords=self.coords, num_steps=3, seed=0)
        b = sample_ode(
            self.decoder, self.cond, torch.randn_like(self.z), coords=self.coords, num_steps=3, seed=0
        )
        self.assertGreater((a - b).abs().max().item(), 1e-6)


class TestPatchifyCodec(unittest.TestCase):
    def test_roundtrip_is_exact(self):
        codec = PatchifyCodec(downsample_factor=4)
        frames = torch.rand(2, 3, 32, 32)
        torch.testing.assert_close(codec.decode(codec.encode(frames)), frames, atol=1e-6, rtol=1e-5)

    def test_latent_shape(self):
        codec = PatchifyCodec(downsample_factor=8)
        self.assertEqual(codec.latent_channels, 192)
        self.assertEqual(codec.encode(torch.rand(1, 3, 64, 64)).shape, (1, 192, 8, 8))
        self.assertEqual(codec.latent_size(256), 32)

    def test_codec_is_frozen(self):
        codec = PatchifyCodec()
        self.assertFalse(codec.training)
        self.assertEqual([p for p in codec.parameters() if p.requires_grad], [])


if __name__ == "__main__":
    unittest.main()
