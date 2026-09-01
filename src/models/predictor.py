# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from src.masks.utils import apply_masks
from src.models.utils.modules import Block, GuidedBlock
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import repeat_interleave_batch, trunc_normal_


class VisionTransformerPredictor(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        out_embed_dim=None,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        chop_last_n_tokens=0,
        use_rope=False,
        is_causal=False,
        use_guidance=False,
        guidance_dim=None,
        guidance_gate_init=0.0,
        guidance_step_ratio=4,
        guidance_window=None,
        **kwargs
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.chop_last_n_tokens = chop_last_n_tokens
        self.is_causal = is_causal
        # -- cross-attention to a frozen, longer-horizon world model
        self.use_guidance = use_guidance
        self.guidance_dim = embed_dim if guidance_dim is None else guidance_dim
        # How many steps of *this* model fit in one step of the guidance model, e.g. a
        # 0.5fps guidance model steps 2s at a time while a 4fps model (tubelet 2) steps
        # 0.5s at a time, so guidance_step_ratio == 4.
        self.guidance_step_ratio = guidance_step_ratio
        # Number of most-recent guidance steps a token may read (None -> every past one)
        self.guidance_window = guidance_window

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for i in range(num_mask_tokens)]
            )

        # Determine positional embedding
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.grid_depth = num_frames // self.tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
            )
        else:
            self.num_patches = num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        # Position embedding
        self.uniform_power = uniform_power

        self.predictor_pos_embed = None
        if not use_rope:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
            )

        # Attention Blocks
        self.use_rope = use_rope
        if use_guidance:
            assert use_rope, "Guidance cross-attention is defined in terms of RoPE positions"
            assert is_causal, "Guidance cross-attention is only defined for the causal predictor"
            block_fn = partial(GuidedBlock, guidance_gate_init=guidance_gate_init)
        else:
            block_fn = Block
        self.predictor_blocks = nn.ModuleList(
            [
                block_fn(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=self.grid_depth,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    is_causal=is_causal
                )
                for i in range(depth)
            ]
        )

        if out_embed_dim is None:
            teacher_embed_dim = kwargs.get("teacher_embed_dim", None)
            if teacher_embed_dim is not None:
                out_embed_dim = teacher_embed_dim
            else:
                out_embed_dim = embed_dim

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, out_embed_dim, bias=True)

        # Project the guidance model's latents (which live in *its* encoder's feature
        # space, e.g. 1024-d for a ViT-L) into this predictor's width. Shared by every
        # block so the projection is paid once per step.
        if use_guidance:
            self.guidance_norm = norm_layer(self.guidance_dim)
            self.guidance_proj = nn.Linear(self.guidance_dim, predictor_embed_dim, bias=True)

        # ------ initialize weights
        if self.predictor_pos_embed is not None:
            self._init_pos_embed(self.predictor_pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size  # TODO: update; currently assumes square input
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def build_guidance_positions(self, num_steps, num_guidance_steps, device):
        """Place this predictor's tokens and the guidance model's tokens on one shared
        (time, height, width) grid, and say which of the latter each of the former may read.

        Time is measured in units of *this* model's step (e.g. 0.5s for a 4fps model with
        tubelet 2), and a token's time coordinate is the time of the frame it *predicts*:

          - token `s` of this predictor has seen tubelets 0..s and predicts tubelet s+1,
            so its coordinate is `s + 1`;
          - guidance token `l` has seen guidance tubelets 0..l -- i.e. up to time
            `R * l` on this axis -- and predicts guidance tubelet l+1, so its coordinate
            is `R * (l + 1)`, with `R = guidance_step_ratio`.

        The rotation therefore sees exactly the gap between the two prediction horizons:
        a token predicting 0.5s ahead that reads a guidance token predicting 2s ahead
        gets a relative offset of 3 steps.

        Causality is preserved by only letting token `s` read guidance token `l` when the
        guidance model's *context* ends no later than this model's, i.e. `R * l <= s`.
        Guidance latents are predictions rather than observations, so nothing about the
        future leaks as long as that holds.

        :return: (q_pos, k_pos, xattn_mask)
        """
        P = self.grid_height * self.grid_width
        R = self.guidance_step_ratio

        # -- spatial coordinates, identical layout for both sides
        patch_ids = torch.arange(P, device=device)
        patch_h = (patch_ids // self.grid_width).float()
        patch_w = (patch_ids % self.grid_width).float()

        step_ids = torch.arange(num_steps, device=device)
        q_step = step_ids.repeat_interleave(P)
        q_pos = (
            (q_step + 1).float(),
            patch_h.repeat(num_steps),
            patch_w.repeat(num_steps),
        )

        guidance_ids = torch.arange(num_guidance_steps, device=device)
        k_step = guidance_ids.repeat_interleave(P)
        k_pos = (
            ((k_step + 1) * R).float(),
            patch_h.repeat(num_guidance_steps),
            patch_w.repeat(num_guidance_steps),
        )

        # -- [N_q, N_k] bool mask, True where the guidance token is readable
        lag = q_step.unsqueeze(1) - R * k_step.unsqueeze(0)
        xattn_mask = lag >= 0
        if self.guidance_window is not None:
            xattn_mask = xattn_mask & (lag < R * self.guidance_window)
        return q_pos, k_pos, xattn_mask

    def forward(self, x, masks_x=None, masks_y=None, mask_index=1, has_cls=False, guidance=None):
        """
        :param x: context tokens
        :param masks_x: indices of context tokens in input
        :params masks_y: indices of target tokens in input
        :param guidance: [B, num_guidance_steps * num_patches_per_frame, guidance_dim]
            latents predicted by a frozen, longer-horizon world model
        """
        if not self.is_causal:
            assert (masks_x is not None) and (masks_y is not None), "Cannot run predictor without mask indices"
            if not isinstance(masks_x, list):
                masks_x = [masks_x]
            if not isinstance(masks_y, list):
                masks_y = [masks_y]

        # Batch Size
        if self.is_causal:
            B = len(x)
        else:
            B = len(x) // len(masks_x)

        # Map context tokens to predictor dimensions
        x = self.predictor_embed(x)
        if has_cls:
            x_cls = x[:, :1, :]
            x = x[:, 1:, :]
        _, N_ctxt, D = x.shape

        # Add positional embedding to ctxt tokens
        if not self.use_rope:
            x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(x_pos_embed, masks_x)

        if not self.is_causal:
            # Make target tokens
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]
            pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
            pred_tokens = apply_masks(pred_tokens, masks_y)
            # -- add pos embed
            if not self.use_rope:
                pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
                pos_embs = apply_masks(pos_embs, masks_y)
                pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
                pred_tokens += pos_embs

            # Concatenate context & target tokens
            x = x.repeat(len(masks_x), 1, 1)
            x = torch.cat([x, pred_tokens], dim=1)

            # Positions of context & target tokens
            masks_x = torch.cat(masks_x, dim=0)
            masks_y = torch.cat(masks_y, dim=0)
            masks = torch.cat([masks_x, masks_y], dim=1)

            # Put tokens in sorted order
            argsort = torch.argsort(masks, dim=1)  # [B, N]
            masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
            x = torch.stack([x[i, row, :] for i, row in enumerate(argsort)], dim=0)

        # Remove the last n tokens of sorted sequence before processing
        if self.chop_last_n_tokens > 0:
            x = x[:, : -self.chop_last_n_tokens]
            masks = masks[:, : -self.chop_last_n_tokens]

        if has_cls:
            x = torch.cat([x_cls, x], dim=1)

        # Prepare guidance tokens and the space-time coordinates the cross-attention
        # rotates queries and keys with
        q_pos = k_pos = xattn_mask = None
        if guidance is not None:
            assert self.use_guidance, "Predictor was not built with use_guidance=True"
            P = self.grid_height * self.grid_width
            assert x.size(1) % P == 0, f"{x.size(1)} predictor tokens is not a whole number of frames"
            assert guidance.size(1) % P == 0, (
                f"{guidance.size(1)} guidance tokens is not a whole number of frames; the guidance model "
                "must use the same spatial grid as this predictor"
            )
            q_pos, k_pos, xattn_mask = self.build_guidance_positions(
                num_steps=x.size(1) // P, num_guidance_steps=guidance.size(1) // P, device=x.device
            )
            guidance = self.guidance_proj(self.guidance_norm(guidance))

        blk_kwargs = {}
        if guidance is not None:
            blk_kwargs = dict(guidance=guidance, q_pos=q_pos, k_pos=k_pos, xattn_mask=xattn_mask)

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, None if self.is_causal else masks, None, use_reentrant=False, **blk_kwargs
                )
            else:
                x = blk(x, mask=None if self.is_causal else masks, attn_mask=None, **blk_kwargs)
        x = self.predictor_norm(x)

        if has_cls:
            x = x[:, 1:, :]

        # Return output corresponding to target tokens
        if not self.return_all_tokens and not self.is_causal:
            reverse_argsort = torch.argsort(argsort, dim=1)  # [B, N]
            x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
            x = x[:, N_ctxt:]

        x = self.predictor_proj(x)

        return x


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
