# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import sys
import warnings

import torch
import yaml

import src.models.predictor as vit_pred
import src.models.vision_transformer as video_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, LinearDecaySchedule, WarmupCosineSchedule
from src.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

MAX_RETRIES = 3

# Buffers that are re-derived from the model config when the model is built (see
# RoPEAttention), so they may legitimately be absent from -- or left over in -- a
# checkpoint without any information being lost. This is what makes it possible to
# post-train starting from the released V-JEPA 2 checkpoints, which predate them.
RECOMPUTED_BUFFERS = ("temp_attn_mask",)

# The predictor's mask-token list is sized `len(mask) * len(dataset_fpcs)`, so a
# checkpoint trained on a different data mixture carries a different number of them
# (the released V-JEPA 2 checkpoints hold 10). Only `mask_index < len(dataset_fpcs)`
# is ever used at train time -- see PredictorMultiSeqWrapper -- so the surplus
# trailing tokens of a bigger checkpoint can be dropped without losing anything
# this run would have used.
MASK_TOKEN_KEY = re.compile(r"(.*\.)?mask_tokens\.\d+$")


def build_eval_args(
    model_name,
    patch_size,
    tubelet_size,
    num_frames,
    logging_folder,
    checkpoint,
    write_tag,
    eval_cfg_paths,
    uniform_power=False,
    use_sdpa=False,
    clip_duration=None,
    use_silu=False,
    wide_silu=True,
    tag=None,
):
    """
    Helper function to parse the pre-training configs to construct the
    evaluation configs, return as a list of eval configs.
    """
    # By convention, the pre-training config should specify any required evals
    # in the 'evals' key
    if eval_cfg_paths is None:
        logger.info("No evaluations specified!")
        return

    eval_nodes = None
    eval_tasks_per_node = None
    args_eval = []
    for i, f in enumerate(eval_cfg_paths):
        with open(f, "r") as y_file:
            _args = yaml.load(y_file, Loader=yaml.FullLoader)
            _tag = _args.get("tag", "")
            _args["tag"] = f"{tag}-{_tag}"
            _nodes = _args.get("nodes", None)
            _tasks = _args.get("tasks_per_node", 8)
            eval_nodes = _nodes if eval_nodes is None else eval_nodes
            eval_tasks_per_node = _tasks if eval_tasks_per_node is None else eval_tasks_per_node
            if (eval_nodes != _nodes) or (eval_tasks_per_node != _tasks):
                warnings.warn("Configs for online evals must use same number of nodes for slurm-batch processing")

            # Model params
            _args["pretrain"] = {}
            _args["pretrain"]["model_name"] = model_name
            _args["pretrain"]["patch_size"] = patch_size
            _args["pretrain"]["tubelet_size"] = tubelet_size
            _args["pretrain"]["uniform_power"] = uniform_power
            _args["pretrain"]["use_sdpa"] = use_sdpa
            _args["pretrain"]["clip_duration"] = clip_duration
            _args["pretrain"]["use_silu"] = use_silu
            _args["pretrain"]["wide_silu"] = wide_silu

            # Data params
            _args["pretrain"]["frames_per_clip"] = num_frames

            # Misc
            _args["pretrain"]["folder"] = logging_folder
            _args["pretrain"]["checkpoint"] = checkpoint
            _args["pretrain"]["write_tag"] = write_tag

            args_eval += [_args]

    return eval_nodes, eval_tasks_per_node, args_eval


def load_module_state_dict(module, pretrained_dict, tag, epoch, allow_missing=None):
    """Load `pretrained_dict` into `module`, tolerating mismatches that only involve
    the buffers listed in RECOMPUTED_BUFFERS or surplus trailing mask tokens. Any
    other missing/unexpected key is still an error, exactly as with a strict load.

    :param allow_missing: optional compiled regex; keys the model has and the checkpoint
        does not are permitted (and logged) when they match it. Used to initialize new
        parameters -- e.g. guidance cross-attention -- from a checkpoint that predates them.
    """
    # Recomputed buffers must be dropped *before* the load, not filtered out of the
    # result: their shape follows the model config (temp_attn_mask is (grid_depth * P)
    # square), so a checkpoint trained on a different clip length carries a differently
    # sized one -- and load_state_dict raises on a shape mismatch even with strict=False.
    # The module built its own from its own config, which is the one that is correct here.
    stale_buffers = [k for k in pretrained_dict if k.endswith(RECOMPUTED_BUFFERS)]
    if stale_buffers:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in stale_buffers}

    missing, unexpected = module.load_state_dict(pretrained_dict, strict=False)
    recomputed = stale_buffers + [k for k in missing + unexpected if k.endswith(RECOMPUTED_BUFFERS)]
    missing = [k for k in missing if not k.endswith(RECOMPUTED_BUFFERS)]
    unexpected = [k for k in unexpected if not k.endswith(RECOMPUTED_BUFFERS)]

    newly_initialized = []
    if allow_missing is not None:
        newly_initialized = [k for k in missing if allow_missing.match(k)]
        missing = [k for k in missing if k not in newly_initialized]
    # Mask tokens are indexed, so the ones this model does have were already loaded
    # from their same-index counterpart in the checkpoint.
    surplus_mask_tokens = [k for k in unexpected if MASK_TOKEN_KEY.match(k)]
    unexpected = [k for k in unexpected if k not in surplus_mask_tokens]
    if missing or unexpected:
        raise RuntimeError(
            f"Error(s) in loading state_dict for {tag}: missing keys {missing}, unexpected keys {unexpected}"
        )
    if newly_initialized:
        logger.warning(
            f"{tag}: {len(newly_initialized)} parameter(s) were not in the checkpoint and are freshly "
            f"initialized, e.g. {newly_initialized[:3]}"
        )
    if surplus_mask_tokens:
        logger.warning(
            f"{tag}: checkpoint holds {len(surplus_mask_tokens)} more mask token(s) than this config builds "
            f"(len(mask) * len(dataset_fpcs)); dropped {surplus_mask_tokens}. Only mask_index < len(dataset_fpcs) "
            f"is ever used, so this is harmless as long as the retained tokens cover every dataset."
        )
    msg = "<All keys matched successfully>" if not recomputed else f"<{len(recomputed)} recomputed buffer(s) ignored>"
    logger.info(f"loaded pretrained {tag} from epoch {epoch} with msg: {msg}")


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    is_anneal=False,
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    if not is_anneal:
        epoch = checkpoint["epoch"]

    # -- loading encoder
    load_module_state_dict(encoder, checkpoint["encoder"], "encoder", epoch)

    # -- loading predictor
    load_module_state_dict(predictor, checkpoint["predictor"], "predictor", epoch)

    # -- loading target_encoder
    if target_encoder is not None:
        load_module_state_dict(target_encoder, checkpoint["target_encoder"], "target encoder", epoch)

    # -- loading optimizer
    try:
        opt.load_state_dict(checkpoint["opt"])
        logger.info(f"loaded optimizers from epoch {epoch}")
    except ValueError as e:
        # The released V-JEPA 2 checkpoints store optimizer param-groups holding one
        # extra (pos-embed) parameter per model, which a use_rope model does not
        # have, so its state cannot be mapped onto our param groups. Restarting the
        # moment estimates is the sane fallback when post-training from them.
        logger.warning(f"could not load optimizer state from {r_path} ({e}); starting from a fresh optimizer state")
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    use_activation_checkpointing=False,
    is_causal=False
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
        is_causal=is_causal
    )
    encoder = MultiSeqWrapper(encoder)
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=is_causal
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {"params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
        {"params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
