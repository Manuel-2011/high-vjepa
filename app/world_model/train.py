# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Goal-conditioned world-model training on top of a frozen V-JEPA 2 encoder.
#
# The frozen backbone turns `tubelet_size` frames into one temporal token. A *chunk* is
# `tokens_per_chunk` of those tokens (4 tokens x 2 frames = 8 frames = 2s at 4fps). A
# trainable chunk encoder collapses one chunk into a latent, and a causal predictor
# reads `context_chunks` of them and predicts the latent of the next chunk -- a JEPA
# objective, with targets produced by an EMA copy of the chunk encoder.
#
# The predictor is additionally conditioned on the latent of a chunk sampled uniformly
# between `goal_min_seconds` and `goal_max_seconds` past the end of the context: the
# goal, standing in for the intention behind the actions that get the agent there. The
# goal is encoded by the target encoder under `no_grad`, so no gradient flows into it,
# and it is dropped on a `goal_drop_prob` fraction of samples so the predictor also
# learns to roll forward with no intention supplied.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel

from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_opt
from app.world_model.utils import (
    gather_window,
    init_frozen_backbone,
    init_world_model,
    load_checkpoint,
    split_into_chunks,
)
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import SimpleCollator
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


@record
def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")

    # -- FROZEN V-JEPA 2 BACKBONE
    cfgs_vjepa = args.get("vjepa")
    vjepa_ckpt = cfgs_vjepa.get("checkpoint")
    vjepa_ckpt_key = cfgs_vjepa.get("checkpoint_key", "target_encoder")
    vjepa_model_name = cfgs_vjepa.get("model_name", "vit_large")
    patch_size = cfgs_vjepa.get("patch_size", 16)
    tubelet_size = cfgs_vjepa.get("tubelet_size", 2)
    vjepa_uniform_power = cfgs_vjepa.get("uniform_power", False)
    vjepa_use_rope = cfgs_vjepa.get("use_rope", True)
    vjepa_use_silu = cfgs_vjepa.get("use_silu", False)
    vjepa_wide_silu = cfgs_vjepa.get("wide_silu", True)
    # Chunks per frozen-backbone forward pass; lower it if the backbone runs out of
    # memory (it changes nothing about the result, only the peak activation footprint).
    frozen_batch_size = cfgs_vjepa.get("batch_size", -1)

    # -- WORLD MODEL (chunking + goal sampling)
    cfgs_wm = args.get("world_model")
    tokens_per_chunk = cfgs_wm.get("tokens_per_chunk", 4)
    context_chunks = cfgs_wm.get("context_chunks", 8)
    goal_min_seconds = float(cfgs_wm.get("goal_min_seconds", 4.0))
    goal_max_seconds = float(cfgs_wm.get("goal_max_seconds", 16.0))
    goal_drop_prob = float(cfgs_wm.get("goal_drop_prob", 0.1))

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    embed_dim = cfgs_model.get("embed_dim", 768)
    enc_depth = cfgs_model.get("enc_depth", 6)
    enc_num_heads = cfgs_model.get("enc_num_heads", 12)
    pred_depth = cfgs_model.get("pred_depth", 12)
    pred_embed_dim = cfgs_model.get("pred_embed_dim", 384)
    pred_num_heads = cfgs_model.get("pred_num_heads", 12)
    goal_gate_init = cfgs_model.get("goal_gate_init", 1.0)
    horizon_embed_dim = cfgs_model.get("horizon_embed_dim", 128)
    drop_path_rate = cfgs_model.get("drop_path_rate", 0.0)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 256)
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    # -- The clip layout, in frames. Chunk c covers frames [c * F, (c + 1) * F); the
    #    model observes chunks 0..context_chunks-1 and predicts chunks 1..context_chunks,
    #    so `context_chunks + 1` chunks have to be encoded. The goal chunk starts a
    #    uniformly sampled offset past the end of the observed context, which is what
    #    fixes the clip length the data loader has to serve.
    frames_per_chunk = tokens_per_chunk * tubelet_size
    chunk_seconds = frames_per_chunk / fps
    num_encoded_chunks = context_chunks + 1
    context_frames = context_chunks * frames_per_chunk
    goal_min_frames = int(round(goal_min_seconds * fps))
    goal_max_frames = int(round(goal_max_seconds * fps))
    clip_frames = context_frames + goal_max_frames + frames_per_chunk

    assert goal_min_frames >= 1, f"a goal {goal_min_seconds}s away is less than one frame at {fps}fps"
    assert goal_max_frames >= goal_min_frames, "goal_max_seconds must be >= goal_min_seconds"
    assert 0.0 <= goal_drop_prob < 1.0, "goal_drop_prob must be in [0, 1)"
    assert crop_size % patch_size == 0, "crop size must be a whole number of patches"
    cfg_fpcs = cfgs_data.get("dataset_fpcs", None)
    assert cfg_fpcs is None or all(f == clip_frames for f in cfg_fpcs), (
        f"dataset_fpcs {cfg_fpcs} does not match the {clip_frames} frames this configuration needs "
        f"({context_chunks} context chunks of {frames_per_chunk} frames + a goal up to {goal_max_seconds}s away)"
    )
    dataset_fpcs = [clip_frames for _ in dataset_paths]

    grid_size = crop_size // patch_size
    patches_per_chunk = grid_size * grid_size

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()

    log_file = os.path.join(folder, f"log_r{rank}.log")
    logger = get_logger(__name__, force=True, filename=log_file)
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    logger.info(
        f"Chunk: {tokens_per_chunk} V-JEPA tokens = {frames_per_chunk} frames = {chunk_seconds:.2f}s at {fps}fps | "
        f"context: {context_chunks} chunks ({context_frames / fps:.1f}s) | "
        f"goal: {goal_min_seconds}-{goal_max_seconds}s past the context "
        f"({goal_min_frames}-{goal_max_frames} frames), dropped {goal_drop_prob:.0%} of the time | "
        f"clip: {clip_frames} frames ({clip_frames / fps:.1f}s)"
    )

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pt"
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        load_path = r_file if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "goal-gate"),
        ("%.2f", "goal-dist(s)"),
        ("%.3f", "goal-drop"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # -- init the frozen V-JEPA 2 feature extractor
    frozen_backbone = init_frozen_backbone(
        device=device,
        checkpoint=vjepa_ckpt,
        checkpoint_key=vjepa_ckpt_key,
        model_name=vjepa_model_name,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_chunk=frames_per_chunk,
        uniform_power=vjepa_uniform_power,
        use_rope=vjepa_use_rope,
        use_sdpa=use_sdpa,
        use_silu=vjepa_use_silu,
        wide_silu=vjepa_wide_silu,
    )

    # -- init the trainable world model
    encoder, predictor = init_world_model(
        device=device,
        frozen_dim=frozen_backbone.embed_dim,
        grid_height=grid_size,
        grid_width=grid_size,
        tokens_per_chunk=tokens_per_chunk,
        context_chunks=context_chunks,
        embed_dim=embed_dim,
        enc_depth=enc_depth,
        enc_num_heads=enc_num_heads,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        pred_num_heads=pred_num_heads,
        goal_gate_init=goal_gate_init,
        horizon_embed_dim=horizon_embed_dim,
        drop_path_rate=drop_path_rate,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        logger.info("Compiling frozen backbone, encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        frozen_backbone.compile()
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    collator = SimpleCollator(
        dataset_fpcs=dataset_fpcs,
        use_pretrained_model=False,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
    )
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=datasets_weights,
        persistent_workers=persistent_workers,
        collator=collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler (the V-JEPA 2 backbone is frozen and not passed in)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=False,
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    # -- load training checkpoint
    if load_model or os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            collator.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    def frozen_features(video):
        """Frozen V-JEPA 2 tokens for a batch of single chunks, [N, C, F, H, W] ->
        [N, tokens_per_chunk * P, D]. Encoding one chunk at a time is what keeps the
        backbone's bidirectional attention from leaking the future into a chunk latent."""
        step = frozen_batch_size if frozen_batch_size > 0 else video.size(0)
        return torch.cat([frozen_backbone(video[i : i + step]) for i in range(0, video.size(0), step)], dim=0)

    # Goal distances are drawn from their own generator so that every rank sees a
    # different set of horizons for the same iteration -- with the global seed alone
    # each rank would draw the same offsets, and a global batch would cover only
    # `batch_size` distinct horizons instead of `world_size * batch_size`.
    goal_rng = torch.Generator(device=device)
    goal_rng.manual_seed(seed + rank)

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        goal_dist_meter = AverageMeter()
        goal_drop_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            # Every dataset is served at the same clip length, so the collator groups the
            # whole batch under a single frames-per-clip key.
            assert len(sample) == 1, f"expected a single clip length, got {len(sample)}"
            udata, _, _ = sample[0]
            clip = udata[0][0].to(device, non_blocking=True)
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --
                B = clip.size(0)

                # -- Where the goal sits, per sample: uniform over [goal_min, goal_max]
                #    frames past the end of the observed context.
                offset = torch.randint(
                    goal_min_frames, goal_max_frames + 1, (B,), device=device, generator=goal_rng
                )
                goal_start = context_frames + offset
                # -- and which samples get to see it at all
                keep_goal = (
                    torch.rand(B, device=device, generator=goal_rng) >= goal_drop_prob
                )
                goal_clip = gather_window(clip, goal_start, frames_per_chunk)
                chunks = split_into_chunks(clip, frames_per_chunk, num_encoded_chunks)

                def forward_frozen():
                    """Frozen features for chunks 0..context_chunks and for the goal chunk."""
                    with torch.no_grad():
                        feats = frozen_features(torch.cat([chunks, goal_clip], dim=0))
                    n_ctxt = B * num_encoded_chunks
                    return (
                        feats[:n_ctxt].view(B, num_encoded_chunks, tokens_per_chunk * patches_per_chunk, -1),
                        feats[n_ctxt:],
                    )

                def forward_target(chunk_feats, goal_feats):
                    """Prediction targets (chunks 1..context_chunks) and the goal latent,
                    both from the EMA encoder. Running the goal here rather than through
                    the online encoder is what keeps the loss from backpropagating into it."""
                    with torch.no_grad():
                        tgt = chunk_feats[:, 1:].reshape(B * context_chunks, -1, chunk_feats.size(-1))
                        h = target_encoder(tgt).view(B, context_chunks * patches_per_chunk, -1)
                        h = F.layer_norm(h, (h.size(-1),))
                        g = target_encoder(goal_feats)
                        g = F.layer_norm(g, (g.size(-1),))
                        return h, g.detach()

                def forward_context(chunk_feats, goal_latent):
                    ctxt = chunk_feats[:, :context_chunks].reshape(B * context_chunks, -1, chunk_feats.size(-1))
                    z = encoder(ctxt).view(B, context_chunks * patches_per_chunk, -1)
                    # The goal's time coordinate, in chunks, on the same axis as the
                    # predictor's steps (see GoalConditionedPredictor.build_positions).
                    goal_pos = goal_start.float() / frames_per_chunk
                    return predictor(z, goal=goal_latent, goal_pos=goal_pos, keep_goal=keep_goal)

                def loss_fn(z, h):
                    return torch.mean(torch.abs(z - h) ** loss_exp) / loss_exp

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    chunk_feats, goal_feats = forward_frozen()
                    h, goal_latent = forward_target(chunk_feats, goal_feats)
                    z = forward_context(chunk_feats, goal_latent)
                    loss = loss_fn(z, h)  # jepa next-chunk prediction loss

                # Step 2. Backward & step
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    params_k = []
                    params_q = []
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        params_k.append(param_k)
                        params_q.append(param_q)
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1 - m)

                return (
                    float(loss.detach()),
                    float(offset.float().mean()) / fps,
                    1.0 - float(keep_goal.float().mean()),
                    _new_lr,
                    _new_wd,
                )

            (
                loss,
                goal_dist,
                goal_drop,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            goal_dist_meter.update(goal_dist)
            goal_drop_meter.update(goal_drop)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                gate = predictor.module.goal_gate()
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    gate,
                    goal_dist,
                    goal_drop,
                    iter_elapsed_time_ms,
                    gpu_etime_ms,
                    data_elapsed_time_ms,
                )
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "[goal-gate: %.2e] "
                        "[goal-dist: %.1f s] "
                        "[goal-drop: %.2f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            gate,
                            goal_dist_meter.avg,
                            goal_drop_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
