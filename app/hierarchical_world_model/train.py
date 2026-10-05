# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Hierarchical world-model training: a frozen goal-conditioned world model plans, a
# trainable causal world model fills in the detail.
#
# Both levels start predicting from the same instant, T -- the end of the high-level
# context, 16s into the clip.
#
#   high level (frozen, app/world_model)
#       One step is a *chunk* of `tokens_per_chunk` V-JEPA tokens -- 8 frames = 2s at
#       4fps. At T it has observed its full context and is given the latent of the chunk
#       `goal_seconds` (16s) past T: a fixed goal, as if holding a photograph of where
#       this is heading. From there it *rolls out*: it predicts the chunk covering
#       [T, T+2s), feeds that back as though observed, predicts [T+2s, T+4s), and so on
#       for `guidance_chunks` steps. Every one of those sub-goals rests on the same
#       observations, the ones up to T -- a plan made once, not a running commentary.
#
#   low level (trained here)
#       A standard causal V-JEPA 2 world model: one step is one tubelet, 2 frames = 0.5s.
#       Its encoder is frozen and only the predictor is trained, with every block
#       cross-attending to the plan exactly as in app/vjepa_guided. Four low-level steps
#       fit in one chunk, so the first sub-goal guides the four steps covering
#       [T, T+2s); the conditioning then *switches* to the second sub-goal for the four
#       covering [T+2s, T+4s), and so on. Each step sees exactly the one sub-goal for the
#       chunk it is predicting into -- `guidance_window: 1` is what makes it switch
#       rather than accumulate. What switches is what a step *reads*: the predictor's
#       self-attention is causal, so a step still inherits the hidden states of the steps
#       before it and with them the memory of the sub-goals they were given.
#
# The window carries one chunk of lead-in before T so that the step predicting the first
# tubelet after T has real context behind it. Those lead-in steps predict frames before
# T, which no sub-goal covers, so they run unconditioned and the cross-attention branch
# returns exactly zero for them.
#
# Causality holds because the sub-goals are predictions resting only on frames up to T,
# and no step that reads one has a context ending before T.

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

import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel

from app.hierarchical_world_model.utils import (
    HierarchyLayout,
    encode_low_level_window,
    init_low_level_model,
    load_checkpoint,
    plan_sub_goals,
    warm_start_predictor,
)
from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_opt
from app.world_model.utils import load_goal_world_model
from src.datasets.video_window_dataset import make_videowindowdataset
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

    # -- HIGH-LEVEL (frozen goal-conditioned world model)
    cfgs_hl = args.get("high_level")
    hl_config_path = cfgs_hl.get("config")
    hl_checkpoint = cfgs_hl.get("checkpoint")
    goal_seconds = float(cfgs_hl.get("goal_seconds", 16.0))
    # Chunks the high-level model rolls forward from T, i.e. how far the plan reaches.
    # One is the default: the low-level model is conditioned on a single goal, the
    # nearest high-level step. The window is one chunk longer than this, so raising it
    # lengthens the low-level clip and the predictor's attention quadratically.
    guidance_chunks = int(cfgs_hl.get("guidance_chunks", 1))
    guidance_gate_init = cfgs_hl.get("gate_init", 0.0)
    # 1 makes each step read exactly the sub-goal for the chunk it is predicting into and
    # switch to the next one four steps later. Raise it to let a step also keep the
    # sub-goals it has already passed; null lets it see the whole plan at once.
    guidance_window = cfgs_hl.get("window", 1)

    # -- LOW-LEVEL ENCODER (frozen)
    cfgs_enc = args.get("encoder")
    enc_checkpoint = cfgs_enc.get("checkpoint")
    enc_checkpoint_key = cfgs_enc.get("checkpoint_key", "target_encoder")
    model_name = cfgs_enc.get("model_name", "vit_large")
    patch_size = int(cfgs_enc.get("patch_size", 16))
    tubelet_size = int(cfgs_enc.get("tubelet_size", 2))
    uniform_power = cfgs_enc.get("uniform_power", True)
    use_rope = cfgs_enc.get("use_rope", True)
    use_silu = cfgs_enc.get("use_silu", False)
    wide_silu = cfgs_enc.get("wide_silu", True)

    # -- LOW-LEVEL PREDICTOR (trained)
    cfgs_model = args.get("model")
    pred_depth = cfgs_model.get("pred_depth", 12)
    pred_num_heads = cfgs_model.get("pred_num_heads", 12)
    pred_embed_dim = cfgs_model.get("pred_embed_dim", 384)
    num_mask_tokens = int(cfgs_model.get("num_mask_tokens", 4))
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    warm_start = cfgs_model.get("warm_start", None)
    warm_start_key = cfgs_model.get("warm_start_key", "predictor")

    # -- DATA
    cfgs_data = args.get("data")
    dataset_paths = cfgs_data.get("datasets", [])
    index_cache = cfgs_data.get("index_cache", None)
    distinct_videos_per_batch = cfgs_data.get("distinct_videos_per_batch", True)
    batch_size = cfgs_data.get("batch_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 256)
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)
    window_stride_chunks = cfgs_data.get("window_stride_chunks", None)

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
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

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

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- the high-level model's own training config decides its architecture, so the
    #    hierarchy can never be assembled out of two mismatched halves
    with open(hl_config_path, "r") as f:
        hl_config = yaml.load(f, Loader=yaml.FullLoader)

    hl_fps = float(hl_config["data"]["fps"])
    hl_crop = int(hl_config["data"].get("crop_size", 256))
    hl_patch = int(hl_config["vjepa"].get("patch_size", 16))
    hl_tubelet = int(hl_config["vjepa"].get("tubelet_size", 2))
    assert hl_fps == fps, f"the high-level model runs at {hl_fps}fps but this run samples at {fps}fps"
    assert hl_crop == crop_size, f"crop size {hl_crop} != {crop_size}; the two models must share a patch grid"
    assert hl_patch == patch_size, f"patch size {hl_patch} != {patch_size}; the two models must share a patch grid"
    assert hl_tubelet == tubelet_size, (
        f"the high-level model's chunks are built from {hl_tubelet}-frame tubelets but the low-level model "
        f"steps by {tubelet_size} frames; one chunk would not be a whole number of low-level steps"
    )

    hl_goal_max = float(hl_config["world_model"].get("goal_max_seconds", 16.0))
    hl_goal_min = float(hl_config["world_model"].get("goal_min_seconds", 4.0))
    assert hl_goal_min - 1e-6 <= goal_seconds <= hl_goal_max + 1e-6, (
        f"goal_seconds ({goal_seconds:g}s) is outside the {hl_goal_min:g}-{hl_goal_max:g}s range the "
        "high-level model was trained to place goals in"
    )

    logger.info(f"Loading the frozen high-level world model from {hl_checkpoint}")
    high_level = load_goal_world_model(hl_config, hl_checkpoint, device)

    # -- the layout, in frames.
    #    chunk c of the high level covers frames [c * fpc, (c + 1) * fpc);
    #    low-level step s covers frames [s * tubelet, (s + 1) * tubelet).
    fpc = high_level.frames_per_chunk  # 8 frames = 2s at 4fps
    steps_per_chunk = high_level.tokens_per_chunk  # 4 low-level steps per chunk
    context_chunks = high_level.context_chunks  # 8 chunks = 16s of high-level context

    assert 1 <= guidance_chunks <= context_chunks, (
        f"guidance_chunks ({guidance_chunks}) must be between 1 and the high-level context "
        f"({context_chunks} chunks)"
    )
    layout = HierarchyLayout(
        frames_per_chunk=fpc,
        steps_per_chunk=steps_per_chunk,
        tubelet_size=tubelet_size,
        context_chunks=context_chunks,
        guidance_chunks=guidance_chunks,
        patches_per_chunk=high_level.patches_per_chunk,
        goal_start_frame=context_chunks * fpc + int(round(goal_seconds * fps)),
        goal_position=high_level.goal_position(goal_seconds, fps),
    )
    low_level_frames = layout.low_level_frames
    goal_start_frame = layout.goal_start_frame
    clip_frames = layout.clip_frames
    if window_stride_chunks is None:
        # T sits at a fixed point inside the clip, so the low-level model only ever
        # trains on the window around it. Hopping by exactly that window makes
        # consecutive clips of a video tile its footage, at no extra cost per step --
        # a clip is decoded whole either way, there are simply more distinct ones.
        window_stride_chunks = layout.low_level_chunks
    stride_frames = window_stride_chunks * fpc

    logger.info(
        f"Split T at frame {layout.split_frame} ({layout.split_frame / fps:.0f}s into the clip): "
        f"{context_chunks} chunks of {fpc} frames ({fpc / fps:.1f}s) of high-level context before it, "
        f"goal fixed {goal_seconds:g}s past it (frames {goal_start_frame}-{goal_start_frame + fpc}, "
        f"goal_pos {layout.goal_position:g} chunks)"
    )
    logger.info(
        f"Plan: {guidance_chunks} chunk(s) rolled out from T, covering "
        f"T+0s..T+{guidance_chunks * fpc / fps:.0f}s"
    )
    logger.info(
        f"Low level: {steps_per_chunk} steps of {tubelet_size} frames ({tubelet_size / fps:.2f}s) per chunk | "
        f"window frames {layout.low_level_start_frame}-{layout.low_level_start_frame + low_level_frames} "
        f"({layout.low_level_chunks} chunks = {low_level_frames // tubelet_size} steps: "
        f"{steps_per_chunk - 1} unconditioned lead-in, then {guidance_chunks * steps_per_chunk} guided) | "
        f"clip {clip_frames} frames ({clip_frames / fps:.1f}s) every {stride_frames} frames"
    )

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
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
        ("%.5f", "xattn-gate"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # -- init the low-level model (frozen encoder + trainable guided predictor)
    encoder, predictor = init_low_level_model(
        device=device,
        encoder_checkpoint=enc_checkpoint,
        encoder_checkpoint_key=enc_checkpoint_key,
        guidance_dim=high_level.embed_dim,
        num_frames=low_level_frames,
        model_name=model_name,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        num_mask_tokens=num_mask_tokens,
        guidance_step_ratio=steps_per_chunk,
        # A sub-goal only becomes safe once the low-level context reaches the end of the
        # chunk the high-level model conditioned on -- see this file's header.
        guidance_context_offset=layout.guidance_context_offset,
        guidance_gate_init=guidance_gate_init,
        guidance_window=guidance_window,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_rope=use_rope,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    # `warm_start` only ever initializes. By this point `load_model` is true only if a
    # checkpoint of this run will actually be read back, and that one wins.
    if warm_start is not None and not load_model:
        warm_start_predictor(predictor, warm_start, warm_start_key)

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
    (_, unsupervised_loader, unsupervised_sampler) = make_videowindowdataset(
        data_paths=dataset_paths,
        batch_size=batch_size,
        clip_frames=clip_frames,
        stride_frames=stride_frames,
        fps=fps,
        transform=transform,
        rank=rank,
        world_size=world_size,
        seed=seed,
        distinct_videos_per_batch=distinct_videos_per_batch,
        index_cache=index_cache,
        persistent_workers=persistent_workers,
        num_workers=num_workers,
        pin_mem=pin_mem,
    )
    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler. Only the predictor is trained: the low-level
    #    encoder and the whole high-level model are frozen, so `init_opt` gets an empty
    #    stand-in where it expects an encoder.
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=False,
        encoder=torch.nn.Module(),
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
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)

    start_epoch = 0
    # -- load training checkpoint
    if load_model and load_path is not None:
        predictor, optimizer, scaler, start_epoch = load_checkpoint(
            r_path=load_path, predictor=predictor, opt=optimizer, scaler=scaler
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        # The frozen encoder is saved under both keys a V-JEPA checkpoint carries: with
        # nothing training it, the online and EMA target encoders are the same weights,
        # which keeps this checkpoint loadable by the shared evaluation tooling.
        encoder_state = encoder.state_dict()
        save_dict = {
            "encoder": encoder_state,
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": encoder_state,
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

    def mean_gate():
        """Average magnitude of the sub-goal cross-attention gates -- how much the
        low-level model is actually leaning on the plan. Starts at `gate_init`."""
        gates = [b.gamma_xattn.detach() for b in predictor.module.backbone.predictor_blocks]
        return float(torch.stack([g.abs().mean() for g in gates]).mean())

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

            # (clips, video index, first frame) -- the last two are only for debugging
            clip = sample[0].to(device, non_blocking=True)
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def loss_fn(z, h):
                    return torch.mean(torch.abs(z - h) ** loss_exp) / loss_exp

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    sub_goals = plan_sub_goals(high_level, clip, layout)
                    context, targets = encode_low_level_window(encoder, clip, layout)
                    z = predictor([context], is_causal=True, guidance=[sub_goals])[0]
                    loss = loss_fn(z, targets)  # jepa next-step prediction loss

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

                # No momentum update: the encoder is frozen, so its EMA target would be
                # the very same weights and there is nothing for a target encoder to track.

                return (
                    float(loss.detach()),
                    _new_lr,
                    _new_wd,
                )

            (
                loss,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                gate = mean_gate()
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    gate,
                    iter_elapsed_time_ms,
                    gpu_etime_ms,
                    data_elapsed_time_ms,
                )
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "[xattn-gate: %.2e] "
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
