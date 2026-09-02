# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import pathlib
import warnings
from collections import defaultdict
from logging import getLogger

import numpy as np
import pandas as pd
import torch
from decord import cpu, VideoReader

logger = getLogger()

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def make_videowindowdataset(
    data_paths,
    batch_size,
    clip_frames,
    stride_frames,
    fps=None,
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    seed=0,
    distinct_videos_per_batch=True,
    index_cache=None,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    drop_last=True,
):
    dataset = VideoWindowDataset(
        data_paths=data_paths,
        clip_frames=clip_frames,
        stride_frames=stride_frames,
        fps=fps,
        transform=transform,
        shared_transform=shared_transform,
        index_cache=index_cache,
    )
    sampler = WindowSampler(
        video_ids=dataset.window_video_ids,
        batch_size=batch_size,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        distinct_videos_per_batch=distinct_videos_per_batch,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("VideoWindowDataset data loader created")
    return dataset, data_loader, sampler


class VideoWindowDataset(torch.utils.data.Dataset):
    """Every window of every video is a sample.

    `VideoDataset` draws *one* random clip per video per epoch, so an epoch over 540
    long videos is 540 clips and almost all of the footage is never seen. Here the
    windows of every video are enumerated up front, so an epoch walks each video end to
    end. `stride_frames` is the hop between consecutive windows: for the goal-conditioned
    world model it is set to the part of the window the model is actually trained to
    predict, which is shorter than the window itself (the tail is only there to hold the
    furthest goal), so consecutive windows overlap by design.

    :param data_paths: directories to search for videos, individual video files, or
        `.csv` manifests whose first column is a video path
    :param clip_frames: frames per sample, after resampling to `fps`
    :param stride_frames: hop between consecutive windows, in resampled frames
    :param fps: frame rate to resample to (None keeps the video's own)
    :param index_cache: json file caching each video's length and frame rate, so the
        whole collection does not have to be probed on every run
    """

    def __init__(
        self,
        data_paths,
        clip_frames,
        stride_frames,
        fps=None,
        transform=None,
        shared_transform=None,
        index_cache=None,
    ):
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.clip_frames = clip_frames
        self.stride_frames = stride_frames
        self.fps = fps
        self.transform = transform
        self.shared_transform = shared_transform

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        files = _collect_video_files(data_paths)
        if not files:
            raise ValueError(f"no videos found under {data_paths}")
        probed = _probe_videos(files, index_cache)

        # -- lay the windows out: (video, first frame), with the frame indices of a
        #    window being `start + arange(clip_frames) * frame_step`
        self.videos = []
        self.window_video_ids = []
        self.window_starts = []
        self.window_steps = []
        num_short = 0
        for path in files:
            num_frames, video_fps = probed[path]
            frame_step = 1 if fps is None else max(1, int(round(video_fps / fps)))
            span = (clip_frames - 1) * frame_step + 1  # source frames one window covers
            if num_frames < span:
                num_short += 1
                continue
            video_id = len(self.videos)
            self.videos.append(path)
            for start in range(0, num_frames - span + 1, stride_frames * frame_step):
                self.window_video_ids.append(video_id)
                self.window_starts.append(start)
                self.window_steps.append(frame_step)
        if not self.window_starts:
            raise ValueError(
                f"no video is long enough for a {clip_frames}-frame window at {fps}fps "
                f"(shortest requirement: {clip_frames / (fps or 1):.1f}s of footage)"
            )

        self.window_video_ids = np.asarray(self.window_video_ids, dtype=np.int64)
        self.window_starts = np.asarray(self.window_starts, dtype=np.int64)
        self.window_steps = np.asarray(self.window_steps, dtype=np.int64)
        total_hours = sum(probed[p][0] / probed[p][1] for p in self.videos) / 3600.0
        logger.info(
            f"VideoWindowDataset: {len(self.window_starts)} windows of {clip_frames} frames "
            f"(stride {stride_frames}) over {len(self.videos)} videos ({total_hours:.1f}h of footage); "
            f"skipped {num_short} video(s) shorter than one window"
        )

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, index):
        # Keep trying until a window loads, exactly as VideoDataset does: a single
        # unreadable file should not take down a training run.
        for _ in range(10):
            clip = self.load_window(index)
            if clip is not None:
                return clip, int(self.window_video_ids[index]), int(self.window_starts[index])
            index = np.random.randint(len(self))
        raise RuntimeError("failed to load 10 consecutive video windows")

    def load_window(self, index):
        path = self.videos[self.window_video_ids[index]]
        start, step = int(self.window_starts[index]), int(self.window_steps[index])
        try:
            vr = VideoReader(path, num_threads=-1, ctx=cpu(0))
            indices = start + np.arange(self.clip_frames, dtype=np.int64) * step
            buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
        except Exception as e:
            warnings.warn(f"could not read {self.clip_frames} frames from {path} at {start}: {e}")
            return None

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        if self.transform is not None:
            buffer = self.transform(buffer)  # [C, T, H, W]
        return buffer


class WindowSampler(torch.utils.data.Sampler):
    """Distributed sampler over windows that keeps a batch's clips on distinct videos.

    Windows are split across ranks first, so every window is visited once per epoch by
    exactly one rank. Within a rank, `distinct_videos_per_batch` then emits the windows
    in rounds: each round shuffles the videos that still have windows left and takes one
    from each, truncated to a whole number of batches so a batch never straddles two
    rounds. Any `batch_size` consecutive indices therefore come from `batch_size`
    different videos -- otherwise a batch would often be several windows of one video,
    which are highly correlated and make for a poor gradient estimate.

    The tail of an epoch (fewer than `batch_size` videos still holding windows) is
    dropped; a different permutation next epoch drops a different tail.
    """

    def __init__(
        self,
        video_ids,
        batch_size,
        num_replicas=1,
        rank=0,
        seed=0,
        distinct_videos_per_batch=True,
    ):
        self.video_ids = np.asarray(video_ids)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.distinct_videos_per_batch = distinct_videos_per_batch
        self.epoch = 0
        self._order = None

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            self._order = None
        self.epoch = epoch

    def _build_order(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        # The permutation is identical on every rank, so striding it hands each rank a
        # disjoint share of the windows.
        mine = rng.permutation(len(self.video_ids))[self.rank :: self.num_replicas]
        if not self.distinct_videos_per_batch:
            return mine.tolist()

        by_video = defaultdict(list)
        for i in mine:
            by_video[int(self.video_ids[i])].append(int(i))

        order = []
        while True:
            active = [v for v, w in by_video.items() if w]
            if len(active) < self.batch_size:
                break
            rng.shuffle(active)
            # Truncate to whole batches so no batch spans two rounds (and so no video
            # can appear at the end of one round and the start of the next).
            for v in active[: (len(active) // self.batch_size) * self.batch_size]:
                order.append(by_video[v].pop())
        return order

    def _get_order(self):
        if self._order is None:
            self._order = self._build_order()
        return self._order

    def __iter__(self):
        return iter(self._get_order())

    def __len__(self):
        return len(self._get_order())


def _collect_video_files(data_paths):
    """Expand directories, `.csv` manifests and plain paths into a sorted file list."""
    files = []
    for data_path in data_paths:
        if os.path.isdir(data_path):
            found = [
                str(p)
                for p in sorted(pathlib.Path(data_path).rglob("*"))
                if p.suffix.lower() in VIDEO_EXTENSIONS
            ]
            logger.info(f"found {len(found)} videos under {data_path}")
            files += found
        elif str(data_path).endswith(".csv"):
            try:
                data = pd.read_csv(data_path, header=None, delimiter=" ")
            except pd.errors.ParserError:
                data = pd.read_csv(data_path, header=None, delimiter="::")
            files += [str(v) for v in data.values[:, 0]]
        else:
            files += [str(data_path)]
    # De-duplicate while keeping the order stable, so the window layout (and therefore
    # the meaning of a cached index) does not depend on the order paths were listed in.
    return sorted(set(files))


def _probe_videos(files, index_cache=None):
    """Length and frame rate of every video, reading `index_cache` where it can.

    Probing 500+ long videos takes about a minute, and nothing about it changes between
    runs, so results are cached to disk. Videos absent from the cache are probed and
    folded back in, which makes adding footage cheap.
    """
    cache = {}
    if index_cache is not None and os.path.exists(index_cache):
        try:
            with open(index_cache, "r") as f:
                cache = json.load(f)
            logger.info(f"loaded {len(cache)} video index entries from {index_cache}")
        except Exception as e:
            logger.warning(f"could not read video index cache {index_cache} ({e}); rebuilding it")
            cache = {}

    probed, missing = {}, []
    for path in files:
        entry = cache.get(path)
        if entry is not None:
            probed[path] = (int(entry[0]), float(entry[1]))
        else:
            missing.append(path)

    if missing:
        logger.info(f"probing {len(missing)} video(s) for length and frame rate...")
        for path in missing:
            try:
                vr = VideoReader(path, num_threads=1, ctx=cpu(0))
                probed[path] = (len(vr), float(vr.get_avg_fps()))
            except Exception as e:
                warnings.warn(f"skipping unreadable video {path}: {e}")
                probed[path] = (0, 1.0)
        if index_cache is not None:
            cache.update({p: list(probed[p]) for p in missing})
            _write_index_cache(index_cache, cache)

    return probed


def _write_index_cache(index_cache, cache):
    """Write the cache through a temp file so concurrent ranks cannot leave a torn one."""
    try:
        pathlib.Path(index_cache).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{index_cache}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, index_cache)
        logger.info(f"wrote {len(cache)} video index entries to {index_cache}")
    except Exception as e:
        logger.warning(f"could not write video index cache {index_cache} ({e})")
