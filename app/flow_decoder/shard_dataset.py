# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Readers for the latent shards written by `app/flow_decoder/latent_cache.py`.

Two readers, because training and the panels want opposite things:

  * `ShardStream` (iterable) - streams whole shards in a random order and
    shuffles a bounded buffer within them. One shard is ~230 MB, so a map-style
    dataset with a global random sampler would fault a fresh shard in for almost
    every sample; streaming keeps exactly one resident per worker. Training does
    not care that two samples from the same clip land in nearby batches - it is
    a fixed-step-budget regression, not a curriculum - so this is free.

  * `ShardSet` (map-style) - random access by global index, one shard cached at a
    time. The panels need to fetch a *named* sample (this clip, this step) and
    fetch it identically for every world model, which a stream cannot do.

Neither reader touches the codec. They hand out uint8 frames and let the caller
encode on the GPU, because a DataLoader worker that initializes CUDA is a
well-known way to lose an afternoon.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from src.models.flow_decoder.latent_adapter import build_token_coords

logger = logging.getLogger(__name__)

LATENT_KEYS = {"target_encoder": "z_target", "predictor": "z_pred"}


@dataclass
class ShardSetInfo:
    """The manifest, plus the derived things a decoder needs to be built."""

    root: Path
    manifest: dict
    shards: List[Path]

    @property
    def latent_dim(self) -> int:
        return int(self.manifest["latent_dim"])

    @property
    def latent_grid(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.manifest["latent_grid"])

    @property
    def crop_size(self) -> int:
        return int(self.manifest["crop_size"])

    @property
    def model_name(self) -> str:
        return str(self.manifest["model_name"])

    @property
    def num_samples(self) -> int:
        return int(self.manifest["num_samples"])

    @property
    def has_predictor_latents(self) -> bool:
        return bool(self.manifest.get("has_predictor_latents", False))

    def statistics(self, latent_source: str = "target_encoder") -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """The fitted (mean, std) buffers for a latent source, if they were written."""
        path = self.root / "statistics.pt"
        if not path.exists():
            return None
        stats = torch.load(path, map_location="cpu", weights_only=False)
        key = LATENT_KEYS[latent_source]
        if key not in stats:
            return None
        return stats[key]["mean"].float(), stats[key]["std"].float()

    def describe(self) -> str:
        m = self.manifest
        return (
            f"{m['model_name']}: {m['num_samples']} sample(s) in {m['num_shards']} shard(s), "
            f"d_m={m['latent_dim']} grid={tuple(m['latent_grid'])} crop={m['crop_size']} "
            f"step={m['step_seconds']:.3f}s predictor_latents={self.has_predictor_latents}"
        )


def load_shard_set(root: str | Path) -> ShardSetInfo:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run app.flow_decoder.latent_cache to build a shard set first."
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    shards = [root / name for name in manifest["shards"]]
    missing = [str(p) for p in shards if not p.exists()]
    if missing:
        raise FileNotFoundError(f"manifest lists {len(missing)} missing shard(s), e.g. {missing[0]}")
    return ShardSetInfo(root=root, manifest=manifest, shards=shards)


def _select_latents(payload: dict, latent_source: str) -> torch.Tensor:
    key = LATENT_KEYS.get(latent_source)
    if key is None:
        raise ValueError(f"unknown latent source {latent_source!r}; expected one of {sorted(LATENT_KEYS)}")
    if key not in payload:
        raise KeyError(
            f"shard has no {key!r}; re-run latent_cache with --store-predictor-latents to cache "
            "predictor outputs."
        )
    return payload[key]


class ShardStream(IterableDataset):
    """Streams samples for training. See the module docstring for why.

    Each worker takes a strided slice of the shard list, reshuffles it every
    epoch from a seed that includes the epoch and the worker id, and shuffles a
    `buffer_size` window within it. `epochs` is unbounded: the training loop stops
    on its fixed step budget, never on an epoch count or a validation curve.
    """

    def __init__(
        self,
        info: ShardSetInfo,
        latent_source: str = "target_encoder",
        buffer_size: int = 512,
        seed: int = 0,
    ):
        super().__init__()
        self.info = info
        self.latent_source = latent_source
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        shards = self.info.shards[worker_id::num_workers]
        if not shards:
            return
        rng = random.Random(self.seed * 100003 + worker_id)
        epoch = 0
        while True:
            order = list(shards)
            rng.shuffle(order)
            for path in order:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                z = _select_latents(payload, self.latent_source)
                indices = list(range(z.size(0)))
                rng.shuffle(indices)
                for i in indices:
                    yield {
                        "frame_prev": payload["frame_prev"][i],
                        "frame_next": payload["frame_next"][i],
                        "z": z[i],
                    }
                del payload, z
            epoch += 1


class ShardSet(Dataset):
    """Map-style access with a one-shard cache. Used by the panel harness."""

    def __init__(self, info: ShardSetInfo, latent_source: str = "target_encoder"):
        self.info = info
        self.latent_source = latent_source
        self._sizes: List[int] = []
        self._offsets: List[int] = []
        self._cached_path: Optional[Path] = None
        self._cached_payload: Optional[dict] = None

        total = 0
        for path in info.shards:
            # Read only the metadata list to size each shard. Cheaper than
            # loading the tensors, and it fails loudly if a shard is truncated.
            payload = torch.load(path, map_location="cpu", weights_only=False)
            size = len(payload["meta"])
            self._sizes.append(size)
            self._offsets.append(total)
            total += size
            del payload
        self.total = total

    def __len__(self) -> int:
        return self.total

    def _locate(self, index: int) -> Tuple[int, int]:
        if not 0 <= index < self.total:
            raise IndexError(f"index {index} out of range for {self.total} sample(s)")
        for shard_idx, (offset, size) in enumerate(zip(self._offsets, self._sizes)):
            if index < offset + size:
                return shard_idx, index - offset
        raise IndexError(index)  # unreachable

    def _payload(self, shard_idx: int) -> dict:
        path = self.info.shards[shard_idx]
        if self._cached_path != path:
            self._cached_payload = torch.load(path, map_location="cpu", weights_only=False)
            self._cached_path = path
        return self._cached_payload

    def __getitem__(self, index: int) -> dict:
        shard_idx, local = self._locate(index)
        payload = self._payload(shard_idx)
        item = {
            "frame_prev": payload["frame_prev"][local],
            "frame_next": payload["frame_next"][local],
            "z": _select_latents(payload, self.latent_source)[local],
            "meta": payload["meta"][local],
            "index": index,
        }
        if "z_pred" in payload:
            item["z_pred"] = payload["z_pred"][local]
        item["z_target"] = payload["z_target"][local]
        return item


def collate(batch: Sequence[dict]) -> dict:
    """Stack a batch, keeping frames as uint8 for the GPU-side conversion."""
    out = {
        "frame_prev": torch.stack([b["frame_prev"] for b in batch]),
        "frame_next": torch.stack([b["frame_next"] for b in batch]),
        "z": torch.stack([b["z"] for b in batch]).float(),
    }
    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]
    return out


def frames_to_unit(frames: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 3, H, W) -> float [0, 1], the codec's input convention."""
    return frames.float().div_(255.0)


def coords_for(info: ShardSetInfo, device: torch.device) -> torch.Tensor:
    """(L, 6) continuous coordinates for a shard set's own token grid."""
    return build_token_coords(info.latent_grid, device=device)
