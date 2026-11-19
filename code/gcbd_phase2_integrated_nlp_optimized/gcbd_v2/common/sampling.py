"""Shard-local batch shuffling to make lazy geometry-cache reads efficient."""
from __future__ import annotations

import math
import torch
from torch.utils.data import Sampler


class ShardAwareBatchSampler(Sampler[list[int]]):
    """Shuffle contiguous cache-sized blocks, then shuffle within each block.

    Offline cache extraction writes contiguous dataset IDs in batches of the
    training batch size. This sampler randomizes block order while keeping each
    training batch inside one cache shard, avoiding hundreds of random shard
    loads per batch. Every sample is emitted exactly once per epoch.
    """
    def __init__(self, dataset_size: int, batch_size: int, seed: int = 0,
                 drop_last: bool = False):
        if dataset_size < 0 or batch_size < 1:
            raise ValueError("dataset_size must be >= 0 and batch_size >= 1")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003)
        starts = list(range(0, self.dataset_size, self.batch_size))
        if self.drop_last and starts and self.dataset_size % self.batch_size:
            starts = starts[:-1]
        if starts:
            order = torch.randperm(len(starts), generator=generator).tolist()
            for block_index in order:
                start = starts[block_index]
                stop = min(self.dataset_size, start + self.batch_size)
                block = torch.arange(start, stop)
                if block.numel() > 1:
                    block = block[torch.randperm(block.numel(), generator=generator)]
                yield [int(x) for x in block.tolist()]

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)
