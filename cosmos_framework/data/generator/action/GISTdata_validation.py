# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Finite, repeatable validation passes using native single-sample token packing."""

import math

import torch.distributed as dist
from torch import Generator
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn


class GISTdataValidationLoader:
    def __init__(self, dataset, num_workers=0, max_sequence_length=8192):
        if len(dataset) == 0:
            raise ValueError("GISTdata validation split is empty")
        self.dataset = dataset
        self.num_workers = num_workers
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        world = dist.get_world_size() if dist.is_initialized() else 1
        return math.ceil(len(self.dataset) / world)

    def __iter__(self):
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Equal forward counts are required by FSDP collectives. DistributedSampler
        # pads at most world-1 samples; the callback reports unique/total counts.
        sampler = DistributedSampler(self.dataset, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
        inner = DataLoader(
            self.dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=custom_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=Generator().manual_seed(42),
        )
        # PackingDataLoader owns iterator state. Recreate it on EVERY pass so
        # validation always starts from the same sample and terminates at EOF.
        packed = PackingDataLoader(
            inner,
            dataset_name="GISTdata_val",
            max_samples_per_batch=1,
            max_sequence_length=self.max_sequence_length,
            patch_spatial=2,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
        )
        yield from packed
