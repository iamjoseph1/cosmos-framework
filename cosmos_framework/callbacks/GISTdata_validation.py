# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Log held-out total/action/vision losses together at one W&B iteration."""

import torch
import torch.distributed as dist
import wandb

from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback


class GISTdataValidationCallback(Callback):
    def __init__(self, save_s3=False):
        super().__init__()
        # Existing callback defaults supply this field when Hydra merges recipes.
        if save_s3:
            raise ValueError("GISTdata validation supports W&B/local logs; set save_s3=false")

    def on_validation_start(self, model, dataloader, iteration=0):
        self.totals = None
        self.keys = []

    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        values = torch.stack(
            [
                loss.detach().double().mean(),
                output_batch["flow_matching_loss_action"].detach().double().mean(),
                output_batch["flow_matching_loss_vision"].detach().double().mean(),
            ]
        )
        if self.totals is None:
            self.totals = values.new_zeros(5)  # loss sums, valid count, invalid count
        if torch.isfinite(values).all():
            self.totals[:3] += values
            self.totals[3] += 1
        else:
            self.totals[4] += 1
        self.keys.extend(data_batch["__key__"])

    def on_validation_end(self, model, iteration=0):
        if self.totals is None:
            raise RuntimeError("Validation produced no batches")
        all_keys = [self.keys]
        rank = 0
        if dist.is_initialized():
            dist.all_reduce(self.totals)
            rank = dist.get_rank()
            all_keys = [None] * dist.get_world_size()
            dist.all_gather_object(all_keys, self.keys)
        if self.totals[3] == 0:
            raise RuntimeError("All validation losses are non-finite")
        if rank == 0:
            keys = [key for group in all_keys for key in group]
            means = (self.totals[:3] / self.totals[3]).tolist()
            metrics = {
                "iteration": iteration,
                "val/loss": means[0],
                "val/action_loss": means[1],
                "val/vision_loss": means[2],
                "val/unstable_count": self.totals[4].item(),
                "val/num_unique_url_key": len(set(keys)),
                "val/total_url_key": len(keys),
            }
            log.info(f"GISTdata validation: {metrics}")
            if wandb.run is not None:
                wandb.log(metrics, step=iteration)
        self.totals = None
        self.keys = []
