# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in held-out flow-matching evaluation for the GISTdata action recipes."""

import hashlib

import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


class GISTdataValidationModel(OmniMoTModel):
    def _update_train_stats(self, gen_data_clean):
        if self.training:
            super()._update_train_stats(gen_data_clean)

    @torch.no_grad()
    def validation_step(self, data_batch, iteration):
        if self.training:
            raise RuntimeError("Validation requires model.eval()")
        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            raise ValueError("GISTdata validation supports DP/FSDP with context_parallel_degree=1")
        # A sample gets the same RF noise/timestep at every checkpoint, without
        # consuming training RNG state. This is a denoising objective, not a rollout.
        identity = repr(data_batch["__key__"]).encode()
        seed = int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            return self.training_step(data_batch, iteration)
