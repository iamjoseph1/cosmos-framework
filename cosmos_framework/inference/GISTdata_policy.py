# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Persistent native WAM inference for the trained GISTdata Edge policies."""

import gc
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.inference.GISTdata_action import build_GISTdata_action_batch
from cosmos_framework.inference.GISTdata_checkpoint import resolve_GISTdata_checkpoint


class GISTdataPolicy:
    def __init__(
        self, checkpoint_root: Path, num_cameras: int, output_dir: Path, *, num_steps=35, guidance=3.0, checkpoint=None
    ):
        selected = resolve_GISTdata_checkpoint(checkpoint, checkpoint_root, num_cameras=num_cameras)
        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.inference import OmniInference
        from cosmos_framework.scripts.action_policy_server_utils import (
            disable_runtime_ema_for_frozen_config,
            maybe_init_distributed,
        )

        maybe_init_distributed()
        setup = OmniSetupOverrides(
            checkpoint_path=str(selected.checkpoint),
            config_file=str(selected.config_file),
            output_dir=output_dir,
            cp_size=1,
            dp_shard_size=1,
            cfgp_size=1,
            use_torch_compile=False,
            use_cuda_graphs=False,
            guardrails=False,
        ).build_setup()
        self.pipe = OmniInference.create(disable_runtime_ema_for_frozen_config(setup))
        self.model = self.pipe.model.eval()
        config = self.model.config
        if not config.action_gen or config.num_embodiment_domains != 34 or config.max_action_dim != 64:
            raise ValueError("Expected the trained GISTdata joint18/domain33 action contract")
        if str(config.resolution) != "256" or 33 not in config.tokenizer.encode_exact_durations:
            raise ValueError("Expected GISTdata resolution256 and 32-action horizon")
        self.num_cameras, self.num_steps, self.guidance = num_cameras, num_steps, guidance
        self.checkpoint = str(selected.checkpoint)

    def predict(self, arrays, instruction, seed):
        def rgb(name):
            return torch.from_numpy(np.array(arrays[f"observation.images.{name}"], copy=True)).permute(2, 0, 1)

        batch = build_GISTdata_action_batch(
            rgb("head"),
            arrays["observation.state"],
            instruction,
            device="cuda",
            left_wrist_rgb=rgb("left_wrist") if self.num_cameras == 3 else None,
            right_wrist_rgb=rgb("right_wrist") if self.num_cameras == 3 else None,
        )
        with torch.inference_mode():
            result = self.model.generate_samples_from_batch(
                batch,
                seed=[seed],
                num_steps=self.num_steps,
                guidance=self.guidance,
                has_negative_prompt=False,
            )
            actions = result["action"][0].squeeze(0).detach().float().cpu().numpy()
        if actions.shape == (33, 18):
            actions = actions[1:]
        if actions.shape != (32, 18) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid generated action chunk: {actions.shape}")
        return np.ascontiguousarray(actions, dtype=np.float32)

    def close(self):
        self.model = self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
