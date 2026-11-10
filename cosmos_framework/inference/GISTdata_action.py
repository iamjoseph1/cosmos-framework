# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Build WAM inputs from one current image, an instruction and measured joints."""

import json

import torch

from cosmos_framework.data.generator.action.utils.action_processing import make_batched_action_processing_fields
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.utils.GISTdata_action_contract import ACTION_DIM, DOMAIN_NAME, FPS


def build_GISTdata_action_batch(head_rgb, proprioception, instruction, horizon=32, resolution="256", device="cuda"):
    head_rgb = torch.as_tensor(head_rgb)
    q = torch.as_tensor(proprioception, dtype=torch.float32)
    if head_rgb.ndim != 3 or head_rgb.shape[0] != 3 or head_rgb.dtype != torch.uint8:
        raise ValueError("head_rgb must be uint8 [3,H,W] RGB")
    if q.shape != (ACTION_DIM,) or not torch.isfinite(q).all():
        raise ValueError("proprioception must contain 18 finite absolute joint angles in canonical order")
    if not instruction.strip() or horizon < 4 or horizon % 4:
        raise ValueError("A nonempty instruction and a positive horizon divisible by four are required")
    action = torch.zeros(horizon + 1, ACTION_DIM)
    action[0] = q
    sample = ActionTransformPipeline(max_action_dim=64, format_prompt_as_json=True, format_prompt_float_seconds=True)(
        {
            "video": head_rgb[:, None].repeat(1, horizon + 1, 1, 1),
            "action": action,
            "ai_caption": instruction,
            "mode": "wam",
            "viewpoint": "ego_view",
            "domain_id": torch.tensor(get_domain_id(DOMAIN_NAME)),
            "conditioning_fps": torch.tensor(FPS),
        },
        resolution,
    )
    prompt = json.dumps(sample["ai_caption"]) if isinstance(sample["ai_caption"], dict) else sample["ai_caption"]
    return {
        "video": [[sample["video"]]],
        "action": [[sample["action"]]],
        **make_batched_action_processing_fields(sample["action_processing_record"], 1),
        "mode": ["wam"],
        "ai_caption": [prompt],
        "prompt": [instruction],
        "conditioning_fps": [sample["conditioning_fps"]],
        "domain_id": [sample["domain_id"]],
        "image_size": sample["image_size"].unsqueeze(0).to(device),
        "sequence_plan": [sample["sequence_plan"]],
    }
