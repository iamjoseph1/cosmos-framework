# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Single-GPU GISTdata projection tuning with a frozen MoT language network.

The WAM video/action targets are unchanged. Only the video/action projections,
time embedder and action modality embedding train. This is a smaller training
scope than action_policy_GISTdata_nano, which also trains the generation tower.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_GISTdata_nano import (
    action_policy_GISTdata_nano,
)

action_policy_GISTdata_nano_heads = copy.deepcopy(action_policy_GISTdata_nano)
c = action_policy_GISTdata_nano_heads
c.job.name = "action_policy_GISTdata_nano_heads"
# Retain the native FP32 parameter storage and BF16 compute policy, including
# FP32 timestep embedding. Freezing the generation tower removes its large
# gradient and optimizer allocations without changing the native dtype policy.
c.optimizer.keys_to_select = [
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
]
ConfigStore.instance().store(group="experiment", package="_global_", name="action_policy_GISTdata_nano_heads", node=c)
