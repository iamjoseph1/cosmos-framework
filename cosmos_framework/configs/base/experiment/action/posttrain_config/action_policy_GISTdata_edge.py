# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GISTdata WAM SFT of the entire Edge generation pathway and robot projections.

Follows the official DROID-to-Edge recipe's generation/projection training scope,
adding Edge's generation-specific cross-attention K norm. The understanding
pathway and VAE remain frozen. LR, batch, resolution and schedule are GISTdata
adaptation settings, not a reproduction of the large-scale DROID training run.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_GISTdata_nano import (
    action_policy_GISTdata_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.utils.GISTdata_edge_contract import (
    GISTDATA_EDGE_REVISION,
    GISTDATA_EDGE_TRAINABLE_KEYS,
)

action_policy_GISTdata_edge = copy.deepcopy(action_policy_GISTdata_nano)
c = action_policy_GISTdata_edge
c.job.name = "action_policy_GISTdata_edge"
c.model.config = copy.deepcopy(EDGE_MODEL_CONFIG)
c.model.config.num_embodiment_domains = 34
c.model.config.resolution = "256"
c.model.config.max_num_tokens_after_packing = 8192
c.model.config.tokenizer.encode_exact_durations = [33]
c.model.config.ema.enabled = False
c.model.config.compile.enabled = False
c.model.config.parallelism.data_parallel_shard_degree = -1
c.model.config.parallelism.data_parallel_replicate_degree = 1
c.model.config.vlm_config.tokenizer.revision = GISTDATA_EDGE_REVISION
c.optimizer.keys_to_select = list(GISTDATA_EDGE_TRAINABLE_KEYS)
# Fresh embodiment projections use the official action recipe's 5x multiplier.
# The smaller 2e-5 base LR is an initial GISTdata small-batch setting.
c.optimizer.lr_multipliers = {"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0}
c.scheduler.f_max = [1.0]
c.scheduler.f_min = [0.0]
c.scheduler.f_start = [0.0]
ConfigStore.instance().store(group="experiment", package="_global_", name="action_policy_GISTdata_edge", node=c)
