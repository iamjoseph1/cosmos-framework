# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Head-camera Edge SFT with held-out validation enabled."""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_GISTdata_edge import (
    action_policy_GISTdata_edge,
)
from cosmos_framework.configs.base.experiment.action.posttrain_config.GISTdata_validation import (
    with_GISTdata_validation,
)

action_policy_GISTdata_edge_eval = with_GISTdata_validation(action_policy_GISTdata_edge)
ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_GISTdata_edge_eval",
    node=action_policy_GISTdata_edge_eval,
)
