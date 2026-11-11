# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Three-camera Edge SFT with the same held-out validation episodes."""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_GISTdata_edge_threecam import (
    action_policy_GISTdata_edge_threecam,
)
from cosmos_framework.configs.base.experiment.action.posttrain_config.GISTdata_validation import (
    with_GISTdata_validation,
)

action_policy_GISTdata_edge_threecam_eval = with_GISTdata_validation(action_policy_GISTdata_edge_threecam)
ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_GISTdata_edge_threecam_eval",
    node=action_policy_GISTdata_edge_threecam_eval,
)
