# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Three synchronized RGB views with the same Edge generation/action training scope."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_GISTdata_edge import (
    action_policy_GISTdata_edge,
)

action_policy_GISTdata_edge_threecam = copy.deepcopy(action_policy_GISTdata_edge)
c = action_policy_GISTdata_edge_threecam
c.job.group = "GISTdata_threecam_joint18"
c.job.name = "action_policy_GISTdata_edge_threecam"
c.dataloader_train.dataset_name = "GISTdata_threecam_joint18"
c.dataloader_train.dataloader.datasets.GISTdata.dataset.camera_mode = "threecam"
c.checkpoint.save_iter = 200
ConfigStore.instance().store(
    group="experiment", package="_global_", name="action_policy_GISTdata_edge_threecam", node=c
)
