# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Validation recipes keep training splits and camera modes separate."""

from pathlib import Path

import pytest
import tomllib

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_GISTdata_edge import (
    action_policy_GISTdata_edge,
)
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_GISTdata_edge_eval import (
    action_policy_GISTdata_edge_eval,
)
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_GISTdata_edge_threecam_eval import (
    action_policy_GISTdata_edge_threecam_eval,
)
from cosmos_framework.model.generator.GISTdata_validation import GISTdataValidationModel

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize(
    "config,mode", [(action_policy_GISTdata_edge_eval, "head"), (action_policy_GISTdata_edge_threecam_eval, "threecam")]
)
def test_opt_in_validation_keeps_original_training_recipe(config, mode):
    assert config.model._target_ is GISTdataValidationModel
    assert config.trainer.run_validation and config.trainer.run_validation_on_start
    assert config.trainer.max_val_iter is None and config.trainer.validation_iter == 200
    train = config.dataloader_train.dataloader.datasets.GISTdata.dataset
    val = config.dataloader_val.dataset
    assert train.split == "train" and val.split == "val"
    assert train.iterable_shuffle and not val.iterable_shuffle
    assert val.cfg_dropout_rate == 0
    assert (val.get("camera_mode") or "head") == mode
    assert not action_policy_GISTdata_edge.trainer.run_validation
    assert action_policy_GISTdata_edge.dataloader_val is None
    root = Path(__file__).resolve().parents[6]
    stem = "action_policy_GISTdata_edge" + ("_threecam" if mode == "threecam" else "") + "_eval"
    recipe = tomllib.loads((root / f"examples/toml/sft_config/{stem}.toml").read_text())
    assert recipe["job"]["experiment"] == stem
    assert recipe["trainer"]["run_validation"]
    assert recipe["trainer"]["validation_iter"] == 200
