# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Guard short finite schedules and preserve progress when extending a resumed run."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from cosmos_framework.utils.functional.lr_scheduler import validate_lr_schedule_horizon
from cosmos_framework.utils.generator.optimizer import LRSchedulersContainer

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("kind", ["LambdaLinear", "LambdaCosine"])
def test_schedule_shorter_than_training_is_rejected(kind):
    config = OmegaConf.create({"lr_scheduler_type": kind, "cycle_lengths": [200]})
    with pytest.raises(ValueError, match="covers 200 optimizer steps, but training requires 600"):
        validate_lr_schedule_horizon(config, 600)
    validate_lr_schedule_horizon(config, 200)
    config.cycle_lengths = [200, 400]
    validate_lr_schedule_horizon(config, 600)


def test_nonfinite_schedules_are_not_subject_to_lambda_cycle_limits():
    validate_lr_schedule_horizon(SimpleNamespace(lr_scheduler_type="Constant", cycle_lengths=[200]), 600)


def test_resume_extends_lambda_schedule_without_resetting_step_or_base_lrs():
    def create(length):
        optimizer = torch.optim.SGD(
            [
                {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 2e-5},
                {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 1e-4},
            ],
        )
        scheduler = LRSchedulersContainer(
            [optimizer],
            "LambdaLinear",
            cycle_lengths=[length],
            warm_up_steps=[20],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[0.0],
        )
        return optimizer, scheduler

    old_optimizer, old_scheduler = create(200)
    for _ in range(200):
        old_optimizer.step()
        old_scheduler.step()
    saved = old_scheduler.state_dict()
    assert saved["last_epoch"] == 200
    optimizer, scheduler = create(600)
    optimizer.load_state_dict(old_optimizer.state_dict())
    scheduler.load_state_dict(saved)
    assert scheduler.state_dict()["last_epoch"] == 200
    assert scheduler.get_last_lr() == pytest.approx([2e-5 * 400 / 580, 1e-4 * 400 / 580])
    for _ in range(400):
        optimizer.step()
        scheduler.step()
    assert scheduler.state_dict()["last_epoch"] == 600
    assert scheduler.get_last_lr() == pytest.approx([0.0, 0.0])
    assert saved["last_epoch"] == 200


def test_toml_horizon_guard_runs_after_command_line_overrides(monkeypatch, tmp_path):
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    for name in ("GISTDATA_ACTION_BUNDLE", "BASE_CHECKPOINT_PATH", "WAN_VAE_PATH"):
        monkeypatch.setenv(name, str(tmp_path))
    path = "examples/toml/sft_config/action_policy_GISTdata_edge.toml"
    with pytest.raises(ValueError, match="training requires 600"):
        load_experiment_from_toml(path, extra_overrides=["trainer.max_iter=600", "scheduler.cycle_lengths=[200]"])
    c = load_experiment_from_toml(path, extra_overrides=["trainer.max_iter=600", "scheduler.cycle_lengths=[600]"])
    assert c.trainer.max_iter == 600
    assert c.scheduler.cycle_lengths == [600]
