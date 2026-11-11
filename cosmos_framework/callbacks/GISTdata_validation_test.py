# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Held-out modality metrics use the same step, finite denominator and sample IDs."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks import GISTdata_validation as module

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_validation_metrics_aggregate_and_reset(monkeypatch):
    logged = []
    monkeypatch.setattr(
        module, "wandb", SimpleNamespace(run=object(), log=lambda data, step: logged.append((data, step)))
    )
    callback = module.GISTdataValidationCallback()
    callback.on_validation_start(None, None)
    for loss, action, vision, key in [(2, 0.4, 1.6, "a"), (4, 0.6, 3.4, "b"), (float("nan"), 1, 1, "c")]:
        callback.on_validation_step_end(
            None,
            {"__key__": [key]},
            {
                "flow_matching_loss_action": torch.tensor(action),
                "flow_matching_loss_vision": torch.tensor(vision),
            },
            torch.tensor(loss),
            iteration=200,
        )
    callback.on_validation_end(None, iteration=200)
    metrics, step = logged[0]
    assert step == 200 and metrics["val/loss"] == 3
    assert metrics["val/action_loss"] == pytest.approx(0.5)
    assert metrics["val/vision_loss"] == pytest.approx(2.5)
    assert metrics["val/unstable_count"] == 1
    assert metrics["val/num_unique_url_key"] == metrics["val/total_url_key"] == 3
    assert callback.totals is None and callback.keys == []
    callback.on_validation_start(None, None)
    with pytest.raises(RuntimeError, match="no batches"):
        callback.on_validation_end(None)
