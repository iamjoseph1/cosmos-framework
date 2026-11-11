# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Validation is repeatable, has no gradients and cannot change training counters."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.GISTdata_validation import GISTdataValidationModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class TinyValidationModel(GISTdataValidationModel):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.parameter = torch.nn.Parameter(torch.tensor(2.0))
        self.parallel_dims = None

    def training_step(self, batch, iteration):
        assert not torch.is_grad_enabled()
        value = self.parameter * torch.rand(())
        return {"prediction": value}, value


def test_validation_noise_is_repeatable_and_preserves_training_rng(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = TinyValidationModel().eval()
    state = torch.random.get_rng_state().clone()
    _, first = model.validation_step({"__key__": ["bowl:0"]}, 0)
    _, second = model.validation_step({"__key__": ["bowl:0"]}, 200)
    _, other = model.validation_step({"__key__": ["mug:0"]}, 200)
    assert first == second and first != other
    assert not first.requires_grad and model.parameter.grad is None
    assert torch.equal(state, torch.random.get_rng_state())
    assert not model.training
    model.train()
    with pytest.raises(RuntimeError, match="model.eval"):
        model.validation_step({"__key__": ["bowl:0"]}, 0)
    model.eval()
    model.parallel_dims = SimpleNamespace(cp_enabled=True)
    with pytest.raises(ValueError, match="context_parallel_degree=1"):
        model.validation_step({"__key__": ["bowl:0"]}, 0)


def test_validation_does_not_increment_training_sample_counters(monkeypatch):
    called = []
    monkeypatch.setattr(OmniMoTModel, "_update_train_stats", lambda self, data: called.append(data))
    model = TinyValidationModel().eval()
    model._update_train_stats("validation")
    assert called == []
    model.train()
    model._update_train_stats("training")
    assert called == ["training"]
