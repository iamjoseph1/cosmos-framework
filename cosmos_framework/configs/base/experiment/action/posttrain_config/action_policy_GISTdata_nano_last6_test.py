# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Check the real Nano parameter scope before and after activation checkpointing."""

import re

import pytest
import torch

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_GISTdata_nano_last6 import (
    action_policy_GISTdata_nano_last6,
    last_generation_layer_keys,
)
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import apply_ac
from cosmos_framework.utils.generator.optimizer import _build_params_with_metadata
from cosmos_framework.utils.lazy_config import instantiate

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("checkpointed", [False, True])
def test_native_optimizer_trains_only_last_six_generation_layers_and_projections(checkpointed):
    c = action_policy_GISTdata_nano_last6
    # Meta tensors retain the released architecture and parameter names without
    # allocating its 15B weights or fetching any model checkpoints.
    with torch.device("meta"):
        language = instantiate(c.model.config.vlm_config.model_instance)
        network_config = Cosmos3VFMNetworkConfig(
            vlm_config=language.config,
            latent_patch_size=2,
            latent_channel_size=48,
            num_embodiment_domains=34,
            action_gen=True,
            action_dim=64,
            joint_attn_implementation="two_way",
        )
        network_config._attn_implementation_internal = "eager"
        net = Cosmos3VFMNetwork(language_model=language, config=network_config)
        if checkpointed:
            apply_ac(language, ActivationCheckpointingConfig(mode="full"))
        model = torch.nn.Module()
        model.net = net
    selected = _build_params_with_metadata(
        model,
        list(c.optimizer.keys_to_select),
        dict(c.optimizer.lr_multipliers),
        base_lr=c.optimizer.lr,
        base_weight_decay=c.optimizer.weight_decay,
        disable_weight_decay_for_1d_params=False,
    )
    trained_layers = set()
    for name, parameter in net.named_parameters():
        if name.startswith("language_model."):
            match = re.search(r"\.layers\.(\d+)\.", name)
            expected = bool(match and int(match[1]) >= 30 and "moe_gen" in name)
            assert parameter.requires_grad == expected, name
            if expected:
                trained_layers.add(int(match[1]))
        else:
            assert parameter.requires_grad, name
    assert trained_layers == set(range(30, 36))
    assert sum(parameter.numel() for parameter, _ in selected) == 1_195_061_056
    assert {id(parameter) for parameter, _ in selected} == {
        id(parameter) for parameter in net.parameters() if parameter.requires_grad
    }


@pytest.mark.parametrize("last_n", [-1, 0, 37])
def test_out_of_range_layer_counts_are_rejected(last_n):
    with pytest.raises(ValueError, match="last_n"):
        last_generation_layer_keys(36, last_n)
