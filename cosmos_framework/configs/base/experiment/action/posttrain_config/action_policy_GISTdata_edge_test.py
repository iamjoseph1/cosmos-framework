# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Verify all Edge generation layers, cross-attention K norms and robot heads train."""

import json
import re

import pytest
import torch

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_GISTdata_edge import (
    action_policy_GISTdata_edge,
)
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import apply_ac
from cosmos_framework.scripts.prepare_GISTdata_edge_checkpoint import prepare
from cosmos_framework.utils.generator.optimizer import _build_params_with_metadata
from cosmos_framework.utils.GISTdata_edge_contract import GISTDATA_EDGE_REPOSITORY, GISTDATA_EDGE_REVISION
from cosmos_framework.utils.lazy_config import instantiate

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("checkpointed", [False, True])
def test_native_edge_generation_and_projections_train_with_understanding_frozen(checkpointed):
    c = action_policy_GISTdata_edge
    assert c.model.config.action_gen
    assert c.model.config.num_embodiment_domains == 34
    assert c.model.config.tokenizer.encode_exact_durations == [33]
    assert c.model.config.vlm_config.tokenizer.revision == GISTDATA_EDGE_REVISION
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
    generation_layers, cross_attention_norm_layers = set(), set()
    for name, parameter in net.named_parameters():
        if name.startswith("language_model."):
            expected = "moe_gen" in name or "k_norm_und_for_gen" in name
            assert parameter.requires_grad == expected, name
            if expected:
                match = re.search(r"\.layers\.(\d+)\.", name)
                # Edge also has a trainable final generation norm outside its layers.
                if match:
                    generation_layers.add(int(match[1]))
                if "k_norm_und_for_gen" in name:
                    assert match is not None, name
                    cross_attention_norm_layers.add(int(match[1]))
        else:
            assert parameter.requires_grad, name
    assert generation_layers == cross_attention_norm_layers == set(range(28))
    assert sum(parameter.numel() for parameter, _ in selected) == 1_423_911_744
    # Newly initialized action encoders/decoders use the robot recipe's 5x LR.
    action_weights = {id(p) for n, p in net.named_parameters() if "action2llm" in n or "llm2action" in n}
    assert all(metadata.lr == pytest.approx(1e-4) for p, metadata in selected if id(p) in action_weights)


def test_checkpoint_preparation_rejects_unmatched_dcp(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model/.metadata").touch()
    (tmp_path / "GISTdata_edge_base.json").write_text(
        json.dumps({"repository": "nvidia/Cosmos3-Nano", "revision": "x"})
    )
    with pytest.raises(ValueError, match="different model"):
        prepare(tmp_path)


def test_checkpoint_preparation_preserves_partial_outputs(tmp_path):
    marker = tmp_path / "user_file"
    marker.write_text("preserve me")
    with pytest.raises(FileExistsError, match="nonempty"):
        prepare(tmp_path)
    assert marker.read_text() == "preserve me"


def test_checkpoint_preparation_reuses_a_matched_completed_conversion(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model/.metadata").touch()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    report = {"repository": GISTDATA_EDGE_REPOSITORY, "revision": GISTDATA_EDGE_REVISION, "snapshot": str(snapshot)}
    (tmp_path / "GISTdata_edge_base.json").write_text(json.dumps(report))
    assert prepare(tmp_path) == report
