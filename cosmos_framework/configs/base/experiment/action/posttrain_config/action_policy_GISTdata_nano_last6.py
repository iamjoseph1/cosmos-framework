# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GISTdata SFT of generation layers 30--35 and the video/action projections.

The understanding pathway and generation layers 0--29 remain frozen. FSDP
shards the model across two GPUs; FP32 parameter storage and BF16 compute are
inherited from the native Nano recipe.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_GISTdata_nano_heads import (
    action_policy_GISTdata_nano_heads,
)

GENERATION_MODULES = (
    "self_attn.q_proj_moe_gen",
    "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen",
    "self_attn.o_proj_moe_gen",
    "self_attn.q_norm_moe_gen",
    "self_attn.k_norm_moe_gen",
    "mlp_moe_gen",
    "input_layernorm_moe_gen",
    "post_attention_layernorm_moe_gen",
)


def last_generation_layer_keys(num_layers: int, last_n: int) -> list[str]:
    """Select complete generation submodules, with or without full AC wrappers.

    The optimizer matches substrings, so include both layer and module path
    boundaries. Selecting just 'layers.30.' would also train understanding
    weights; selecting an unwrapped path alone misses checkpointed layers.
    """
    if not 1 <= last_n <= num_layers:
        raise ValueError("last_n must be between one and num_layers")
    return [
        f".layers.{index}.{wrapper}{module}."
        for index in range(num_layers - last_n, num_layers)
        for wrapper in ("", "_checkpoint_wrapped_module.")
        for module in GENERATION_MODULES
    ]


action_policy_GISTdata_nano_last6 = copy.deepcopy(action_policy_GISTdata_nano_heads)
c = action_policy_GISTdata_nano_last6
c.job.name = "action_policy_GISTdata_nano_last6"
c.model.config.parallelism.data_parallel_shard_degree = 2
c.model.config.parallelism.data_parallel_replicate_degree = 1
c.optimizer.keys_to_select = list(c.optimizer.keys_to_select) + last_generation_layer_keys(36, 6)
# 1 window/rank x 2 ranks x 4 accumulations = 8 windows per optimizer step.
c.trainer.grad_accum_iter = 4
ConfigStore.instance().store(group="experiment", package="_global_", name="action_policy_GISTdata_nano_last6", node=c)
