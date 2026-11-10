# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""WAM adaptation for 20-Hz head RGB and 18-D absolute arm/hand joint targets."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.data.generator.action.datasets.GISTdata_npz_dataset import get_GISTdata_action_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L

action_policy_GISTdata_nano = copy.deepcopy(action_policy_droid_nano)
c = action_policy_GISTdata_nano
c.job.update(
    project="rb_xhand_dish_pnp", group="GISTdata_head_joint18", name="action_policy_GISTdata_nano", wandb_mode="online"
)
c.model.config.num_embodiment_domains = 34  # new domain 33; do not recycle retired IDs
c.model.config.resolution = "256"
c.model.config.max_num_tokens_after_packing = 8192
c.model.config.tokenizer.encode_exact_durations = [33]
c.model.config.ema.enabled = False
c.model.config.compile.enabled = False
c.optimizer.lr = 2e-5
c.optimizer.lr_multipliers = {"action2llm": 1.0, "llm2action": 1.0, "action_modality_embed": 1.0}
c.scheduler.cycle_lengths = [200]
c.scheduler.warm_up_steps = [20]
c.trainer.update(max_iter=200, grad_accum_iter=8, logging_iter=1)
c.checkpoint.update(save_iter=50, load_path="${oc.env:BASE_CHECKPOINT_PATH}")
c.dataloader_train.dataset_name = "GISTdata_head_joint18"
c.dataloader_train.max_samples_per_batch = 1
c.dataloader_train.max_sequence_length = 8192
c.dataloader_train.dataloader.update(batch_size=1, num_workers=2, prefetch_factor=2)
c.dataloader_train.dataloader.datasets = dict(
    GISTdata=dict(
        ratio=1,
        dataset=L(get_GISTdata_action_sft_dataset)(
            bundle="${oc.env:GISTDATA_ACTION_BUNDLE}",
            split="train",
            resolution="256",
            max_action_dim="${model.config.max_action_dim}",
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            successful_only=False,
            iterable_shuffle=True,
            horizon=32,
        ),
    )
)
ConfigStore.instance().store(group="experiment", package="_global_", name="action_policy_GISTdata_nano", node=c)
