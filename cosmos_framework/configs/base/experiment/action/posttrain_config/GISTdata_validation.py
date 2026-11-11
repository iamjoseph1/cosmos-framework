# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in validation wiring shared by the head and three-camera Edge recipes."""

import copy

from cosmos_framework.callbacks.GISTdata_validation import GISTdataValidationCallback
from cosmos_framework.data.generator.action.GISTdata_validation import GISTdataValidationLoader
from cosmos_framework.model.generator.GISTdata_validation import GISTdataValidationModel
from cosmos_framework.utils.lazy_config import LazyCall as L


def with_GISTdata_validation(recipe):
    config = copy.deepcopy(recipe)
    config.model._target_ = GISTdataValidationModel
    config.job.name += "_eval"
    config.trainer.update(run_validation=True, run_validation_on_start=True, validation_iter=200, max_val_iter=None)
    config.trainer.callbacks.wandb_val = L(GISTdataValidationCallback)()
    train_dataset = config.dataloader_train.dataloader.datasets.GISTdata.dataset
    val_dataset = copy.deepcopy(train_dataset)
    val_dataset.update(split="val", cfg_dropout_rate=0.0, iterable_shuffle=False)
    config.dataloader_val = L(GISTdataValidationLoader)(dataset=val_dataset, num_workers=0)
    return config
