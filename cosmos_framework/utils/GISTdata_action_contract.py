# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GISTdata contract for original right-arm/right-hand absolute joint targets."""

import hashlib
from pathlib import Path

import numpy as np

DOMAIN_NAME = "gistdata_joint_pos"
ACTION_DIM = 18
FPS = 20
INSTRUCTIONS = {
    "ycb_bowl": "Using your right arm and right hand, pick up the red bowl and place it at the designated drop-off position on the table.",
    "ycb_mug": "Using your right arm and right hand, pick up the red mug and place it at the designated drop-off position on the table.",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_sha256(action: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest()


def window_starts(transitions: int, horizon: int, stride: int) -> list[int]:
    """Cover the final transition without padding commands or crossing episodes."""
    if horizon < 1 or horizon % 4 or stride < 1:
        raise ValueError("horizon must be positive and divisible by four; stride must be positive")
    if transitions < horizon:
        raise ValueError(f"Only {transitions} transitions for a {horizon}-step horizon")
    starts = list(range(0, transitions - horizon + 1, stride))
    last = transitions - horizon
    if starts[-1] != last:
        starts.append(last)
    return starts
