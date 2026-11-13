# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Resolve prepared robot-policy checkpoints and their trained camera contract."""

import re
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class GISTdataCheckpoint:
    bundle: Path
    checkpoint: Path
    config_file: Path
    camera_mode: str
    iteration: int

    @property
    def num_cameras(self):
        return 1 if self.camera_mode == "head" else 3

    def describe(self):
        return {
            "bundle": str(self.bundle),
            "checkpoint": str(self.checkpoint),
            "config_file": str(self.config_file),
            "camera_mode": self.camera_mode,
            "num_cameras": self.num_cameras,
            "iteration": self.iteration,
        }


def resolve_GISTdata_checkpoint(selection, checkpoint_root=Path("outputs/checkpoints/GISTdata"), *, num_cameras=None):
    """Accept a prepared bundle, an explicit iter_* directory, or a bundle name.

    ``auto`` selects the v2 bundle matching the client's camera count. An explicit
    selection always stays fixed, and a mismatched client is rejected before CUDA.
    """
    if num_cameras is not None and num_cameras not in (1, 3):
        raise ValueError("num_cameras must be 1 or 3")
    selection = str(selection) if selection is not None else "auto"
    if selection == "auto":
        if num_cameras is None:
            raise ValueError("auto checkpoint selection requires num_cameras=1 or 3")
        selection = "edge_v2" if num_cameras == 1 else "edge_threecam_v2"
    candidate = Path(selection).expanduser()
    if not candidate.exists() and not candidate.is_absolute() and len(candidate.parts) == 1:
        candidate = Path(checkpoint_root) / candidate
    candidate = candidate.resolve()
    if (candidate / "config.local.yaml").is_file():
        bundle = candidate
        checkpoints = bundle / "checkpoints"
        marker = checkpoints / "latest_checkpoint.txt"
        if marker.exists():
            name = marker.read_text().strip()
            if not re.fullmatch(r"iter_\d{9}", name):
                raise ValueError(f"Invalid checkpoint marker: {marker}")
            checkpoint = checkpoints / name
        else:
            complete = sorted(
                p
                for p in checkpoints.glob("iter_*")
                if re.fullmatch(r"iter_\d{9}", p.name) and (p / "model/.metadata").is_file()
            )
            if not complete:
                raise FileNotFoundError(f"No complete model checkpoint in {checkpoints}")
            checkpoint = complete[-1]
    else:
        checkpoint = candidate
        bundle = candidate.parent.parent
    config_file = bundle / "config.local.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"Prepared config missing: {config_file}; run prepare_GISTdata_local_inference first")
    if not re.fullmatch(r"iter_\d{9}", checkpoint.name) or not (checkpoint / "model/.metadata").is_file():
        raise ValueError(f"Incomplete or unsupported DCP checkpoint: {checkpoint}")
    saved = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)
    mode = saved.get("GISTdata_inference", {}).get("camera_mode")
    if mode not in ("head", "threecam"):
        raise ValueError(f"Missing trained head/threecam camera mode in {config_file}")
    result = GISTdataCheckpoint(bundle, checkpoint, config_file, mode, int(checkpoint.name[5:]))
    if num_cameras is not None and result.num_cameras != num_cameras:
        raise ValueError(
            f"Checkpoint {bundle.name} requires --num-cameras {result.num_cameras}; client sent {num_cameras}"
        )
    return result
