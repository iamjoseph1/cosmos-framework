# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Head or synchronized three-camera RGB + task instruction + measured joints -> absolute joint commands."""

import json
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.action.utils.viewpoint_utils import compose_multiview
from cosmos_framework.utils.GISTdata_action_contract import ACTION_DIM, DOMAIN_NAME, FPS


@lru_cache(maxsize=6)
def decode_head(path: str) -> np.ndarray:
    with av.open(path) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


@lru_cache(maxsize=8)
def load_trajectory(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["state"], data["action"]


class GISTdataActionDataset(Dataset):
    def __init__(self, bundle: str, split="train", successful_only=False):
        self.root = Path(bundle).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if (m["schema_version"], m["domain_name"], m["raw_action_dim"], m["fps"], m["normalization"]) != (
            1,
            DOMAIN_NAME,
            ACTION_DIM,
            FPS,
            "none",
        ):
            raise ValueError("Unsupported GISTdata action contract")
        self.camera_mode = m.get("camera_mode", "head")
        if self.camera_mode not in ("head", "threecam"):
            raise ValueError("Unsupported GISTdata camera mode")
        if self.camera_mode == "threecam" and m["camera_order"] != ["head", "left_wrist", "right_wrist"]:
            raise ValueError("Unexpected three-camera order")
        if split not in ("train", "val", "all"):
            raise ValueError("split must be train, val, or all")
        self.episodes = {e["episode_id"]: e for e in m["episodes"]}
        self.windows = [json.loads(s) for s in (self.root / f"{split}.jsonl").read_text().splitlines() if s.strip()]
        if successful_only:
            self.windows = [w for w in self.windows if self.episodes[w["episode_id"]]["task_success"]]
        if not self.windows:
            raise ValueError(f"No windows in split {split}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        w = self.windows[index]
        ep = self.episodes[w["episode_id"]]
        start, horizon = w["start"], self.manifest["horizon"]
        state, commands = load_trajectory(str(self.root / ep["trajectory"]))
        if self.camera_mode == "threecam":
            views = [
                torch.from_numpy(
                    decode_head(str(self.root / ep["camera_videos"][camera]))[start : start + horizon + 1].copy()
                )
                .permute(0, 3, 1, 2)
                .float()
                for camera in self.manifest["camera_order"]
            ]
            video = compose_multiview(*views).round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3)
        else:
            frames = decode_head(str(self.root / ep["head_video"]))[start : start + horizon + 1].copy()
            video = torch.from_numpy(frames).permute(3, 0, 1, 2)
        action = np.concatenate([state[start : start + 1], commands[start : start + horizon]], axis=0)
        if video.shape[1] != horizon + 1 or action.shape != (horizon + 1, ACTION_DIM):
            raise ValueError(f"Broken temporal alignment: {ep['episode_id']}:{start}")
        sample = {
            "__key__": f"{ep['episode_id']}:{start}",
            "video": video,
            "action": torch.from_numpy(action),
            "ai_caption": ep["instruction"],
            "conditioning_fps": torch.tensor(FPS, dtype=torch.long),
            "domain_id": torch.tensor(get_domain_id(DOMAIN_NAME), dtype=torch.long),
            "mode": "wam",
            "viewpoint": "concat_view" if self.camera_mode == "threecam" else "ego_view",
        }
        if self.camera_mode == "threecam":
            sample["additional_view_description"] = self.manifest["camera_layout"]["description"]
        return sample

    def get_shuffle_blocks(self):
        blocks = []
        start = 0
        while start < len(self.windows):
            end = start + 1
            while end < len(self.windows) and self.windows[end]["episode_id"] == self.windows[start]["episode_id"]:
                end += 1
            blocks.append((start, end - start))
            start = end
        return blocks


def get_GISTdata_action_sft_dataset(
    bundle: str,
    split="train",
    resolution="256",
    max_action_dim=64,
    tokenizer_config=None,
    successful_only=False,
    cfg_dropout_rate=0.1,
    iterable_shuffle=False,
    horizon=None,
    camera_mode=None,
):
    dataset = GISTdataActionDataset(bundle, split, successful_only)
    if camera_mode is not None and dataset.camera_mode != camera_mode:
        raise ValueError("Bundle camera mode differs from the training recipe")
    if horizon is not None and dataset.manifest["horizon"] != horizon:
        raise ValueError("Bundle horizon differs from the training recipe's VAE/action horizon")
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        max_action_dim=max_action_dim,
        cfg_dropout_rate=cfg_dropout_rate,
        append_idle_frames=False,
        format_prompt_as_json=True,
        format_prompt_float_seconds=True,
    )
    wrapped = ActionSFTDataset(dataset, transform, resolution)
    return ActionIterableShuffleDataset(wrapped, seed=dataset.manifest["seed"]) if iterable_shuffle else wrapped
