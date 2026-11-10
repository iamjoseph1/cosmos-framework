# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Validate every portable trajectory and materialize WAM examples without weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.datasets.GISTdata_npz_dataset import get_GISTdata_action_sft_dataset
from cosmos_framework.utils.GISTdata_action_contract import action_sha256, file_sha256


def validate(bundle: Path, tokenizer_folder: Path | None = None, tokenizer_model: str = "nano"):
    if tokenizer_model not in ("nano", "edge"):
        raise ValueError("tokenizer_model must be nano or edge")
    m = json.loads((bundle / "manifest.json").read_text())
    groups = {s: {r["group"] for r in m["episodes"] if r["split"] == s} for s in ("train", "val")}
    if groups["train"] & groups["val"]:
        raise ValueError("Initial-condition group leakage between train and val")
    for r in m["episodes"]:
        with np.load(bundle / r["trajectory"], allow_pickle=False) as a:
            if a["state"].shape != (r["transitions"] + 1, 18) or a["action"].shape != (r["transitions"], 18):
                raise ValueError(f"Broken trajectory: {r['episode_id']}")
            if not np.isfinite(a["state"]).all() or not np.isfinite(a["action"]).all():
                raise ValueError(f"Nonfinite trajectory: {r['episode_id']}")
            if a["action"].dtype != np.float32 or action_sha256(a["action"]) != r["original_action_sha256"]:
                raise ValueError(f"Commands no longer match the original GISTdata: {r['episode_id']}")
        if not (bundle / r["head_video"]).is_file():
            raise FileNotFoundError(r["head_video"])
        if file_sha256(bundle / r["head_video"]) != r["source_head_sha256"]:
            raise ValueError(f"Head video was changed: {r['episode_id']}")
    tokenizer = None
    if tokenizer_folder is not None and tokenizer_model == "edge":
        from cosmos_framework.data.generator.processors import build_processor

        tokenizer = {"_target_": build_processor, "tokenizer_type": str(tokenizer_folder)}
    elif tokenizer_folder is not None:
        from cosmos_framework.configs.base.defaults.reasoner import create_qwen2_tokenizer_with_download

        tokenizer = {
            "_target_": create_qwen2_tokenizer_with_download,
            "pretrained_model_name": str(tokenizer_folder),
            "config_variant": "hf",
        }
    examples = []
    for split in ("train", "val"):
        ds = get_GISTdata_action_sft_dataset(str(bundle), split=split, tokenizer_config=tokenizer, cfg_dropout_rate=0)
        # Both ends of every episode: include tail windows and both object types.
        for start, length in ds.get_shuffle_blocks():
            for index in sorted({start, start + length - 1}):
                sample = ds[index]
                plan = sample["sequence_plan"]
                assert plan.condition_frame_indexes_vision == [0]
                assert plan.condition_frame_indexes_action == [0]
                assert tuple(sample["action"].shape) == (m["horizon"] + 1, 64)
                assert torch.count_nonzero(sample["action"][:, 18:]) == 0
                assert torch.isfinite(sample["action"]).all()
                assert sample["action_processing_record"].raw_action_dim == 18
                if tokenizer is not None:
                    assert sample["text_token_ids"].numel() > 0
        examples.append(
            {
                "split": split,
                "windows": len(ds),
                "video_shape": list(sample["video"].shape),
                "action_shape": list(sample["action"].shape),
                "state_condition_rows": [0],
            }
        )
    return {
        "passed": True,
        "episodes": len(m["episodes"]),
        "tokenized": tokenizer is not None,
        "tokenizer_model": tokenizer_model,
        "samples": examples,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--tokenizer-folder", type=Path)
    p.add_argument("--tokenizer-model", choices=("nano", "edge"), default="nano")
    a = p.parse_args()
    print(json.dumps(validate(a.bundle, a.tokenizer_folder, a.tokenizer_model), indent=2))


if __name__ == "__main__":
    main()
