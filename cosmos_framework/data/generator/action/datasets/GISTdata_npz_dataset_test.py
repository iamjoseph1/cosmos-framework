# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Regression checks for command/state alignment, split leakage and WAM conditioning."""

import json

import av
import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.GISTdata_npz_dataset import (
    GISTdataActionDataset,
    get_GISTdata_action_sft_dataset,
)
from cosmos_framework.inference.GISTdata_action import build_GISTdata_action_batch
from cosmos_framework.scripts.prepare_GISTdata_action_data import prepare
from cosmos_framework.scripts.validate_GISTdata_action_data import validate
from cosmos_framework.utils.GISTdata_action_contract import INSTRUCTIONS, window_starts

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "source"
    for obj in INSTRUCTIONS:
        for i in range(1, 3):
            inputs = root / "episodes" / obj
            inputs.mkdir(parents=True, exist_ok=True)
            ep = root / "results/red_objects_defaultscene_3cam" / obj / f"data_{i:04d}"
            (ep / "videos").mkdir(parents=True)
            action = np.arange(12 * 18, dtype=np.float32).reshape(12, 18) / 100
            state = np.arange(13 * 18, dtype=np.float32).reshape(13, 18) / 50
            np.savez(
                inputs / f"data_{i:04d}.npz",
                action=action,
                object_pose_world=[i, 0, 1, 1, 0, 0, 0],
                place_xy=[0.52, -0.38],
            )
            np.savez(ep / "observations.npz", state=state, timestamp=np.arange(13) / 20)
            (ep / "metadata.json").write_text(
                json.dumps(
                    {
                        "state_names": [f"j{x}" for x in range(18)],
                        "input_semantics": "absolute joint-position targets; T commands and T+1 observations",
                        "control_hz": 20,
                        "cameras": {"head": {"resolution": [64, 48]}},
                    }
                )
            )
            (ep / "replay_report.json").write_text(
                json.dumps(
                    {
                        "runtime_completed": True,
                        "executed_actions": 12,
                        "task_success": not (obj == "ycb_mug" and i == 2),
                    }
                )
            )
            with av.open(str(ep / "videos/head.mp4"), "w") as c:
                s = c.add_stream("libx264", rate=20)
                s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
                for t in range(13):
                    f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), t * 10, dtype=np.uint8), format="rgb24")
                    for packet in s.encode(f):
                        c.mux(packet)
                for packet in s.encode():
                    c.mux(packet)
    output = tmp_path / "bundle"
    summary = prepare(root, output, expected_episodes=4, horizon=8, stride=3)
    return root, output, summary


def test_group_split_and_all_episode_retention(prepared):
    _, bundle, summary = prepared
    assert summary["episodes"] == 4
    assert summary["unsuccessful_episodes"] == ["ycb_mug/data_0002"]
    assert summary["group_overlap"] == []
    m = json.loads((bundle / "manifest.json").read_text())
    assert len(m["episodes"]) == 4
    assert {r["object"] for r in m["episodes"] if r["split"] == "val"} == set(INSTRUCTIONS)


def test_tail_windows_do_not_invent_commands():
    assert window_starts(12, 8, 3) == [0, 3, 4]
    with pytest.raises(ValueError):
        window_starts(6, 8, 3)
    with pytest.raises(ValueError):
        window_starts(12, 7, 3)


def test_state_is_measured_before_the_first_command(prepared):
    _, bundle, _ = prepared
    ds = GISTdataActionDataset(str(bundle), "all")
    sample = ds[1]  # start=3
    record = ds.episodes[ds.windows[1]["episode_id"]]
    with np.load(bundle / record["trajectory"]) as t:
        np.testing.assert_array_equal(sample["action"][0].numpy(), t["state"][3])
        np.testing.assert_array_equal(sample["action"][1:].numpy(), t["action"][3:11])
    assert sample["video"].shape == (3, 9, 48, 64)


def test_training_and_inference_share_prompt_and_conditioning(prepared):
    _, bundle, _ = prepared
    raw = GISTdataActionDataset(str(bundle), "all")[0]
    train = get_GISTdata_action_sft_dataset(str(bundle), "all", cfg_dropout_rate=0)[0]
    infer = build_GISTdata_action_batch(
        raw["video"][:, 0], raw["action"][0], raw["ai_caption"], horizon=8, device="cpu"
    )
    assert train["ai_caption"] == json.loads(infer["ai_caption"][0])
    assert train["sequence_plan"].condition_frame_indexes_action == [0]
    assert infer["sequence_plan"][0].condition_frame_indexes_vision == [0]
    torch.testing.assert_close(train["action"][0], infer["action"][0][0][0])
    assert train["action_processing_record"].raw_action_dim == 18
    assert torch.count_nonzero(train["action"][:, 18:]) == 0
    assert torch.count_nonzero(infer["action"][0][0][1:]) == 0  # future commands are never supplied


def test_missing_episode_and_overwrite_are_rejected(prepared):
    root, bundle, _ = prepared
    with pytest.raises(ValueError, match="Expected 200"):
        prepare(root, bundle.parent / "partial")
    with pytest.raises(FileExistsError):
        prepare(root, bundle, expected_episodes=4, horizon=8)


def test_original_command_integrity_is_checked(prepared):
    _, bundle, _ = prepared
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["dataset_name"] == "GISTdata"
    path = bundle / manifest["episodes"][0]["trajectory"]
    with np.load(path) as data:
        values = {key: data[key].copy() for key in data.files}
    values["action"][0, 0] += 0.1
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="original GISTdata"):
        validate(bundle)
