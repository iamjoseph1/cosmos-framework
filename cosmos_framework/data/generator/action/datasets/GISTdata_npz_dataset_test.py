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
    decode_head,
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
            for camera, offset in (("head", 0), ("left_wrist", 35), ("right_wrist", 70)):
                with av.open(str(ep / f"videos/{camera}.mp4"), "w") as c:
                    stream = c.add_stream("libx264", rate=20)
                    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                    for t in range(13):
                        frame = av.VideoFrame.from_ndarray(
                            np.full((48, 64, 3), t * 10 + offset, dtype=np.uint8), format="rgb24"
                        )
                        for packet in stream.encode(frame):
                            c.mux(packet)
                    for packet in stream.encode():
                        c.mux(packet)
            metadata_path = ep / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["cameras"].update({name: {"resolution": [64, 48]} for name in ("left_wrist", "right_wrist")})
            metadata_path.write_text(json.dumps(metadata))
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


def test_validation_loader_is_finite_repeatable_and_excludes_training_episodes(prepared, monkeypatch):
    from types import SimpleNamespace

    from cosmos_framework.data.generator.action.GISTdata_validation import GISTdataValidationLoader
    from cosmos_framework.data.generator.augmentors import text_tokenizer

    _, bundle, _ = prepared
    monkeypatch.setattr(
        text_tokenizer, "lazy_instantiate", lambda config: SimpleNamespace(tokenize_text=lambda *a, **k: [1, 2, 3])
    )
    dataset = get_GISTdata_action_sft_dataset(
        str(bundle), "val", cfg_dropout_rate=0, horizon=8, tokenizer_config={"fake": True}
    )
    loader = GISTdataValidationLoader(dataset)
    state = torch.random.get_rng_state().clone()
    first = list(loader)
    second = list(loader)
    assert len(first) == len(second) == len(dataset)
    assert [b["__key__"] for b in first] == [b["__key__"] for b in second]
    assert torch.equal(state, torch.random.get_rng_state())
    manifest = json.loads((bundle / "manifest.json").read_text())
    train_episodes = {e["episode_id"] for e in manifest["episodes"] if e["split"] == "train"}
    assert not {key.rsplit(":", 1)[0] for b in first for key in b["__key__"]} & train_episodes


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


@pytest.fixture
def threecam(prepared):
    root, head_bundle, _ = prepared
    bundle = head_bundle.parent / "threecam"
    prepare(root, bundle, expected_episodes=4, horizon=8, stride=3, camera_mode="threecam")
    return root, head_bundle, bundle


def test_threecam_contains_all_synchronized_views_and_same_actions(threecam):
    _, head_bundle, bundle = threecam
    head = GISTdataActionDataset(str(head_bundle), "all")
    multi = GISTdataActionDataset(str(bundle), "all")
    assert head.windows == multi.windows
    assert [(r["episode_id"], r["split"]) for r in head.episodes.values()] == [
        (r["episode_id"], r["split"]) for r in multi.episodes.values()
    ]
    for index in (0, 1, len(multi) - 1):
        sample = multi[index]
        assert sample["video"].shape == (3, 9, 72, 64)
        torch.testing.assert_close(sample["action"], head[index]["action"])
        torch.testing.assert_close(sample["video"][:, :, :48], head[index]["video"])
        # Each wrist has a distinctive brightness and stays at the matching time.
        for t in (0, 8):
            primary = sample["video"][:, t, 0, 0].float().mean()
            left = sample["video"][:, t, 50, 0].float().mean()
            right = sample["video"][:, t, 50, 40].float().mean()
            assert 30 < left - primary < 40
            assert 65 < right - primary < 75
        assert sample["viewpoint"] == "concat_view"
        assert "left-wrist" in sample["additional_view_description"]
    assert validate(bundle)["camera_order"] == ["head", "left_wrist", "right_wrist"]


def test_threecam_recipe_refuses_head_only_data(prepared):
    _, bundle, _ = prepared
    with pytest.raises(ValueError, match="camera mode"):
        get_GISTdata_action_sft_dataset(str(bundle), camera_mode="threecam")


def test_threecam_wrist_integrity_is_checked(threecam):
    _, _, bundle = threecam
    manifest = json.loads((bundle / "manifest.json").read_text())
    video = bundle / manifest["episodes"][0]["camera_videos"]["right_wrist"]
    video.write_bytes(video.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="right_wrist"):
        validate(bundle)


def test_threecam_rejects_unsynchronized_camera_before_output_creation(prepared):
    root, bundle, _ = prepared
    video = root / "results/red_objects_defaultscene_3cam/ycb_bowl/data_0001/videos/left_wrist.mp4"
    with av.open(str(video), "w") as c:
        stream = c.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for t in range(13):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    output = bundle.parent / "bad_threecam"
    with pytest.raises(ValueError, match="clock mismatch.*left_wrist"):
        prepare(root, output, expected_episodes=4, horizon=8, camera_mode="threecam")
    assert not output.exists()


def test_threecam_training_and_inference_preserve_layout_prompt_and_conditions(threecam):
    _, _, bundle = threecam
    raw_dataset = GISTdataActionDataset(str(bundle), "all")
    index = 1
    raw = raw_dataset[index]
    window = raw_dataset.windows[index]
    episode = raw_dataset.episodes[window["episode_id"]]
    views = {
        name: torch.from_numpy(decode_head(str(bundle / path))[window["start"]].copy()).permute(2, 0, 1)
        for name, path in episode["camera_videos"].items()
    }
    train = get_GISTdata_action_sft_dataset(str(bundle), "all", cfg_dropout_rate=0)[index]
    infer = build_GISTdata_action_batch(
        views["head"],
        raw["action"][0],
        raw["ai_caption"],
        horizon=8,
        device="cpu",
        left_wrist_rgb=views["left_wrist"],
        right_wrist_rgb=views["right_wrist"],
    )
    assert train["ai_caption"] == json.loads(infer["ai_caption"][0])
    torch.testing.assert_close(train["video"][:, 0], infer["video"][0][0][:, 0])
    torch.testing.assert_close(train["action"][0], infer["action"][0][0][0])
    assert torch.count_nonzero(infer["action"][0][0][1:]) == 0
    assert infer["sequence_plan"][0].condition_frame_indexes_vision == [0]
    assert infer["sequence_plan"][0].condition_frame_indexes_action == [0]


def test_inference_rejects_partial_or_inconsistent_camera_set():
    head = torch.zeros(3, 48, 64, dtype=torch.uint8)
    state = torch.zeros(18)
    with pytest.raises(ValueError, match="both wrist"):
        build_GISTdata_action_batch(head, state, "Pick up the mug.", device="cpu", left_wrist_rgb=head)
    with pytest.raises(ValueError, match="matching native dimensions"):
        build_GISTdata_action_batch(
            head,
            state,
            "Pick up the mug.",
            device="cpu",
            left_wrist_rgb=head,
            right_wrist_rgb=torch.zeros(3, 24, 32, dtype=torch.uint8),
        )
