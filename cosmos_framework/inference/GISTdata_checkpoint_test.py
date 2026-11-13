# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpoint selection must preserve the trained camera contract and iteration."""

import json
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from cosmos_framework.inference.GISTdata_checkpoint import resolve_GISTdata_checkpoint

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def bundle(root, name, mode, steps=(10000,)):
    folder = root / name
    folder.mkdir()
    (folder / "config.local.yaml").write_text(OmegaConf.to_yaml({"GISTdata_inference": {"camera_mode": mode}}))
    for step in steps:
        model = folder / f"checkpoints/iter_{step:09d}/model"
        model.mkdir(parents=True)
        (model / ".metadata").write_bytes(b"test")
    return folder


def test_same_entry_point_selects_both_v2_models_and_preserves_explicit_iteration(tmp_path):
    head = bundle(tmp_path, "edge_v2", "head", (4000, 10000))
    three = bundle(tmp_path, "edge_threecam_v2", "threecam")
    for count, expected in ((1, head), (3, three)):
        resolved = resolve_GISTdata_checkpoint("auto", tmp_path, num_cameras=count)
        assert resolved.bundle == expected and resolved.num_cameras == count and resolved.iteration == 10000
        assert resolve_GISTdata_checkpoint(expected.name, tmp_path) == resolved
    earlier = resolve_GISTdata_checkpoint(head / "checkpoints/iter_000004000", num_cameras=1)
    assert earlier.iteration == 4000


def test_camera_mode_comes_from_saved_config_and_rejects_wrong_client(tmp_path):
    folder = bundle(tmp_path, "custom_name", "threecam")
    with pytest.raises(ValueError, match="requires --num-cameras 3; client sent 1"):
        resolve_GISTdata_checkpoint(folder, num_cameras=1)
    assert resolve_GISTdata_checkpoint(folder, num_cameras=3).camera_mode == "threecam"


def test_incomplete_checkpoint_cannot_override_valid_latest(tmp_path):
    folder = bundle(tmp_path, "headcam_edge_600", "head", (600,))
    partial = folder / "checkpoints/iter_000010000/model"
    partial.mkdir(parents=True)
    assert resolve_GISTdata_checkpoint(folder).iteration == 600
    marker = folder / "checkpoints/latest_checkpoint.txt"
    marker.write_text("iter_000010000\n")
    with pytest.raises(ValueError, match="Incomplete"):
        resolve_GISTdata_checkpoint(folder)
    marker.write_text("../unexpected")
    with pytest.raises(ValueError, match="Invalid checkpoint marker"):
        resolve_GISTdata_checkpoint(folder)


def test_check_only_prints_checkpoint_without_loading_cuda_or_server(tmp_path):
    folder = bundle(tmp_path, "edge_v2", "head")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cosmos_framework.scripts.serve_GISTdata_policy",
            "--checkpoint",
            str(folder),
            "--check-only",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout)["selected_checkpoints"][0]
    assert report["num_cameras"] == 1 and report["iteration"] == 10000


@pytest.mark.parametrize("mode", ["head", "threecam"])
def test_validation_training_config_is_localized_to_native_inference_model(tmp_path, mode):
    from cosmos_framework.scripts.prepare_GISTdata_local_inference import prepare

    folder = bundle(tmp_path, "edge_v2", "head")
    assets = tmp_path / "assets"
    vision = assets / "Cosmos3-Edge/vision_encoder"
    vision.mkdir(parents=True)
    (assets / "Wan2.2_VAE.pth").touch()
    (vision.parent / "config.json").write_text("{}")
    (vision / "model.safetensors").touch()
    source = {
        "model": {
            "_target_": "cosmos_framework.model.generator.GISTdata_validation.GISTdataValidationModel",
            "config": {
                "vlm_config": {"model_name": "nvidia/Cosmos3-Edge"},
                "tokenizer": {},
                "ema": {},
                "compile": {},
                "activation_checkpointing": {},
                "parallelism": {},
            },
        },
        "dataloader_train": {"dataloader": {"datasets": {"GISTdata": {"dataset": {"camera_mode": None}}}}},
    }
    (folder / "config.remote.yaml").write_text(OmegaConf.to_yaml(source))
    (folder / "data_manifest.remote.json").write_text(json.dumps({"camera_mode": mode}))
    (folder / "config.local.yaml").unlink()  # prepare creates its own config, preserving existing modified ones.
    prepare(folder, assets)
    result = OmegaConf.load(folder / "config.local.yaml")
    assert result.model._target_ == "cosmos_framework.model.generator.omni_mot_model.OmniMoTModel"
    assert result.GISTdata_inference.camera_mode == mode
    assert OmegaConf.load(folder / "config.remote.yaml").model._target_ == source["model"]["_target_"]
    assert (folder / "checkpoints/iter_000010000/model/vision_encoder").resolve() == vision


@pytest.mark.parametrize(
    "saved_mode,manifest_mode,message",
    [(None, None, "camera_mode is unset"), ("head", "threecam", "manifest disagree")],
)
def test_localization_requires_unambiguous_training_camera_contract(tmp_path, saved_mode, manifest_mode, message):
    from cosmos_framework.scripts.prepare_GISTdata_local_inference import prepare

    source = {"dataloader_train": {"dataloader": {"datasets": {"GISTdata": {"dataset": {"camera_mode": saved_mode}}}}}}
    (tmp_path / "config.remote.yaml").write_text(OmegaConf.to_yaml(source))
    if manifest_mode is not None:
        (tmp_path / "data_manifest.remote.json").write_text(json.dumps({"camera_mode": manifest_mode}))
    with pytest.raises(ValueError, match=message):
        prepare(tmp_path, tmp_path / "assets")
