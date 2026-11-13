# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Localize a downloaded GISTdata checkpoint without changing its DCP weights."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


def prepare(checkpoint_dir: Path, assets_dir: Path):
    checkpoint_dir, assets_dir = checkpoint_dir.resolve(), assets_dir.resolve()
    source = checkpoint_dir / "config.remote.yaml"
    original = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    dataset = original["dataloader_train"]["dataloader"]["datasets"]["GISTdata"]["dataset"]
    mode = dataset.get("camera_mode")
    manifest_file = checkpoint_dir / "data_manifest.remote.json"
    if manifest_file.is_file():
        # camera_mode=None delegates to the dataset manifest during training.
        # Inference must use that same manifest, including nullable threecam recipes.
        manifest_mode = json.loads(manifest_file.read_text()).get("camera_mode", "head")
        if mode is not None and mode != manifest_mode:
            raise ValueError("Saved camera mode and training data manifest disagree")
        mode = manifest_mode
    elif mode is None:
        if "camera_mode" in dataset:
            raise ValueError("camera_mode is unset; copy the training manifest.json as data_manifest.remote.json")
        mode = "head"  # Original head-only configs predate the camera_mode field.
    if mode not in ("head", "threecam"):
        raise ValueError(f"Unsupported camera mode: {mode}")
    processor = assets_dir / "Cosmos3-Edge"
    vae = assets_dir / "Wan2.2_VAE.pth"
    for path in (vae, processor / "config.json", processor / "vision_encoder/model.safetensors"):
        if not path.is_file():
            raise FileNotFoundError(path)
    model = copy.deepcopy(original["model"])
    # Validation adds no parameters. Inference uses the native model with the
    # identical state-dict schema, without importing a training-only subclass.
    if model.get("_target_") == "cosmos_framework.model.generator.GISTdata_validation.GISTdataValidationModel":
        model["_target_"] = "cosmos_framework.model.generator.omni_mot_model.OmniMoTModel"
    config = model["config"]
    if config["vlm_config"]["model_name"] != "nvidia/Cosmos3-Edge":
        raise ValueError("This local asset bundle is for the two trained GISTdata Edge policies")
    model_dirs = sorted((checkpoint_dir / "checkpoints").glob("iter_*/model"))
    if not model_dirs:
        raise FileNotFoundError(f"No DCP model checkpoints in {checkpoint_dir}")
    config["tokenizer"]["vae_path"] = str(vae)
    config["vlm_config"]["tokenizer"] = {
        "_target_": "cosmos_framework.data.generator.processors.build_processor",
        "tokenizer_type": str(processor),
    }
    config["ema"]["enabled"] = False
    config["compile"]["enabled"] = False
    config["activation_checkpointing"]["mode"] = "none"
    config["parallelism"]["enable_inference_mode"] = True
    local = {
        "model": model,
        "GISTdata_inference": {
            "camera_mode": mode,
            "source_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "camera_order": ["head"] if mode == "head" else ["head", "left_wrist", "right_wrist"],
            "action_semantics": "absolute_joint_position",
            "action_units": "rad",
            "horizon": 32,
            "fps": 20,
        },
    }
    # The lazy Edge vision tower reads this checkpoint-local bundle. Keep one
    # shared copy of the frozen vision weights for both trained policies.
    vision = processor / "vision_encoder"
    description = vision / "config.json"
    if not description.exists():
        description.write_bytes((processor / "config.json").read_bytes())
    for model_dir in model_dirs:
        if not (model_dir / ".metadata").is_file():
            raise ValueError(f"Incomplete DCP checkpoint: {model_dir}")
        link = model_dir / "vision_encoder"
        if link.is_symlink() or link.exists():
            if link.resolve() != vision:
                raise FileExistsError(f"Refusing to replace another vision tower: {link}")
        else:
            link.symlink_to(vision, target_is_directory=True)
    destination = checkpoint_dir / "config.local.yaml"
    text = OmegaConf.to_yaml(OmegaConf.create(local), resolve=True)
    if destination.exists() and destination.read_text() != text:
        raise FileExistsError(f"Refusing to replace modified local config: {destination}")
    destination.write_text(text)
    return {"config_file": str(destination), "camera_mode": mode, "assets": str(assets_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.checkpoint_dir, args.assets_dir), indent=2))


if __name__ == "__main__":
    main()
