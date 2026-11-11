# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Predict absolute right-arm/hand commands from current synchronized RGB images and joints.

--dry-run validates the input contract without weights or CUDA. Real inference
requires a checkpoint trained with a GISTdata action-policy recipe; the base model
does not have a trained projector for this robot. No robot commands are sent.
"""

import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from cosmos_framework.inference.GISTdata_action import build_GISTdata_action_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True, help="Current head-camera RGB image")
    p.add_argument("--left-wrist-image", type=Path, help="Current left-wrist RGB, synchronized with --image")
    p.add_argument("--right-wrist-image", type=Path, help="Current right-wrist RGB, synchronized with --image")
    p.add_argument("--actions-only", action="store_true", help="Skip VAE decoding/saving of predicted future RGB")
    p.add_argument(
        "--proprioception", type=Path, required=True, help="JSON list of 18 measured angles in manifest joint order"
    )
    p.add_argument("--instruction", required=True)
    p.add_argument("--output", type=Path, required=True, help="New output directory")
    p.add_argument("--checkpoint-path", type=Path)
    p.add_argument("--config-file", type=Path, help="Matching saved training config.yaml (or exported config.json)")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-steps", type=int, default=35)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if (a.left_wrist_image is None) != (a.right_wrist_image is None):
        p.error("Provide both --left-wrist-image and --right-wrist-image for the three-camera model")
    if not a.dry_run and (a.checkpoint_path is None or a.config_file is None):
        p.error("Real inference requires --checkpoint-path and --config-file for the trained robot model")
    if a.num_steps < 1:
        p.error("--num-steps must be positive")
    threecam = a.left_wrist_image is not None
    if a.config_file is not None:
        from omegaconf import OmegaConf

        saved = OmegaConf.to_container(OmegaConf.load(a.config_file), resolve=True)
        expected = saved.get("GISTdata_inference", {}).get("camera_mode")
        if expected is None:
            dataset = saved.get("dataloader_train", {}).get("dataloader", {}).get("datasets", {}).get("GISTdata", {})
            expected = dataset.get("dataset", {}).get("camera_mode", "head")
        if expected != ("threecam" if threecam else "head"):
            p.error(f"Camera inputs do not match checkpoint config camera_mode={expected}")

    def load_rgb(path):
        return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)

    image = load_rgb(a.image)
    state = json.loads(a.proprioception.read_text())
    batch = build_GISTdata_action_batch(
        image,
        state,
        a.instruction,
        a.horizon,
        device="cpu" if a.dry_run else "cuda",
        left_wrist_rgb=load_rgb(a.left_wrist_image) if threecam else None,
        right_wrist_rgb=load_rgb(a.right_wrist_image) if threecam else None,
    )
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "instruction": a.instruction,
        "camera_mode": "threecam" if threecam else "head",
        "image": str(a.image),
        "left_wrist_image": str(a.left_wrist_image) if threecam else None,
        "right_wrist_image": str(a.right_wrist_image) if threecam else None,
        "cpu_offload": False,
        "save_future_video": not a.actions_only,
        "joint_units": "rad",
        "action_semantics": "absolute_joint_position",
        "raw_action_dim": 18,
        "fps": 20,
        "horizon": a.horizon,
        "seed": a.seed,
        "num_steps": a.num_steps,
        "guidance": a.guidance,
        "vision_condition_rows": batch["sequence_plan"][0].condition_frame_indexes_vision,
        "action_condition_rows": batch["sequence_plan"][0].condition_frame_indexes_action,
        "checkpoint": str(a.checkpoint_path),
        "config": str(a.config_file),
    }
    (a.output / "request.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if a.dry_run:
        print(
            json.dumps(
                {
                    "passed": True,
                    "video_shape": list(batch["video"][0][0].shape),
                    "action_shape": list(batch["action"][0][0].shape),
                    **metadata,
                },
                indent=2,
            )
        )
        return
    from cosmos_framework.data.generator.action.utils.transforms import remove_reflection_padding
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.scripts.action_policy_server_utils import (
        disable_runtime_ema_for_frozen_config,
        maybe_init_distributed,
    )

    maybe_init_distributed()
    setup = OmniSetupOverrides(
        checkpoint_path=str(a.checkpoint_path),
        config_file=str(a.config_file),
        output_dir=a.output,
        cp_size=1,
        dp_shard_size=1,
        cfgp_size=1,
        use_torch_compile=False,
        use_cuda_graphs=False,
        guardrails=False,
        enable_cpu_offload=False,
    ).build_setup()
    load_start = time.monotonic()
    pipe = OmniInference.create(disable_runtime_ema_for_frozen_config(setup))
    model = pipe.model.eval()
    if not model.config.action_gen or model.config.num_embodiment_domains <= 33 or model.config.max_action_dim != 64:
        raise ValueError("Checkpoint does not declare the trained GISTdata robot's 34-domain/64-channel contract")
    durations = model.config.tokenizer.encode_exact_durations
    if durations is not None and a.horizon + 1 not in durations:
        raise ValueError("Requested horizon does not match the trained checkpoint's VAE duration")
    if str(model.config.resolution) != "256":
        raise ValueError("This input adapter expects the GISTdata recipe's resolution=256")
    metadata["model_load_seconds"] = time.monotonic() - load_start
    start = time.monotonic()
    with torch.inference_mode():
        result = model.generate_samples_from_batch(
            batch, guidance=a.guidance, seed=[a.seed], num_steps=a.num_steps, has_negative_prompt=False
        )
        commands = result["action"][0].squeeze(0).detach().float().cpu().numpy()
        if commands.shape == (a.horizon + 1, 18):
            commands = commands[1:]  # clean state prefix is NOT a robot command
        if commands.shape != (a.horizon, 18) or not np.isfinite(commands).all():
            raise ValueError(f"Invalid generated commands: {commands.shape}")
        if not a.actions_only:
            video = model.decode(result["vision"][0]).squeeze(0)
            video = remove_reflection_padding(video, batch["image_size"])
            frames = ((video.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    np.savez_compressed(a.output / "predicted_actions.npz", action=commands, timestamp=np.arange(a.horizon) / 20)
    (a.output / "predicted_actions.json").write_text(json.dumps(commands.tolist()) + "\n")
    if not a.actions_only:
        with av.open(str(a.output / "predicted_future.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=20)
            stream.height, stream.width, stream.pix_fmt = frames.shape[1], frames.shape[2], "yuv420p"
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    torch.cuda.synchronize()
    metadata.update(seconds=time.monotonic() - start, peak_cuda_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30)
    (a.output / "request.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps({"action_shape": list(commands.shape), "seconds": time.monotonic() - start, "output": str(a.output)})
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
