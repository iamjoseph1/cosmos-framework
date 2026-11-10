# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Package original GISTdata commands with aligned head-camera observations.

Run with Python + numpy + ffprobe; importing this module does not require CUDA.
All episodes, including failed demonstrations, are retained and explicitly tagged.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from cosmos_framework.utils.GISTdata_action_contract import (
    ACTION_DIM,
    DOMAIN_NAME,
    FPS,
    INSTRUCTIONS,
    action_sha256,
    file_sha256,
    window_starts,
)


def prepare(source: Path, output: Path, expected_episodes=200, horizon=32, stride=8, val_fraction=0.1, seed=42):
    source, output = source.resolve(), output.resolve()
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    # Require the authoritative 100+100 inputs, rather than accepting partial render directories.
    inputs = [(obj, p) for obj in INSTRUCTIONS for p in sorted((source / "episodes" / obj).glob("data_*.npz"))]
    if len(inputs) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} input episodes, found {len(inputs)}")
    renders = source / "results" / "red_objects_defaultscene_3cam"
    episodes, joint_names = [], None
    for obj, path in inputs:
        ep = renders / obj / path.stem
        meta = json.loads((ep / "metadata.json").read_text())
        # Preserve the input recording format's original report filename.
        report = json.loads((ep / "replay_report.json").read_text())
        if not report.get("runtime_completed"):
            raise ValueError(f"Incomplete rendering: {ep}")
        names = meta["state_names"]
        if joint_names is None:
            joint_names = names
        if names != joint_names or len(names) != ACTION_DIM:
            raise ValueError(f"Inconsistent canonical joint order: {ep}")
        if meta["input_semantics"] != "absolute joint-position targets; T commands and T+1 observations":
            raise ValueError(f"Unexpected command semantics: {ep}")
        with np.load(path, allow_pickle=False) as z:
            action = z["action"].copy()
            if action.dtype != np.float32:
                raise ValueError(f"Expected original float32 commands; refusing a silent conversion: {path}")
            pose = z["object_pose_world"].astype(np.float64)
            goal = z["place_xy"].astype(np.float64)
        with np.load(ep / "observations.npz", allow_pickle=False) as z:
            state, timestamp = z["state"].astype(np.float32), z["timestamp"].astype(np.float64)
        if action.ndim != 2 or action.shape[1] != ACTION_DIM or state.shape != (len(action) + 1, ACTION_DIM):
            raise ValueError(f"Expected T x 18 commands and (T+1) x 18 observations: {ep}")
        if not all(np.isfinite(a).all() for a in (action, state, timestamp)):
            raise ValueError(f"Nonfinite observations/actions: {ep}")
        if timestamp.shape != (len(state),) or not np.allclose(timestamp, np.arange(len(state)) / FPS, atol=1e-7):
            raise ValueError(f"Expected observation-before-action timestamps at {FPS} Hz: {ep}")
        if meta["control_hz"] != FPS or report["executed_actions"] != len(action):
            raise ValueError(f"Command count or control rate mismatch: {ep}")
        video = ep / "videos/head.mp4"
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_frames,avg_frame_rate",
                    "-of",
                    "json",
                    str(video),
                ],
                text=True,
            )
        )["streams"][0]
        num, den = map(float, probe["avg_frame_rate"].split("/"))
        if int(probe["nb_frames"]) != len(state) or abs(num / den - FPS) > 1e-7:
            raise ValueError(f"RGB/action clock mismatch: {ep}")
        if (probe["width"], probe["height"]) != tuple(meta["cameras"]["head"]["resolution"]):
            raise ValueError(f"Head-camera dimensions differ from metadata: {ep}")
        window_starts(len(action), horizon, stride)
        # q and -q represent the same orientation. Group repeated initial conditions.
        if pose[3] < 0:
            pose[3:] *= -1
        group = obj + ":" + ",".join(f"{x:.4f}" for x in np.concatenate([pose, goal]))
        episodes.append(
            dict(
                episode_id=f"{obj}/{path.stem}",
                object=obj,
                group=group,
                instruction=INSTRUCTIONS[obj],
                task_success=bool(report["task_success"]),
                transitions=len(action),
                source_action=str(path),
                source_head=str(video),
                source_action_file_sha256=file_sha256(path),
                original_action_sha256=action_sha256(action),
                source_head_sha256=file_sha256(video),
                original_action_dtype=str(action.dtype),
                state=state,
                action=action,
                timestamp=timestamp,
                goal_xy=goal.tolist(),
            )
        )
    # Deterministic, object-stratified split of INITIAL-CONDITION GROUPS.
    validation = set()
    for obj in INSTRUCTIONS:
        groups = sorted(
            {e["group"] for e in episodes if e["object"] == obj},
            key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(),
        )
        if len(groups) < 2:
            raise ValueError(f"Need two distinct initial-condition groups for {obj}")
        validation.update(groups[: max(1, min(len(groups) - 1, round(len(groups) * val_fraction)))])
    # Reserve a new directory only after the entire input passes validation.
    output.mkdir(parents=True, exist_ok=False)
    rows, windows = [], {"train": [], "val": [], "all": []}
    for entry in episodes:
        dest = output / "episodes" / entry["episode_id"]
        dest.mkdir(parents=True)
        np.savez_compressed(
            dest / "trajectory.npz",
            state=entry.pop("state"),
            action=entry.pop("action"),
            timestamp=entry.pop("timestamp"),
        )
        shutil.copy2(entry["source_head"], dest / "head.mp4")
        with np.load(dest / "trajectory.npz", allow_pickle=False) as stored:
            if action_sha256(stored["action"]) != entry["original_action_sha256"]:
                raise ValueError(f"Packaged commands differ from original GISTdata: {dest}")
        if file_sha256(dest / "head.mp4") != entry["source_head_sha256"]:
            raise ValueError(f"Copied head video differs from the source: {dest}")
        row = {
            **entry,
            "split": "val" if entry["group"] in validation else "train",
            "trajectory": str((dest / "trajectory.npz").relative_to(output)),
            "head_video": str((dest / "head.mp4").relative_to(output)),
        }
        rows.append(row)
        for start in window_starts(row["transitions"], horizon, stride):
            w = {"episode_id": row["episode_id"], "start": start}
            windows[row["split"]].append(w)
            windows["all"].append(w)
    manifest = dict(
        schema_version=1,
        dataset_name="GISTdata",
        provenance={
            "provider": "GIST",
            "actions": "Original absolute joint-position commands; identical float32 array bytes",
            "head_rgb": "Isaac Sim rendering of original trajectories: red_objects_defaultscene_3cam/head",
            "proprioception": "Measured joint positions before each command in the same Isaac Sim recording",
            "augmentation": "none; no AI-generated, recolored, or edited video",
            "training_executed": False,
        },
        domain_name=DOMAIN_NAME,
        raw_action_dim=ACTION_DIM,
        action_semantics="absolute_joint_position",
        units="rad",
        joint_names=joint_names,
        fps=FPS,
        horizon=horizon,
        stride=stride,
        use_state=True,
        normalization="none",
        camera="head",
        resolution="256",
        seed=seed,
        validation_fraction=val_fraction,
        instructions=INSTRUCTIONS,
        episodes=rows,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "instructions.json").write_text(json.dumps(INSTRUCTIONS, indent=2) + "\n")
    for split, entries in windows.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(w) + "\n" for w in entries))
    summary = dict(
        episodes=len(rows),
        objects=dict(Counter(r["object"] for r in rows)),
        splits=dict(Counter(r["split"] for r in rows)),
        windows={k: len(v) for k, v in windows.items()},
        unsuccessful_episodes=[r["episode_id"] for r in rows if not r["task_success"]],
        group_overlap=sorted({r["group"] for r in rows if r["split"] == "train"} & validation),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, required=True, help="GIST data root with original episodes/ and recorded results/"
    )
    p.add_argument(
        "--output", type=Path, required=True, help="New portable output directory; existing paths are refused"
    )
    p.add_argument("--expected-episodes", type=int, default=200)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    print(
        json.dumps(
            prepare(a.source, a.output, a.expected_episodes, a.horizon, a.stride, a.val_fraction, a.seed), indent=2
        )
    )


if __name__ == "__main__":
    main()
