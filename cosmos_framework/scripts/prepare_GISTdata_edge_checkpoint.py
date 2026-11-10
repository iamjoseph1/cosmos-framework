# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Download the pinned Edge release and convert it to a fresh CPU DCP checkpoint."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from cosmos_framework.utils.GISTdata_edge_contract import GISTDATA_EDGE_REPOSITORY, GISTDATA_EDGE_REVISION


def prepare(output: Path) -> dict:
    output = output.resolve()
    receipt = output / "GISTdata_edge_base.json"
    if (output / "model/.metadata").is_file():
        if not receipt.is_file():
            raise ValueError("Existing DCP lacks matching GISTdata Edge provenance; choose a fresh output")
        report = json.loads(receipt.read_text())
        if (report["repository"], report["revision"]) != (GISTDATA_EDGE_REPOSITORY, GISTDATA_EDGE_REVISION):
            raise ValueError("Existing DCP belongs to a different model/revision; choose a fresh output")
        if not Path(report["snapshot"]).is_dir():
            from huggingface_hub import snapshot_download

            report["snapshot"] = snapshot_download(
                GISTDATA_EDGE_REPOSITORY, revision=GISTDATA_EDGE_REVISION, max_workers=4
            )
            receipt.write_text(json.dumps(report, indent=2) + "\n")
        return report
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a partial/nonempty checkpoint directory: {output}")
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(GISTDATA_EDGE_REPOSITORY, revision=GISTDATA_EDGE_REVISION, max_workers=4))
    if json.loads((snapshot / "config.json").read_text()).get("model_type") != "cosmos3_edge":
        raise ValueError("Downloaded checkpoint does not declare the Edge architecture")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", COSMOS_DEVICE="cpu")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cosmos_framework.scripts.convert_model_to_dcp",
            "--checkpoint-path",
            str(snapshot),
            "-o",
            str(output),
        ],
        env=env,
        check=True,
    )
    if not (output / "model/.metadata").is_file():
        raise RuntimeError("Converter finished without a complete model DCP")
    report = {
        "repository": GISTDATA_EDGE_REPOSITORY,
        "revision": GISTDATA_EDGE_REVISION,
        "snapshot": str(snapshot),
        "output": str(output),
        "training_executed": False,
    }
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
