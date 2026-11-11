# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Serve the two trained Edge policies to rb_xhand.py over its existing ZMQ protocol."""

import argparse
import os
import signal
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/checkpoints/GISTdata"))
    parser.add_argument("--output", type=Path, help="New server log directory")
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--execute-steps", type=int, default=32, help="Use this many actions before replanning (1..32)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_steps < 1 or not 1 <= args.execute_steps <= 32:
        parser.error("num-steps must be positive; execute-steps must be 1..32")
    os.environ.setdefault("COSMOS_INTERNAL", "0")
    os.environ.setdefault("COSMOS_TRAINING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from cosmos_framework.inference.GISTdata_policy import GISTdataPolicy
    from cosmos_framework.inference.GISTdata_policy_server import GISTdataPolicyServer

    output = args.output or Path("outputs/GISTdata_policy_server") / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    )
    output.mkdir(parents=True, exist_ok=False)
    loads = 0

    def factory(count):
        nonlocal loads
        loads += 1
        return GISTdataPolicy(
            args.checkpoint_root,
            count,
            output / f"load_{loads:03d}_cameras{count}",
            num_steps=args.num_steps,
            guidance=args.guidance,
        )

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    server = GISTdataPolicyServer(
        factory, execute_steps=args.execute_steps, seed=args.seed, events_path=output / "events.jsonl"
    )
    try:
        server.serve(args.host, args.port)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        import torch

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
