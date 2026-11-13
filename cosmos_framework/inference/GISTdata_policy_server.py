# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""ZMQ robot_interface protocol, automatic dish instruction, and action-chunk queue."""

import json
import re
import time
from collections import deque
from pathlib import Path

import numpy as np
from robot_interface import wire
from robot_interface.schema.GISTdata.schema import GISTdataSchema

from cosmos_framework.utils.GISTdata_action_contract import INSTRUCTIONS


class GISTdataPolicyServer:
    def __init__(self, policy_factory, *, execute_steps=32, seed=42, events_path: Path | None = None):
        if not 1 <= execute_steps <= 32:
            raise ValueError("execute_steps must be between 1 and 32")
        self.policy_factory, self.execute_steps, self.seed = policy_factory, execute_steps, seed
        self.events_path = events_path
        self.policy = self.schema = None
        self.num_cameras = None
        self.pending = deque()
        self.instruction = None
        self.generations = self.steps = 0
        self.ready = False

    def log(self, event, **fields):
        record = {"event": event, "time": time.time(), **fields}
        print("[GISTdata] " + json.dumps(record, ensure_ascii=False), flush=True)
        if self.events_path is not None:
            with self.events_path.open("a") as out:
                out.write(json.dumps(record) + "\n")

    def reset(self, meta):
        self.ready = False
        self.pending.clear()
        dish = meta.get("dish")
        if dish is None:  # Backward compatibility with the original head-only launcher.
            names = set(re.findall(r"\b(bowl|mug)\b", meta.get("task", "").lower()))
            if len(names) != 1:
                raise ValueError("RESET must identify dish=bowl or mug")
            dish = names.pop()
        if dish not in ("bowl", "mug"):
            raise ValueError("dish must be bowl or mug")
        count = int(meta.get("num_cameras", "1"))
        schema = GISTdataSchema(count, dish)
        if meta.get("fps", "20") != "20":
            raise ValueError("GISTdata policy requires the 20-Hz joint18 client")
        if self.policy is None or self.num_cameras != count:
            if self.policy is not None:
                self.policy.close()
                self.policy = None
            self.num_cameras = None
            self.log("loading", num_cameras=count)
            self.policy = self.policy_factory(count)
            self.num_cameras = count
        self.schema = schema
        self.instruction = INSTRUCTIONS[f"ycb_{dish}"]
        self.generations = self.steps = 0
        self.ready = True
        checkpoint = str(getattr(self.policy, "checkpoint", "unknown"))
        self.log("reset", dish=dish, num_cameras=count, instruction=self.instruction, checkpoint=checkpoint)
        return wire.encode(
            wire.MessageType.RESET,
            meta={
                "dish": dish,
                "num_cameras": str(count),
                "instruction": self.instruction,
                "execute_steps": str(self.execute_steps),
                "checkpoint": checkpoint,
            },
        )

    def handle(self, buf):
        kind, arrays, meta = wire.decode(buf)
        if kind is wire.MessageType.RESET:
            return self.reset(meta)
        if kind is not wire.MessageType.OBS or not self.ready:
            raise ValueError("A successful RESET is required before OBS requests")
        self.schema.validate_obs(arrays)
        if not self.pending:
            started = time.monotonic()
            chunk = self.policy.predict(arrays, self.instruction, self.seed + self.generations)
            if chunk.shape != (32, 18) or chunk.dtype != np.float32 or not np.isfinite(chunk).all():
                raise ValueError("Policy must return a finite float32 [32,18] action chunk")
            self.pending.extend(row.copy() for row in chunk[: self.execute_steps])
            self.generations += 1
            self.log(
                "chunk",
                generation=self.generations,
                step=self.steps,
                num_cameras=self.num_cameras,
                seconds=time.monotonic() - started,
                queued_actions=len(self.pending),
                instruction=self.instruction,
            )
        action = self.pending.popleft()
        self.steps += 1
        return self.schema.encode_action(action)

    def serve(self, host, port):
        import zmq

        context = zmq.Context()
        sock = context.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.bind(f"tcp://{host}:{port}")
            self.log("listening", address=f"tcp://{host}:{port}", model_selection="automatic on RESET")
            while True:
                buf = sock.recv()
                try:
                    reply = self.handle(buf)
                except Exception as error:
                    import traceback

                    traceback.print_exc()
                    self.pending.clear()
                    self.ready = False
                    try:
                        kind, _, _ = wire.decode(buf)
                    except Exception:
                        kind = wire.MessageType.OBS
                    reply = wire.encode(
                        wire.MessageType.RESET if kind is wire.MessageType.RESET else wire.MessageType.ACTION,
                        meta={"error": str(error)},
                    )
                sock.send(reply)
        finally:
            sock.close()
            context.term()
            self.close()

    def close(self):
        self.pending.clear()
        if self.policy is not None:
            self.policy.close()
            self.policy = None
