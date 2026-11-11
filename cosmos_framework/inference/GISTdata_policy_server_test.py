# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Protocol contracts, automatic instructions, chunk reuse/reset and camera switching."""

import numpy as np
import pytest
from robot_interface import wire
from robot_interface.schema.GISTdata.schema import GISTdataSchema

from cosmos_framework.inference.GISTdata_policy_server import GISTdataPolicyServer
from cosmos_framework.utils.GISTdata_action_contract import INSTRUCTIONS

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class FakePolicy:
    def __init__(self, count):
        self.count, self.calls, self.closed = count, [], False

    def predict(self, arrays, instruction, seed):
        self.calls.append((instruction, seed))
        return np.arange(32 * 18, dtype=np.float32).reshape(32, 18)

    def close(self):
        self.closed = True


def observation(schema):
    return schema.pack_observation(
        np.arange(18, dtype=np.float32),
        {name: np.full((240, 320, 3), i + 1, dtype=np.uint8) for i, name in enumerate(schema.CAMERA_NAMES)},
    )


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("dish", ["bowl", "mug"])
def test_wire_camera_pixels_dish_instruction_and_chunk_contract(count, dish):
    schema = GISTdataSchema(count, dish)
    server = GISTdataPolicyServer(FakePolicy, seed=42)
    kind, _, meta = wire.decode(server.handle(schema.encode_reset("Ignore this custom client text")))
    assert kind == wire.MessageType.RESET
    assert meta["instruction"] == INSTRUCTIONS[f"ycb_{dish}"]
    encoded = schema.encode_obs(observation(schema))
    for step in range(33):
        kind, arrays, _ = wire.decode(server.handle(encoded))
        assert kind == wire.MessageType.ACTION and arrays["action"].shape == (18,)
        np.testing.assert_array_equal(arrays["action"], np.arange(18, dtype=np.float32) + (step % 32) * 18)
    assert server.policy.calls == [(INSTRUCTIONS[f"ycb_{dish}"], 42), (INSTRUCTIONS[f"ycb_{dish}"], 43)]
    _, decoded, _ = wire.decode(encoded)
    for i, name in enumerate(schema.CAMERA_NAMES):
        assert np.all(decoded[f"observation.images.{name}"] == i + 1)


def test_reset_clears_pending_actions_switches_model_and_updates_instruction():
    server = GISTdataPolicyServer(FakePolicy)
    head = GISTdataSchema(1, "bowl")
    server.handle(head.encode_reset())
    policy = server.policy
    server.handle(head.encode_obs(observation(head)))
    assert len(server.pending) == 31
    mug = GISTdataSchema(1, "mug")
    server.handle(mug.encode_reset())
    assert server.policy is policy and not server.pending
    server.handle(mug.encode_obs(observation(mug)))
    assert policy.calls[-1] == (INSTRUCTIONS["ycb_mug"], 42)
    three = GISTdataSchema(3, "mug")
    server.handle(three.encode_reset())
    assert policy.closed and server.policy.count == 3 and not server.pending


def test_partial_chunk_replans_and_requires_explicit_matching_observations():
    server = GISTdataPolicyServer(FakePolicy, execute_steps=2)
    schema = GISTdataSchema(3, "bowl")
    with pytest.raises(ValueError, match="RESET"):
        server.handle(schema.encode_obs(observation(schema)))
    server.handle(schema.encode_reset())
    bad = observation(schema)
    del bad["observation.images.left_wrist"]
    with pytest.raises(ValueError, match="Expected exactly"):
        server.handle(wire.encode(wire.MessageType.OBS, bad))
    for _ in range(3):
        server.handle(schema.encode_obs(observation(schema)))
    assert len(server.policy.calls) == 2
    with pytest.raises(ValueError, match="dish"):
        server.handle(wire.encode(wire.MessageType.RESET, meta={"dish": "plate"}))
    assert not server.ready and not server.pending


def test_legacy_head_client_reset_is_supported():
    from robot_interface.schema.rb_xhand_replay import schema

    server = GISTdataPolicyServer(FakePolicy)
    server.handle(schema.encode_reset(schema.TASKS["mug"]))
    assert server.instruction == INSTRUCTIONS["ycb_mug"] and server.num_cameras == 1
