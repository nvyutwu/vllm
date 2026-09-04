# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol tests for rank-complete worker-owned KVCR mmap segments."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm.v1.kv_offload.tiering.p2p.segment import (
    SegmentedNixlTransport,
    SegmentOperationResult,
    SegmentRegistration,
    _layout_digest,
    decode_registrations,
    encode_registrations,
)

pytestmark = pytest.mark.cpu_test


def _view() -> memoryview:
    return memoryview(bytearray(2 * 16)).cast("B", shape=(2, 16))


def _registration(
    transport: SegmentedNixlTransport, segment_id: int, base_addr: int
) -> SegmentRegistration:
    rank_end = segment_id + 4
    return SegmentRegistration(
        segment_id=segment_id,
        rank_start=segment_id,
        rank_end=rank_end,
        local_slots=4,
        base_addr=base_addr,
        num_blocks=transport.num_blocks,
        block_len=transport.block_len,
        config_fingerprint=transport.config_fingerprint,
        layout_digest=_layout_digest(
            rank_start=segment_id,
            rank_end=rank_end,
            local_slots=4,
            num_blocks=transport.num_blocks,
            block_len=transport.block_len,
            config_fingerprint=transport.config_fingerprint,
        ),
        agent_metadata=f"nixl-{segment_id}-{base_addr}".encode(),
    )


def _ready_transport() -> tuple[SegmentedNixlTransport, list[SegmentRegistration]]:
    transport = SegmentedNixlTransport(
        _view(),
        config_fields={"model": "kimi-k3", "layout": "mla+mamba"},
        world_size=8,
        local_world_size=4,
    )
    registrations = [
        _registration(transport, 0, 0x1000),
        _registration(transport, 4, 0x2000),
    ]
    transport.update_worker_metadata(registrations, ())
    assert transport.ready
    return transport, registrations


def test_registration_wire_round_trip_preserves_rank_geometry():
    transport, registrations = _ready_transport()

    decoded = decode_registrations(encode_registrations(reversed(registrations)))

    assert list(decoded) == [0, 4]
    assert decoded[4] == registrations[1]
    assert transport.get_agent_metadata() == encode_registrations(registrations)


def test_logical_transfer_commits_only_after_every_segment_ack():
    transport, registrations = _ready_transport()
    remote = [
        _registration(transport, 0, 0x3000),
        _registration(transport, 4, 0x4000),
    ]
    transport.add_remote_peer("peer:5710", encode_registrations(remote), 0, 2, 16)

    transfer_id = transport.write_blocks("peer:5710", [0, 1], [1, 0])
    assert transfer_id is not None
    commands = transport.take_worker_commands()
    assert [command.segment_id for command in commands] == [0, 4]
    assert all(command.local_indices == (0, 1) for command in commands)
    assert all(command.remote_indices == (1, 0) for command in commands)

    transport.update_worker_metadata(
        (),
        (SegmentOperationResult(commands[0].operation_id, 0, True),),
    )
    assert transport.poll() == ((), ())

    transport.update_worker_metadata(
        (),
        (SegmentOperationResult(commands[1].operation_id, 4, True),),
    )
    assert transport.poll().done == (transfer_id,)


def test_failed_or_late_segment_ack_never_commits_a_logical_transfer():
    transport, _ = _ready_transport()
    remote = [_registration(transport, 0, 0x3000), _registration(transport, 4, 0x4000)]
    transport.add_remote_peer("peer:5710", encode_registrations(remote), 0, 2, 16)
    transfer_id = transport.write_blocks("peer:5710", [0], [1])
    assert transfer_id is not None
    commands = transport.take_worker_commands()

    transport.update_worker_metadata(
        (),
        (
            SegmentOperationResult(commands[0].operation_id, 0, True),
            SegmentOperationResult(commands[1].operation_id, 4, False),
        ),
    )
    assert transport.poll().failed == (transfer_id,)

    # A duplicate / late completion from an old generation has no owner and
    # cannot affect a subsequent primary-tier reservation.
    transport.update_worker_metadata(
        (), (SegmentOperationResult(commands[1].operation_id, 4, True),)
    )
    assert transport.poll() == ((), ())


def test_rejects_rank_or_layout_mismatch_before_commands_are_issued():
    transport, _ = _ready_transport()
    incompatible = _registration(transport, 0, 0x3000)
    incompatible = replace(incompatible, rank_end=3, local_slots=3)
    remote = [incompatible, _registration(transport, 4, 0x4000)]

    with pytest.raises(ValueError, match="invalid|missing|incompatible"):
        transport.add_remote_peer("peer:5710", encode_registrations(remote), 0, 2, 16)
