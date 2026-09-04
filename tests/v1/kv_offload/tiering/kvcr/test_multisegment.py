# SPDX-License-Identifier: Apache-2.0
"""Wire-contract tests for the KVCR rank-complete extra mmap row."""

from __future__ import annotations

import time

import pytest

from vllm.v1.kv_offload.tiering.base import JobResult
from vllm.v1.kv_offload.tiering.kvcr.manager import _FrameworkPinAdapter
from vllm.v1.kv_offload.tiering.kvcr.multisegment import (
    KVCRSegmentManifest,
    decode_manifest,
    decode_terminal,
    encode_ack,
    encode_manifest,
    encode_terminal,
)
from vllm.v1.kv_offload.tiering.p2p.segment import SegmentRegistration, _layout_digest

pytestmark = pytest.mark.cpu_test


def _registration() -> SegmentRegistration:
    fingerprint = "kimi-k3-mla-mamba"
    return SegmentRegistration(
        segment_id=4,
        rank_start=4,
        rank_end=8,
        local_slots=4,
        base_addr=0xDEADBEEF,
        num_blocks=1024,
        block_len=4096,
        config_fingerprint=fingerprint,
        layout_digest=_layout_digest(
            rank_start=4,
            rank_end=8,
            local_slots=4,
            num_blocks=1024,
            block_len=4096,
            config_fingerprint=fingerprint,
        ),
        agent_metadata=b"target-nixl-agent",
    )


def test_manifest_round_trip_preserves_rank_and_target_block() -> None:
    manifest = KVCRSegmentManifest(
        operation_tag="generation-9",
        request_id="request-4",
        reply_endpoint="tcp://target:17772",
        source_keys=(b"physical-group-0", b"physical-group-mamba"),
        target_block_id=77,
        target_segment=_registration(),
        expires_at=time.monotonic() + 30,
    )

    assert decode_manifest(encode_manifest(manifest)) == manifest


@pytest.mark.parametrize(
    "payload",
    [encode_ack("generation-9", True), encode_terminal("generation-9", False)],
)
def test_terminal_wire_distinguishes_ack_from_completion(payload: bytes) -> None:
    message_type, operation_tag, success = decode_terminal(payload)

    assert message_type in {"ack", "done"}
    assert operation_tag == "generation-9"
    assert success is (message_type == "ack")


def test_manifest_rejects_expired_target_reservation() -> None:
    manifest = KVCRSegmentManifest(
        operation_tag="expired",
        request_id="request-4",
        reply_endpoint="tcp://target:17772",
        source_keys=(b"physical-group-0",),
        target_block_id=77,
        target_segment=_registration(),
        expires_at=time.monotonic() - 1,
    )

    with pytest.raises(ValueError, match="expired"):
        encode_manifest(manifest)


def test_source_pin_result_waits_for_extra_segment_and_base_release() -> None:
    """The source primary block cannot be recycled before both writes finish."""
    adapter = _FrameworkPinAdapter(None)  # type: ignore[arg-type]
    adapter._pin_jobs["pin-7"] = 7

    assert adapter.defer_pins("operation-7", ("pin-7",))
    adapter.complete_deferred_operation("operation-7", True)
    assert adapter.take_pin_job_results() == []

    assert adapter.release_pin("pin-7")
    assert adapter.take_pin_job_results() == [JobResult(job_id=7, success=True)]
    assert not adapter._segment_outcomes
