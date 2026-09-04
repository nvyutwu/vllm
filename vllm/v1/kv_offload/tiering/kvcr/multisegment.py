# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire contract for KVCR's missing pod-local mmap segment.

KVCR's framework-DRAM agent transfers the scheduler pod's row.  Kimi-K3's
second pod has another private row, so the target announces that row before a
source KVCR operation begins.  The source then binds its pinned source row to
the manifest by opaque operation tag and queues exactly one worker-owned NIXL
write.  Cache promotion remains pending until KVCR's row and this segment both
finish.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any

from vllm.v1.kv_offload.tiering.p2p.segment import SegmentRegistration

_WIRE_VERSION = 2


@dataclass(frozen=True, slots=True)
class KVCRSegmentManifest:
    """Target-owned rendezvous data for one rank-complete KVCR transfer."""

    operation_tag: str
    request_id: str
    reply_endpoint: str
    source_keys: tuple[bytes, ...]
    # ``TieringOffloadingManager`` batches all promotions for one request into
    # a single TransferJob.  This tuple stays positionally aligned with
    # ``source_keys`` so the source agent writes every rank-4 row page into
    # the matching target primary slot before the batched base delivery is
    # allowed to complete.
    target_block_ids: tuple[int, ...]
    target_segment: SegmentRegistration
    expires_at: float

    def validate(self) -> None:
        if not self.operation_tag or not self.request_id or not self.reply_endpoint:
            raise ValueError("missing KVCR segment manifest identity")
        if not self.source_keys or any(not key for key in self.source_keys):
            raise ValueError("missing physical KVCR keys")
        if len(set(self.source_keys)) != len(self.source_keys):
            raise ValueError("duplicate physical KVCR keys")
        if len(self.source_keys) != len(self.target_block_ids):
            raise ValueError("source keys and target primary blocks disagree")
        if any(
            isinstance(block_id, bool)
            or not isinstance(block_id, int)
            or block_id < 0
            for block_id in self.target_block_ids
        ):
            raise ValueError("invalid target primary blocks")
        if self.target_segment.rank_start == 0:
            raise ValueError("KVCR owns row zero; manifest must name extra row")
        self.target_segment.validate()
        if self.expires_at <= time.monotonic():
            raise ValueError("expired KVCR segment manifest")

    def to_wire(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": _WIRE_VERSION,
            "type": "prepare",
            "operation_tag": self.operation_tag,
            "request_id": self.request_id,
            "reply_endpoint": self.reply_endpoint,
            "source_keys": [
                base64.b64encode(key).decode("ascii") for key in self.source_keys
            ],
            "target_block_ids": list(self.target_block_ids),
            "target_segment": self.target_segment.to_wire(),
            "expires_at": self.expires_at,
        }


def encode_manifest(manifest: KVCRSegmentManifest) -> bytes:
    return json.dumps(
        manifest.to_wire(), sort_keys=True, separators=(",", ":")
    ).encode()


def decode_manifest(payload: bytes) -> KVCRSegmentManifest:
    try:
        raw = json.loads(payload.decode())
        if not isinstance(raw, dict) or raw.get("version") != _WIRE_VERSION:
            raise ValueError("unsupported KVCR segment manifest version")
        if raw.get("type") != "prepare":
            raise ValueError("not a KVCR segment prepare message")
        keys = raw["source_keys"]
        target_block_ids = raw["target_block_ids"]
        if not isinstance(keys, list) or not isinstance(target_block_ids, list):
            raise ValueError("source_keys and target_block_ids must be lists")
        manifest = KVCRSegmentManifest(
            operation_tag=raw["operation_tag"],
            request_id=raw["request_id"],
            reply_endpoint=raw["reply_endpoint"],
            source_keys=tuple(base64.b64decode(key, validate=True) for key in keys),
            target_block_ids=tuple(target_block_ids),
            target_segment=SegmentRegistration.from_wire(raw["target_segment"]),
            expires_at=raw["expires_at"],
        )
    except (KeyError, TypeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid KVCR segment manifest: {exc}") from exc
    if (
        isinstance(manifest.expires_at, bool)
        or not isinstance(manifest.expires_at, (int, float))
    ):
        raise ValueError("invalid KVCR segment manifest field types")
    manifest.validate()
    return manifest


def encode_terminal(operation_tag: str, success: bool) -> bytes:
    if not operation_tag:
        raise ValueError("missing KVCR segment operation tag")
    return json.dumps(
        {
            "version": _WIRE_VERSION,
            "type": "done",
            "operation_tag": operation_tag,
            "success": success,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def encode_ack(operation_tag: str, success: bool) -> bytes:
    if not operation_tag:
        raise ValueError("missing KVCR segment operation tag")
    return json.dumps(
        {
            "version": _WIRE_VERSION,
            "type": "ack",
            "operation_tag": operation_tag,
            "success": success,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def decode_terminal(payload: bytes) -> tuple[str, str, bool]:
    try:
        raw = json.loads(payload.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid KVCR segment terminal: {exc}") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("version") != _WIRE_VERSION
        or raw.get("type") not in {"ack", "done"}
        or not isinstance(raw.get("operation_tag"), str)
        or not raw["operation_tag"]
        or not isinstance(raw.get("success"), bool)
    ):
        raise ValueError("invalid KVCR segment terminal")
    return raw["type"], raw["operation_tag"], raw["success"]
