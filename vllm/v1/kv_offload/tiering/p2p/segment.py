# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-complete, worker-owned P2P segments for pod-local KVCR mmap files.

``/dev/shm/vllm_offload_<engine>.mmap`` is local to one pod.  A scheduler
process can consequently see only one row of a multi-node TP/DCP engine.  This
module keeps the scheduler-side P2P protocol, but moves registration and RDMA
submission for every pod-local row to a worker which shares that row.

The scheduler-facing transport deliberately reports one transfer complete only
after *every* required segment agent has reported completion.  Therefore the
normal primary-tier ``complete_write(..., success=True)`` contract remains the
single point that makes a transferred key reusable.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.p2p.data.base import (
    CancelMode,
    DataTransport,
    PollResult,
)
from vllm.v1.kv_offload.tiering.p2p.data.nixl import NixlTransport

logger = init_logger(__name__)

_WIRE_VERSION = 1
_EMPTY_POLL_RESULT = PollResult(done=(), failed=())


@dataclass(frozen=True, slots=True)
class SegmentRegistration:
    """One worker-leader's registration for a complete pod-local mmap row.

    ``rank_start``/``rank_end`` describe the logical DCP ranks represented by
    the row.  The bytes themselves are not reorganized: a source row is copied
    verbatim into the matching target row, preserving both MLA and Mamba
    rank-owned bytes.
    """

    segment_id: int
    rank_start: int
    rank_end: int
    local_slots: int
    base_addr: int
    num_blocks: int
    block_len: int
    config_fingerprint: str
    layout_digest: str
    agent_metadata: bytes

    def validate(self) -> None:
        if self.segment_id != self.rank_start:
            raise ValueError("segment_id must equal rank_start")
        if self.rank_start < 0 or self.rank_end <= self.rank_start:
            raise ValueError("invalid global-rank range")
        if self.local_slots != self.rank_end - self.rank_start:
            raise ValueError("rank range and local_slots disagree")
        if self.base_addr <= 0 or self.num_blocks <= 0 or self.block_len <= 0:
            raise ValueError("invalid mmap registration geometry")
        if not self.agent_metadata:
            raise ValueError("missing NIXL agent metadata")

    def compatible_with(self, remote: SegmentRegistration) -> bool:
        """Return whether this row can copy directly to *remote*.

        The base address and agent metadata are intentionally excluded: they
        are process-local rendezvous details.  Everything governing row/rank
        interpretation must match before a transfer is submitted.
        """

        return (
            self.segment_id == remote.segment_id
            and self.rank_start == remote.rank_start
            and self.rank_end == remote.rank_end
            and self.local_slots == remote.local_slots
            and self.num_blocks == remote.num_blocks
            and self.block_len == remote.block_len
            and self.config_fingerprint == remote.config_fingerprint
            and self.layout_digest == remote.layout_digest
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "rank_start": self.rank_start,
            "rank_end": self.rank_end,
            "local_slots": self.local_slots,
            "base_addr": self.base_addr,
            "num_blocks": self.num_blocks,
            "block_len": self.block_len,
            "config_fingerprint": self.config_fingerprint,
            "layout_digest": self.layout_digest,
            "agent_metadata": base64.b64encode(self.agent_metadata).decode("ascii"),
        }

    @classmethod
    def from_wire(cls, raw: object) -> SegmentRegistration:
        if not isinstance(raw, dict):
            raise ValueError("segment registration must be an object")
        try:
            registration = cls(
                segment_id=raw["segment_id"],
                rank_start=raw["rank_start"],
                rank_end=raw["rank_end"],
                local_slots=raw["local_slots"],
                base_addr=raw["base_addr"],
                num_blocks=raw["num_blocks"],
                block_len=raw["block_len"],
                config_fingerprint=raw["config_fingerprint"],
                layout_digest=raw["layout_digest"],
                agent_metadata=base64.b64decode(raw["agent_metadata"], validate=True),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid segment registration: {exc}") from exc
        if not all(
            isinstance(value, int)
            for value in (
                registration.segment_id,
                registration.rank_start,
                registration.rank_end,
                registration.local_slots,
                registration.base_addr,
                registration.num_blocks,
                registration.block_len,
            )
        ) or not all(
            isinstance(value, str)
            for value in (registration.config_fingerprint, registration.layout_digest)
        ):
            raise ValueError("segment registration has invalid field types")
        registration.validate()
        return registration


def _layout_digest(
    *,
    rank_start: int,
    rank_end: int,
    local_slots: int,
    num_blocks: int,
    block_len: int,
    config_fingerprint: str,
) -> str:
    fields = {
        "rank_start": rank_start,
        "rank_end": rank_end,
        "local_slots": local_slots,
        "num_blocks": num_blocks,
        "block_len": block_len,
        "config_fingerprint": config_fingerprint,
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def encode_registrations(registrations: Iterable[SegmentRegistration]) -> bytes:
    """Encode the worker registrations for the existing P2P ConnectMsg."""

    ordered = sorted(registrations, key=lambda registration: registration.segment_id)
    return json.dumps(
        {"version": _WIRE_VERSION, "segments": [item.to_wire() for item in ordered]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def decode_registrations(payload: bytes) -> dict[int, SegmentRegistration]:
    try:
        decoded = json.loads(payload.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid rank-complete segment metadata: {exc}") from exc
    if not isinstance(decoded, dict) or decoded.get("version") != _WIRE_VERSION:
        raise ValueError("unsupported rank-complete segment metadata version")
    raw_segments = decoded.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("rank-complete segment metadata has no segments")
    segments: dict[int, SegmentRegistration] = {}
    for raw in raw_segments:
        registration = SegmentRegistration.from_wire(raw)
        if registration.segment_id in segments:
            raise ValueError(f"duplicate segment {registration.segment_id}")
        segments[registration.segment_id] = registration
    return segments


@dataclass(frozen=True, slots=True)
class SegmentTransferCommand:
    """Scheduler → worker-leader command for one mmap-row RDMA write."""

    operation_id: int
    segment_id: int
    peer_id: str
    remote: SegmentRegistration
    local_indices: tuple[int, ...]
    remote_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SegmentOperationResult:
    """Worker-leader → scheduler acknowledgement for one segment command."""

    operation_id: int
    segment_id: int
    success: bool


class SegmentAgent:
    """A NIXL owner colocated with one pod-local mmap row.

    The agent is instantiated only by the local-rank leader.  All other GPU
    ranks in that pod share the same mmap and must not create duplicate NIXL
    registrations for it.
    """

    def __init__(
        self,
        *,
        engine_id: str,
        rank_start: int,
        local_slots: int,
        view: memoryview,
        config_fields: dict[str, Any],
        backends: list[str] | None,
        num_threads: int,
    ) -> None:
        if view.ndim != 2:
            raise ValueError("segment mmap view must be two-dimensional")
        self._transport = NixlTransport(
            agent_name=(
                f"{engine_id}-kvcr-segment-{rank_start}-{uuid.uuid4().hex[:12]}"
            ),
            view=view,
            config_fields=config_fields,
            backends=backends,
            num_threads=num_threads,
        )
        if not self._transport.available:
            raise RuntimeError("NIXL is unavailable for KVCR segment registration")
        rank_end = rank_start + local_slots
        self.registration = SegmentRegistration(
            segment_id=rank_start,
            rank_start=rank_start,
            rank_end=rank_end,
            local_slots=local_slots,
            base_addr=self._transport.base_addr,
            num_blocks=self._transport.num_blocks,
            block_len=self._transport.block_len,
            config_fingerprint=self._transport.config_fingerprint,
            layout_digest=_layout_digest(
                rank_start=rank_start,
                rank_end=rank_end,
                local_slots=local_slots,
                num_blocks=self._transport.num_blocks,
                block_len=self._transport.block_len,
                config_fingerprint=self._transport.config_fingerprint,
            ),
            agent_metadata=self._transport.get_agent_metadata(),
        )
        self.registration.validate()
        self._remote_peers: dict[str, SegmentRegistration] = {}
        self._operations: dict[int, SegmentTransferCommand] = {}

    def submit(self, command: SegmentTransferCommand) -> SegmentOperationResult | None:
        """Submit one non-blocking row transfer, or return an immediate failure."""

        if command.segment_id != self.registration.segment_id:
            return None
        if not self.registration.compatible_with(command.remote):
            logger.error(
                "KVCR segment %d rejected incompatible target layout %s",
                self.registration.segment_id,
                command.remote.layout_digest,
            )
            return SegmentOperationResult(
                command.operation_id, command.segment_id, False
            )
        if len(command.local_indices) != len(command.remote_indices):
            return SegmentOperationResult(
                command.operation_id, command.segment_id, False
            )
        peer_key = f"{command.peer_id}/segment-{command.segment_id}"
        try:
            existing = self._remote_peers.get(peer_key)
            if existing != command.remote:
                if existing is not None:
                    self._transport.remove_remote_peer(peer_key)
                self._transport.add_remote_peer(
                    peer_key,
                    command.remote.agent_metadata,
                    command.remote.base_addr,
                    command.remote.num_blocks,
                    command.remote.block_len,
                )
                self._remote_peers[peer_key] = command.remote
            transfer_id = self._transport.write_blocks(
                peer_key,
                list(command.local_indices),
                list(command.remote_indices),
            )
        except Exception:
            logger.exception(
                "KVCR segment %d failed to submit operation %d",
                self.registration.segment_id,
                command.operation_id,
            )
            return SegmentOperationResult(
                command.operation_id, command.segment_id, False
            )
        if transfer_id is None:
            return SegmentOperationResult(
                command.operation_id, command.segment_id, False
            )
        self._operations[transfer_id] = command
        return None

    def poll(self) -> list[SegmentOperationResult]:
        results: list[SegmentOperationResult] = []
        try:
            poll_result = self._transport.poll()
        except Exception:
            logger.exception(
                "KVCR segment %d poll failed", self.registration.segment_id
            )
            for command in self._operations.values():
                results.append(
                    SegmentOperationResult(
                        command.operation_id, command.segment_id, False
                    )
                )
            self._operations.clear()
            return results
        for transfer_id in poll_result.done:
            command = self._operations.pop(transfer_id, None)
            if command is not None:
                results.append(
                    SegmentOperationResult(
                        command.operation_id, command.segment_id, True
                    )
                )
        for transfer_id in poll_result.failed:
            command = self._operations.pop(transfer_id, None)
            if command is not None:
                results.append(
                    SegmentOperationResult(
                        command.operation_id, command.segment_id, False
                    )
                )
        return results

    def close(self) -> None:
        self._operations.clear()
        self._remote_peers.clear()
        self._transport.close()


@dataclass(slots=True)
class _LogicalTransfer:
    peer_id: str
    required_segments: set[int]
    pending_segments: set[int]
    failed: bool = False


class SegmentedNixlTransport(DataTransport):
    """Scheduler facade whose one transfer covers all pod-local mmap rows.

    Worker metadata supplies segment registrations and operation results.
    ``write_blocks`` simply creates one worker command per local segment; it
    never touches a pod-local mmap from the scheduler process.
    """

    def __init__(
        self,
        view: memoryview,
        *,
        config_fields: dict[str, Any],
        world_size: int,
        local_world_size: int,
    ) -> None:
        super().__init__(view, config_fields=config_fields)
        if world_size <= local_world_size or world_size % local_world_size:
            raise ValueError(
                "rank-complete transport requires integral multi-node rows"
            )
        self._expected_segments = set(range(0, world_size, local_world_size))
        self._local_world_size = local_world_size
        self._local_segments: dict[int, SegmentRegistration] = {}
        self._remote_segments: dict[str, dict[int, SegmentRegistration]] = {}
        self._commands: list[SegmentTransferCommand] = []
        self._transfers: dict[int, _LogicalTransfer] = {}
        self._finished_done: list[int] = []
        self._finished_failed: list[int] = []
        self._next_transfer_id = itertools.count()
        self._next_operation_id = itertools.count()
        self._operation_to_transfer: dict[int, int] = {}

    @property
    def ready(self) -> bool:
        return set(self._local_segments) == self._expected_segments

    def _validate_local_registration(self, registration: SegmentRegistration) -> None:
        registration.validate()
        if registration.segment_id not in self._expected_segments:
            raise ValueError(f"unexpected local segment {registration.segment_id}")
        if registration.local_slots != self._local_world_size:
            raise ValueError("local worker row width changed")
        if registration.num_blocks != self.num_blocks:
            raise ValueError("segment and scheduler block counts differ")
        if registration.block_len != self.block_len:
            raise ValueError("segment and scheduler row sizes differ")
        if registration.config_fingerprint != self.config_fingerprint:
            raise ValueError("segment and scheduler configuration differs")

    def update_worker_metadata(
        self,
        registrations: Iterable[SegmentRegistration],
        results: Iterable[SegmentOperationResult],
    ) -> None:
        for registration in registrations:
            self._validate_local_registration(registration)
            old = self._local_segments.get(registration.segment_id)
            if old is not None and old != registration:
                raise ValueError(
                    f"segment {registration.segment_id} registration changed "
                    "during runtime"
                )
            self._local_segments[registration.segment_id] = registration

        for result in results:
            if result.operation_id < 0:
                continue
            self._settle_operation(result)

    def _settle_operation(self, result: SegmentOperationResult) -> None:
        transfer_id = getattr(self, "_operation_to_transfer", {}).pop(
            result.operation_id, None
        )
        if transfer_id is None:
            # Duplicate/late acknowledgements are idempotently ignored.  A
            # reused primary slot can therefore never be completed by an old
            # worker response.
            return
        logical = self._transfers.get(transfer_id)
        if logical is None or result.segment_id not in logical.pending_segments:
            return
        logical.pending_segments.remove(result.segment_id)
        logical.failed |= not result.success
        if logical.pending_segments:
            return
        self._transfers.pop(transfer_id, None)
        if logical.failed:
            self._finished_failed.append(transfer_id)
        else:
            self._finished_done.append(transfer_id)

    def get_agent_metadata(self) -> bytes:
        if not self.ready:
            raise RuntimeError("all local KVCR mmap segments are not registered")
        return encode_registrations(self._local_segments.values())

    def add_remote_peer(
        self,
        peer_id: str,
        agent_metadata: bytes,
        base_addr: int,
        num_blocks: int,
        block_len: int,
    ) -> None:
        # The legacy scalar geometry is still handshake-validated by
        # P2PSession.  Authoritative rank geometry comes from the manifest.
        del base_addr, num_blocks, block_len
        remote_segments = decode_registrations(agent_metadata)
        if set(remote_segments) != self._expected_segments:
            raise ValueError("remote engine is missing required KVCR segments")
        if not self.ready:
            raise ValueError("local KVCR segments are not registered")
        for segment_id, remote in remote_segments.items():
            local = self._local_segments[segment_id]
            if not local.compatible_with(remote):
                raise ValueError(
                    f"remote KVCR segment {segment_id} has incompatible rank/layout"
                )
        self._remote_segments[peer_id] = remote_segments

    def remove_remote_peer(self, peer_id: str) -> None:
        self._remote_segments.pop(peer_id, None)

    def write_blocks(
        self,
        peer_id: str,
        local_idxs: list[int],
        remote_idxs: list[int],
    ) -> int | None:
        if len(local_idxs) != len(remote_idxs) or not local_idxs:
            return None
        remote_segments = self._remote_segments.get(peer_id)
        if remote_segments is None or not self.ready:
            return None
        if any(
            index < 0 or index >= self.num_blocks
            for index in local_idxs + remote_idxs
        ):
            return None
        transfer_id = next(self._next_transfer_id)
        required = set(self._expected_segments)
        self._transfers[transfer_id] = _LogicalTransfer(
            peer_id=peer_id,
            required_segments=required,
            pending_segments=set(required),
        )
        for segment_id in sorted(required):
            operation_id = next(self._next_operation_id)
            self._operation_to_transfer[operation_id] = transfer_id
            self._commands.append(
                SegmentTransferCommand(
                    operation_id=operation_id,
                    segment_id=segment_id,
                    peer_id=peer_id,
                    remote=remote_segments[segment_id],
                    local_indices=tuple(local_idxs),
                    remote_indices=tuple(remote_idxs),
                )
            )
        return transfer_id

    def take_worker_commands(self) -> tuple[SegmentTransferCommand, ...]:
        commands = tuple(self._commands)
        self._commands.clear()
        return commands

    def poll(self, peer_id: str | None = None) -> PollResult:
        del peer_id
        if not self._finished_done and not self._finished_failed:
            return _EMPTY_POLL_RESULT
        result = PollResult(
            done=tuple(self._finished_done), failed=tuple(self._finished_failed)
        )
        self._finished_done.clear()
        self._finished_failed.clear()
        return result

    def cancel(
        self,
        transfer_ids: Iterable[int],
        mode: CancelMode = "immediate",
    ) -> list[int]:
        del mode
        for transfer_id in transfer_ids:
            logical = self._transfers.pop(transfer_id, None)
            if logical is None:
                continue
            for command in list(self._commands):
                if self._operation_to_transfer.get(command.operation_id) == transfer_id:
                    self._commands.remove(command)
                    self._operation_to_transfer.pop(command.operation_id, None)
            for operation_id, owner in list(self._operation_to_transfer.items()):
                if owner == transfer_id:
                    self._operation_to_transfer.pop(operation_id, None)
            self._finished_failed.append(transfer_id)
        return []

    def close(self) -> None:
        self._commands.clear()
        self._transfers.clear()
        self._remote_segments.clear()
        self._local_segments.clear()
        self._finished_done.clear()
        self._finished_failed.clear()
