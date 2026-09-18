# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional, observational CPU-root cache projection.

Native keys remain the only residency authority. Complete-row metadata retains
bounded interior prefix identities, so checkpoints can arrive in either order.

The wire carries a ``LogicalSnapshot`` baseline followed by contiguous
``LogicalUpdate`` cuts, as the reviewed design requires: a consumer applies a
snapshot and the contiguous updates that follow it, and never a half-restored
index. A snapshot is republished whenever continuity cannot be guaranteed --
initial view, CPU reset, confidence change, replay-buffer turnover or publisher
backpressure -- so a consumer that misses an update recovers from the next
authoritative cut rather than from a reconstructed one.

This module is imported only when logical_cache_events is explicitly enabled.
"""

import queue
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from vllm.distributed.kv_events import EventBatch, EventPublisher, ZmqEventPublisher
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingEvent,
    OffloadKey,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.request import Request

logger = init_logger(__name__)

# Replay depth for the isolated logical topic. A late subscriber asking for
# sequence 0 must still receive a snapshot plus every update that followed it,
# so the producer re-bases before this many updates accumulate.
DEFAULT_BUFFER_STEPS = 32
# Publisher backlog. Bounded so a stalled consumer cannot grow scheduler memory;
# an overflow is reported to the producer instead of silently losing a cut.
DEFAULT_QUEUE_SIZE = 8
# Scheduler observations between re-announcements of a sticky unknown view. An
# unknown view has nothing new to say, but saying it exactly once means a single
# dropped frame leaves a consumer holding stale positive credit forever.
DEFAULT_UNKNOWN_REANNOUNCE_STEPS = 1000


class LogicalCachePublisher(ZmqEventPublisher):
    """Bounded publisher that never silently breaks protocol continuity.

    Incremental cuts must arrive contiguously, so a dropped message cannot be
    repaired by the wire. On overflow the whole backlog is discarded and the
    caller is told, which lets the projector re-base the stream with a snapshot.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._overflowed = False

    def publish(self, events: EventBatch) -> None:
        """Enqueue a cut, discarding the backlog rather than blocking.

        The base class promises at-least-once delivery. This topic deliberately
        does not: a logical cut is only ever a view of state the engine still
        holds, so dropping one and re-basing is correct where blocking the
        scheduler would not be. The signature matches the base; a discarded
        backlog is reported through :meth:`take_overflow` instead.
        """
        if not self._running:
            raise RuntimeError("Publisher is closed")
        events.data_parallel_rank = self._data_parallel_rank
        try:
            self._event_queue.put_nowait(events)
            return
        except queue.Full:
            pass
        # Drop the entire backlog rather than an arbitrary middle cut: a partial
        # backlog would leave the consumer applying a discontiguous update.
        while True:
            try:
                self._event_queue.get_nowait()
                self._event_queue.task_done()
            except queue.Empty:
                break
        self._overflowed = True
        self._event_queue.put_nowait(events)

    def take_overflow(self) -> bool:
        """True once per backlog discard, so the producer can re-base."""
        overflowed, self._overflowed = self._overflowed, False
        return overflowed

    def shutdown(self) -> None:
        # The base inserts its stop sentinel with put_nowait, which raises on a
        # full queue. Make room only when it IS full: dropping a queued cut
        # otherwise is exactly the arbitrary mid-stream loss this class exists
        # to avoid, and at shutdown there is no later cut to re-base from.
        if self._event_queue.full():
            try:
                self._event_queue.get_nowait()
                self._event_queue.task_done()
            except queue.Empty:
                pass
        super().shutdown()


@dataclass(frozen=True)
class _Row:
    segment: dict[str, Any]
    # At most B/h bindings per complete row; one for an exact-tail row.
    endpoints: tuple[tuple[int, bytes], ...]


class LogicalCPUProjector:
    def __init__(
        self,
        manager: CPUOffloadingManager,
        publisher: EventPublisher,
        *,
        namespace: str,
        full_group: int,
        recurrent_groups: tuple[int, ...],
        block_size: int,
        hash_unit: int,
        recurrent_chunk: int,
        buffer_steps: int = DEFAULT_BUFFER_STEPS,
        unknown_reannounce_steps: int = DEFAULT_UNKNOWN_REANNOUNCE_STEPS,
    ) -> None:
        self.manager = manager
        self.publisher = publisher
        self.namespace = namespace
        self.full_group = full_group
        self.recurrent_groups = recurrent_groups
        self.block_size = block_size
        self.hash_unit = hash_unit
        self.recurrent_chunk = recurrent_chunk
        self.max_rows = manager.logical_cache_capacity
        # Keep the covering snapshot inside the replay window at all times.
        self.snapshot_interval = max(1, buffer_steps - 1)
        self.unknown_reannounce_steps = max(1, unknown_reannounce_steps)
        self._unknown_silence = 0
        self.rows: dict[OffloadKey, _Row] = {}
        self._epoch = uuid4().hex
        self._cursor = 0
        self._reason = ""
        # Last view actually put on the wire, keyed by identity, so a cut can be
        # expressed as a delta without re-deriving the consumer's state.
        self._sent_segments: dict[str, dict[str, Any]] = {}
        self._sent_plans: dict[str, dict[str, Any]] = {}
        self._sent_confidence: tuple[str, str] | None = None
        self._since_snapshot = 0
        self._need_snapshot = True
        self._dirty = True
        self.observe(())

    # ── view maintenance ────────────────────────────────────────────────────

    def _invalidate(self, reason: str) -> None:
        self.rows.clear()
        self._reason = reason
        self._dirty = True

    def _prune_missing(self) -> None:
        """Drop rows the native cache no longer holds. Read-only, no promotion."""
        for key in list(self.rows):
            if self.manager.peek(key) == LookupResult.MISS:
                del self.rows[key]

    def register_complete(self, req: Request, end: int) -> None:
        self._register(req, end, "complete")

    def register_tail(self, req: Request, end: int) -> None:
        self._register(req, end, "tail")

    def _register(self, req: Request, end: int, kind: str) -> None:
        if self._reason:
            return
        try:
            # Presence, not truthiness. `prompt_embeds` is a tensor: truthiness
            # raises on more than one element, and -- worse -- a single zero
            # element is FALSY, so the guard would admit a request whose hash
            # domain this profile does not support. `mm_features` is a list, so
            # emptiness is genuinely absence.
            if any(
                getattr(req, field, None) is not None
                for field in ("lora_request", "cache_salt", "prompt_embeds")
            ) or getattr(req, "mm_features", None):
                self._invalidate("unsupported request hash inputs")
                return
            b, h = self.block_size, self.hash_unit
            start = (end - 1) // b * b
            # Explicit raises, not asserts: this is the producer's only guard
            # against publishing a row whose geometry it does not actually have,
            # and it must survive -O / PYTHONOPTIMIZE.
            if end <= 0 or end % h:
                raise ValueError(f"endpoint {end} is not a positive multiple of {h}")
            if (end % b == 0) != (kind == "complete"):
                raise ValueError(f"endpoint {end} does not match a {kind} row")
            hashes = req.block_hashes[start // h : end // h]
            if len(hashes) != (end - start) // h:
                raise ValueError("request is missing block hashes for this row")
            if not all(isinstance(x, bytes) for x in hashes):
                raise TypeError("block hashes are not native prefix bytes")
            end_hash = bytes(hashes[-1])
            key = make_offload_key(end_hash, self.full_group)
            parent = bytes(req.block_hashes[start // h - 1]) if start else None
            tokens = list(req.all_token_ids[start:end])
            if len(tokens) != end - start:
                raise ValueError("request is missing token IDs for this row")
            segment = dict(
                id=f"{kind}:{start}:{end}:{end_hash.hex()}",
                kind=kind,
                start=start,
                end=end,
                parent_hash=parent,
                end_hash=end_hash,
                token_ids=tokens,
                hash_unit=b if kind == "complete" else h,
                hashes=[end_hash] if kind == "complete" else list(hashes),
            )
            endpoints = (
                tuple(
                    (pos, bytes(hashes[(pos - start) // h - 1]))
                    for pos in range(start + h, end + 1, h)
                    if pos % self.recurrent_chunk == 0
                )
                if kind == "complete"
                else ((end, end_hash),)
            )
            row = _Row(segment, endpoints)
            existing = self.rows.get(key)
            if existing is not None:
                if existing != row:
                    self._invalidate("conflicting row identity")
                return
            # Bound retained metadata at admission, not only at observation:
            # registration runs ahead of the native store and must not let the
            # projection outgrow the pool it describes.
            if len(self.rows) >= self.max_rows:
                self._prune_missing()
                if len(self.rows) >= self.max_rows:
                    self._invalidate("row bound exceeded")
                    return
            self.rows[key] = row
            self._dirty = True
        except Exception:
            logger.exception("Logical CPU metadata unavailable")
            self._invalidate("invalid row metadata")

    def _derive(self) -> tuple[dict[str, dict], dict[str, dict]]:
        """Read-only pass over the native CPU view. Never promotes or pins."""
        segments: dict[str, dict[str, Any]] = {}
        plans: dict[str, dict[str, Any]] = {}
        for key, row in list(self.rows.items()):
            status = self.manager.peek(key)
            if status == LookupResult.MISS:
                del self.rows[key]
                continue
            if status != LookupResult.HIT:
                continue
            segment = row.segment
            segments[segment["id"]] = segment
            for end, terminal_hash in row.endpoints:
                if not all(
                    self.manager.peek(make_offload_key(terminal_hash, group))
                    == LookupResult.HIT
                    for group in self.recurrent_groups
                ):
                    continue
                complete = segment["kind"] == "complete"
                anchor = segment["end"] if complete else segment["start"]
                anchor_hash = (
                    segment["end_hash"] if complete else segment["parent_hash"]
                )
                plan_id = f"{segment['id']}:{end}"
                plans[plan_id] = dict(
                    id=plan_id,
                    kind=segment["kind"],
                    end=end,
                    anchor=anchor,
                    anchor_hash=anchor_hash,
                    terminal_hash=terminal_hash,
                    minimum_prompt_tokens=max(end, anchor) + 1,
                    binding=segment["id"],
                )
        return segments, plans

    # ── wire composition ────────────────────────────────────────────────────

    def _header(self, message_type: str) -> dict[str, Any]:
        return dict(
            type=message_type,
            profile="native-cpu-root",
            version=1,
            namespace=self.namespace,
            epoch=self._epoch,
            cursor=self._cursor,
            confidence="unknown" if self._reason else "known",
            reason=self._reason,
            block_size=self.block_size,
            hash_unit=self.hash_unit,
            max_rows=self.max_rows,
            max_plans=self.max_rows * (self.block_size // self.hash_unit + 1),
        )

    def _send(self, message: dict[str, Any]) -> bool:
        """Publish one cut and advance the cursor. True if a backlog was lost."""
        self.publisher.publish(EventBatch(ts=time.time(), events=[message]))
        self._cursor += 1
        # Only the bounded logical publisher can discard a backlog. Any other
        # EventPublisher is lossless by its own contract.
        take_overflow = getattr(self.publisher, "take_overflow", None)
        return take_overflow() is True if callable(take_overflow) else False

    def _publish_snapshot(
        self, segments: dict[str, dict], plans: dict[str, dict]
    ) -> None:
        message = dict(
            self._header("LogicalSnapshot"),
            segments=sorted(segments.values(), key=lambda s: s["id"]),
            plans=sorted(plans.values(), key=lambda p: p["id"]),
        )
        # A snapshot is self-sufficient: an overflow that dropped the backlog
        # cannot invalidate the cut we just enqueued, so its report is ignored.
        self._send(message)
        self._sent_segments = dict(segments)
        self._sent_plans = dict(plans)
        self._sent_confidence = (message["confidence"], self._reason)
        self._since_snapshot = 0
        self._need_snapshot = False

    def _publish_update(
        self,
        segments: dict[str, dict],
        plans: dict[str, dict],
        added_segments: list[dict],
        removed_segments: list[str],
        added_plans: list[dict],
        removed_plans: list[str],
    ) -> None:
        message = dict(
            self._header("LogicalUpdate"),
            added_segments=sorted(added_segments, key=lambda s: s["id"]),
            removed_segments=sorted(removed_segments),
            added_plans=sorted(added_plans, key=lambda p: p["id"]),
            removed_plans=sorted(removed_plans),
        )
        overflowed = self._send(message)
        self._sent_segments = dict(segments)
        self._sent_plans = dict(plans)
        self._sent_confidence = (message["confidence"], self._reason)
        self._since_snapshot += 1
        if overflowed:
            # The discarded backlog may have held the cut this update continues
            # from. Re-base immediately rather than leaving the consumer to
            # detect the gap and wait for an unrelated change.
            self._publish_snapshot(segments, plans)

    def observe(self, events: Iterable[OffloadingEvent]) -> None:
        """Revalidate with a read-only CPU view after the single native drain."""
        # An unknown view still has to be counted towards its re-announcement,
        # so it does not take the idle short-circuit. That path is O(1): it
        # never reaches the native scan.
        if not tuple(events) and not self._dirty and not self._reason:
            return
        try:
            segments, plans = ({}, {}) if self._reason else self._derive()
            if len(self.rows) > self.max_rows:
                self._invalidate("row bound exceeded")
                segments, plans = {}, {}
            confidence = ("unknown" if self._reason else "known", self._reason)

            added_segments = [
                s for i, s in segments.items() if self._sent_segments.get(i) != s
            ]
            removed_segments = [i for i in self._sent_segments if i not in segments]
            added_plans = [p for i, p in plans.items() if self._sent_plans.get(i) != p]
            removed_plans = [i for i in self._sent_plans if i not in plans]
            changes = (
                len(added_segments)
                + len(removed_segments)
                + len(added_plans)
                + len(removed_plans)
            )

            if (
                not changes
                and confidence == self._sent_confidence
                and not self._need_snapshot
            ):
                self._dirty = False
                # A view that has gone unknown is sticky until a CPU reset, so
                # this is the one state where "nothing more to say" is unsafe:
                # if the single cut that announced the loss was dropped on the
                # wire, silence leaves the consumer holding stale POSITIVE
                # credit. Re-announce on a bounded interval so a lost loss is
                # always eventually repaired, whatever the transport did.
                if self._reason:
                    self._unknown_silence += 1
                    if self._unknown_silence >= self.unknown_reannounce_steps:
                        self._need_snapshot = True
                        self._publish_snapshot({}, {})
                        self._unknown_silence = 0
                return
            self._unknown_silence = 0
            # A delta that already carries every live segment and plan rebuilds
            # the whole view; sending it as a snapshot costs the same bytes and
            # drops the continuity requirement.
            resends_whole_view = len(added_segments) >= len(segments) and len(
                added_plans
            ) >= len(plans)
            if (
                self._need_snapshot
                or confidence != self._sent_confidence
                or self._since_snapshot >= self.snapshot_interval
                or resends_whole_view
            ):
                self._publish_snapshot(segments, plans)
            else:
                self._publish_update(
                    segments,
                    plans,
                    added_segments,
                    removed_segments,
                    added_plans,
                    removed_plans,
                )
            self._dirty = False
        except Exception:
            logger.exception("Logical CPU observation failed; invalidating view")
            self._invalidate("observation failure")
            # Never suppress a native error or reset physical state, but do not
            # withhold the loss either: a consumer holding positive credit must
            # be told now, not at whatever observation happens to come next.
            try:
                self._need_snapshot = True
                self._publish_snapshot({}, {})
            except Exception:
                logger.exception(
                    "Logical CPU unknown notification failed; "
                    "retrying on the next observation"
                )
                self._need_snapshot = True

    def reset(self) -> None:
        self.rows.clear()
        self._epoch = uuid4().hex
        self._cursor = 0
        self._reason = ""
        self._sent_segments = {}
        self._sent_plans = {}
        self._sent_confidence = None
        self._since_snapshot = 0
        self._need_snapshot = True
        self._dirty = True
        self.observe(())

    def shutdown(self) -> None:
        self.publisher.shutdown()


def create_logical_projector(scheduler, vllm_config, kv_cache_config, options):
    """Return None for unsupported native profiles; never replace cache policy."""
    manager = scheduler.manager
    configs = scheduler.config.kv_group_configs
    specs = [g.kv_cache_spec for g in kv_cache_config.kv_cache_groups]
    full = [i for i, s in enumerate(specs) if isinstance(s, FullAttentionSpec)]
    recurrent = [i for i, s in enumerate(specs) if isinstance(s, MambaSpec)]
    modes = {specs[i].mamba_cache_mode for i in recurrent}
    chunks = {configs[i].tokens_per_chunk for i in recurrent}
    if (
        type(manager) is not CPUOffloadingManager
        or len(full) != 1
        or not recurrent
        or len(full) + len(recurrent) != len(specs)
        or len(modes) != 1
        or not modes <= {"align", "all"}
        or len(chunks) != 1
        or scheduler.config.blocks_per_chunk != 1
        or any(c.is_eagle_group for c in configs)
    ):
        logger.warning("Logical CPU profile unsupported; native behavior unchanged")
        return None
    b = configs[full[0]].tokens_per_chunk
    h = scheduler.config.tokens_per_hash
    r = next(iter(chunks))
    if not (h > 0 and b % h == 0 and r % h == 0 and b % r == 0):
        logger.warning("Logical CPU geometry unsupported")
        return None
    try:
        # Certifies the CPU policy without calling a tiered/native lookup.
        manager.peek(make_offload_key(b"logical-capability-probe", full[0]))
        endpoint = options["endpoint"]
        replay = options["replay_endpoint"]
        namespace = options["namespace"]
        # Explicit raises, not asserts, for the same reason as _register: these
        # must survive -O / PYTHONOPTIMIZE. buffer_steps is the damaging one --
        # the replay deque is maxlen=buffer_steps, so 1 leaves a window that can
        # hold a single message and 0 leaves one that can never serve a late
        # subscriber at all.
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("logical_cache_events.namespace must be a nonempty string")
        if not endpoint or not replay:
            raise ValueError(
                "logical_cache_events needs both endpoint and replay_endpoint"
            )
        buffer_steps = int(options.get("buffer_steps", DEFAULT_BUFFER_STEPS))
        queue_size = int(options.get("max_queue_size", DEFAULT_QUEUE_SIZE))
        if buffer_steps < 2 or queue_size < 1:
            raise ValueError(
                "logical_cache_events needs buffer_steps >= 2 and "
                f"max_queue_size >= 1, got buffer_steps={buffer_steps}, "
                f"max_queue_size={queue_size}"
            )
        endpoints = [endpoint, replay]
        legacy = vllm_config.kv_events_config
        if legacy is not None:
            endpoints.extend(e for e in (legacy.endpoint, legacy.replay_endpoint) if e)
        dp_size = vllm_config.parallel_config.data_parallel_size
        addresses = []
        for address in endpoints:
            for rank in range(dp_size):
                resolved = ZmqEventPublisher.offset_endpoint_port(address, rank)
                if resolved is None:
                    raise ValueError(f"cannot resolve endpoint {address!r}")
                # All TCP hosts at the same port are conservatively considered
                # colliding, including wildcard/localhost spelling differences.
                addresses.append(
                    resolved.rsplit(":", 1)[-1]
                    if resolved.startswith("tcp://")
                    else resolved
                )
        if len(addresses) != len(set(addresses)):
            raise ValueError("Logical and legacy endpoints/DP ports must be disjoint")
        publisher = LogicalCachePublisher(
            data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
            endpoint=endpoint,
            replay_endpoint=replay,
            topic="logical-cache-v1",
            buffer_steps=buffer_steps,
            max_queue_size=queue_size,
        )
    except Exception:
        logger.exception(
            "Logical CPU publication unavailable; native behavior unchanged"
        )
        return None
    # Native events are exact keys; enabling their observation changes no policy.
    # They are still drained once, and are not exposed on a disabled legacy stream.
    if manager.events is None:
        manager.events = []
    return LogicalCPUProjector(
        manager,
        publisher,
        namespace=namespace,
        full_group=full[0],
        recurrent_groups=tuple(recurrent),
        block_size=b,
        hash_unit=h,
        recurrent_chunk=r,
        buffer_steps=buffer_steps,
    )
