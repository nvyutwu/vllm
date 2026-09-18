# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.config import KVEventsConfig, KVTransferConfig
from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    BlockRemoved,
    BlockStored,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
)
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpecKind,
)
from vllm.v1.kv_offload.base import (
    Locality,
    Medium,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


def _logical_case(
    capacity=16,
    threshold=1,
    buffer_steps=32,
    publisher=None,
    unknown_reannounce_steps=1000,
):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.logical import (
        LogicalCPUProjector,
    )
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    manager = CPUOffloadingManager(
        capacity, enable_events=True, store_threshold=threshold
    )
    if publisher is None:
        publisher = MagicMock()
        publisher.publish.return_value = None
        publisher.take_overflow.return_value = False
    projector = LogicalCPUProjector(
        manager,
        publisher,
        namespace="test-model",
        full_group=3,
        recurrent_groups=(0, 1, 2),
        block_size=12288,
        hash_unit=128,
        recurrent_chunk=1536,
        buffer_steps=buffer_steps,
        unknown_reannounce_steps=unknown_reannounce_steps,
    )
    req = SimpleNamespace(
        block_hashes=[BlockHash(f"h{i}".encode()) for i in range(288)],
        all_token_ids=list(range(36864)),
        lora_request=None,
        mm_features=[],
        cache_salt=None,
        prompt_embeds=None,
    )
    return manager, publisher, projector, req


def _logical_keys(req, end, groups):
    return [make_offload_key(req.block_hashes[end // 128 - 1], g) for g in groups]


def _logical_ready(manager, keys):
    from vllm.v1.kv_offload.base import ReqContext

    ctx = ReqContext(req_id="logical-test")
    out = manager.prepare_store(keys, ctx)
    assert out is not None
    manager.complete_store(out.keys_to_store, ctx)


def logical_view(publisher, strict=True):
    """Fold a published cut sequence exactly as a conforming consumer must.

    A snapshot re-bases the view; an update applies only when it is contiguous
    with the accepted cut in the same epoch. Reading the last message instead
    would let a producer that silently broke continuity still pass.
    """
    view = None
    for published in publisher.publish.call_args_list:
        message = published.args[0].events[0]
        if view is not None and message["epoch"] != view["epoch"]:
            view = None
        if message["type"] == "LogicalSnapshot":
            view = dict(
                message,
                segments={s["id"]: s for s in message["segments"]},
                plans={p["id"]: p for p in message["plans"]},
            )
            continue
        assert message["type"] == "LogicalUpdate"
        if view is None or message["cursor"] != view["cursor"] + 1:
            assert not strict, "discontiguous logical update"
            view = None
            continue
        segments = dict(view["segments"])
        plans = dict(view["plans"])
        for plan_id in message["removed_plans"]:
            plans.pop(plan_id, None)
        for segment_id in message["removed_segments"]:
            segments.pop(segment_id, None)
            for plan_id in [i for i, p in plans.items() if p["binding"] == segment_id]:
                del plans[plan_id]
        for segment in message["added_segments"]:
            segments[segment["id"]] = segment
        for plan in message["added_plans"]:
            plans[plan["id"]] = plan
        view = dict(message, segments=segments, plans=plans)
    if view is None:
        return None
    return dict(
        view,
        segments=sorted(view["segments"].values(), key=lambda s: s["id"]),
        plans=sorted(view["plans"].values(), key=lambda p: p["id"]),
    )


def _logical_snapshot(projector, manager, publisher):
    projector.observe(tuple(manager.take_events()))
    return logical_view(publisher)


@pytest.mark.parametrize("tail", [False, True])
def test_logical_complete_and_exact_tail_have_distinct_prompt_predicates(tail):
    manager, publisher, projector, req = _logical_case()
    projector.register_complete(req, 12288)
    projector.register_tail(req, 10752)
    keys = _logical_keys(req, 10752, range(3)) + _logical_keys(req, 12288, [3])
    if tail:
        keys += _logical_keys(req, 10752, [3])
    _logical_ready(manager, keys)
    snap = _logical_snapshot(projector, manager, publisher)
    plans = [p for p in snap["plans"] if p["end"] == 10752]
    assert {(p["kind"], p["anchor"], p["minimum_prompt_tokens"]) for p in plans} == (
        {("complete", 12288, 12289), ("tail", 0, 10753)}
        if tail
        else {("complete", 12288, 12289)}
    )


def test_logical_readiness_joins_existing_keys_and_separate_jobs():
    from vllm.v1.kv_offload.base import ReqContext

    manager, publisher, projector, req = _logical_case(threshold=2)
    keys = _logical_keys(req, 23296, range(4))
    ctx = ReqContext(req_id="logical-test")
    manager.prepare_store(keys[:2], ctx)
    _logical_ready(manager, keys[:2])
    projector.register_tail(req, 23296)
    partial = manager.prepare_store(keys, ctx)
    assert partial is not None and partial.keys_to_store == []
    assert not _logical_snapshot(projector, manager, publisher)["plans"]
    partial = manager.prepare_store(keys, ctx)
    assert partial is not None and partial.keys_to_store == keys[2:]
    manager.complete_store(keys[2:3], ctx)
    assert not _logical_snapshot(projector, manager, publisher)["plans"]
    # The attention row was absent at the first offer: register its metadata again.
    projector.register_tail(req, 23296)
    manager.complete_store(keys[3:], ctx)
    snap = _logical_snapshot(projector, manager, publisher)
    assert [(p["end"], p["anchor"]) for p in snap["plans"]] == [(23296, 12288)]


def test_logical_prefix_loss_changes_reachability_not_downstream_plan():
    manager, publisher, projector, req = _logical_case(capacity=5)
    projector.register_complete(req, 12288)
    projector.register_tail(req, 23296)
    _logical_ready(manager, _logical_keys(req, 12288, [3]))
    _logical_ready(manager, _logical_keys(req, 23296, range(4)))
    snap = _logical_snapshot(projector, manager, publisher)
    assert len(snap["segments"]) == 2 and len(snap["plans"]) == 1
    _logical_ready(manager, [make_offload_key(BlockHash(b"other"), 3)])
    snap = _logical_snapshot(projector, manager, publisher)
    assert len(snap["segments"]) == 1
    assert snap["plans"][0]["end"] == 23296


def test_logical_companion_loss_withdraws_plan_without_coverage_delta():
    manager, publisher, projector, req = _logical_case(capacity=4)
    projector.register_tail(req, 15104)
    _logical_ready(manager, _logical_keys(req, 15104, range(4)))
    before = _logical_snapshot(projector, manager, publisher)
    _logical_ready(manager, [make_offload_key(BlockHash(b"other"), 0)])
    after = _logical_snapshot(projector, manager, publisher)
    assert after["segments"] == before["segments"]
    assert before["plans"] and not after["plans"]


def test_logical_duplicate_events_and_reset_are_atomic_cpu_snapshots():
    manager, publisher, projector, req = _logical_case()
    projector.register_tail(req, 10752)
    _logical_ready(manager, _logical_keys(req, 10752, range(4)))
    events = tuple(manager.take_events())
    projector.observe(events)
    before = publisher.publish.call_args.args[0].events[0]
    count = publisher.publish.call_count
    projector.observe(events + events)
    assert publisher.publish.call_count == count
    manager.reset_cache()
    projector.reset()
    after = publisher.publish.call_args.args[0].events[0]
    assert after["epoch"] != before["epoch"]
    assert after["cursor"] == 0 and after["namespace"] == "test-model"
    assert after["segments"] == after["plans"] == []
    projector.observe(events)
    assert publisher.publish.call_args.args[0].events[0] == after


def test_logical_complete_bindings_are_bounded_and_die_with_enclosing_row():
    manager, publisher, projector, req = _logical_case(capacity=4)
    projector.register_complete(req, 12288)
    _logical_ready(manager, _logical_keys(req, 12288, [3]))
    _logical_ready(manager, _logical_keys(req, 10752, range(3)))
    assert _logical_snapshot(projector, manager, publisher)["plans"]
    _logical_ready(manager, [make_offload_key(BlockHash(b"other"), 3)])
    snap = _logical_snapshot(projector, manager, publisher)
    assert not snap["plans"] and not projector.rows


def _logical_messages(publisher):
    return [c.args[0].events[0] for c in publisher.publish.call_args_list]


def test_logical_change_after_a_baseline_is_an_incremental_cut():
    """The reviewed contract is a snapshot plus contiguous updates, not a
    complete view on every publication."""
    manager, publisher, projector, req = _logical_case()
    projector.register_complete(req, 12288)
    _logical_ready(manager, _logical_keys(req, 12288, [3]))
    baseline = _logical_snapshot(projector, manager, publisher)
    assert len(baseline["segments"]) == 1 and not baseline["plans"]
    assert _logical_messages(publisher)[-1]["type"] == "LogicalSnapshot"

    _logical_ready(manager, _logical_keys(req, 10752, range(3)))
    view = _logical_snapshot(projector, manager, publisher)
    cut = _logical_messages(publisher)[-1]
    assert cut["type"] == "LogicalUpdate"
    assert cut["cursor"] == baseline["cursor"] + 1
    assert cut["epoch"] == baseline["epoch"]
    # Only the newly certified endpoint travels; the resident row does not.
    assert cut["added_segments"] == [] and cut["removed_segments"] == []
    assert cut["removed_plans"] == []
    assert [p["end"] for p in cut["added_plans"]] == [10752]
    assert len(view["segments"]) == 1 and [p["end"] for p in view["plans"]] == [10752]


def test_logical_update_is_never_applied_without_its_baseline():
    manager, publisher, projector, req = _logical_case()
    projector.register_complete(req, 12288)
    _logical_ready(manager, _logical_keys(req, 12288, [3]))
    _logical_snapshot(projector, manager, publisher)
    _logical_ready(manager, _logical_keys(req, 10752, range(3)))
    _logical_snapshot(projector, manager, publisher)

    messages = _logical_messages(publisher)
    baseline = max(i for i, m in enumerate(messages) if m["type"] == "LogicalSnapshot")
    assert any(m["type"] == "LogicalUpdate" for m in messages[baseline + 1 :])
    orphaned = MagicMock()
    orphaned.publish.call_args_list = publisher.publish.call_args_list[baseline + 1 :]
    # A consumer that joined after the baseline holds no view at all; it must
    # not synthesise one from the deltas it happens to have seen.
    assert logical_view(orphaned, strict=False) is None


def test_logical_updates_never_outrun_the_replay_window():
    manager, publisher, projector, req = _logical_case(capacity=32, buffer_steps=3)
    for end in (10752, 11136, 21504, 23040):
        projector.register_tail(req, end)
        _logical_ready(manager, _logical_keys(req, end, range(4)))
        _logical_snapshot(projector, manager, publisher)

    kinds = [m["type"] for m in _logical_messages(publisher)]
    assert kinds.count("LogicalUpdate") >= 2, kinds
    run = 0
    for kind in kinds:
        run = run + 1 if kind == "LogicalUpdate" else 0
        # buffer_steps=3 retains three cuts, so at most two updates may separate
        # consecutive snapshots or a late subscriber cannot rebuild.
        assert run <= 2, kinds


def test_logical_publisher_backpressure_rebases_with_a_snapshot():
    overflow: dict[str, int | None] = {"at": None}
    publisher = MagicMock()
    publisher.publish.return_value = None
    publisher.take_overflow.side_effect = lambda: (
        len(publisher.publish.call_args_list) == overflow["at"]
    )
    manager, publisher, projector, req = _logical_case(publisher=publisher)
    projector.register_complete(req, 12288)
    _logical_ready(manager, _logical_keys(req, 12288, [3]))
    _logical_snapshot(projector, manager, publisher)

    overflow["at"] = len(publisher.publish.call_args_list) + 1
    _logical_ready(manager, _logical_keys(req, 10752, range(3)))
    view = _logical_snapshot(projector, manager, publisher)
    tail = _logical_messages(publisher)[-2:]
    assert [m["type"] for m in tail] == ["LogicalUpdate", "LogicalSnapshot"]
    # The dropped backlog may have held the cut the update continued from, so
    # the repair is authoritative and carries the same view.
    assert [p["end"] for p in tail[-1]["plans"]] == [10752]
    assert [p["end"] for p in view["plans"]] == [10752]


def test_logical_observation_failure_reports_the_loss_in_the_same_pass():
    manager, publisher, projector, req = _logical_case()
    projector.register_complete(req, 12288)
    _logical_ready(manager, _logical_keys(req, 12288, [3]))
    _logical_snapshot(projector, manager, publisher)
    before = publisher.publish.call_count

    _logical_ready(manager, _logical_keys(req, 10752, range(3)))
    events = tuple(manager.take_events())
    assert events
    with patch.object(manager, "peek", side_effect=RuntimeError("native failure")):
        projector.observe(events)
    assert publisher.publish.call_count == before + 1
    loss = _logical_messages(publisher)[-1]
    assert loss["type"] == "LogicalSnapshot"
    assert loss["confidence"] == "unknown" and loss["reason"] == "observation failure"
    assert loss["segments"] == loss["plans"] == []
    assert logical_view(publisher)["confidence"] == "unknown"


@pytest.mark.parametrize("embeds", [torch.zeros(4, 8), torch.zeros(1)])
def test_logical_prompt_embeds_are_refused_by_presence_not_truthiness(embeds):
    """A tensor's truthiness is the wrong question in both directions.

    More than one element raises, which the catch-all would report as "invalid
    row metadata" rather than an unsupported hash domain. A single zero element
    is FALSY, which is worse: the request would pass the guard and be registered
    as if it were token-only.
    """
    manager, publisher, projector, req = _logical_case()
    req.prompt_embeds = embeds
    projector.register_complete(req, 12288)
    assert not projector.rows
    view = _logical_snapshot(projector, manager, publisher)
    assert view["confidence"] == "unknown"
    assert view["reason"] == "unsupported request hash inputs"


def test_logical_shutdown_keeps_a_queued_cut_when_there_is_room():
    """Only a FULL queue may lose a cut. At shutdown there is no later cut to
    re-base from, so an unconditional drop would be a silent mid-stream loss."""
    from uuid import uuid4

    from vllm.distributed.kv_events import EventBatch
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.logical import (
        LogicalCachePublisher,
    )

    suffix = uuid4().hex
    publisher = LogicalCachePublisher(
        0,
        endpoint=f"inproc://shutdown-{suffix}",
        replay_endpoint=f"inproc://shutdown-replay-{suffix}",
        topic="logical-cache-v1",
        max_queue_size=8,
        buffer_steps=4,
    )
    try:
        publisher.publish(EventBatch(ts=0.0, events=[{"type": "LogicalSnapshot"}]))
        assert not publisher.take_overflow()
    finally:
        publisher.shutdown()
    # The publisher thread drains before exiting, so the cut reached the wire.
    assert len(publisher._buffer) == 1


def test_logical_unknown_view_keeps_announcing_itself():
    """Saying "unknown" exactly once is unsafe: a dropped frame would leave a
    consumer holding stale POSITIVE credit, and a sticky unknown producer has
    nothing else to publish that would ever reveal it."""
    manager, publisher, projector, req = _logical_case(unknown_reannounce_steps=3)
    projector.register_complete(req, 12288)
    _logical_ready(manager, _logical_keys(req, 12288, [3]))
    _logical_ready(manager, _logical_keys(req, 10752, range(3)))
    assert _logical_snapshot(projector, manager, publisher)["plans"]

    _logical_ready(manager, _logical_keys(req, 21504, range(4)))
    events = tuple(manager.take_events())
    with patch.object(manager, "peek", side_effect=RuntimeError("native failure")):
        projector.observe(events)
    announced = _logical_messages(publisher)[-1]
    assert announced["confidence"] == "unknown"
    count = publisher.publish.call_count

    # Idle observations: the view has nothing new to say, but silence is not
    # safe here, so the loss is restated on a bounded interval.
    for _ in range(3):
        projector.observe(())
    assert publisher.publish.call_count == count + 1
    repeat = _logical_messages(publisher)[-1]
    assert repeat["type"] == "LogicalSnapshot"
    assert repeat["confidence"] == "unknown"
    assert repeat["reason"] == announced["reason"]
    assert repeat["epoch"] == announced["epoch"]
    assert repeat["cursor"] == announced["cursor"] + 1
    assert repeat["segments"] == repeat["plans"] == []


def test_logical_churn_state_is_bounded_by_the_pool_not_by_history():
    """Fixed capacity, long history: nothing retained may grow with the history."""
    manager, publisher, projector, req = _logical_case(capacity=4)
    ends = [e for e in range(128, 36864, 128) if e % 12288][:40]
    assert len(ends) == 40
    widths = []
    for end in ends:
        projector.register_tail(req, end)
        _logical_ready(manager, _logical_keys(req, end, range(4)))
        view = _logical_snapshot(projector, manager, publisher)
        widths.append((len(projector.rows), len(view["segments"]), len(view["plans"])))

    rows, segments, plans = max(widths)
    assert rows <= 4, widths
    # The consumer's view is what the producer actually said; it must stay
    # bounded by the pool too, or a long-lived worker grows the frontend.
    assert segments <= 4 and plans <= 4, widths
    assert len(projector._sent_segments) <= 4
    assert len(projector._sent_plans) <= 4
    # And the stream stays inside its replay window the whole way.
    assert projector._since_snapshot <= projector.snapshot_interval


def test_logical_registration_is_bounded_before_native_admission():
    manager, publisher, projector, req = _logical_case(capacity=2)
    for end in (12288, 24576):
        projector.register_complete(req, end)
        _logical_ready(manager, _logical_keys(req, end, [3]))
    _logical_snapshot(projector, manager, publisher)
    assert len(projector.rows) == 2

    # Registration runs ahead of the native store: the pool is full and no row
    # can be pruned, so the projection is retired instead of growing past it.
    projector.register_complete(req, 36864)
    assert not projector.rows
    view = _logical_snapshot(projector, manager, publisher)
    assert view["confidence"] == "unknown" and view["reason"] == "row bound exceeded"
    assert view["segments"] == view["plans"] == []


def test_logical_transport_isolates_wildcard_legacy_and_replays_latest_snapshot():
    import time
    from uuid import uuid4

    import msgspec
    import zmq

    from vllm.distributed.kv_events import EventBatch, ZmqEventPublisher
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.logical import (
        LogicalCachePublisher,
    )

    suffix = uuid4().hex
    legacy_address = f"inproc://legacy-{suffix}"
    logical_address = f"inproc://logical-{suffix}"
    replay_address = f"inproc://replay-{suffix}"
    legacy = ZmqEventPublisher(0, endpoint=legacy_address)
    logical = LogicalCachePublisher(
        2,
        endpoint=logical_address,
        replay_endpoint=replay_address,
        topic="logical-cache-v1",
        max_queue_size=1,
        buffer_steps=1,
    )
    context = zmq.Context.instance()
    wildcard = context.socket(zmq.SUB)
    wildcard.setsockopt(zmq.SUBSCRIBE, b"")
    wildcard.connect(legacy_address)
    replay = context.socket(zmq.DEALER)
    replay.setsockopt(zmq.RCVTIMEO, 2000)
    replay.connect(f"{replay_address}_dp2")
    try:
        legacy.publish(EventBatch(ts=0.0, events=[{"type": "AllBlocksCleared"}]))
        assert wildcard.poll(2000)
        assert msgspec.msgpack.decode(wildcard.recv_multipart()[2])[1] == [
            {"type": "AllBlocksCleared"}
        ]
        for cursor in range(20):
            logical.publish(
                EventBatch(
                    ts=0.0,
                    events=[
                        dict(
                            type="LogicalSnapshot", cursor=cursor, segments=[], plans=[]
                        )
                    ],
                )
            )
        deadline = time.monotonic() + 5
        observed = None
        while time.monotonic() < deadline:
            replay.send_multipart([b"", (0).to_bytes(8, "big")])
            while True:
                delimiter, topic, sequence, payload = replay.recv_multipart()
                assert delimiter == b""
                if sequence == ZmqEventPublisher.END_SEQ:
                    break
                assert topic == b"logical-cache-v1"
                observed = msgspec.msgpack.decode(payload)
            if observed and observed[1][0]["cursor"] == 19:
                break
        assert observed is not None
        assert observed[2] == 2 and observed[1][0]["cursor"] == 19
        assert not wildcard.poll(100)
    finally:
        wildcard.close(linger=0)
        replay.close(linger=0)
        logical.shutdown()
        legacy.shutdown()


_CPU_MEDIUM = Medium.CPU
_FULL_ATTENTION_EVENT_SPEC = OffloadingEventGroupSpec(
    kv_cache_spec_kind=KVCacheSpecKind.FULL_ATTENTION.value,
    kv_cache_spec_sliding_window=None,
)


def _tracker(
    *,
    enable_kv_cache_events: bool = True,
    self_describing_kv_events: bool = True,
) -> OffloadingEventsTracker:
    return OffloadingEventsTracker(
        OffloadingKVEventsConfig(
            enable_kv_cache_events=enable_kv_cache_events,
            self_describing_kv_events=self_describing_kv_events,
        )
    )


def _hash(i: int) -> BlockHash:
    return BlockHash(str(i).encode())


def _wire_hash(block_hash: BlockHash):
    return maybe_convert_block_hash(block_hash)


def _request(*, block_hashes: list[BlockHash], token_count: int, req_id: str = "req"):
    req = MagicMock()
    req.request_id = req_id
    req.block_hashes = block_hashes
    req.all_token_ids = list(range(1, token_count + 1))
    req.lora_request = None
    return req


def _group_config(
    *,
    group_idx: int = 0,
    block_size: int = 4,
    blocks_per_chunk: int = 1,
    tokens_per_hash: int | None = None,
    sliding_window_size_in_chunks: int | None = None,
) -> GroupOffloadConfig:
    if tokens_per_hash is None:
        tokens_per_hash = block_size
    tokens_per_chunk = block_size * blocks_per_chunk
    assert tokens_per_chunk % tokens_per_hash == 0
    return GroupOffloadConfig(
        group_idx=group_idx,
        tokens_per_block=block_size,
        tokens_per_chunk=tokens_per_chunk,
        hashes_per_chunk=tokens_per_chunk // tokens_per_hash,
        sliding_window_size_in_chunks=sliding_window_size_in_chunks,
        kv_event_group_spec=_FULL_ATTENTION_EVENT_SPEC,
    )


def _record_chunks(
    tracker: OffloadingEventsTracker,
    req,
    group_config: GroupOffloadConfig,
    num_chunks: int,
) -> list[OffloadKey]:
    keys: list[OffloadKey] = []
    hbf = group_config.hashes_per_chunk
    for chunk_idx in range(num_chunks):
        tail_hash = req.block_hashes[(chunk_idx + 1) * hbf - 1]
        assert tail_hash is not None
        key = make_offload_key(tail_hash, group_config.group_idx)
        tracker.record_store(req, group_config, chunk_idx, key)
        keys.append(key)
    return keys


def _record_lookup_chunks(
    tracker: OffloadingEventsTracker,
    req,
    group_config: GroupOffloadConfig,
    num_chunks: int,
) -> list[OffloadKey]:
    keys: list[OffloadKey] = []
    hbf = group_config.hashes_per_chunk
    for chunk_idx in range(num_chunks):
        tail_hash = req.block_hashes[(chunk_idx + 1) * hbf - 1]
        assert tail_hash is not None
        key = make_offload_key(tail_hash, group_config.group_idx)
        tracker.record_lookup(
            req,
            group_config,
            chunk_idx,
            key,
        )
        keys.append(key)
    return keys


def _stored_event(
    keys: list[OffloadKey],
    medium: Medium = _CPU_MEDIUM,
    locality: Locality | None = None,
    ownership: str | None = None,
    removal_expected: bool = False,
) -> OffloadingEvent:
    return OffloadingEvent(
        keys=keys,
        medium=medium,
        removed=False,
        locality=locality,
        ownership=ownership,
        removal_expected=removal_expected,
    )


def _removed_event(
    keys: list[OffloadKey],
    medium: Medium = _CPU_MEDIUM,
    locality: Locality | None = None,
    ownership: str | None = None,
) -> OffloadingEvent:
    return OffloadingEvent(
        keys=keys,
        medium=medium,
        removed=True,
        locality=locality,
        ownership=ownership,
    )


def _lookup_chunk() -> tuple[
    OffloadingEventsTracker, MagicMock, GroupOffloadConfig, OffloadKey
]:
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4)
    group_config = _group_config()
    key = _record_lookup_chunks(
        tracker,
        req,
        group_config,
        num_chunks=1,
    )[0]
    return tracker, req, group_config, key


def test_take_events_forwards_locality_to_rich_store():
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4)
    key = _record_chunks(tracker, req, _group_config(), num_chunks=1)[0]

    events = list(
        tracker.take_events(
            [_stored_event([key], locality=Locality.LOCAL, medium=Medium.STORAGE)]
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].token_ids == [1, 2, 3, 4]
    assert events[0].block_size == 4
    assert events[0].locality == "LOCAL"


def test_take_events_forwards_locality_to_placeholder_store():
    tracker = _tracker(self_describing_kv_events=False)
    req = _request(block_hashes=[_hash(0)], token_count=4)
    key = _record_chunks(tracker, req, _group_config(), num_chunks=1)[0]

    events = list(
        tracker.take_events(
            [_stored_event([key], locality=Locality.REMOTE, medium=Medium.STORAGE)]
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].block_size == 0
    assert events[0].locality == "REMOTE"


def test_partial_tail_event_describes_hash_aligned_physical_block_prefix():
    tracker = _tracker()
    group_config = _group_config(block_size=16, blocks_per_chunk=1)._replace(
        hashes_per_chunk=4
    )
    req = _request(block_hashes=[_hash(i) for i in range(8)], token_count=32)
    key = make_offload_key(req.block_hashes[6], group_config.group_idx)

    tracker.record_partial_store(req, group_config, 28, key)
    [event] = tracker.take_events([_stored_event([key])])

    assert isinstance(event, BlockStored)
    assert event.block_hashes == [_wire_hash(_hash(i)) for i in range(4, 7)]
    assert event.parent_block_hash == _wire_hash(_hash(3))
    assert event.token_ids == list(range(17, 29))
    assert event.block_size == 4


def test_partial_tail_lookup_does_not_overwrite_store_metadata():
    tracker = _tracker()
    group_config = _group_config()
    stored_req = _request(block_hashes=[_hash(0)], token_count=4)
    lookup_req = _request(block_hashes=[_hash(0)], token_count=4)
    lookup_req.all_token_ids = [9, 9, 9, 9]
    key = make_offload_key(stored_req.block_hashes[0], group_config.group_idx)

    tracker.record_partial_store(stored_req, group_config, 4, key)
    tracker.record_partial_lookup(lookup_req, group_config, 4, key)
    [event] = tracker.take_events([_stored_event([key])])

    assert isinstance(event, BlockStored)
    assert event.token_ids == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "record_method", ["record_partial_store", "record_partial_lookup"]
)
def test_partial_tail_sliding_window_event_uses_placeholder(record_method):
    tracker = _tracker()
    group_config = _group_config(sliding_window_size_in_chunks=1)
    req = _request(block_hashes=[_hash(0)], token_count=4)
    key = make_offload_key(req.block_hashes[0], group_config.group_idx)

    getattr(tracker, record_method)(req, group_config, 4, key)
    [event] = tracker.take_events([_stored_event([key])])

    assert isinstance(event, BlockStored)
    assert event.block_hashes == [_wire_hash(_hash(0))]
    assert event.token_ids == []
    assert event.block_size == 0


def test_take_events_forwards_locality_to_remove():
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4)
    key = _record_chunks(tracker, req, _group_config(), num_chunks=1)[0]

    events = list(
        tracker.take_events(
            [_removed_event([key], locality=Locality.LOCAL, medium=Medium.STORAGE)]
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], BlockRemoved)
    assert events[0].locality == "LOCAL"


def test_take_events_publishes_routable_block_stored():
    block_size = 4
    tracker = _tracker()
    group_config = _group_config(block_size=block_size)
    req = _request(
        block_hashes=[_hash(i) for i in range(6)],
        token_count=block_size * 6,
    )
    keys = _record_chunks(tracker, req, group_config, num_chunks=6)

    batch1 = list(tracker.take_events([_stored_event(keys[:3])]))
    assert len(batch1) == 3

    for i, event in enumerate(batch1):
        assert isinstance(event, BlockStored)
        assert event.medium == _CPU_MEDIUM.value
        assert event.block_hashes == [_wire_hash(_hash(i))]
        assert event.block_size == block_size
        assert event.token_ids == list(
            range(i * block_size + 1, (i + 1) * block_size + 1)
        )
        if i == 0:
            assert event.parent_block_hash is None
        else:
            assert event.parent_block_hash == _wire_hash(_hash(i - 1))
        assert event.lora_id is None
        assert event.lora_name is None
        assert event.extra_keys is None
        assert event.group_idx == 0
        assert event.kv_cache_spec_kind == KVCacheSpecKind.FULL_ATTENTION.value
        assert event.kv_cache_spec_sliding_window is None

    batch2 = list(tracker.take_events([_stored_event(keys[3:])]))
    assert len(batch2) == 3
    assert batch2[0].parent_block_hash == batch1[-1].block_hashes[-1]

    assert len(tracker._pending_event_metadata) == 6


def test_promotion_emits_full_cpu_stored_event():
    tracker, _, _, key = _lookup_chunk()

    [event] = tracker.take_events([_stored_event([key])])

    assert isinstance(event, BlockStored)
    assert event.medium == MEDIUM_CPU
    assert event.block_hashes == [_wire_hash(_hash(0))]
    assert event.parent_block_hash is None
    assert event.token_ids == [1, 2, 3, 4]
    assert event.block_size == 4
    assert event.lora_id is None
    assert event.lora_name is None
    assert event.extra_keys is None
    assert event.group_idx == 0
    assert event.kv_cache_spec_kind == KVCacheSpecKind.FULL_ATTENTION.value
    assert event.kv_cache_spec_sliding_window is None


@pytest.mark.parametrize(
    ("blocks_per_chunk", "expected_hash_indices"),
    [(1, [63]), (2, [63, 127])],
)
def test_event_hashes_use_group_block_size(
    blocks_per_chunk: int, expected_hash_indices: list[int]
):
    tokens_per_hash = 4
    block_size = 256
    hashes_per_block = block_size // tokens_per_hash
    tracker = _tracker()
    group_config = _group_config(
        block_size=block_size,
        blocks_per_chunk=blocks_per_chunk,
        tokens_per_hash=tokens_per_hash,
    )
    req = _request(
        block_hashes=[_hash(i) for i in range(hashes_per_block * blocks_per_chunk)],
        token_count=block_size * blocks_per_chunk,
    )
    [key] = _record_chunks(tracker, req, group_config, num_chunks=1)

    [event] = tracker.take_events([_stored_event([key])])

    assert isinstance(event, BlockStored)
    assert event.block_hashes == [_wire_hash(_hash(i)) for i in expected_hash_indices]
    assert event.block_size == block_size
    assert len(event.token_ids) == block_size * blocks_per_chunk


def test_lookup_promotion_factor_gt_1_store_and_remove():
    block_size = 4
    blocks_per_chunk = 2
    tracker = _tracker()
    group_config = _group_config(
        block_size=block_size, blocks_per_chunk=blocks_per_chunk
    )
    req = _request(
        block_hashes=[_hash(i) for i in range(4)],
        token_count=block_size * blocks_per_chunk * 2,
    )
    keys = _record_lookup_chunks(tracker, req, group_config, num_chunks=2)

    stored = list(tracker.take_events([_stored_event(keys)]))
    assert len(stored) == 2

    expected_hashes = []
    for chunk_idx, event in enumerate(stored):
        assert isinstance(event, BlockStored)
        expected_chunk_hashes = [
            _wire_hash(_hash(i))
            for i in range(
                chunk_idx * blocks_per_chunk,
                (chunk_idx + 1) * blocks_per_chunk,
            )
        ]
        assert event.block_hashes == expected_chunk_hashes
        assert event.block_size == block_size
        assert len(event.token_ids) == block_size * blocks_per_chunk
        if chunk_idx == 0:
            assert event.parent_block_hash is None
        else:
            assert event.parent_block_hash == _wire_hash(_hash(blocks_per_chunk - 1))
        expected_hashes.extend(expected_chunk_hashes)

    assert len(tracker._pending_event_metadata) == 2

    removed = list(tracker.take_events([_removed_event(keys)]))
    assert len(removed) == 1
    assert isinstance(removed[0], BlockRemoved)
    assert removed[0].block_hashes == expected_hashes
    assert removed[0].medium == _CPU_MEDIUM.value
    assert removed[0].group_idx == 0
    assert not tracker._pending_event_metadata


def test_take_events_factor_gt_1_store_is_order_independent():
    blocks_per_chunk = 3
    tracker = _tracker()
    group_config = _group_config(blocks_per_chunk=blocks_per_chunk)
    req = _request(
        block_hashes=[_hash(i) for i in range(6)],
        token_count=4 * blocks_per_chunk * 2,
    )
    keys = _record_chunks(tracker, req, group_config, num_chunks=2)
    unknown_key = make_offload_key(_hash(12345), 0)

    events = list(tracker.take_events([_stored_event([keys[1], unknown_key, keys[0]])]))

    assert len(events) == 3
    chunk1, placeholder, chunk0 = events
    assert [len(event.block_hashes) for event in events] == [3, 1, 3]
    assert placeholder.block_size == 0
    assert placeholder.token_ids == []
    assert chunk0.parent_block_hash is None
    assert chunk1.parent_block_hash == chunk0.block_hashes[-1]


def test_take_events_opt_out_keeps_placeholders():
    tracker = _tracker(self_describing_kv_events=False)
    group_config = _group_config()
    req = _request(block_hashes=[_hash(i) for i in range(3)], token_count=12)
    keys = _record_chunks(tracker, req, group_config, num_chunks=3)
    _record_lookup_chunks(tracker, req, group_config, num_chunks=3)

    assert not tracker.self_describing_enabled
    assert not tracker._pending_event_metadata

    events = list(
        tracker.take_events(
            [
                _stored_event(keys),
                _removed_event(keys),
            ]
        )
    )
    assert len(events) == 4
    for event in events[:3]:
        assert isinstance(event, BlockStored)
        assert event.block_size == 0
        assert event.token_ids == []
        assert event.parent_block_hash is None
    assert isinstance(events[3], BlockRemoved)
    assert len(events[3].block_hashes) == 3


@pytest.mark.parametrize(
    "sliding_window_size_in_chunks",
    [1, 2],
    ids=["ssm", "sliding-window"],
)
def test_event_metadata_skips_non_full_attention_group(
    sliding_window_size_in_chunks: int,
):
    tracker = _tracker()
    group_config = _group_config(
        sliding_window_size_in_chunks=sliding_window_size_in_chunks
    )
    req = _request(block_hashes=[_hash(i) for i in range(3)], token_count=12)
    keys = _record_chunks(tracker, req, group_config, num_chunks=3)
    _record_lookup_chunks(tracker, req, group_config, num_chunks=3)

    assert not tracker._pending_event_metadata

    events = list(tracker.take_events([_stored_event(keys[:1])]))
    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].block_size == 0


def test_pending_cpu_removal_consumes_hit_backfill_until_next_hit():
    tracker = _tracker()
    block_hashes = [_hash(0), _hash(1)]
    req = _request(block_hashes=block_hashes, token_count=8)
    group_config = _group_config(blocks_per_chunk=2)
    key = _record_chunks(tracker, req, group_config, num_chunks=1)[0]
    confirmed_meta = tracker._pending_event_metadata[key]
    lookup_req = _request(
        block_hashes=block_hashes,
        token_count=8,
        req_id="new-request",
    )

    tracker.record_lookup(
        lookup_req,
        group_config,
        0,
        key,
    )
    assert tracker._pending_event_metadata[key] is confirmed_meta

    list(
        tracker.take_events([_stored_event([key], Medium.STORAGE, ownership="custom")])
    )
    removed = list(tracker.take_events([_removed_event([key])]))
    assert len(removed) == 1
    assert removed[0].block_hashes == [
        _wire_hash(_hash(0)),
        _wire_hash(_hash(1)),
    ]

    stored = list(tracker.take_events([_stored_event([key])]))
    assert len(stored) == 1
    assert stored[0].block_size == 0
    assert stored[0].token_ids == []

    tracker.record_lookup(lookup_req, group_config, 0, key)
    removed = list(tracker.take_events([_removed_event([key])]))
    assert removed[0].block_hashes == [
        _wire_hash(_hash(0)),
        _wire_hash(_hash(1)),
    ]


@pytest.mark.parametrize(
    ("record_method", "position"),
    [("record_store", 0), ("record_partial_store", 4)],
)
def test_reoffload_preserves_secondary_residency(record_method, position):
    tracker, req, group_config, key = _lookup_chunk()
    [stored] = tracker.take_events(
        [
            _stored_event(
                [key],
                Medium.STORAGE,
                ownership="custom",
                removal_expected=True,
            )
        ]
    )

    getattr(tracker, record_method)(req, group_config, position, key)

    assert tracker._pending_event_metadata[key].active_residencies == {
        (Medium.CPU, None),
        (Medium.STORAGE, "custom"),
    }
    [removed] = tracker.take_events(
        [_removed_event([key], Medium.STORAGE, ownership="custom")]
    )

    assert stored.token_ids == [1, 2, 3, 4]
    assert stored.ownership == removed.ownership == "custom"
    assert key in tracker._pending_event_metadata


def test_take_events_groups_removed_hashes_by_kv_group():
    tracker = _tracker()
    group0_config = _group_config(group_idx=0, blocks_per_chunk=2)
    group1_config = _group_config(group_idx=1, blocks_per_chunk=2)
    req0 = _request(block_hashes=[_hash(0), _hash(1)], token_count=8)
    req1 = _request(block_hashes=[_hash(10), _hash(11)], token_count=8)
    key0 = _record_chunks(tracker, req0, group0_config, num_chunks=1)[0]
    key1 = _record_chunks(tracker, req1, group1_config, num_chunks=1)[0]

    removed = list(tracker.take_events([_removed_event([key0, key1])]))

    assert len(removed) == 2
    by_group = {event.group_idx: event.block_hashes for event in removed}
    assert by_group == {
        0: [_wire_hash(_hash(0)), _wire_hash(_hash(1))],
        1: [_wire_hash(_hash(10)), _wire_hash(_hash(11))],
    }


def test_take_events_supports_restore_after_eviction():
    block_size = 4
    tracker = _tracker()
    group_config = _group_config(block_size=block_size)
    req = _request(block_hashes=[_hash(0)], token_count=block_size)
    key = _record_chunks(tracker, req, group_config, num_chunks=1)[0]

    first_store = list(tracker.take_events([_stored_event([key])]))
    assert len(first_store) == 1
    assert isinstance(first_store[0], BlockStored)
    assert first_store[0].token_ids == [1, 2, 3, 4]

    removed = list(tracker.take_events([_removed_event([key])]))
    assert len(removed) == 1
    assert isinstance(removed[0], BlockRemoved)
    assert not tracker._pending_event_metadata

    req.all_token_ids = [5, 6, 7, 8]
    tracker.record_store(req, group_config, chunk_idx=0, offload_key=key)

    second_store = list(tracker.take_events([_stored_event([key])]))
    assert len(second_store) == 1
    assert isinstance(second_store[0], BlockStored)
    assert second_store[0].token_ids == [5, 6, 7, 8]


def test_reset_cache_clears_side_table():
    tracker = _tracker()
    group_config = _group_config()
    req = _request(block_hashes=[_hash(i) for i in range(3)], token_count=12)
    _record_lookup_chunks(tracker, req, group_config, num_chunks=3)

    assert tracker._pending_event_metadata

    tracker.reset()

    assert not tracker._pending_event_metadata


def test_tiering_accepts_self_describing_kv_events():
    vllm_config = create_vllm_config(
        block_size=4,
        max_num_batched_tokens=16,
        disable_hybrid_kv_cache_manager=False,
    )
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": 1 << 20,
            "self_describing_kv_events": True,
            "secondary_tiers": [{"type": "example"}],
        },
    )
    vllm_config.kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="null",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )

    spec = TieringOffloadingSpec(build_offloading_config(vllm_config, kv_cache_config))
    tracker = OffloadingEventsTracker(spec.kv_events_config)

    assert spec.kv_events_config.enable_kv_cache_events
    assert spec.kv_events_config.self_describing_kv_events
    assert tracker.self_describing_enabled
