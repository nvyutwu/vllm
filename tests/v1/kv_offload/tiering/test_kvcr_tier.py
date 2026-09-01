# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection, Iterable, Mapping
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("kvcr")

from kvcr import KVCRBindings
from kvcr.config import G3Options, KVCRBackendConfigs, KVCRConfig, KVCRGuardConfig
from kvcr.policy import FIFOPolicy, G3FIFOPolicy, G3LRUPolicy, LRUPolicy
from kvcr.types import (
    BlockKey,
    CacheTier,
    InventoryEvent,
    MemDescriptor,
    OpEntryResult,
    OpEntryStatus,
    OpHandle,
    QueryStatus,
)

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    ExternalKVSourceState,
    LookupResult,
    Medium,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob
from vllm.v1.kv_offload.tiering.kvcr import manager as kvcr_manager
from vllm.v1.kv_offload.tiering.kvcr.manager import KVCRSecondaryTierManager


def _op_entries(
    entries: Mapping[BlockKey, bool],
) -> dict[BlockKey, OpEntryResult]:
    return {
        key: OpEntryResult(OpEntryStatus.SUCCESS if success else OpEntryStatus.FAILED)
        for key, success in entries.items()
    }


class _StubControlChannel:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def send(self, endpoint: str, message: bytes) -> bool:
        return True

    def recv(self) -> list[bytes]:
        return []

    def close(self) -> None:
        pass


class RecordingKVCR:
    def __init__(self) -> None:
        self.config: KVCRConfig | None = None
        self.guard_config: KVCRGuardConfig | None = None
        self.backend_configs = KVCRBackendConfigs()
        self.constructor_bindings: KVCRBindings | None = None
        self.nixl_agent_name = "recording"
        self.framework_control: _StubControlChannel | None = None
        self.inventory_sink = None
        self.query_status = QueryStatus.MISS
        self.stats: OffloadingConnectorStats | None = None
        self.submit_hint_calls: list[
            tuple[
                list[BlockKey],
                str | None,
                str,
                object | None,
                str | None,
                int | None,
            ]
        ] = []
        self.discard_hint_calls: list[str] = []
        self.deliver_calls: list[
            tuple[OpHandle, dict[BlockKey, MemDescriptor], str | None]
        ] = []
        self.deposit_calls: list[tuple[OpHandle, dict[BlockKey, MemDescriptor]]] = []
        self.completed: list[tuple[OpHandle, dict[BlockKey, OpEntryResult]]] = []
        self._next_op_handle = 1
        self.closed = False

    def submit_hint(
        self,
        block_key_list: Collection[BlockKey],
        src: str | None = None,
        mode: str = "copy",
        hints: object | None = None,
        request_id: str | None = None,
        source_inventory_epoch: int | None = None,
    ) -> None:
        self.submit_hint_calls.append(
            (
                list(block_key_list),
                src,
                mode,
                hints,
                request_id,
                source_inventory_epoch,
            )
        )

    def discard_hint(self, request_id: str) -> None:
        self.discard_hint_calls.append(request_id)

    def query(
        self,
        keys: Collection[BlockKey],
        request_id: str | None = None,
    ) -> list[tuple[QueryStatus, CacheTier | None]]:
        tier = {
            QueryStatus.HIT: CacheTier.LOCAL_G2,
            QueryStatus.FETCHING: CacheTier.LOCAL_G2,
            QueryStatus.FETCHABLE: CacheTier.REMOTE_G2,
            QueryStatus.MISS: None,
        }[self.query_status]
        return [(self.query_status, tier) for _ in keys]

    def deliver(
        self,
        blocks: Mapping[BlockKey, MemDescriptor],
        request_id: str | None = None,
    ) -> OpHandle:
        op_handle = self._next_op_handle
        self._next_op_handle += 1
        self.deliver_calls.append((op_handle, dict(blocks), request_id))
        self.complete(op_handle, {key: True for key in blocks})
        return op_handle

    def deposit(
        self,
        blocks: Mapping[BlockKey, MemDescriptor],
    ) -> OpHandle:
        op_handle = self._next_op_handle
        self._next_op_handle += 1
        self.deposit_calls.append((op_handle, dict(blocks)))
        self.complete(op_handle, {key: True for key in blocks})
        return op_handle

    def complete(
        self,
        op_handle: OpHandle,
        entries: Mapping[BlockKey, bool],
    ) -> None:
        self.completed.append((op_handle, _op_entries(entries)))

    def poll_completed(
        self,
    ) -> Iterable[tuple[OpHandle, dict[BlockKey, OpEntryResult]]]:
        completed = self.completed
        self.completed = []
        return completed

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self.stats
        self.stats = None
        return stats

    def close(self) -> None:
        self.closed = True


class _ExternalPolicy(FIFOPolicy):
    pass


def _job(
    job_id: int,
    req_context: ReqContext,
    key: OffloadKey | None = None,
    block_id: int = 0,
) -> TransferJob:
    return TransferJob(
        job_id=job_id,
        keys=[key if key is not None else OffloadKey(b"k0")],
        block_ids=np.array([block_id], dtype=np.int64),
        is_promotion=True,
        req_context=req_context,
    )


def _make_tier(
    monkeypatch,
    kvcr: RecordingKVCR,
    *,
    enable_telemetry: bool = False,
    secondary_g2_slots: int = 0,
    kvcr_service_socket_path: str | None = None,
    compatibility_digest: str | None = None,
    enable_kv_cache_events: bool = False,
    self_describing_kv_events: bool = False,
    policy: str | None = None,
    g3: dict[str, object] | None = None,
    control_ports: list[int] | None = None,
    data_parallel_rank_local: int | None = None,
    inventory_epoch: int | None = None,
    operation_timeout_ms: int = 1000,
    drain_timeout_ms: int | None = None,
) -> KVCRSecondaryTierManager:
    def make_control(_bind_host, bind_port, advertise_host):
        return _StubControlChannel(f"tcp://{advertise_host}:{int(bind_port)}")

    def make_kvcr(config, bindings, backend_configs, guard_config):
        kvcr.config = config
        kvcr.guard_config = guard_config
        kvcr.nixl_agent_name = config.nixl_agent_name
        kvcr.backend_configs = backend_configs
        kvcr.constructor_bindings = bindings
        kvcr.framework_control = bindings.framework_control
        kvcr.inventory_sink = bindings.inventory_sink
        return kvcr

    monkeypatch.setattr(kvcr_manager, "KVCR", make_kvcr)
    monkeypatch.setattr(kvcr_manager, "ZmqPeerControlChannel", make_control)
    return KVCRSecondaryTierManager(
        offloading_spec=SimpleNamespace(
            config=SimpleNamespace(
                parallel=SimpleNamespace(
                    data_parallel_rank_local=data_parallel_rank_local,
                )
            ),
            kv_events_config=SimpleNamespace(
                enable_kv_cache_events=enable_kv_cache_events,
                self_describing_kv_events=self_describing_kv_events,
            ),
        ),
        primary_kv_view=memoryview(np.zeros((4, 16), dtype=np.int8)),
        tier_type="kvcr",
        router_capabilities=["router_hint"],
        control_host="127.0.0.1",
        control_ports=control_ports if control_ports is not None else [7777],
        control_advertise_host="127.0.0.1",
        enable_telemetry=enable_telemetry,
        secondary_g2_slots=secondary_g2_slots,
        kvcr_service_socket_path=kvcr_service_socket_path,
        compatibility_digest=compatibility_digest,
        policy=policy,
        g3=g3,
        inventory_epoch=inventory_epoch,
        operation_timeout_ms=operation_timeout_ms,
        drain_timeout_ms=drain_timeout_ms,
    )


def test_kvcr_tier_configures_service_for_local_dp_rank(monkeypatch):
    """Keep the control endpoint and guard pool aligned to the local DP rank."""
    kvcr = RecordingKVCR()
    tier = _make_tier(
        monkeypatch,
        kvcr,
        kvcr_service_socket_path="/tmp/kvcr.sock",
        compatibility_digest="Opaque-Digest",
        secondary_g2_slots=1,
        control_ports=[7001, 7002],
        data_parallel_rank_local=1,
        inventory_epoch=8080,
    )

    assert kvcr.framework_control is not None
    assert kvcr.framework_control.endpoint == "tcp://127.0.0.1:7002"
    assert kvcr.guard_config == KVCRGuardConfig(
        kvcr_service_socket_path="/tmp/kvcr.sock",
        pool_index=1,
        row_stride=tier._primary_row_stride,
        compatibility_digest="Opaque-Digest",
    )
    assert kvcr.backend_configs.local_dram is None
    assert kvcr.config is not None
    assert kvcr.config.inventory_epoch == 8080


def test_kvcr_tier_converts_g3_paths(monkeypatch, tmp_path):
    """Convert user-provided G3 paths to KVCR's typed configuration."""
    kvcr = RecordingKVCR()
    path = tmp_path / "g3.data"

    _make_tier(
        monkeypatch,
        kvcr,
        g3={"paths": [str(path)], "capacity_bytes_per_file": 64},
    )

    assert kvcr.backend_configs.g3 == G3Options(
        paths=(path,),
        capacity_bytes_per_file=64,
    )


def test_kvcr_tier_maps_router_hint_to_load(monkeypatch):
    """Exercise the complete vLLM router-hint-to-KVCR load translation."""
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    router_hint = {
        "source_control_endpoint": "tcp://source:1234",
        "source_inventory_epoch": 77,
        "block_hashes": [121, 122, 123, 124],
        "start_block": 2,
        "hinted_blocks": 1,
    }
    ctx = ReqContext(req_id="req", kv_transfer_params={"router_hint": router_hint})
    key = make_offload_key((123).to_bytes(8, "big"), 0)
    same_hash_other_group = make_offload_key((123).to_bytes(8, "big"), 7)
    other_key = make_offload_key((124).to_bytes(8, "big"), 0)

    tier.on_new_request(ctx)
    assert len(kvcr.submit_hint_calls) == 1
    _, source, mode, hint, request_id, source_epoch = kvcr.submit_hint_calls[0]
    assert (source, mode, request_id, source_epoch) == (
        "tcp://source:1234",
        "copy",
        "req",
        77,
    )
    bindings = kvcr.constructor_bindings
    assert bindings is not None
    assert bindings.key_hint_adapter is not None
    assert bindings.key_hint_adapter.matches(BlockKey(bytes(key)), hint)
    assert bindings.key_hint_adapter.matches(
        BlockKey(bytes(same_hash_other_group)), hint
    )
    assert not bindings.key_hint_adapter.matches(BlockKey(bytes(other_key)), hint)
    outside_slice = make_offload_key((124).to_bytes(8, "big"), 0)
    assert not bindings.key_hint_adapter.matches(BlockKey(bytes(outside_slice)), hint)

    kvcr.query_status = QueryStatus.HIT
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert tier.lookup(key, ctx) is LookupResult.HIT
    hint_stats = tier.get_stats()
    assert hint_stats is not None
    assert hint_stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_already_local": 1,
    }

    tier.submit_load(_job(7, ctx, key=key, block_id=2))

    assert len(kvcr.submit_hint_calls) == 1
    _, blocks, request_id = kvcr.deliver_calls[0]
    assert request_id == "req"
    assert list(blocks) == [key]
    assert blocks[key].end_point_name == kvcr.nixl_agent_name
    assert blocks[key].addr == tier._primary_base_addr + 2 * 16
    assert blocks[key].size == 16
    assert list(tier.get_finished_jobs()) == [JobResult(7, True)]

    tier.on_request_finished(ctx)
    assert kvcr.discard_hint_calls == ["req"]


def test_kvcr_tier_allows_request_without_router_hint(monkeypatch):
    """Keep router hints optional for requests from non-hint-aware routers."""
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr)

    tier.on_new_request(ReqContext(req_id="req"))

    assert kvcr.submit_hint_calls == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (QueryStatus.MISS, LookupResult.MISS),
        (QueryStatus.HIT, LookupResult.HIT),
        (QueryStatus.FETCHABLE, LookupResult.HIT),
        (QueryStatus.FETCHING, LookupResult.RETRY),
    ],
)
def test_kvcr_tier_maps_query_status(monkeypatch, status, expected):
    """Map KVCR cache states to the scheduler's lookup contract."""
    kvcr = RecordingKVCR()
    kvcr.query_status = status
    tier = _make_tier(monkeypatch, kvcr)

    assert tier.lookup(OffloadKey(b"k0"), ReqContext(req_id="req")) is expected


def test_kvcr_hint_reconciliation_conserves_h_equals_a_plus_n(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123, 124],
                "start_block": 0,
                "hinted_blocks": 2,
            }
        },
    )
    ctx.set_state(ExternalKVSourceState())
    local_key = make_offload_key((123).to_bytes(8, "big"), 0)
    remote_key = make_offload_key((124).to_bytes(8, "big"), 0)

    tier.on_new_request(ctx)
    kvcr.query_status = QueryStatus.HIT
    assert tier.lookup(local_key, ctx) is LookupResult.HIT
    kvcr.query_status = QueryStatus.MISS
    assert tier.lookup(remote_key, ctx) is LookupResult.MISS
    tier.on_request_finished(ctx)

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 2,
        "vllm:kvcr_blocks_already_local": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_source_validation_started": 1,
        "vllm:kvcr_source_blocks_missing:('source_missing',)": 1,
    }
    sources = ctx.get_state(ExternalKVSourceState)
    assert sources is not None
    assert sources.lookup_sources[local_key] == "local_cpu"


def test_kvcr_counts_framework_cpu_hit_as_already_local(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )
    key = make_offload_key((123).to_bytes(8, "big"), 0)

    tier.on_new_request(ctx)
    tier.record_primary_hit(key, ctx)
    tier.on_request_finished(ctx)

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_already_local": 1,
    }


def test_kvcr_remote_promotion_and_use_are_logical_and_conserving(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )
    ctx.set_state(ExternalKVSourceState())
    key = make_offload_key((123).to_bytes(8, "big"), 0)
    same_hash_other_group = make_offload_key((123).to_bytes(8, "big"), 1)

    tier.on_new_request(ctx)
    kvcr.query_status = QueryStatus.FETCHABLE
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert tier.lookup(same_hash_other_group, ctx) is LookupResult.HIT
    sources = ctx.get_state(ExternalKVSourceState)
    assert sources is not None
    assert sources.lookup_sources[key] == "kvcr_p2p"
    tier.submit_load(
        TransferJob(
            job_id=17,
            keys=[key, same_hash_other_group],
            block_ids=np.array([0, 1], dtype=np.int64),
            is_promotion=True,
            req_context=ctx,
        )
    )
    assert list(tier.get_finished_jobs()) == [JobResult(17, True)]
    tier.record_blocks_used([key, same_hash_other_group], ctx)
    tier.on_request_finished(ctx)

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_source_validation_started": 1,
        "vllm:kvcr_blocks_promoted": 1,
        "vllm:kvcr_blocks_used": 1,
    }


def test_kvcr_logical_metrics_exclude_opportunistic_non_hint_blocks(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )
    ctx.set_state(ExternalKVSourceState())
    hinted = make_offload_key((123).to_bytes(8, "big"), 0)
    opportunistic = make_offload_key((999).to_bytes(8, "big"), 0)

    tier.on_new_request(ctx)
    kvcr.query_status = QueryStatus.FETCHABLE
    assert tier.lookup(hinted, ctx) is LookupResult.HIT
    sources = ctx.get_state(ExternalKVSourceState)
    assert sources is not None
    sources.lookup_sources[opportunistic] = "kvcr_p2p"
    tier.submit_load(
        TransferJob(
            job_id=17,
            keys=[hinted, opportunistic],
            block_ids=np.array([0, 1], dtype=np.int64),
            is_promotion=True,
            req_context=ctx,
        )
    )
    _, delivered, request_id = kvcr.deliver_calls[-1]
    assert request_id == "req"
    assert set(delivered) == {
        BlockKey(bytes(hinted)),
        BlockKey(bytes(opportunistic)),
    }
    assert list(tier.get_finished_jobs()) == [JobResult(17, True)]
    tier.record_blocks_used([hinted, opportunistic], ctx)
    tier.on_request_finished(ctx)

    # The transport may opportunistically move more data, but logical metrics
    # describe only the router opportunity represented by the hint.
    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_source_validation_started": 1,
        "vllm:kvcr_blocks_promoted": 1,
        "vllm:kvcr_blocks_used": 1,
    }


def test_kvcr_pure_opportunistic_job_does_not_hold_hint_lifecycle(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )
    sources = ExternalKVSourceState()
    ctx.set_state(sources)
    opportunistic = make_offload_key((999).to_bytes(8, "big"), 0)
    sources.lookup_sources[opportunistic] = "kvcr_p2p"

    tier.on_new_request(ctx)
    tier.submit_load(
        TransferJob(
            job_id=18,
            keys=[opportunistic],
            block_ids=np.array([0], dtype=np.int64),
            is_promotion=True,
            req_context=ctx,
        )
    )
    tier.on_request_finished(ctx)
    assert "req" not in tier._hint_metric_states
    assert list(tier.get_finished_jobs()) == [JobResult(18, True)]

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_blocks_cancelled:('before_source_validation',)": 1,
    }


def test_kvcr_partial_physical_groups_do_not_promote_logical_block(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )
    ctx.set_state(ExternalKVSourceState())
    key = make_offload_key((123).to_bytes(8, "big"), 0)
    other_group = make_offload_key((123).to_bytes(8, "big"), 1)

    tier.on_new_request(ctx)
    kvcr.query_status = QueryStatus.FETCHABLE
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert tier.lookup(other_group, ctx) is LookupResult.HIT
    tier.submit_load(
        TransferJob(
            job_id=17,
            keys=[key, other_group],
            block_ids=np.array([0, 1], dtype=np.int64),
            is_promotion=True,
            req_context=ctx,
        )
    )
    op_handle, _, _ = kvcr.deliver_calls[-1]
    kvcr.completed = [
        (
            op_handle,
            _op_entries(
                {
                    BlockKey(bytes(key)): True,
                    BlockKey(bytes(other_group)): False,
                }
            ),
        )
    ]

    assert list(tier.get_finished_jobs()) == [
        JobResult(17, False, successful_keys={key})
    ]
    tier.on_request_finished(ctx)

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_source_validation_started": 1,
    }


def test_kvcr_failed_then_successful_retry_promotes_realized_outcome(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )
    ctx.set_state(ExternalKVSourceState())
    key = make_offload_key((123).to_bytes(8, "big"), 0)

    tier.on_new_request(ctx)
    kvcr.query_status = QueryStatus.FETCHABLE
    assert tier.lookup(key, ctx) is LookupResult.HIT
    first = TransferJob(
        job_id=17,
        keys=[key],
        block_ids=np.array([0], dtype=np.int64),
        is_promotion=True,
        req_context=ctx,
    )
    tier.submit_load(first)
    first_op, _, _ = kvcr.deliver_calls[-1]
    kvcr.completed = [(first_op, _op_entries({BlockKey(bytes(key)): False}))]
    assert list(tier.get_finished_jobs()) == [
        JobResult(17, False, successful_keys=set())
    ]

    tier.submit_load(
        TransferJob(
            job_id=18,
            keys=[key],
            block_ids=np.array([0], dtype=np.int64),
            is_promotion=True,
            req_context=ctx,
        )
    )
    assert list(tier.get_finished_jobs()) == [JobResult(18, True)]
    tier.on_request_finished(ctx)

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_source_validation_started": 1,
        "vllm:kvcr_blocks_promoted": 1,
        "vllm:kvcr_blocks_cancelled:('after_promotion',)": 1,
    }


def test_kvcr_remote_promotion_not_used_is_cancelled_after_promotion(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )
    ctx.set_state(ExternalKVSourceState())
    key = make_offload_key((123).to_bytes(8, "big"), 0)

    tier.on_new_request(ctx)
    kvcr.query_status = QueryStatus.FETCHABLE
    assert tier.lookup(key, ctx) is LookupResult.HIT
    tier.submit_load(
        TransferJob(
            job_id=17,
            keys=[key],
            block_ids=np.array([0], dtype=np.int64),
            is_promotion=True,
            req_context=ctx,
        )
    )
    assert list(tier.get_finished_jobs()) == [JobResult(17, True)]
    tier.on_request_finished(ctx)

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_source_validation_started": 1,
        "vllm:kvcr_blocks_promoted": 1,
        "vllm:kvcr_blocks_cancelled:('after_promotion',)": 1,
    }


def test_kvcr_destination_capacity_decline_is_counted_once(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )
    ctx.set_state(ExternalKVSourceState())
    key = make_offload_key((123).to_bytes(8, "big"), 0)

    tier.on_new_request(ctx)
    kvcr.query_status = QueryStatus.FETCHABLE
    assert tier.lookup(key, ctx) is LookupResult.HIT
    tier.record_promotion_allocation_failure(key, ctx)
    tier.record_promotion_allocation_failure(key, ctx)
    tier.on_request_finished(ctx)

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_blocks_policy_declined:('destination_capacity',)": 1,
    }


def test_kvcr_declined_logical_block_excludes_other_physical_group_attempt(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )
    ctx.set_state(ExternalKVSourceState())
    key = make_offload_key((123).to_bytes(8, "big"), 0)
    other_group = make_offload_key((123).to_bytes(8, "big"), 1)

    tier.on_new_request(ctx)
    kvcr.query_status = QueryStatus.FETCHABLE
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert tier.lookup(other_group, ctx) is LookupResult.HIT
    tier.record_promotion_allocation_failure(other_group, ctx)
    tier.submit_load(
        TransferJob(
            job_id=17,
            keys=[key],
            block_ids=np.array([0], dtype=np.int64),
            is_promotion=True,
            req_context=ctx,
        )
    )
    assert list(tier.get_finished_jobs()) == [JobResult(17, True)]
    tier.on_request_finished(ctx)

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_blocks_policy_declined:('destination_capacity',)": 1,
    }


def test_kvcr_declined_logical_block_excludes_other_physical_group_miss(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )
    ctx.set_state(ExternalKVSourceState())
    key = make_offload_key((123).to_bytes(8, "big"), 0)
    other_group = make_offload_key((123).to_bytes(8, "big"), 1)

    tier.on_new_request(ctx)
    kvcr.query_status = QueryStatus.FETCHABLE
    assert tier.lookup(key, ctx) is LookupResult.HIT
    tier.record_promotion_allocation_failure(key, ctx)
    kvcr.query_status = QueryStatus.MISS
    assert tier.lookup(other_group, ctx) is LookupResult.MISS
    tier.on_request_finished(ctx)

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_blocks_policy_declined:('destination_capacity',)": 1,
    }


def test_kvcr_started_logical_block_excludes_other_group_capacity_decline(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )
    ctx.set_state(ExternalKVSourceState())
    key = make_offload_key((123).to_bytes(8, "big"), 0)
    other_group = make_offload_key((123).to_bytes(8, "big"), 1)

    tier.on_new_request(ctx)
    kvcr.query_status = QueryStatus.FETCHABLE
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert tier.lookup(other_group, ctx) is LookupResult.HIT
    tier.submit_load(
        TransferJob(
            job_id=17,
            keys=[key],
            block_ids=np.array([0], dtype=np.int64),
            is_promotion=True,
            req_context=ctx,
        )
    )
    tier.record_promotion_allocation_failure(other_group, ctx)
    assert list(tier.get_finished_jobs()) == [JobResult(17, True)]
    tier.record_blocks_used([key], ctx)
    tier.on_request_finished(ctx)

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_source_validation_started": 1,
        "vllm:kvcr_blocks_promoted": 1,
        "vllm:kvcr_blocks_used": 1,
    }


def test_kvcr_unattempted_hint_is_pre_validation_cancellation(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={
            "router_hint": {
                "source_control_endpoint": "tcp://source:1234",
                "source_inventory_epoch": 7,
                "block_hashes": [123],
                "start_block": 0,
                "hinted_blocks": 1,
            }
        },
    )

    tier.on_new_request(ctx)
    tier.on_request_finished(ctx)

    stats = tier.get_stats()
    assert stats is not None
    assert stats.reduce() == {
        "vllm:kvcr_hint_blocks_received": 1,
        "vllm:kvcr_blocks_remote_needed": 1,
        "vllm:kvcr_blocks_cancelled:('before_source_validation',)": 1,
    }


def test_kvcr_tier_serves_primary_pin_request(monkeypatch):
    """Hold primary-tier hits until KVCR releases the corresponding pin."""
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr)
    keys = (BlockKey(b"k0"), BlockKey(b"k1"), BlockKey(b"k2"))
    hit_keys = (keys[0], keys[2])
    block_ids = {keys[0]: 1, keys[2]: 5}
    lifecycle: list[str] = []

    class Parent:
        def on_new_request(self, req_context):
            lifecycle.append("new")
            return SimpleNamespace()

        def lookup(self, key, req_context):
            return LookupResult.HIT if key in hit_keys else LookupResult.MISS

        def create_store_job(self, requested_keys, req_context):
            return TransferJob(
                job_id=11,
                keys=requested_keys,
                block_ids=np.array([block_ids[key] for key in requested_keys]),
                is_promotion=True,
                req_context=req_context,
            )

        def on_request_finished(self, req_context):
            lifecycle.append("finished")

    bindings = kvcr.constructor_bindings
    assert bindings is not None
    request = bindings.request_pin(keys)

    tier.serve_external_requests(Parent())

    [(queued_request, result)] = bindings.poll_pin_results()
    assert queued_request == request
    assert result is not None
    pin_handle, descriptors = result
    assert descriptors[keys[1]] is None
    descriptor = descriptors[keys[2]]
    assert descriptor is not None
    assert descriptor.addr == (tier._primary_base_addr + 5 * tier._primary_row_stride)
    assert lifecycle == ["new", "finished"]

    polls = 0

    def release_pin():
        nonlocal polls
        polls += 1
        bindings.release_pin(pin_handle)
        return []

    monkeypatch.setattr(kvcr, "poll_completed", release_pin)
    tier.drain_jobs()

    assert polls == 1
    assert list(tier.get_finished_jobs()) == [JobResult(11, True)]


def test_kvcr_telemetry_is_opt_in_and_namespaced_at_vllm_boundary(monkeypatch):
    """Keep telemetry opt-in and namespace metrics only at the vLLM boundary."""
    assert KVCRSecondaryTierManager.build_metric_definitions({}) == {}
    definitions = KVCRSecondaryTierManager.build_metric_definitions(
        {"enable_telemetry": True}
    )
    assert "vllm:kvcr_duration_seconds" in definitions
    assert "vllm:kvcr_transfer_blocks" in definitions
    assert "vllm:kvcr_transfer_blocks_submitted" in definitions
    assert "vllm:kvcr_transfer_blocks_failed" in definitions
    assert "vllm:kvcr_source_blocks_available" in definitions
    assert "vllm:kvcr_source_blocks_missing" in definitions
    assert "vllm:kvcr_blocks_cancelled" in definitions
    assert "vllm:kvcr_hint_blocks_received" in definitions
    assert "vllm:kvcr_blocks_already_local" in definitions
    assert "vllm:kvcr_blocks_remote_needed" in definitions
    assert "vllm:kvcr_source_validation_started" in definitions
    assert "vllm:kvcr_blocks_policy_declined" in definitions
    assert "vllm:kvcr_blocks_promoted" in definitions
    assert "vllm:kvcr_blocks_promotion_failed" not in definitions
    assert "vllm:kvcr_blocks_used" in definitions

    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    bindings = kvcr.constructor_bindings
    assert bindings is not None
    assert bindings.stats_factory is not None
    stats = bindings.stats_factory()
    stats.increase_counter(
        "kvcr_transfer_blocks",
        2,
        ("remote_deliver",),
    )
    kvcr.stats = stats

    returned = tier.get_stats()

    assert returned is stats
    assert returned.reduce() == {"vllm:kvcr_transfer_blocks:('remote_deliver',)": 2}


@pytest.mark.parametrize(
    ("policy", "expected_type"),
    [
        ("fifo", FIFOPolicy),
        ("lru", LRUPolicy),
        ("g3_fifo", G3FIFOPolicy),
        ("g3_lru", G3LRUPolicy),
        (f"{__name__}._ExternalPolicy", _ExternalPolicy),
    ],
)
def test_kvcr_tier_passes_selected_policy(monkeypatch, policy, expected_type):
    """Resolve every built-in and fully qualified external policy."""
    kvcr = RecordingKVCR()
    _make_tier(monkeypatch, kvcr, policy=policy)

    bindings = kvcr.constructor_bindings
    assert bindings is not None
    assert type(bindings.policy) is expected_type


@pytest.mark.parametrize(
    ("socket_path", "digest"),
    [("/tmp/kvcr.sock", None), (None, "Opaque-Digest")],
)
def test_kvcr_tier_requires_complete_service_config(monkeypatch, socket_path, digest):
    """Reject partial guard configuration before connecting to the service."""
    with pytest.raises(ValueError, match="configured together"):
        _make_tier(
            monkeypatch,
            RecordingKVCR(),
            kvcr_service_socket_path=socket_path,
            compatibility_digest=digest,
        )


def test_kvcr_tier_stores_and_emits_inventory(monkeypatch):
    """Cover store descriptors and inventory translation at the KVCR boundary."""
    kvcr = RecordingKVCR()
    tier = _make_tier(
        monkeypatch,
        kvcr,
        secondary_g2_slots=2,
        enable_kv_cache_events=True,
        self_describing_kv_events=True,
    )
    local_dram = kvcr.backend_configs.local_dram
    assert local_dram is not None
    assert local_dram.length == 2 * tier._primary_row_stride
    assert local_dram.slot_count == 2

    key = OffloadKey(b"k0")
    tier.submit_store(_job(11, ReqContext(req_id="req"), key=key, block_id=2))

    _, blocks = kvcr.deposit_calls[0]
    assert blocks[key].addr == tier._primary_base_addr + 2 * 16
    assert list(tier.get_finished_jobs()) == [JobResult(11, True)]

    assert kvcr.inventory_sink is not None
    kvcr.inventory_sink(
        InventoryEvent((BlockKey(bytes(key)),), CacheTier.LOCAL_G2, False)
    )
    kvcr.inventory_sink(InventoryEvent((BlockKey(bytes(key)),), CacheTier.G3, False))
    assert [
        (event.keys, event.medium, event.ownership, event.removed)
        for event in tier.take_events()
    ] == [
        ([key], Medium.CPU, "kvcr", False),
        ([key], Medium.STORAGE, "kvcr", False),
    ]
    tier.shutdown()


def test_kvcr_tier_requires_self_describing_inventory_events(monkeypatch):
    """Require tier-aware events whenever KVCR owns local cache inventory."""
    with pytest.raises(ValueError, match="self_describing_kv_events"):
        _make_tier(
            monkeypatch,
            RecordingKVCR(),
            secondary_g2_slots=1,
            enable_kv_cache_events=True,
        )


def test_kvcr_tier_waits_for_all_completions_and_drains(monkeypatch):
    """Wait for every block result and preserve partial success while draining."""

    class DrainingKVCR(RecordingKVCR):
        def __init__(self):
            super().__init__()
            self.polls: list[list[tuple[OpHandle, dict[BlockKey, OpEntryResult]]]] = []

        def deliver(
            self,
            blocks: Mapping[BlockKey, MemDescriptor],
            request_id: str | None = None,
        ) -> OpHandle:
            op_handle = self._next_op_handle
            self._next_op_handle += 1
            self.deliver_calls.append((op_handle, dict(blocks), request_id))
            keys = list(blocks)
            self.polls = [
                [(op_handle, _op_entries({keys[0]: True}))],
                [(op_handle, _op_entries({keys[1]: False}))],
            ]
            return op_handle

        def poll_completed(
            self,
        ) -> Iterable[tuple[OpHandle, dict[BlockKey, OpEntryResult]]]:
            return self.polls.pop(0) if self.polls else []

    kvcr = DrainingKVCR()
    tier = _make_tier(monkeypatch, kvcr)
    keys = [OffloadKey(b"k0"), OffloadKey(b"k1")]
    tier.submit_load(
        TransferJob(
            job_id=13,
            keys=keys,
            block_ids=np.array([0, 1], dtype=np.int64),
            is_promotion=True,
            req_context=ReqContext(req_id="req"),
        )
    )

    assert list(tier.get_finished_jobs()) == []
    tier.drain_jobs()

    [result] = tier.get_finished_jobs()
    assert result.job_id == 13
    assert not result.success
    assert result.successful_keys == {keys[0]}


def test_kvcr_tier_drain_is_bounded_under_peer_loss(monkeypatch):
    class StuckKVCR(RecordingKVCR):
        def deliver(self, blocks, request_id=None):
            op_handle = self._next_op_handle
            self._next_op_handle += 1
            self.deliver_calls.append((op_handle, dict(blocks), request_id))
            return op_handle

    kvcr = StuckKVCR()
    tier = _make_tier(
        monkeypatch,
        kvcr,
        operation_timeout_ms=5,
        drain_timeout_ms=10,
    )
    tier.submit_load(_job(23, ReqContext(req_id="req")))

    with pytest.raises(TimeoutError, match="KVCR drain timed out"):
        tier.drain_jobs()

    tier.shutdown()
    assert kvcr.closed
