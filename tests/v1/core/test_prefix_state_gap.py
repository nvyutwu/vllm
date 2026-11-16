# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid prefix-cache state-gap metrics: how far full-attention KV matched
past the resume point because a Mamba state was missing."""

from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, Counter, Histogram

from tests.v1.core.test_hybrid_resumable_cache import G84, Sim, tokens
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.cpu_test

R = G84.row  # 9,216-token MLA row; Sim retains a state every 2 rows


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _record(
    *,
    local_end: int,
    num_local: int,
    num_external: int = 0,
    external_end: int | None = None,
    interval: int | None = 2 * R,
    connector: bool = True,
    preempted: bool = False,
) -> PrefixCacheStats:
    stats = PrefixCacheStats()
    stub = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            prefix_cache_stats=stats,
            prefix_cache_lookup_enabled=lambda request: True,
            coordinator=SimpleNamespace(retention_interval=interval),
        ),
        connector=(
            SimpleNamespace(get_external_full_attention_hit_end=lambda r: external_end)
            if connector
            else None
        ),
    )
    request = SimpleNamespace(
        local_full_attention_hit_end=local_end,
        num_preemptions=int(preempted),
    )
    Scheduler._record_prefix_state_gaps(stub, request, num_local, num_external)
    return stats


def test_no_gap_records_nothing():
    stats = _record(local_end=3 * R, num_local=3 * R)
    assert stats.state_gap_tokens == {}
    assert stats.state_gap_requests == {}
    assert stats.state_gap_samples == {}


def test_local_gap_past_last_grid_state():
    # MLA matched 3.5 rows; resume at the grid state 2R: nothing kept was lost.
    stats = _record(local_end=3 * R + 4608, num_local=2 * R, connector=False)
    gap = R + 4608
    assert stats.state_gap_tokens == {
        "local/past_last_grid_state": gap,
        "combined/past_last_grid_state": gap,
    }
    assert stats.state_gap_requests == {
        "local/past_last_grid_state": 1,
        "combined/past_last_grid_state": 1,
    }
    assert stats.state_gap_samples == {"local": [gap], "combined": [gap]}


def test_local_gap_over_a_grid_state_is_missing_grid_state():
    # Grid state at 4R was evicted: resume falls back to 2R.
    stats = _record(local_end=5 * R, num_local=2 * R)
    assert stats.state_gap_tokens == {
        "local/missing_grid_state": 3 * R,
        "combined/missing_grid_state": 3 * R,
    }


def test_external_tier_fills_the_local_gap():
    # Device: MLA to 5R, state at 2R. Connector loads to 5R (MLA + state).
    stats = _record(
        local_end=5 * R, num_local=2 * R, num_external=3 * R, external_end=5 * R
    )
    assert stats.state_gap_tokens == {"local/missing_grid_state": 3 * R}
    assert "combined" not in stats.state_gap_samples


def test_external_gap_and_combined_take_the_deeper_tier():
    # Device: no gap at 2R. Connector: MLA to 7R, deepest state at 4R.
    stats = _record(
        local_end=2 * R, num_local=2 * R, num_external=2 * R, external_end=7 * R
    )
    assert stats.state_gap_tokens == {
        "external/missing_grid_state": 3 * R,
        "combined/missing_grid_state": 3 * R,
    }
    # External MLA found nothing past the local hit end of 6R.
    stats = _record(local_end=6 * R, num_local=2 * R, num_external=2 * R)
    assert stats.state_gap_tokens == {
        "local/missing_grid_state": 4 * R,
        "combined/missing_grid_state": 2 * R,
    }


@pytest.mark.parametrize(
    ("interval", "cause"),
    [(None, "missing_grid_state"), (0, "past_last_grid_state")],
)
def test_dense_and_semantic_only_retention(interval, cause):
    stats = _record(local_end=3 * R, num_local=R, interval=interval, connector=False)
    assert stats.state_gap_tokens == {
        f"local/{cause}": 2 * R,
        f"combined/{cause}": 2 * R,
    }


def test_preempted_requests_are_not_recorded():
    stats = _record(local_end=5 * R, num_local=2 * R, preempted=True)
    assert stats.state_gap_tokens == {}


def test_kv_cache_manager_reports_full_attention_hit_end():
    """End to end through the hybrid KV cache manager: evict the grid state at
    4R and the prompt-tail state; the revisit resumes at 2R while MLA still
    matches the whole first prompt."""
    sim = Sim(G84)
    first = tokens(5 * R + 2000, seed=1)
    req, _ = sim.run(first)
    states = sim.states(req)
    assert 2 * R in states and 4 * R in states
    for pos in states:
        if pos > 2 * R:
            sim.evict_state(req, pos)

    revisit = sim.request(first + tokens(3000, seed=2))
    hit = sim.admit(revisit)
    assert hit == 2 * R
    end = revisit.local_full_attention_hit_end
    # MLA still holds every full row of the first prompt.
    assert 5 * R <= end <= len(first)

    stub_stats = _record(local_end=end, num_local=hit, connector=False)
    assert stub_stats.state_gap_tokens == {
        "local/missing_grid_state": end - hit,
        "combined/missing_grid_state": end - hit,
    }


def test_kv_cache_manager_no_gap_when_state_present():
    sim = Sim(G84)
    first = tokens(3 * R + 500, seed=3)
    sim.run(first)
    revisit = sim.request(first + tokens(1000, seed=4))
    hit = sim.admit(revisit)
    assert hit > 2 * R
    assert revisit.local_full_attention_hit_end == hit


def test_state_gap_stats_survive_engine_core_serialization():
    stats = PrefixCacheStats()
    stats.record_state_gap("external", "missing_grid_state", 4096)
    stats.record_state_gap("external", "missing_grid_state", 1024)
    stats.record_state_gap("combined", "past_last_grid_state", 512)
    outputs = EngineCoreOutputs(
        scheduler_stats=SchedulerStats(prefix_cache_stats=stats)
    )
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(MsgpackEncoder().encode(outputs))
    got = decoded.scheduler_stats.prefix_cache_stats
    assert got.state_gap_tokens == {
        "external/missing_grid_state": 5120,
        "combined/past_last_grid_state": 512,
    }
    assert got.state_gap_requests == {
        "external/missing_grid_state": 2,
        "combined/past_last_grid_state": 1,
    }
    assert got.state_gap_samples == {"external": [4096, 1024], "combined": [512]}


def test_prometheus_observes_state_gaps():
    registry = CollectorRegistry()
    labels = ["model_name", "engine"]
    fake = SimpleNamespace(
        counter_prefix_cache_state_gap_tokens=Counter(
            "vllm:prefix_cache_state_gap_tokens",
            "",
            labels + ["tier", "cause"],
            registry=registry,
        ),
        counter_prefix_cache_state_gap_requests=Counter(
            "vllm:prefix_cache_state_gap_requests",
            "",
            labels + ["tier", "cause"],
            registry=registry,
        ),
        histogram_prefix_cache_state_gap=Histogram(
            "vllm:prefix_cache_state_gap_tokens_per_request",
            "",
            labels + ["tier"],
            buckets=[1024, 8192, 65536],
            registry=registry,
        ),
    )
    stats = PrefixCacheStats()
    stats.record_state_gap("local", "missing_grid_state", 9216)
    stats.record_state_gap("local", "missing_grid_state", 1000)
    PrometheusStatLogger._observe_prefix_cache_state_gaps(fake, stats, ["m", "0"])

    base = {"model_name": "m", "engine": "0", "tier": "local"}
    assert registry.get_sample_value(
        "vllm:prefix_cache_state_gap_tokens_total",
        {**base, "cause": "missing_grid_state"},
    ) == pytest.approx(10216)
    assert registry.get_sample_value(
        "vllm:prefix_cache_state_gap_requests_total",
        {**base, "cause": "missing_grid_state"},
    ) == pytest.approx(2)
    assert registry.get_sample_value(
        "vllm:prefix_cache_state_gap_tokens_per_request_bucket",
        {**base, "le": "1024.0"},
    ) == pytest.approx(1)
    assert registry.get_sample_value(
        "vllm:prefix_cache_state_gap_tokens_per_request_count", base
    ) == pytest.approx(2)


def test_prometheus_skips_state_gaps_without_hybrid_metrics():
    # Non-hybrid models register no state-gap metrics; observing is a no-op.
    fake = SimpleNamespace(
        counter_prefix_cache_state_gap_tokens=None,
        counter_prefix_cache_state_gap_requests=None,
        histogram_prefix_cache_state_gap=None,
    )
    stats = PrefixCacheStats()
    stats.record_state_gap("local", "missing_grid_state", 9216)
    PrometheusStatLogger._observe_prefix_cache_state_gaps(fake, stats, ["m", "0"])
