# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the zero-reclaim preemption cascade.

Companion to test_deferred_block_free.py. That file tests the *defer mechanism*
(by calling _preempt_request directly). This file tests the **running-extend
allocation retry loop**: when the policy-selected victim's free is deferred
(same-step), the loop must not chain-preempt it (zero-reclaim cascade).

The retry loop is forced deterministically by patching allocate_slots to fail
after the running set is established (standard control-flow unit test; no GPU,
no exact-saturation recreation).
"""

import os
from unittest.mock import patch

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

MODEL = os.environ.get("VLLM_TEST_DEFER_FREE_MODEL", "facebook/opt-125m")
STOP_TOKEN_ID = 42
BLOCK_SIZE = 16
NUM_PROMPT_TOKENS = 33  # 3 blocks @ block_size=16


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
    token_id: int = 0,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _deferring_scheduler(num_blocks=1000, max_num_seqs=16):
    """Async scheduler with deferred block freeing forced on (the production
    gate additionally requires a KV-consumer connector; the mechanism itself is
    independent of it, like test_deferred_block_free._create_deferring_scheduler)."""
    s = create_scheduler(
        model=MODEL,
        async_scheduling=True,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=False,
    )
    s.defer_block_free = True
    return s


def _running_same_step(scheduler, n_requests, max_tokens=5):
    """Schedule depth-1 async (prefill + 1 over-scheduled decode) so every
    running request has last_sched_seq > processed_step_seq -> its free is
    deferred this step."""
    reqs = create_requests(
        num_requests=n_requests,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=max_tokens,
        stop_token_ids=[STOP_TOKEN_ID],
    )
    for r in reqs:
        scheduler.add_request(r)
    scheduler.schedule()  # prefill (step 1)
    scheduler.schedule()  # 1 decode (step 2 in flight) -> all same-step
    running = list(scheduler.running)
    assert running
    for r in running:
        assert r.last_sched_seq > scheduler.processed_step_seq
    return running


def _force_retry_loop(scheduler):
    """Patch allocate_slots to fail, run one schedule() -> forces the
    running-extend preempt retry loop. Returns the SchedulerOutput."""
    with patch.object(scheduler.kv_cache_manager, "allocate_slots", return_value=None):
        return scheduler.schedule()


def test_zero_reclaim_guard_preserves_trigger_state():
    """A deferred (zero-reclaim) victim is NOT preempted, and the trigger's
    scheduler state stays fully consistent (no half-schedule, no spurious
    output, no bookkeeping corruption)."""
    s = _deferring_scheduler()
    running = _running_same_step(s, n_requests=6)
    trigger = s.running[0]
    running_rids = {r.request_id for r in running}
    waiting_before = len(s.waiting)

    out = _force_retry_loop(s)

    # No same-step (deferred-free) victim is preempted.
    assert not [r for r in out.preempted_req_ids if r in running_rids]
    # Trigger stays RUNNING, still in running, not scheduled this step.
    assert trigger.status == RequestStatus.RUNNING
    assert trigger in s.running
    assert trigger.request_id not in out.num_scheduled_tokens
    # No spurious waiting churn.
    assert len(s.waiting) == waiting_before
    # The regression itself: a single allocation failure must not empty the
    # running queue. Pre-fix this dropped from 6 to 0 in one scheduling pass
    # (the loop chained victims until it reached `trigger`).
    assert list(s.running) == running, (
        f"running queue was evicted: {len(running)} -> {len(s.running)}"
    )
    # And no blocks were withheld from the pool by a pointless preemption.
    assert not s.deferred_frees


def test_no_starvation_after_fence_advances():
    """The guard is a conservative stop-and-wait, not a stall: once the
    in-flight step that fences a victim's deferred free is processed via
    update_from_output, that victim becomes reclaimable and the scheduler
    resumes normal preemption progress under a renewed allocation failure."""
    s = _deferring_scheduler()
    reqs = create_requests(
        num_requests=4,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=5,
        stop_token_ids=[STOP_TOKEN_ID],
    )
    for r in reqs:
        s.add_request(r)
    out1 = s.schedule()  # prefill (step 1)
    out2 = s.schedule()  # 1 decode in flight (step 2) -> all running same-step
    victim = s.running[-1]
    victim_fence = victim.last_sched_seq
    assert s.processed_step_seq < victim_fence  # free is deferred this step

    # Guard stops the cascade under allocation failure (zero preemptions).
    out_fail = _force_retry_loop(s)
    assert len(out_fail.preempted_req_ids) == 0

    # Process both in-flight steps' outputs in order: the fence advances past
    # the victim's last_sched_seq, so its deferred free becomes reclaimable.
    s.update_from_output(out1, _make_model_runner_output(out1))
    s.update_from_output(out2, _make_model_runner_output(out2))
    assert s.processed_step_seq >= victim_fence

    # Renewed allocation failure: the formerly-deferred victim is now
    # reclaimable, so normal preemption progress resumes (no starvation).
    out3 = _force_retry_loop(s)
    assert len(out3.preempted_req_ids) >= 1


def test_priority_policy_guard():
    """Under PRIORITY the victim is max(priority, arrival); the guard before
    removing it is bookkeeping-safe and still stops the cascade."""
    from vllm.v1.core.sched.scheduler import SchedulingPolicy

    s = _deferring_scheduler()
    s.policy = SchedulingPolicy.PRIORITY
    running = _running_same_step(s, n_requests=6)
    running_rids = {r.request_id for r in running}
    out = _force_retry_loop(s)
    assert not [r for r in out.preempted_req_ids if r in running_rids], (
        "PRIORITY: guard should not preempt a deferred victim, "
        f"got {out.preempted_req_ids}"
    )


def test_guard_inert_without_defer():
    """When defer_block_free=False (no connector / no async), the guard
    predicate is never true, so normal preemption still happens under
    allocation failure (no-connector path unaffected)."""
    s = _deferring_scheduler()
    s.defer_block_free = False
    reqs = create_requests(
        num_requests=6,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=5,
        stop_token_ids=[STOP_TOKEN_ID],
    )
    for r in reqs:
        s.add_request(r)
    s.schedule()  # prefill -> running
    assert s.running
    out = _force_retry_loop(s)
    assert len(out.preempted_req_ids) >= 1, (
        "defer OFF: normal preemption should still happen, guard must be inert"
    )


def test_reclaimable_victim_preempted_normally():
    """Even with defer_block_free=True, a victim scheduled in an already-
    processed step (last_sched_seq <= processed_step_seq) frees immediately, so
    the guard must NOT block its preemption (no false positive)."""
    s = _deferring_scheduler()
    _running_same_step(s, n_requests=6)
    # Make all running victims reclaimable: advance processed past last_sched_seq.
    s.processed_step_seq = max(r.last_sched_seq for r in s.running) + 1
    running_rids = {r.request_id for r in s.running}
    out = _force_retry_loop(s)
    preempted_running = [r for r in out.preempted_req_ids if r in running_rids]
    assert len(preempted_running) >= 1, (
        "reclaimable victim: guard must allow normal preemption"
    )


def test_mixed_fences_guard_does_not_scan_past_the_policy_victim():
    """Pins the deliberate conservatism of the guard.

    With PP (v2 runner sets next_decode_eligible_step = current_step + pp_size)
    the running set carries several distinct last_sched_seq values at once, so
    the policy-selected victim can be fenced while a deeper one is reclaimable.
    The guard stops anyway rather than scanning, because scanning would invert
    the FCFS victim order (preempt an older request to spare a newer one).

    If a future change adds a bounded scan, this test is the one to revisit --
    it is documenting a trade-off, not a correctness requirement.
    """
    s = _deferring_scheduler()
    running = _running_same_step(s, n_requests=6)
    # Simulate PP microbatching: an older running request was last scheduled in
    # an already-processed step, so ITS free would be immediate.
    running[1].last_sched_seq = s.processed_step_seq
    assert s._can_reclaim_request_blocks_now(running[1])
    # ... but the policy victim (FCFS tail) is still fenced.
    assert not s._can_reclaim_request_blocks_now(s.running[-1])

    out = _force_retry_loop(s)

    assert not out.preempted_req_ids, (
        "guard must stop at the policy victim, not scan for a reclaimable one"
    )
    assert list(s.running) == running


def test_guard_can_only_fire_while_a_step_is_in_flight():
    """Termination invariant behind the guard: it fires only when
    processed_step_seq < sched_step_seq, i.e. a non-empty step is still in the
    engine's batch queue and will advance the fence. A guard that could fire
    with the fence fully caught up would be a stall.
    """
    s = _deferring_scheduler()
    running = _running_same_step(s, n_requests=6)
    assert s.processed_step_seq < s.sched_step_seq
    for r in running:
        # last_sched_seq is stamped from sched_step_seq, so it can never lead it.
        assert r.last_sched_seq <= s.sched_step_seq
        # Guard fires => a step is in flight.
        if not s._can_reclaim_request_blocks_now(r):
            assert s.processed_step_seq < s.sched_step_seq

    # Once the fence catches up, the guard is provably inert for every request.
    s.processed_step_seq = s.sched_step_seq
    for r in running:
        assert s._can_reclaim_request_blocks_now(r)
