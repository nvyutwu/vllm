# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid (MLA + 3 KDA state groups) prefix cache: router events for
unresumable rows (``publish_unresumable_removals``) and superseded-state
demotion with a junction guard (``evict_superseded_mamba_states``).

Geometry follows the TP8 + DCP8 deployment: MLA per-GPU block 1152 (1024 with
the DSpark drafter) x DCP 8 = one 9216 (8192) token row, which is also the
scheduler block and the router block; three KDA "align" groups with
1152 (1024) token state blocks, FlashKDA prefill checkpoints, 128-token
hashes, periodic retention every two rows, the replay-boundary state and the
128-token partial-tail state. The DSpark variant adds EAGLE-style block drop
and speculative state slots.

Resume rule: a request resumes at ``B`` only with every MLA row covering
``[0, B)`` and a state at ``B`` in all 3 KDA groups.
"""

import random
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
    maybe_convert_block_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.unresumable_rows import UnresumableRowPublisher
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

HASH = 128
DCP = 8
MLA_GROUP = 0
KDA_GROUPS = (1, 2, 3)


class Geometry(NamedTuple):
    name: str
    kda: int  # per-GPU MLA block = KDA state block
    eagle: bool
    spec_blocks: int

    @property
    def row(self) -> int:
        return self.kda * DCP


G84 = Geometry("8.4", 1152, eagle=False, spec_blocks=0)
DSPARK = Geometry("dspark", 1024, eagle=True, spec_blocks=5)
GEOMETRIES = [pytest.param(G84, id="8.4"), pytest.param(DSPARK, id="dspark")]


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def tokens(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 150_000) for _ in range(n)]


class Sim:
    def __init__(
        self,
        geo: Geometry,
        *,
        publish: bool = True,
        evict_superseded: bool = False,
        num_blocks: int = 4096,
        retention_rows: int = 2,
    ):
        self.geo = geo
        self.R = geo.row
        kda = MambaSpec(
            block_size=geo.kda,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_speculative_blocks=geo.spec_blocks,
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=16,
        )
        cfg = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["mla"],
                    MLAAttentionSpec(
                        block_size=geo.kda,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.bfloat16,
                    ),
                ),
                *(KVCacheGroupSpec([f"kda{i}"], kda) for i in KDA_GROUPS),
            ],
            prefix_cache_retention_interval=retention_rows * self.R,
        )
        self.mgr = KVCacheManager(
            cfg,
            max_model_len=1 << 20,
            scheduler_block_size=self.R,
            hash_block_size=HASH,
            enable_caching=True,
            enable_kv_cache_events=True,
            dcp_world_size=DCP,
            use_eagle=geo.eagle,
            evict_superseded_mamba_states=evict_superseded,
            publish_unresumable_removals=publish,
        )
        self.pool = self.mgr.block_pool
        assert (self.mgr.unresumable_rows is not None) == publish
        self.events: list = []
        self._next = 0
        self.hits: list[int] = []

    # -- requests ----------------------------------------------------------
    def request(self, toks: list[int]) -> Request:
        params = SamplingParams(max_tokens=1 << 16)
        params.update_from_generation_config({}, eos_token_id=-1)
        req = Request(
            request_id=f"r{self._next}",
            prompt_token_ids=list(toks),
            mm_features=None,
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(HASH, sha256),
        )
        self._next += 1
        return req

    def _split(self, req: Request, num_new: int, num_local: int) -> int:
        stub = SimpleNamespace(
            block_size=self.R,
            cache_config=SimpleNamespace(
                block_size=self.geo.kda, replay_boundary_alias=False
            ),
            use_eagle=self.geo.eagle,
            use_eagle_block_drop=self.geo.eagle,
            num_hash_lookahead_tokens=0,
            max_num_scheduled_tokens=self.R,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
            mamba_partial_cache_hit=True,
            mamba_shared_prefix_checkpoint=False,
            hash_block_size=HASH,
            mamba_has_prefill_checkpoint_blocks=True,
            mamba_prefill_checkpoint_alignment=16,
        )
        return Scheduler._mamba_block_aligned_split(stub, req, num_new, num_local)

    def _step(self, req, num_new, num_local=0, computed_blocks=None) -> None:
        self.mgr.new_step_starts()
        start = req.num_computed_tokens + num_local
        out = self.mgr.allocate_slots(req, num_new, num_local, computed_blocks)
        assert out is not None, req.request_id
        req.status = RequestStatus.RUNNING
        _, retained = self.mgr.take_kv_cache_block_copies()
        self.pool.free_blocks(retained)
        req.num_computed_tokens = start + num_new
        self.drain()

    def admit(self, req: Request) -> int:
        blocks, hit, boundary = self.mgr.get_computed_blocks(req)
        req.shared_prefix_boundary = boundary
        self._pending = (blocks, hit)
        self.hits.append(hit)
        return hit

    def prefill(self, req: Request, until: int | None = None) -> int:
        """Admit (if new) and prefill ``req`` up to ``until`` tokens."""
        first = req.num_computed_tokens == 0 and req.status == RequestStatus.WAITING
        hit = self.admit(req) if first else 0
        until = req.num_prompt_tokens if until is None else until
        while req.num_computed_tokens < until:
            local = hit if first else 0
            remaining = until - req.num_computed_tokens - local
            num_new = self._split(req, min(remaining, self.R), local)
            assert num_new > 0
            self._step(req, num_new, local, self._pending[0] if first else None)
            first = False
        return hit

    def decode(self, req: Request, toks: list[int]) -> None:
        for tok in toks:
            req.append_output_token_ids(tok)
            self._step(req, 1)

    def finish(self, req: Request) -> None:
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        self.mgr.free(req)
        self.drain()

    def run(self, toks, decode=(), free=True) -> tuple[Request, int]:
        req = self.request(toks)
        hit = self.prefill(req)
        self.decode(req, list(decode))
        if free:
            self.finish(req)
        return req, hit

    def drain(self) -> list:
        new = self.mgr.take_events()
        self.events.extend(new)
        return new

    # -- introspection -----------------------------------------------------
    def row_hash(self, req: Request, row: int):
        """External (router) hash of ``req``'s MLA row ``row``."""
        return maybe_convert_block_hash(
            req.block_hashes[(row + 1) * self.R // HASH - 1]
        )

    def num_full_rows(self, req: Request) -> int:
        return len(req.block_hashes) * HASH // self.R

    def state_blocks(self, req: Request, pos: int, groups=KDA_GROUPS):
        key = req.block_hashes[pos // HASH - 1]
        return [self.pool.get_cached_block(key, [g]) for g in groups]

    def states(self, req: Request) -> list[int]:
        """Positions on ``req``'s tokens with a complete (3-group) state."""
        return [
            (i + 1) * HASH
            for i, key in enumerate(req.block_hashes)
            if self.pool.get_cached_block(key, list(KDA_GROUPS))
        ]

    def evict_state(self, req: Request, pos: int, groups=KDA_GROUPS) -> None:
        ids = set()
        for blocks in self.state_blocks(req, pos, groups):
            assert blocks, f"no state at {pos}"
            ids.add(blocks[0].block_id)
        self.mgr.evict_blocks(ids)
        self.drain()

    def mla_row_block(self, req: Request, row: int):
        index = (row + 1) * self.R // HASH - 1
        if index >= len(req.block_hashes):
            return None
        key = req.block_hashes[index]
        blocks = self.pool.get_cached_block(key, [MLA_GROUP])
        return blocks[0] if blocks else None

    def resume_point(self, req: Request) -> int:
        """Expected resume point from the resume rule, computed from the
        cached keys directly: the deepest complete state with every MLA row
        below it cached."""
        best = 0
        for pos in self.states(req):
            rows_needed = cdiv(pos, self.R)
            full_rows = pos // self.R
            ok = all(self.mla_row_block(req, r) for r in range(full_rows))
            if ok and rows_needed > full_rows:
                # The row holding ``pos`` is a full row or a partial key.
                ok = self.mla_row_block(req, full_rows) is not None or bool(
                    self.pool.get_cached_block(
                        req.block_hashes[pos // HASH - 1], [MLA_GROUP]
                    )
                )
            if ok:
                best = max(best, pos)
        return best


def removed_rows(events) -> list:
    return [
        h
        for e in events
        if isinstance(e, BlockRemoved) and e.group_idx == MLA_GROUP
        for h in e.block_hashes
    ]


def stored_rows(events, row: int) -> list[BlockStored]:
    return [
        e
        for e in events
        if isinstance(e, BlockStored)
        and e.group_idx == MLA_GROUP
        and e.block_size == row
    ]


def router_replay(events, row: int) -> tuple[set, int]:
    """A router's view of full MLA rows: (hashes, removes of unknown hashes)."""
    index: set = set()
    missing = 0
    full: set = set()
    for e in events:
        if isinstance(e, BlockStored) and e.group_idx == MLA_GROUP:
            if e.block_size == row:
                index.update(e.block_hashes)
                full.update(e.block_hashes)
        elif isinstance(e, BlockRemoved) and e.group_idx == MLA_GROUP:
            for h in e.block_hashes:
                if h not in full:
                    continue  # partial keys: never indexed
                if h in index:
                    index.discard(h)
                else:
                    missing += 1
    return index, missing


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("geo", GEOMETRIES)
def test_geometry(geo: Geometry):
    sim = Sim(geo)
    R = sim.R
    assert R in (9216, 8192)
    assert sim.mgr.coordinator.single_type_managers[MLA_GROUP].block_size == R
    a, _ = sim.run(tokens(5 * R + 3000, 1))
    states = sim.states(a)
    # Periodic states every two rows, a state in the last row (replay
    # boundary / prompt tail) and a 128-token partial tail state.
    assert 2 * R in states and 4 * R in states
    assert max(states) > 5 * R
    assert any(p % geo.kda for p in states), states
    # The router sees only full rows: 5 of them; the 6th is a partial key.
    stored = {h for e in stored_rows(sim.events, R) for h in e.block_hashes}
    assert stored == {sim.row_hash(a, k) for k in range(5)}
    assert removed_rows(sim.events) == []


# ---------------------------------------------------------------------------
# P4a: router events for dead rows
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("geo", GEOMETRIES)
def test_rows_removed_exactly_when_their_states_die(geo: Geometry):
    sim = Sim(geo)
    R = sim.R
    a, _ = sim.run(tokens(5 * R + 3000, 2), decode=tokens(40, 3))
    sim.events.clear()
    published = set(range(5))
    for pos in sorted(sim.states(a), reverse=True):
        sim.evict_state(a, pos)
        after = sim.resume_point(a)
        # Exactly the still-published rows that start at or above the new
        # resume point; nothing for rows a surviving state still covers.
        newly = {k for k in published if k * R >= after}
        assert set(removed_rows(sim.events)) == {sim.row_hash(a, k) for k in newly}
        sim.events.clear()
        published -= newly
        # The data stays cached: every MLA row is still keyed.
        assert all(sim.mla_row_block(a, k) for k in range(5))
        # And a repeat of the conversation resumes no deeper than that.
        repeat = sim.request(a.all_token_ids)
        assert sim.mgr.get_computed_blocks(repeat)[1] <= after
    assert published == set()


def _removed_since(sim: Sim, mark: int) -> set:
    return set(removed_rows(sim.events[mark:]))


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_removals_per_eviction(geo: Geometry):
    """Evicting a state that a deeper one still covers removes nothing;
    evicting the deepest removes the rows down to the next survivor; splitting
    one of the 3 groups kills the state."""
    sim = Sim(geo)
    R = sim.R
    a, _ = sim.run(tokens(5 * R + 3000, 4))
    mark = len(sim.events)
    sim.evict_state(a, 2 * R)  # covered by 4R and the tail states
    assert _removed_since(sim, mark) == set()
    # Evict only one group of every state in the last row: all incomplete.
    for pos in [p for p in sim.states(a) if p > 4 * R]:
        sim.evict_state(a, pos, groups=(KDA_GROUPS[1],))
    assert max(sim.states(a)) == 4 * R
    assert _removed_since(sim, mark) == {sim.row_hash(a, 4)}
    mark = len(sim.events)
    sim.evict_state(a, 4 * R)  # 2R is gone too: nothing left
    assert _removed_since(sim, mark) == {sim.row_hash(a, k) for k in range(4)}


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_no_double_removal_on_physical_eviction(geo: Geometry):
    sim = Sim(geo)
    R = sim.R
    a, _ = sim.run(tokens(5 * R + 3000, 5))
    for pos in [p for p in sim.states(a) if p > 4 * R]:
        sim.evict_state(a, pos)
    assert removed_rows(sim.events) == [sim.row_hash(a, 4)]
    sim.events.clear()
    # LRU reaches the dead row and a live one: only the live one is announced.
    dead, live = sim.mla_row_block(a, 4), sim.mla_row_block(a, 3)
    sim.mgr.evict_blocks({dead.block_id, live.block_id})
    sim.drain()
    assert removed_rows(sim.events) == [sim.row_hash(a, 3)]
    assert sim.pool.unpublished_block_hashes == {}
    # Draining the whole pool: a router never sees an unknown removal.
    sim.pool.get_new_blocks(sim.pool.get_num_free_blocks())
    sim.drain()
    index, missing = router_replay(sim.events, R)
    assert missing == 0


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_fork_rows_are_skipped(geo: Geometry):
    """F forks A inside row 3, resuming from A's 2R state. A's rows that F
    resumed through (0, 1) are F's now; A's row 2 is also held by F's own copy
    (same tokens); only rows 3 and 4 die with A's deeper states."""
    sim = Sim(geo)
    R = sim.R
    toks_a = tokens(5 * R + 3000, 6)
    a, _ = sim.run(toks_a)
    f, hit_f = sim.run(toks_a[: 3 * R + 500] + tokens(2 * R, 7))
    assert hit_f == 2 * R - (HASH if geo.eagle else 0) or hit_f == 2 * R
    mark = len(sim.events)
    for pos in [p for p in sim.states(a) if p > 2 * R]:
        sim.evict_state(a, pos)
    assert _removed_since(sim, mark) == {sim.row_hash(a, 3), sim.row_hash(a, 4)}
    # F's states still cover rows 0-2; evicting the shared 2R state (F's now)
    # removes nothing.
    mark = len(sim.events)
    sim.evict_state(a, 2 * R)
    assert _removed_since(sim, mark) == set()
    # ...and F's conversation still resumes past them.
    repeat = sim.request(list(f.all_token_ids) + tokens(10, 70))
    assert sim.mgr.get_computed_blocks(repeat)[1] > 4 * R


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_shared_system_prompt_rows_survive(geo: Geometry):
    """Two sessions share a 2-row system prompt and its junction state. One
    session's states all die: its own rows go, the shared rows (touched by
    the other session since) stay."""
    sim = Sim(geo)
    R = sim.R
    system = tokens(2 * R + 300, 8)
    a, _ = sim.run(system + tokens(3 * R, 9))
    b, _ = sim.run(system + tokens(3 * R, 10))
    assert b.shared_prefix_boundary or sim.hits[-1] >= 2 * R - HASH
    mark = len(sim.events)
    for pos in [p for p in sim.states(a) if p > 2 * R + 300]:
        sim.evict_state(a, pos)
    removed = _removed_since(sim, mark)
    assert removed == {sim.row_hash(a, k) for k in (2, 3, 4)}
    assert not removed & {sim.row_hash(a, 0), sim.row_hash(a, 1)}


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_running_request_rows_republished_when_its_state_lands(geo: Geometry):
    """A long prefill's periodic states are evicted mid-prefill: the rows they
    covered are removed (rows still being filled are not). When the prompt's
    tail state is stored the rows are resumable again: BlockStored again, in
    order, with their parents and tokens."""
    sim = Sim(geo)
    R = sim.R
    req = sim.request(tokens(5 * R + 3000, 11))
    sim.prefill(req, until=4 * R + 2 * geo.kda)
    assert 2 * R in sim.states(req) and 4 * R in sim.states(req)
    mark = len(sim.events)
    sim.evict_state(req, 4 * R)
    sim.evict_state(req, 2 * R)
    removed = removed_rows(sim.events[mark:])
    assert set(removed) == {sim.row_hash(req, k) for k in range(4)}
    # Deepest first, as LRU would.
    assert removed == [sim.row_hash(req, k) for k in (3, 2, 1, 0)]
    mark = len(sim.events)
    sim.prefill(req)  # the rest of the prompt: tail states land
    restored = [
        e
        for e in stored_rows(sim.events[mark:], R)
        if e.block_hashes[0] in set(removed)
    ]
    assert [e.block_hashes[0] for e in restored] == [
        sim.row_hash(req, k) for k in range(4)
    ]
    for k, e in enumerate(restored):
        assert e.parent_block_hash == (sim.row_hash(req, k - 1) if k else None)
        assert e.token_ids == req.all_token_ids[k * R : (k + 1) * R]
    sim.finish(req)
    index, missing = router_replay(sim.events, R)
    assert missing == 0
    assert index == {sim.row_hash(req, k) for k in range(5)}


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_long_decode_rows_without_state_die_at_finish(geo: Geometry):
    """Rows filled only by decode, above the last kept state, cannot be resumed
    from: they are removed when the request finishes."""
    sim = Sim(geo)
    R = sim.R
    req = sim.request(tokens(R + 3000, 12))
    sim.prefill(req)
    sim.decode(req, tokens(2 * R, 13))
    top = max(sim.states(req))
    mark = len(sim.events)
    sim.finish(req)
    dead = {k for k in range(sim.num_full_rows(req)) if k * R >= top}
    assert dead, (top, sim.num_full_rows(req))
    assert _removed_since(sim, mark) == {sim.row_hash(req, k) for k in dead}


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_eviction_order_and_hits_unchanged(geo: Geometry):
    """Same workload with and without the flag under LRU pressure: identical
    hits, free-queue order and final router view; the flag only moves
    removals earlier, never repeats one."""
    results = []
    for publish in (False, True):
        sim = Sim(geo, publish=publish, num_blocks=420)
        R = sim.R
        base = tokens(3 * R + 1000, 20)
        sim.run(base, decode=tokens(30, 21))
        for i in range(6):
            sim.run(base[: 2 * R + 700] + tokens(R + 900, 30 + i))
            sim.run(tokens(2 * R + 200, 40 + i))
        sim.run(base + tokens(500, 50))
        queue = [b.block_id for b in sim.pool.free_block_queue.get_all_free_blocks()]
        sim.pool.get_new_blocks(sim.pool.get_num_free_blocks())
        sim.drain()
        index, missing = router_replay(sim.events, R)
        results.append((sim.hits, queue, index, missing))
    (hits0, q0, idx0, miss0), (hits1, q1, idx1, miss1) = results
    assert hits0 == hits1
    assert q0 == q1
    assert idx0 == idx1 == set()
    assert miss1 <= miss0


def test_disabled_without_events():
    sim_cfg = Sim(G84, publish=False)
    assert sim_cfg.mgr.unresumable_rows is None
    assert sim_cfg.pool.cached_block_listener is None


# ---------------------------------------------------------------------------
# P1: superseded-state demotion with a junction guard
# ---------------------------------------------------------------------------
def out_of_lru_order(sim: Sim) -> list[int]:
    """Cached free blocks queued ahead of the last uncached one, in order."""
    queue = sim.pool.free_block_queue.get_all_free_blocks()
    last_uncached = max(i for i, b in enumerate(queue) if b.block_hash is None)
    return [b.block_id for b in queue[:last_uncached] if b.block_hash is not None]


def state_ids(sim: Sim, req: Request, pos: int) -> set[int]:
    return {blocks[0].block_id for blocks in sim.state_blocks(req, pos)}


def _tail_state(sim: Sim, req: Request) -> int:
    return max(sim.states(req))


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_superseded_tip_demoted_with_all_groups(geo: Geometry):
    sim = Sim(geo, publish=False, evict_superseded=True)
    R = sim.R
    a, _ = sim.run(tokens(3 * R + 2000, 60), decode=tokens(50, 61))
    tip = _tail_state(sim, a)
    tip_ids = state_ids(sim, a, tip)
    b, hit_b = sim.run(list(a.all_token_ids) + tokens(R, 62))
    assert hit_b == tip
    demoted = out_of_lru_order(sim)
    # Exactly the 3 group states the turn resumed from, side by side.
    assert set(demoted) == tip_ids and len(demoted) == 3
    assert {get_group_id(sim.pool.blocks[i].block_hash) for i in demoted} == set(
        KDA_GROUPS
    )
    # Still hashed: a regenerate of turn 2 still hits until reuse.
    regen = sim.request(list(a.all_token_ids) + tokens(R, 63))
    assert sim.mgr.get_computed_blocks(regen)[1] == tip


def queue_ids(sim: Sim) -> list[int]:
    return [b.block_id for b in sim.pool.free_block_queue.get_all_free_blocks()]


def _same_order_with_and_without_demotion(geo: Geometry, scenario) -> list[int]:
    """Run ``scenario(sim)`` with demotion off and on; return the queue."""
    queues = []
    for evict in (False, True):
        sim = Sim(geo, publish=False, evict_superseded=evict)
        scenario(sim)
        queues.append(queue_ids(sim))
    assert queues[0] == queues[1], "a kept state was demoted"
    return queues[1]


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_periodic_state_is_kept(geo: Geometry):
    def scenario(sim: Sim) -> None:
        R = sim.R
        a, _ = sim.run(tokens(5 * R + 2000, 64))
        for pos in [p for p in sim.states(a) if p > 4 * R]:
            sim.evict_state(a, pos)
        _, hit_b = sim.run(list(a.all_token_ids) + tokens(R, 65))
        assert hit_b == 4 * R  # resumed from a periodic state

    _same_order_with_and_without_demotion(geo, scenario)


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_junction_state_kept_after_direct_hits(geo: Geometry):
    """A shared system prompt: the second session detects the junction and
    retains a state there. Later sessions hit it directly
    (shared_prefix_boundary 0) and supersede it; plain #64975 would demote it
    after each, so the next new session would miss."""
    sim = Sim(geo, publish=False, evict_superseded=True)
    R = sim.R
    system = tokens(R + 500, 66)
    sim.run(system + tokens(2 * R, 67))
    b, hit_b = sim.run(system + tokens(2 * R, 68))
    junction = b.shared_prefix_boundary
    assert junction and hit_b < junction
    junction_states = [p for p in sim.states(b) if p <= junction and p > hit_b]
    assert junction_states, "the junction state is retained"
    for i in range(3):
        c, hit_c = sim.run(system + tokens(2 * R, 70 + i))
        assert hit_c == max(junction_states)
        if not geo.eagle:
            # A direct hit: plain #64975 (boundary < state < latest) would
            # demote it. (With EAGLE block drop the full-attention hit stays a
            # hash unit above the state, so the boundary already covers it.)
            assert c.shared_prefix_boundary < hit_c
        # Superseded by c's deeper states, yet kept at normal rank.
        assert hit_c < max(sim.states(c))
        assert out_of_lru_order(sim) == []


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_state_hit_by_two_requests_is_kept(geo: Geometry):
    sim = Sim(geo, publish=False, evict_superseded=True)
    R = sim.R
    a, _ = sim.run(tokens(2 * R + 2000, 80))
    tip = _tail_state(sim, a)
    b1 = sim.request(list(a.all_token_ids) + tokens(R, 81))
    b2 = sim.request(list(a.all_token_ids) + tokens(R, 82))
    assert sim.prefill(b1) == tip
    assert sim.prefill(b2) == tip
    sim.finish(b1)
    sim.finish(b2)
    assert out_of_lru_order(sim) == []


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_incomplete_hit_state_is_not_split(geo: Geometry):
    """One group's copy of the hit state loses its key before the turn ends:
    the other two are not demoted on their own (the decision is per
    position, for all groups)."""

    def scenario(sim: Sim) -> None:
        R = sim.R
        a, _ = sim.run(tokens(2 * R + 2000, 83))
        tip = _tail_state(sim, a)
        b = sim.request(list(a.all_token_ids) + tokens(R, 84))
        assert sim.prefill(b) == tip
        (blocks,) = sim.state_blocks(a, tip, groups=(KDA_GROUPS[0],))
        # Simulate a connector dropping that one group's key.
        sim.pool._maybe_evict_cached_block(blocks[0])
        sim.finish(b)

    _same_order_with_and_without_demotion(geo, scenario)


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_demotion_changes_order_only(geo: Geometry):
    """Hits of a follow-up are identical with and without demotion while the
    pool has room."""
    hits = []
    for evict in (False, True):
        sim = Sim(geo, publish=False, evict_superseded=evict)
        R = sim.R
        a, _ = sim.run(tokens(3 * R + 2000, 90), decode=tokens(20, 91))
        b, _ = sim.run(list(a.all_token_ids) + tokens(R, 92))
        sim.run(list(a.all_token_ids) + tokens(R, 93))
        sim.run(list(b.all_token_ids) + tokens(500, 94))
        hits.append(sim.hits)
    assert hits[0] == hits[1]


def test_flags_compose():
    """Demotion keeps hashes, so it publishes nothing; and evicting a demoted
    (superseded) state kills no row, since the deeper state still covers
    them."""
    sim = Sim(G84, publish=True, evict_superseded=True)
    R = sim.R
    a, _ = sim.run(tokens(R + 2000, 95), decode=tokens(50, 96))
    tip = _tail_state(sim, a)
    mark = len(sim.events)
    b, hit_b = sim.run(list(a.all_token_ids) + tokens(2 * R, 97))
    assert hit_b == tip
    assert set(out_of_lru_order(sim)) == state_ids(sim, a, tip)
    assert _removed_since(sim, mark) == set()
    sim.evict_state(a, tip)
    assert _removed_since(sim, mark) == set()


# ---------------------------------------------------------------------------
# End-to-end invariant: router credit never exceeds what vLLM can resume
# ---------------------------------------------------------------------------
def router_credit_rows(index: set, sim: Sim, req: Request) -> int:
    """Rows a router credits ``req``'s prefix with: the longest run of its
    full-row hashes from row 0 present in the router's index."""
    k = 0
    while k < sim.num_full_rows(req) and sim.row_hash(req, k) in index:
        k += 1
    return k


def actual_resume(sim: Sim, toks: list[int]) -> int:
    """What vLLM resumes a follow-up of this conversation from, now."""
    probe = sim.request(toks + tokens(16, 999))
    return sim.mgr.get_computed_blocks(probe)[1]


@pytest.mark.parametrize("geo", GEOMETRIES)
@pytest.mark.parametrize("publish", [True, False])
def test_router_credit_never_exceeds_resume_point(geo: Geometry, publish: bool):
    """Four independent conversations (distinct prefixes, so no row is
    shared); KDA states are then evicted one random (position, group subset)
    at a time. After every eviction the router's view, replayed from the
    event stream, must credit each conversation with no more rows than vLLM
    can resume from (counting the row holding the resume point), and, with
    nothing shared, with no fewer. Without the flag the first check fails:
    the router keeps crediting rows whose states are gone."""
    sim = Sim(geo, publish=publish)
    R = sim.R
    rng = random.Random(7)
    convs = []
    for i in range(4):
        n = rng.randrange(3, 6) * R + rng.randrange(200, 4000)
        req, _ = sim.run(tokens(n, 300 + i), decode=tokens(30, 400 + i))
        convs.append((req, list(req.all_token_ids)))
    over = 0
    for _ in range(40):
        req, toks = rng.choice(convs)
        states = sim.states(req)
        if not states:
            continue
        pos = rng.choice(states)
        groups = tuple(sorted(rng.sample(KDA_GROUPS, rng.randrange(1, 4))))
        sim.evict_state(req, pos, groups=groups)
        index, missing = router_replay(sim.events, R)
        assert missing == 0
        for r, t in convs:
            credit = router_credit_rows(index, sim, r)
            resume = actual_resume(sim, t)
            ceiling = cdiv(resume, R)
            if publish:
                assert credit <= ceiling, (r.request_id, credit, resume)
                assert credit >= resume // R, (r.request_id, credit, resume)
            over += credit > ceiling
    if not publish:
        assert over > 0, "scenario never produced a dead row"


# ---------------------------------------------------------------------------
# Content check: no hit ever reads another prefix's KV or state
# ---------------------------------------------------------------------------
class ContentSim(Sim):
    """``Sim`` plus a shadow model of what every physical block holds (as in
    ``test_replay_boundary_alias``): each MLA slot and each KDA state block
    records the identity of the prefix it was computed from; copy-on-write
    copies apply before a step's writes, as on the worker. Every prefix-cache
    hit is checked against the requester's own prefix."""

    def __init__(self, geo: Geometry, **kwargs):
        super().__init__(geo, **kwargs)
        self.kv: dict[int, dict[int, int]] = {}
        self.state: dict[int, int] = {}
        self._ids: dict[str, list[int]] = {}
        self.mla = self.mgr.coordinator.single_type_managers[MLA_GROUP]
        self.kda = [self.mgr.coordinator.single_type_managers[g] for g in KDA_GROUPS]

    def prefix_ids(self, req: Request) -> list[int]:
        ids = self._ids.setdefault(req.request_id, [0])
        for tok in req.all_token_ids[len(ids) - 1 :]:
            ids.append(hash((ids[-1], tok)))
        return ids

    def admit(self, req: Request) -> int:
        hit = super().admit(req)
        self.check_hit(req, self._pending[0], hit)
        return hit

    def check_hit(self, req: Request, blocks, hit: int) -> None:
        ids = self.prefix_ids(req)
        mla_blocks = blocks.blocks[MLA_GROUP]
        for pos in range(hit):
            got = self.kv.get(mla_blocks[pos // self.R].block_id, {}).get(pos % self.R)
            assert got == ids[pos + 1], (
                f"{req.request_id}: MLA position {pos} of a {hit}-token hit "
                "holds another prefix's KV"
            )
        if hit:
            for g in KDA_GROUPS:
                block = blocks.blocks[g][(hit - 1) // self.geo.kda]
                assert self.state.get(block.block_id) == ids[hit], (
                    f"{req.request_id}: KDA group {g} state for a {hit}-token "
                    "hit is another prefix's"
                )

    def _step(self, req, num_new, num_local=0, computed_blocks=None) -> None:
        self.mgr.new_step_starts()
        start = req.num_computed_tokens + num_local
        out = self.mgr.allocate_slots(req, num_new, num_local, computed_blocks)
        assert out is not None, req.request_id
        req.status = RequestStatus.RUNNING
        copies, retained = self.mgr.take_kv_cache_block_copies()
        for copy in copies:
            if copy.src_block_id in self.state:
                self.state[copy.dst_block_id] = self.state[copy.src_block_id]
            self.kv[copy.dst_block_id] = dict(self.kv.get(copy.src_block_id, {}))
        self._write(req, start, start + num_new)
        self.pool.free_blocks(retained)
        req.num_computed_tokens = start + num_new
        self.drain()

    def _write(self, req: Request, start: int, end: int) -> None:
        from vllm.model_executor.layers.mamba.checkpoint import (
            compute_mamba_prefill_checkpoints,
        )

        ids = self.prefix_ids(req)
        mla_blocks = self.mla.req_to_blocks[req.request_id]
        for pos in range(start, end):
            block = mla_blocks[pos // self.R]
            assert not block.is_null
            self.kv.setdefault(block.block_id, {})[pos % self.R] = ids[pos + 1]
        (offset,), (col,) = compute_mamba_prefill_checkpoints(
            [end],
            [end - start],
            hash_block_size=HASH,
            mamba_block_size=self.geo.kda,
            checkpoint_alignment=16,
            drop_eagle_block=self.geo.eagle,
            cache_hit_alignment_tokens=HASH,
        )
        for manager in self.kda:
            blocks = manager.req_to_blocks[req.request_id]
            running = blocks[cdiv(end, self.geo.kda) - 1]
            assert not running.is_null
            self.state[running.block_id] = ids[end]
            if col >= 0 and not blocks[col].is_null:
                self.state[blocks[col].block_id] = ids[start + offset]


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_hits_read_only_own_prefix_with_both_flags(geo: Geometry):
    """Multi-turn sessions sharing a system prompt, a fork, a regenerate and
    forced KDA evictions under a small pool, with both flags on: every hit
    reads only its own prefix's KV and states, and hit lengths match the run
    without the flags."""

    def workload(sim: ContentSim) -> list[int]:
        R = sim.R
        system = tokens(R + 700, 500)
        convs = [system + tokens(2 * R + 300 * i, 510 + i) for i in range(4)]
        for turn in range(3):
            for i in range(4):
                req, _ = sim.run(convs[i], decode=tokens(40, 600 + 10 * i + turn))
                convs[i] = list(req.all_token_ids) + tokens(R // 2, 700 + 10 * i + turn)
            # Kill one session's deepest state (one group) to create dead rows.
            states = sim.states(req)
            if states:
                sim.evict_state(req, max(states), groups=(KDA_GROUPS[turn % 3],))
        fork = convs[0][: 2 * R + 900] + tokens(R, 800)
        sim.run(fork)
        sim.run(convs[1][: len(convs[1]) - R // 2])  # regenerate a turn
        return sim.hits

    off = workload(ContentSim(geo, publish=False, num_blocks=260))
    on = workload(ContentSim(geo, publish=True, evict_superseded=True, num_blocks=260))
    assert on == off


# ---------------------------------------------------------------------------
# P4a cost: lookups only when the resume point can have moved
# ---------------------------------------------------------------------------
class EveryStepPublisher(UnresumableRowPublisher):
    """Reference: the first P4a version, which ran the hybrid lookup after
    every cache step of a chain with dead rows and on every evaluation."""

    def _check_new_hashes(self, chain) -> None:
        chain.dirty = True

    def _evaluate(self, chain, upper) -> None:
        resume = None
        for group_id, rows in chain.rows.items():
            block_size = self.row_block_size[group_id]
            last = len(rows)
            if upper is not None:
                last = min(last, cdiv(upper, block_size))
            if last <= 0:
                continue
            if resume is None:
                resume = self.resume_point(chain.block_hashes)
            first = cdiv(resume, block_size)
            for row in range(last - 1, first - 1, -1):
                block, key = rows[row]
                if not self._may_die(chain, block, key):
                    continue
                self.block_pool.unpublish_block_hash(block)
                self.dead_rows[block.block_id] = (chain, group_id, row)
                chain.dead.add(block.block_id)


def use_reference_publisher(sim: Sim) -> None:
    old = sim.mgr.unresumable_rows
    assert old is not None
    sim.mgr.unresumable_rows = EveryStepPublisher(
        old.coordinator, old.row_managers, sorted(old.state_group_ids)
    )


class Driver:
    """Interleaves many requests on one ``Sim`` from a seeded RNG: multi-turn
    sessions (some sharing a system prompt), forks, identical concurrent
    prompts, decode across state blocks and rows, preemption, forced KDA
    evictions and pool pressure. Allocation failure finishes the request, so
    the workload depends only on the seed and the cache's decisions."""

    def __init__(self, sim: Sim, seed: int):
        self.sim = sim
        self.rng = random.Random(seed)
        R = sim.R
        system = tokens(R + 500, seed * 7 + 1)
        self.sessions = [
            (system if i % 2 else []) + tokens(R // 2 + 400 * i, seed * 7 + 2 + i)
            for i in range(5)
        ]
        self.active: list[tuple[Request, int, int]] = []  # req, session, decode left
        self.done = 0

    def _start(self) -> None:
        rng, R = self.rng, self.sim.R
        kind = rng.random()
        s = rng.randrange(len(self.sessions))
        base = self.sessions[s]
        if kind < 0.15 and self.active:
            # Identical copy of a running request's prompt.
            other = rng.choice(self.active)[0]
            toks = list(other.prompt_token_ids)
            s = -1
        elif kind < 0.3 and len(base) > R:
            toks = base[: rng.randrange(R // 2, len(base))] + tokens(
                rng.randrange(300, R), rng.randrange(1 << 30)
            )
            s = -1
        else:
            toks = base + tokens(rng.randrange(200, R), rng.randrange(1 << 30))
        req = self.sim.request(toks)
        self.active.append((req, s, rng.choice([5, 200, 1300, 2 * R // 3])))

    def _finish(self, i: int) -> None:
        req, s, _ = self.active.pop(i)
        self.sim.finish(req)
        if s >= 0:
            self.sessions[s] = list(req.all_token_ids)
        self.done += 1

    def _try_step(self, req: Request, num_new: int, num_local=0, blocks=None) -> bool:
        sim = self.sim
        sim.mgr.new_step_starts()
        start = req.num_computed_tokens + num_local
        out = sim.mgr.allocate_slots(req, num_new, num_local, blocks)
        if out is None:
            return False
        req.status = RequestStatus.RUNNING
        _, retained = sim.mgr.take_kv_cache_block_copies()
        sim.pool.free_blocks(retained)
        req.num_computed_tokens = start + num_new
        sim.drain()
        return True

    def _advance(self, i: int) -> None:
        sim = self.sim
        req, s, left = self.active[i]
        if req.num_computed_tokens < req.num_tokens:
            local, blocks = 0, None
            if req.num_computed_tokens == 0:
                local = sim.admit(req)
                blocks = sim._pending[0]
            remaining = req.num_tokens - req.num_computed_tokens - local
            num_new = sim._split(req, min(remaining, sim.R), local)
            ok = num_new > 0 and self._try_step(req, num_new, local, blocks)
        elif left > 0:
            n = min(left, self.rng.choice([1, 1, 1, 64]))
            ok = True
            for tok in tokens(n, self.rng.randrange(1 << 30)):
                req.append_output_token_ids(tok)
                if not self._try_step(req, 1):
                    ok = False
                    break
            self.active[i] = (req, s, left - n)
        else:
            self._finish(i)
            return
        if not ok:
            self._finish(i)

    def _preempt(self, i: int) -> None:
        req = self.active[i][0]
        if req.num_computed_tokens == 0:
            return
        req.status = RequestStatus.PREEMPTED
        self.sim.mgr.free(req)
        self.sim.drain()
        req.status = RequestStatus.WAITING
        req.num_computed_tokens = 0

    def _evict(self) -> None:
        sim, rng = self.sim, self.rng
        req = rng.choice(self.active)[0]
        states = sim.states(req)
        if states:
            pos = rng.choice(states)
            groups = tuple(sorted(rng.sample(KDA_GROUPS, rng.randrange(1, 4))))
            sim.evict_state(req, pos, groups=groups)

    def run(self, num_ops: int) -> None:
        rng = self.rng
        for _ in range(num_ops):
            x = rng.random()
            if len(self.active) < 2 or (x < 0.08 and len(self.active) < 6):
                self._start()
            elif x < 0.12:
                self._preempt(rng.randrange(len(self.active)))
            elif x < 0.18:
                self._evict()
            else:
                self._advance(rng.randrange(len(self.active)))
        while self.active:
            self._finish(0)


@pytest.mark.parametrize("geo", GEOMETRIES)
@pytest.mark.parametrize("seed", range(6))
def test_events_equal_every_step_reference(geo: Geometry, seed: int):
    """The incremental publisher emits exactly the events (same blocks, same
    order) and gives exactly the hits of the reference that runs the lookup
    after every step, on interleaved random workloads, with fewer lookups."""

    def run(reference: bool, publish: bool = True):
        sim = Sim(geo, publish=publish, evict_superseded=bool(seed % 2), num_blocks=300)
        if reference:
            use_reference_publisher(sim)
        driver = Driver(sim, seed)
        driver.run(700)
        return sim, driver

    new, d_new = run(False)
    ref, d_ref = run(True)
    assert d_new.done == d_ref.done
    assert new.hits == ref.hits
    assert new.events == ref.events
    assert removed_rows(new.events), "workload never killed a row"
    # Unknown removals a router sees come only from duplicate keys, as
    # without the flag (two blocks holding one key each publish removed).
    off, _ = run(False, publish=False)
    assert off.hits == new.hits
    assert router_replay(new.events, new.R)[1] == router_replay(off.events, off.R)[1]
    assert new.mgr.unresumable_rows.num_lookups < ref.mgr.unresumable_rows.num_lookups


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_decode_with_dead_rows_runs_no_lookup(geo: Geometry):
    """A running request whose resumed-from state is evicted has dead rows.
    Its following decode steps run no hybrid lookup until a key on its path
    is cached; the reference runs one every step."""
    for reference in (False, True):
        sim = Sim(geo)
        if reference:
            use_reference_publisher(sim)
        R = sim.R
        toks = tokens(4 * R + 3000, 20)
        sim.run(toks)
        req = sim.request(toks + tokens(R // 4, 21))
        hit = sim.prefill(req)
        assert hit > 3 * R
        for pos in [p for p in sim.states(req) if p > R]:
            sim.evict_state(req, pos)
        pub = sim.mgr.unresumable_rows
        assert pub.chains[req.request_id].dead
        # Stay inside one state block: no new key gets cached.
        if -req.num_tokens % geo.kda < 200:
            sim.decode(req, tokens(-req.num_tokens % geo.kda + 1, 23))
            assert pub.chains[req.request_id].dead
        steps = min(geo.kda // 2, -req.num_tokens % geo.kda - 1)
        assert steps >= 200
        before = pub.num_lookups
        sim.decode(req, tokens(steps, 22))
        lookups = pub.num_lookups - before
        if reference:
            assert lookups == steps
        else:
            assert lookups == 0
        sim.finish(req)


@pytest.mark.parametrize("geo", GEOMETRIES)
def test_lookups_bounded_by_keys_cached_on_path(geo: Geometry):
    """Decode across many state blocks with dead rows: at most one lookup per
    step that cached a key on the request's path."""
    sim = Sim(geo)
    R = sim.R
    toks = tokens(3 * R + 1000, 30)
    sim.run(toks)
    req = sim.request(toks + tokens(200, 31))
    sim.prefill(req)
    for pos in [p for p in sim.states(req) if p > R // 2]:
        sim.evict_state(req, pos)
    pub = sim.mgr.unresumable_rows
    assert pub.chains[req.request_id].dead
    num_cached = 0
    on_cached = pub.on_cached

    def counting_on_cached(*args):
        nonlocal num_cached
        num_cached += 1
        on_cached(*args)

    pub.on_cached = counting_on_cached  # type: ignore[method-assign]
    cached_steps = 0
    lookups = 0
    for tok in tokens(3 * R, 32):
        before_cached, before = num_cached, pub.num_lookups
        req.append_output_token_ids(tok)
        sim._step(req, 1)
        assert pub.num_lookups - before <= 1
        lookups += pub.num_lookups - before
        cached_steps += num_cached != before_cached
    assert lookups <= cached_steps
    assert 100 * lookups < 3 * R
    sim.finish(req)
