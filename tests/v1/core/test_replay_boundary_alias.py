# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay-boundary alias for hybrid (full-attention + Mamba ``align``) caching.

Kimi-K3 geometry: KDA (Mamba) block 1536, MLA block 1536 x DCP 8 = 12288,
prefix-match unit 128, 8192-token prefill chunks, retention 0. Retention keeps
the KDA state at the prompt's replay-boundary block end ``R`` (and the exact
tail ``T``), but MLA keys only whole 12288 blocks and ``T``. A follow-up that
drops the previous prompt's last ~150 tokens diverges between ``R`` and ``T``
and could not resume at ``R``; ``CacheConfig.replay_boundary_alias`` keys the
MLA tail block at ``R`` too.

The tests drive the real ``KVCacheManager`` and the real
``Scheduler._mamba_block_aligned_split`` and keep a shadow model of what every
physical block holds: each MLA slot and each KDA state block records the
identity of the prefix it was computed from, copy-on-write copies are applied
before a step's writes (as the worker does), and every prefix-cache hit is
checked against the requester's own prefix. A hit that reads another prefix's
KV or state, or a write into a block that another key still describes, fails
the check.
"""

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

HASH = 128
KDA = 1536
DCP = 8
MLA = KDA * DCP  # 12288
CHUNK = 8192
MLA_GROUP, KDA_GROUP = 0, 1


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def replay_boundary(num_prompt_tokens: int) -> int:
    """Where retention keeps the KDA state: ``num_prompt - 1`` floored to the
    hash unit, then to the KDA block."""
    return (num_prompt_tokens - 1) // HASH * HASH // KDA * KDA


class Sim:
    """One request per scheduler step, with a shadow content model."""

    def __init__(
        self,
        alias: bool,
        retention: int | None = 0,
        flashkda: bool = True,
        num_blocks: int = 512,
        eagle: bool = False,
    ):
        cfg = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["mla"],
                    MLAAttentionSpec(
                        block_size=KDA,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.bfloat16,
                    ),
                ),
                KVCacheGroupSpec(
                    ["kda"],
                    MambaSpec(
                        block_size=KDA,
                        shapes=((1, 1),),
                        dtypes=(torch.float32,),
                        mamba_cache_mode="align",
                        num_prefill_checkpoint_blocks=int(flashkda),
                    ),
                ),
            ],
            prefix_cache_retention_interval=retention,
        )
        if alias:
            cfg = replace(cfg, replay_boundary_alias=True)
        self.mgr = KVCacheManager(
            cfg,
            max_model_len=1 << 20,
            scheduler_block_size=MLA,
            hash_block_size=HASH,
            enable_caching=True,
            enable_kv_cache_events=True,
            dcp_world_size=DCP,
            use_eagle=eagle,
        )
        self.flashkda = flashkda
        self.eagle = eagle
        self.mla = self.mgr.coordinator.single_type_managers[MLA_GROUP]
        self.kda = self.mgr.coordinator.single_type_managers[KDA_GROUP]
        assert self.mla.block_size == MLA and self.kda.block_size == KDA
        # Shadow model: MLA block id -> slot -> prefix id; KDA block -> prefix id.
        self.kv: dict[int, dict[int, int]] = defaultdict(dict)
        self.state: dict[int, int] = {}
        self.events: list = []
        self._ids: dict[str, list[int]] = {}
        self._next_id = 0

    # -- requests ---------------------------------------------------------
    def request(self, tokens: list[int]) -> Request:
        params = SamplingParams(max_tokens=1 << 16)
        params.update_from_generation_config({}, eos_token_id=-1)
        req = Request(
            request_id=f"r{self._next_id}",
            prompt_token_ids=list(tokens),
            mm_features=None,
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(HASH, sha256),
        )
        self._next_id += 1
        return req

    def prefix_ids(self, req: Request) -> list[int]:
        """``ids[p]`` identifies ``req``'s first ``p`` tokens; the KV of
        position ``p`` and the state after ``p`` tokens are computed from
        exactly that prefix."""
        ids = self._ids.setdefault(req.request_id, [0])
        for tok in req.all_token_ids[len(ids) - 1 :]:
            ids.append(hash((ids[-1], tok)))
        return ids

    # -- one scheduler step ----------------------------------------------
    def _split(self, req: Request, num_new: int, num_local: int) -> int:
        stub = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=KDA),
            use_eagle=self.eagle,
            use_eagle_block_drop=self.eagle,
            max_num_scheduled_tokens=CHUNK,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
            mamba_partial_cache_hit=True,
            hash_block_size=HASH,
            mamba_has_prefill_checkpoint_blocks=self.flashkda,
        )
        return Scheduler._mamba_block_aligned_split(stub, req, num_new, num_local)

    def _run_copies(self) -> list:
        copies, retained = self.mgr.take_kv_cache_block_copies()
        for copy in copies:
            if copy.src_block_id in self.state:
                self.state[copy.dst_block_id] = self.state[copy.src_block_id]
            self.kv[copy.dst_block_id] = dict(self.kv.get(copy.src_block_id, {}))
        return retained

    def _write(self, req: Request, start: int, end: int) -> None:
        """The forward pass: MLA KV for ``[start, end)`` and the KDA running
        state (plus the FlashKDA internal checkpoint) at the chunk end."""
        ids = self.prefix_ids(req)
        mla_blocks = self.mla.req_to_blocks[req.request_id]
        for pos in range(start, end):
            block = mla_blocks[pos // MLA]
            assert not block.is_null
            self.kv[block.block_id][pos % MLA] = ids[pos + 1]
        kda_blocks = self.kda.req_to_blocks[req.request_id]
        running = kda_blocks[cdiv(end, KDA) - 1]
        assert not running.is_null
        self.state[running.block_id] = ids[end]
        offset = end // KDA * KDA - start
        if self.flashkda and end % KDA != 0 and 0 < offset < end - start:
            checkpoint = kda_blocks[end // KDA - 1]
            if not checkpoint.is_null:
                self.state[checkpoint.block_id] = ids[end // KDA * KDA]

    def _step(self, req, num_new, num_local=0, computed_blocks=None):
        self.mgr.new_step_starts()
        start = req.num_computed_tokens + num_local
        out = self.mgr.allocate_slots(req, num_new, num_local, computed_blocks)
        assert out is not None, req.request_id
        retained = self._run_copies()
        self._write(req, start, start + num_new)
        self.mgr.block_pool.free_blocks(retained)
        req.num_computed_tokens = start + num_new
        self.events.extend(self.mgr.take_events())

    # -- whole requests ---------------------------------------------------
    def check_hit(self, req: Request, blocks, hit: int) -> None:
        ids = self.prefix_ids(req)
        mla_blocks = blocks.blocks[MLA_GROUP]
        for pos in range(hit):
            got = self.kv.get(mla_blocks[pos // MLA].block_id, {}).get(pos % MLA)
            assert got == ids[pos + 1], (
                f"{req.request_id}: MLA position {pos} of a {hit}-token hit "
                "holds another prefix's KV"
            )
        if hit:
            state_block = blocks.blocks[KDA_GROUP][(hit - 1) // KDA]
            assert self.state.get(state_block.block_id) == ids[hit], (
                f"{req.request_id}: KDA state for the {hit}-token hit is wrong"
            )

    def prefill(self, req: Request) -> int:
        """Admit and prefill ``req``; return its local prefix-cache hit."""
        blocks, hit, boundary = self.mgr.get_computed_blocks(req)
        self.check_hit(req, blocks, hit)
        req.shared_prefix_boundary = boundary
        first = True
        while req.num_computed_tokens < req.num_prompt_tokens:
            local = hit if first else 0
            remaining = req.num_prompt_tokens - req.num_computed_tokens - local
            num_new = self._split(req, min(remaining, CHUNK), local)
            assert num_new > 0
            self._step(req, num_new, local, blocks if first else None)
            first = False
        return hit

    def decode(self, req: Request, tokens: list[int]) -> None:
        """Generate ``tokens`` one step at a time (the sampled token of a step
        becomes part of the request before the next step computes it)."""
        for tok in tokens:
            req.append_output_token_ids(tok)
            self._step(req, 1)

    def run(self, tokens: list[int], decode: Sequence[int] = (), free=True):
        req = self.request(tokens)
        hit = self.prefill(req)
        self.decode(req, list(decode))
        if free:
            self.finish(req)
        return req, hit

    def finish(self, req: Request) -> None:
        self.mgr.free(req)
        self.events.extend(self.mgr.take_events())

    def assert_own_kv_intact(self, req: Request) -> None:
        """Every computed position of a running request still holds its own KV."""
        blocks = self.mla.req_to_blocks[req.request_id]
        ids = self.prefix_ids(req)
        for pos in range(req.num_computed_tokens):
            got = self.kv[blocks[pos // MLA].block_id].get(pos % MLA)
            assert got == ids[pos + 1], (
                f"{req.request_id}: position {pos} was overwritten by another request"
            )

    # -- introspection ----------------------------------------------------
    def mla_key(self, req: Request, num_tokens: int):
        return make_block_hash_with_group_id(
            req.block_hashes[num_tokens // HASH - 1], MLA_GROUP
        )

    def mla_alias_keys(self, req: Request, num_tokens: int) -> list[int]:
        """Block ids an MLA key of ``req``'s prefix at ``num_tokens`` maps to."""
        key = self.mla_key(req, num_tokens)
        block = self.mgr.block_pool.cached_block_hash_to_block.get_one_block(key)
        return [] if block is None else [block.block_id]

    def is_silent_alias(self, req: Request, num_tokens: int) -> bool:
        key = self.mla_key(req, num_tokens)
        return any(
            key in keys
            for keys in self.mgr.block_pool.silent_block_hashes_by_block.values()
        )


def tokens(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 150_000) for _ in range(n)]


N_A = 30_000  # one full MLA block run (24576) + a partial tail block
R_A = replay_boundary(N_A)  # 29184 = 19 x 1536
T_A = N_A // HASH * HASH  # 29952
DIVERGE = N_A - 150  # follow-up drops the previous prompt's last 150 tokens


def test_geometry_of_the_conflux_case():
    assert (MLA * 2 < R_A < T_A < MLA * 3) and R_A <= DIVERGE
    sim = Sim(alias=True)
    assert sim.mla.replay_alias_unit == KDA
    assert sim.kda.replay_alias_unit == 0
    assert Sim(alias=False).mla.replay_alias_unit == 0


@pytest.mark.parametrize("flashkda", [True, False])
def test_follow_up_resumes_at_replay_boundary(flashkda: bool):
    """Prompt A, then B = A minus its last 150 tokens + new content: B resumes
    at R in both groups and copies the partial MLA block."""
    prompt_a = tokens(N_A, seed=1)
    prompt_b = prompt_a[:DIVERGE] + tokens(3_000, seed=2)

    for alias, expect in ((False, 0), (True, R_A)):
        sim = Sim(alias=alias, flashkda=flashkda)
        req_a, hit_a = sim.run(prompt_a)
        assert hit_a == 0
        # Retention 0 keeps KDA states at R and T only.
        assert sim.mgr.block_pool.get_cached_block(
            req_a.block_hashes[R_A // HASH - 1], [KDA_GROUP]
        )
        a_tail_block = sim.mla_alias_keys(req_a, T_A)
        assert a_tail_block, "A's MLA tail key"
        assert (sim.mla_alias_keys(req_a, R_A) == a_tail_block) == alias
        a_tail_before = dict(sim.kv[a_tail_block[0]])

        req_b, hit_b = sim.run(prompt_b)
        assert hit_b == expect
        # A's tail block is untouched by B (B wrote into its own copy) and an
        # exact repeat and an extension of A still resume at A's tail.
        assert sim.kv[a_tail_block[0]] == a_tail_before
        _, hit_repeat = sim.run(prompt_a)
        assert hit_repeat == T_A
        _, hit_ext = sim.run(prompt_a + tokens(500, seed=3))
        assert hit_ext == T_A
        # B's (copied) tail block carries B's own alias for B's follow-ups;
        # without it they fall back to the junction B's miss detected (24576).
        _, hit_bb = sim.run(prompt_b[: len(prompt_b) - 150] + tokens(100, seed=4))
        assert hit_bb == (replay_boundary(len(prompt_b)) if alias else 2 * MLA)


def test_follow_up_while_previous_turn_still_decoding():
    """B arrives while A is still decoding into its tail block: B's copy must
    not see or disturb A's in-flight writes."""
    prompt_a = tokens(N_A, seed=11)
    sim = Sim(alias=True)
    req_a = sim.request(prompt_a)
    sim.prefill(req_a)
    sim.decode(req_a, tokens(300, seed=12))
    req_b, hit_b = sim.run(prompt_a[:DIVERGE] + tokens(2_000, seed=13))
    assert hit_b == R_A
    sim.decode(req_a, tokens(300, seed=14))
    sim.assert_own_kv_intact(req_a)
    sim.finish(req_a)


def test_divergence_inside_last_kda_block_is_unchanged():
    """A follow-up diverging before R has no state there; it falls back as
    before (no partial MLA hit without a matching KDA state)."""
    prompt_a = tokens(N_A, seed=21)
    prompt_b = prompt_a[: R_A - 100] + tokens(2_000, seed=22)
    for alias in (False, True):
        sim = Sim(alias=alias)
        sim.run(prompt_a)
        _, hit_b = sim.run(prompt_b)
        assert hit_b == 0


@pytest.mark.parametrize("retention", [0, None])
def test_previous_turn_decoded_past_its_tail_block(retention):
    """A decodes past the end of its tail MLA block, which promotes the block
    to a full key; the tail key (and the alias) are then re-registered on the
    full block. A hit through either must copy the block: otherwise the
    requester overwrites a block that A is still attending over and that A's
    full key still describes."""
    prompt_a = tokens(N_A, seed=31)
    out_a = tokens(MLA * 3 - N_A + 200, seed=32)  # crosses 36864
    for alias in (False, True):
        sim = Sim(alias=alias, retention=retention)
        req_a = sim.request(prompt_a)
        sim.prefill(req_a)
        sim.decode(req_a, out_a[:-100])
        assert sim.mla_alias_keys(req_a, T_A), "tail key re-registered"
        assert sim.is_silent_alias(req_a, R_A) == alias
        # E extends A's prompt (not A's output): resumes at A's tail key.
        _, hit_e = sim.run(prompt_a + tokens(1_000, seed=33))
        assert hit_e == T_A
        # B diverges just before A's prompt end: R with the alias, an older
        # state without (none under retention 0, a chunk end under dense).
        _, hit_b = sim.run(prompt_a[:DIVERGE] + tokens(1_000, seed=34))
        assert hit_b == R_A if alias else hit_b < 2 * MLA
        sim.decode(req_a, out_a[-100:])
        sim.assert_own_kv_intact(req_a)
        sim.finish(req_a)
        # F continues A's exact output; check_hit verifies every position of
        # its hit against F's own prefix (dense retention reaches past 36864).
        _, hit_f = sim.run(prompt_a + out_a + tokens(500, seed=35))
        assert hit_f >= (MLA * 3 if retention is None else T_A)


@pytest.mark.parametrize("eagle", [False, True])
@pytest.mark.parametrize("past_block_end", [False, True])
def test_extension_of_running_request_copies(eagle: bool, past_block_end: bool):
    """E extends A's prompt while A is still decoding, either inside its tail
    block or after crossing its end (block promoted). EAGLE-style trailing-block
    dropping (DSpark included) lowers E's hit to R, below every key the block
    carries, so a copy that requires the hit to equal the block's primary key
    never fires and E would write over A's output KV even inside the tail
    block. Without the alias."""
    prompt_a = tokens(N_A, seed=51)
    sim = Sim(alias=False, eagle=eagle)
    req_a = sim.request(prompt_a)
    sim.prefill(req_a)
    sim.decode(req_a, tokens(MLA * 3 - N_A + 100 if past_block_end else 300, seed=52))
    _, hit_e = sim.run(prompt_a + tokens(1_000, seed=53))
    assert hit_e == (R_A if eagle else T_A)
    sim.decode(req_a, tokens(100, seed=54))
    sim.assert_own_kv_intact(req_a)
    sim.finish(req_a)


def test_hit_trimmed_by_kda_inside_block_copies():
    """MLA matches A's tail key T but KDA only has a state at R (the tail state
    was evicted): the hit ends at R inside A's block. The requester must copy,
    otherwise it writes over A's block, which A is still decoding into."""
    prompt_a = tokens(N_A, seed=41)
    for alias in (False, True):
        sim = Sim(alias=alias)
        req_a = sim.request(prompt_a)
        sim.prefill(req_a)
        sim.decode(req_a, tokens(64, seed=42))
        pool = sim.mgr.block_pool
        (tail_state,) = pool.get_cached_block(
            req_a.block_hashes[T_A // HASH - 1], [KDA_GROUP]
        )
        assert tail_state.ref_cnt == 0
        pool.evict_blocks({tail_state.block_id})
        _, hit_e = sim.run(prompt_a + tokens(1_000, seed=43))
        assert hit_e == R_A
        sim.decode(req_a, tokens(64, seed=44))
        sim.assert_own_kv_intact(req_a)
        sim.finish(req_a)


def test_alias_is_not_published_as_kv_event():
    """The alias is engine-local: no BlockStored on registration and no
    BlockRemoved when its block's keys go (promotion, eviction, reset)."""
    prompt_a = tokens(N_A, seed=51)
    streams = {}
    for alias in (False, True):
        sim = Sim(alias=alias, num_blocks=48)
        req_a, _ = sim.run(prompt_a)
        alias_hash = maybe_convert_block_hash(req_a.block_hashes[R_A // HASH - 1])
        # Run unrelated prompts until A's blocks are evicted.
        for seed in range(52, 92):
            if not sim.mla_alias_keys(req_a, T_A):
                break
            sim.run(tokens(N_A, seed=seed))
        assert not sim.mla_alias_keys(req_a, T_A), "A's tail was not evicted"
        for event in sim.events:
            if isinstance(event, (BlockStored, BlockRemoved)):
                # The KDA state at R is published by the KDA group as usual.
                assert not (
                    event.group_idx == MLA_GROUP and alias_hash in event.block_hashes
                )
        streams[alias] = [
            (type(e).__name__, tuple(e.block_hashes), e.group_idx)
            for e in sim.events
            if isinstance(e, (BlockStored, BlockRemoved))
        ]
    # Same workload (no follow-ups), same published stream.
    assert streams[False] == streams[True]


def test_alias_registered_once_per_tail_and_dropped_on_eviction():
    sim = Sim(alias=True, num_blocks=64)
    req_a, _ = sim.run(tokens(N_A, seed=71))
    pool = sim.mgr.block_pool
    (tail_block_id,) = sim.mla_alias_keys(req_a, T_A)
    assert sim.mla_alias_keys(req_a, R_A) == [tail_block_id]
    assert len(pool.silent_block_hashes_by_block[tail_block_id]) == 1
    pool.evict_blocks({tail_block_id})
    assert not sim.mla_alias_keys(req_a, R_A)
    assert tail_block_id not in pool.silent_block_hashes_by_block


@pytest.mark.parametrize(
    "num_prompt", [24_576, 24_600, 25_000, 26_700, 30_000, 36_863, 37_000]
)
def test_alias_position_matches_retained_kda_state(num_prompt: int):
    """The alias is registered exactly where the geometry allows (never on a
    block edge or at the tail itself). Where retention 0 kept a KDA state at R
    a follow-up resumes there; where it did not (sparse retention reserves the
    FlashKDA checkpoint only on the prompt-end chunk, so a short final chunk
    that starts above R leaves no state at R, e.g. 36,863 tokens), the alias
    key is inert and the follow-up falls back below R with correct KV."""
    sim = Sim(alias=True)
    req, _ = sim.run(tokens(num_prompt, seed=num_prompt))
    alias = replay_boundary(num_prompt)
    tail = num_prompt // HASH * HASH
    has_alias = sim.is_silent_alias(req, alias)
    expect = (
        tail % MLA != 0
        and alias % MLA != 0
        and alias < tail
        and alias // MLA == tail // MLA
    )
    assert has_alias == expect
    if has_alias:
        has_state = sim.mgr.block_pool.get_cached_block(
            req.block_hashes[alias // HASH - 1], [KDA_GROUP]
        )
        prompt = tokens(num_prompt, seed=num_prompt)
        _, hit_b = sim.run(prompt[: alias + 64] + tokens(500, seed=1))
        assert (hit_b == alias) if has_state else (hit_b < alias)


@pytest.mark.parametrize("alias", [False, True])
def test_prefix_lookup_detail_record(alias: bool, monkeypatch):
    """VLLM_LOG_PREFIX_LOOKUP_DETAIL: per-group walks and the logged record
    for the Conflux follow-up (KDA state at R; MLA key there only with the
    alias)."""
    import vllm.v1.core.sched.scheduler as sched_mod

    prompt_a = tokens(N_A, seed=81)
    sim = Sim(alias=alias)
    sim.run(prompt_a)
    req_b = sim.request(prompt_a[:DIVERGE] + tokens(2_000, seed=82))
    detail = sim.mgr.prefix_lookup_detail(req_b)
    assert detail == {
        "full_attn_hit": R_A if alias else 2 * MLA,
        "full_attn_whole_block_hit": 2 * MLA,
        "full_attn_partial_hit": R_A if alias else None,
        "mamba_state_hit": R_A,
    }

    logged: list[str] = []
    monkeypatch.setattr(
        sched_mod.logger, "info", lambda fmt, *args: logged.append(fmt % args)
    )
    stub = SimpleNamespace(
        kv_cache_manager=sim.mgr,
        _prefix_lookup_detail={},
        connector=SimpleNamespace(
            pop_prefix_lookup_debug=lambda rid: {"partial_probe_pending": True}
        ),
    )
    _, hit, boundary = sim.mgr.get_computed_blocks(req_b)
    req_b.shared_prefix_boundary = boundary
    record = Scheduler._record_prefix_lookup(stub, req_b, hit, False)
    Scheduler._record_connector_lookup(stub, req_b, record, 2 * MLA, hit % MLA, None)
    Scheduler._record_connector_lookup(stub, req_b, record, 2 * MLA, hit % MLA, 0)
    Scheduler._log_admitted_prefix_lookup(stub, req_b, hit, 0, False)
    assert not stub._prefix_lookup_detail
    (line,) = logged
    assert line.startswith("PREFIX_LOOKUP_DETAIL ")
    import json

    rec = json.loads(line.split(" ", 1)[1])
    assert rec["local_hit"] == hit == (R_A if alias else 0)
    assert rec["lookups"] == 1 and rec["deferred"] == 1
    assert rec["partial_tail_decision"] == (
        "local_tail_kept" if alias else "no_local_tail"
    )
    assert rec["flags"]["mamba_state_without_full_attn_key"] is (not alias)
    assert rec["flags"]["cpu_store_pending"] is True
    assert rec["external_request_id"] == req_b.request_id


@pytest.mark.parametrize("alias", [False, True])
def test_prefill_stats_report_hits_found_per_group(alias: bool):
    """F4 telemetry: each group's own hit is reported next to the reconciled
    reuse. The Conflux follow-up finds two whole MLA blocks and the KDA state at
    R; without the alias no MLA key sits at R, so nothing is reused locally."""
    prompt_a = tokens(N_A, seed=91)
    sim = Sim(alias=alias)
    sim.run(prompt_a)
    req_b = sim.request(prompt_a[:DIVERGE] + tokens(2_000, seed=92))
    stub = SimpleNamespace(kv_cache_manager=sim.mgr, connector=None)
    group_hits = Scheduler._group_prefix_hits(stub, req_b, None)
    _, reused, _ = sim.mgr.get_computed_blocks(req_b)

    assert reused == (R_A if alias else 0)
    local_full_attention = R_A if alias else 2 * MLA
    assert Scheduler._found_by_group(group_hits, 0, reused, 0) == (
        local_full_attention,
        local_full_attention,
        R_A,
    )
    assert Scheduler._found_by_group(group_hits, 3 * MLA, 0, R_A) == (
        3 * MLA,
        local_full_attention,
        R_A,
    )


def test_found_by_group_without_stats_or_recurrent_state():
    req = SimpleNamespace(prefill_stats=None, num_preemptions=0)
    assert Scheduler._group_prefix_hits(SimpleNamespace(), req, None) is None
    assert Scheduler._found_by_group(None, 0, 4096, 0) == (4096, 4096, None)
    assert Scheduler._found_by_group({}, 8192, 0, 4096) == (8192, 0, None)


def test_full_attention_external_end_uses_the_connector_lookup():
    req = SimpleNamespace(request_id="r")
    with_hit = SimpleNamespace(
        connector=SimpleNamespace(get_full_attention_external_hit=lambda rid: 1536)
    )
    unknown = SimpleNamespace(
        connector=SimpleNamespace(get_full_attention_external_hit=lambda rid: None)
    )
    assert Scheduler._full_attention_external_end(with_hit, req, MLA) == MLA + 1536
    assert Scheduler._full_attention_external_end(unknown, req, MLA) == 0
    assert (
        Scheduler._full_attention_external_end(
            SimpleNamespace(connector=None), req, MLA
        )
        == 0
    )
