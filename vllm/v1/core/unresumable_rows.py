# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``CacheConfig.publish_unresumable_removals``: KV events that credit only
resumable full-attention blocks in a hybrid (full attention + Mamba "align")
prefix cache.

A request can skip recomputing its first ``B`` tokens only if the cache holds
every full-attention block covering ``[0, B)`` *and* a state at ``B`` in every
state group. Full-attention blocks are cached densely, states sparsely; once
the states above a run of blocks are evicted, those blocks can no longer be
resumed from, yet an external KV-event router still credits them until LRU
reaches them, and sends follow-ups to a worker that recomputes them.

This module publishes ``BlockRemoved`` for such blocks ("dead rows") as soon
as they die, while their data stays cached, and ``BlockStored`` again if they
become resumable again. Eviction order is unchanged.

Chains. Each request is a chain: its full-attention rows (full blocks, the
only ones a router indexes) in order and its block hashes. A row belongs to
the chain that touched it last (cached it or resumed through it). A chain's
resume point ``H`` is the hybrid prefix-cache hit its own tokens would get
now; its owned rows starting at or above ``H`` are dead.

When a chain is evaluated:

- when a state it stored or resumed from is evicted (a real eviction: pool
  reuse or a connector ``evict_blocks``). A finished chain checks all owned
  rows; a running one only the rows below the evicted state, which lost their
  cover (rows it is still filling are not dead yet);
- when its request is freed (finished or preempted): all owned rows;
- after a cache step of its request, if it has dead rows and ``H`` may have
  grown since it was last computed: rows below the new ``H`` are published
  again.

Cost. ``H`` only grows when a key on the chain's path is added to the cache
(a key not cached before; re-registering a cached key changes nothing),
or when the request's hash list grows over keys that are already cached. So a
chain with dead rows is marked dirty when ``on_cached`` adds a key at a
position of its path, or when ``observe`` sees new hashes of its own that are
cached; only a dirty chain runs the hybrid lookup. A decode step that caches
nothing costs O(1) per request. An evaluation first collects the rows that
could die and skips the lookup when there are none.

Shared rows. Only the evaluated chain's own rows are walked; rows another
request touched since belong to that request's chain and are skipped, and so
is a row whose key is also held by another block. A dead row that a request
resumes through (only possible once ``H`` grew) is published again.
"""

from collections.abc import Sequence

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.request import Request

logger = init_logger(__name__)

# Sweep chains that own nothing after this many request frees.
_SWEEP_INTERVAL = 256


class _Chain:
    __slots__ = (
        "block_hashes",
        "dead",
        "dirty",
        "finished",
        "num_hashes",
        "rows",
        "running",
    )

    def __init__(self, block_hashes: Sequence[BlockHash]) -> None:
        # The request's own hash list (not a copy); it outlives the request
        # only while the chain owns cached rows.
        self.block_hashes = block_hashes
        # {group_id: [(block, key) of row 0, row 1, ...]}
        self.rows: dict[int, list[tuple[KVCacheBlock, BlockHashWithGroupId]]] = {}
        # Block ids of this chain's rows whose BlockRemoved was published.
        self.dead: set[int] = set()
        self.running = True
        # Freed for good (not preempted): never observed again.
        self.finished = False
        # ``dead`` is non-empty and a key on the chain's path was added since
        # ``H`` was last computed, so ``H`` may have grown.
        self.dirty = False
        # ``len(block_hashes)`` when ``H`` was last computed (or new hashes
        # were last checked); hashes past it may cover cached keys.
        self.num_hashes = 0


class UnresumableRowPublisher:
    """Publishes ``BlockRemoved``/``BlockStored`` for full-attention rows as
    they stop/start being resumable. See the module docstring."""

    def __init__(
        self,
        coordinator: HybridKVCacheCoordinator,
        row_managers: Sequence[SingleTypeKVCacheManager],
        state_group_ids: Sequence[int],
    ) -> None:
        self.coordinator = coordinator
        self.block_pool: BlockPool = coordinator.block_pool
        self.row_managers = tuple(row_managers)
        self.row_block_size = {m.kv_cache_group_id: m.block_size for m in row_managers}
        self.state_group_ids = frozenset(state_group_ids)
        self.chains: dict[str, _Chain] = {}
        # Row block id -> the chain that touched it last.
        self.row_owner: dict[int, _Chain] = {}
        # Dead row block id -> (chain, group id, row index).
        self.dead_rows: dict[int, tuple[_Chain, int, int]] = {}
        # State prefix hash -> the chain that last stored or resumed from it.
        self.state_owner: dict[BlockHash, _Chain] = {}
        # Evicted states awaiting ``flush``: prefix hash -> position.
        self._evicted_states: dict[BlockHash, int | None] = {}
        # Chains that have dead rows and may be observed again (running or
        # preempted); pruned lazily.
        self._watched: set[_Chain] = set()
        self._group_ids = tuple(range(len(coordinator.kv_cache_config.kv_cache_groups)))
        self._hash_block_size = self.block_pool.hash_block_size
        self._num_frees = 0
        # Hybrid lookups run (``resume_point``); for tests and debugging.
        self.num_lookups = 0
        self.block_pool.cached_block_listener = self

    # -- block pool hooks --------------------------------------------------
    def on_cached(
        self,
        request: Request,
        block_hash: BlockHashWithGroupId,
        num_tokens: int | None = None,
        added: bool = True,
    ) -> None:
        key = get_block_hash(block_hash)
        if get_group_id(block_hash) in self.state_group_ids:
            self.state_owner[key] = self._chain(request)
        # A key already cached (re-registered, or a duplicate block) leaves
        # every resume point unchanged.
        if added and self._watched:
            self._mark_dirty(key, num_tokens)

    def on_evicted(
        self,
        block: KVCacheBlock,
        block_hashes: list[BlockHashWithGroupId],
        num_tokens: int | None,
    ) -> None:
        for block_hash in block_hashes:
            if get_group_id(block_hash) in self.state_group_ids:
                key = get_block_hash(block_hash)
                if key not in self._evicted_states or num_tokens is not None:
                    self._evicted_states[key] = num_tokens
        self.row_owner.pop(block.block_id, None)
        dead = self.dead_rows.pop(block.block_id, None)
        if dead is not None:
            dead[0].dead.discard(block.block_id)

    def flush(self) -> None:
        if not self._evicted_states:
            return
        evicted, self._evicted_states = self._evicted_states, {}
        # The highest evicted position per chain; ``None`` = unknown.
        lost: dict[_Chain, int | None] = {}
        for key, position in evicted.items():
            chain = self.state_owner.pop(key, None)
            if chain is None:
                continue
            if chain in lost:
                prev = lost[chain]
                position = (
                    None if prev is None or position is None else max(prev, position)
                )
            lost[chain] = position
        for chain, position in lost.items():
            self._evaluate(chain, upper=position if chain.running else None)

    # -- KVCacheManager hooks ----------------------------------------------
    def observe(
        self,
        request: Request,
        new_computed_blocks: Sequence[Sequence[KVCacheBlock]] | None = None,
    ) -> None:
        """Record ``request``'s newly cached or resumed rows (and the states it
        resumed from), and publish again rows its new states made resumable."""
        chain = self._chain(request)
        if not chain.running:
            # Resumed after preemption: its block table was rebuilt.
            chain.running = True
            chain.rows = {}
        if new_computed_blocks is not None:
            for group_id in self.state_group_ids:
                hit = new_computed_blocks[group_id]
                if hit and not hit[-1].is_null and hit[-1].block_hash is not None:
                    self.state_owner[get_block_hash(hit[-1].block_hash)] = chain
        revived: list[tuple[KVCacheBlock, int]] = []
        for manager in self.row_managers:
            group_id = manager.kv_cache_group_id
            block_size = manager.block_size
            blocks = manager.req_to_blocks.get(request.request_id, ())
            rows = chain.rows.setdefault(group_id, [])
            row = len(rows)
            while row < len(blocks):
                block = blocks[row]
                key = block.block_hash
                if (
                    block.is_null
                    or key is None
                    or block.block_hash_num_tokens != (row + 1) * block_size
                    or get_group_id(key) != group_id
                ):
                    break
                rows.append((block, key))
                self.row_owner[block.block_id] = chain
                if block.block_id in self.dead_rows:
                    # Resumed through: resumable again.
                    revived.append((block, block_size))
                row += 1
        for block, block_size in revived:
            self._republish(request, block, block_size)
        if chain.dead:
            if not chain.dirty:
                self._check_new_hashes(chain)
            if chain.dirty:
                self._revive(chain, request)
        else:
            chain.num_hashes = len(chain.block_hashes)

    def on_request_freed(self, request: Request) -> None:
        chain = self.chains.get(request.request_id)
        if chain is None or chain.block_hashes is not request.block_hashes:
            return
        chain.running = False
        if request.is_finished():
            chain.finished = True
            self._watched.discard(chain)
        self._evaluate(chain, upper=None)
        if not self._owns_rows(chain):
            del self.chains[request.request_id]
        self._num_frees += 1
        if self._num_frees % _SWEEP_INTERVAL == 0:
            self._sweep()

    def reset(self) -> None:
        self.chains.clear()
        self.row_owner.clear()
        self.dead_rows.clear()
        self.state_owner.clear()
        self._evicted_states.clear()
        self._watched.clear()

    # -- internals ---------------------------------------------------------
    def _chain(self, request: Request) -> _Chain:
        chain = self.chains.get(request.request_id)
        if chain is None or chain.block_hashes is not request.block_hashes:
            chain = self.chains[request.request_id] = _Chain(request.block_hashes)
        return chain

    def resume_point(self, block_hashes: Sequence[BlockHash]) -> int:
        """Tokens a request with these block hashes would resume from now."""
        if not block_hashes:
            return 0
        self.num_lookups += 1
        _, hit_length, _ = self.coordinator.find_longest_cache_hit(
            list(block_hashes), len(block_hashes) * self.block_pool.hash_block_size
        )
        return hit_length

    def _evaluate(self, chain: _Chain, upper: int | None) -> None:
        """Publish ``BlockRemoved`` for ``chain``'s owned rows that start at or
        above its resume point (and, if ``upper`` is set, below ``upper``).

        Rows that cannot die (already dead, re-keyed, owned by another chain
        or duplicated) are filtered first; the hybrid lookup runs only if some
        row could die."""
        candidates: list[tuple[int, list[int]]] = []
        for group_id, rows in chain.rows.items():
            block_size = self.row_block_size[group_id]
            last = len(rows)
            if upper is not None:
                last = min(last, cdiv(upper, block_size))
            live = [
                row
                for row in range(last - 1, -1, -1)
                if self._may_die(chain, *rows[row])
            ]
            if live:
                candidates.append((group_id, live))
        if not candidates:
            return
        had_dead = bool(chain.dead)
        resume = self.resume_point(chain.block_hashes)
        for group_id, live in candidates:
            rows = chain.rows[group_id]
            first = cdiv(resume, self.row_block_size[group_id])
            # Deepest first, as LRU removes a chain.
            for row in live:
                if row < first:
                    break
                block, _ = rows[row]
                self.block_pool.unpublish_block_hash(block)
                self.dead_rows[block.block_id] = (chain, group_id, row)
                chain.dead.add(block.block_id)
        if chain.dead and not chain.finished:
            if not had_dead:
                # ``H`` is fresh: every dead row starts at or above it.
                chain.dirty = False
                chain.num_hashes = len(chain.block_hashes)
            self._watched.add(chain)

    def _may_die(
        self, chain: _Chain, block: KVCacheBlock, key: BlockHashWithGroupId
    ) -> bool:
        return not (
            block.block_id in self.dead_rows
            or block.block_hash != key
            or self.row_owner.get(block.block_id) is not chain
            or self._has_other_holder(key)
        )

    def _revive(self, chain: _Chain, request: Request) -> None:
        chain.dirty = False
        chain.num_hashes = len(chain.block_hashes)
        resume = self.resume_point(chain.block_hashes)
        revive = []
        for block_id in chain.dead:
            _, group_id, row = self.dead_rows[block_id]
            if row * self.row_block_size[group_id] < resume:
                revive.append((row, group_id, block_id))
        for _, group_id, block_id in sorted(revive):
            block = self.block_pool.blocks[block_id]
            self._republish(request, block, self.row_block_size[group_id])

    def _mark_dirty(self, key: BlockHash, num_tokens: int | None) -> None:
        """A key was cached: mark watched chains whose path holds it."""
        index = None if num_tokens is None else num_tokens // self._hash_block_size - 1
        stale = []
        for chain in self._watched:
            if chain.finished or not chain.dead:
                stale.append(chain)
            elif not chain.dirty:
                hashes = chain.block_hashes
                if index is None or (0 <= index < len(hashes) and hashes[index] == key):
                    chain.dirty = True
        for chain in stale:
            self._watched.discard(chain)

    def _check_new_hashes(self, chain: _Chain) -> None:
        """Mark ``chain`` dirty if hashes it gained since ``H`` was computed
        name keys already cached (e.g. by an identical request ahead)."""
        hashes = chain.block_hashes
        num_hashes = len(hashes)
        if num_hashes <= chain.num_hashes:
            return
        cached = self.block_pool.cached_block_hash_to_block
        for index in range(chain.num_hashes, num_hashes):
            block_hash = hashes[index]
            for group_id in self._group_ids:
                key = make_block_hash_with_group_id(block_hash, group_id)
                if cached.get_one_block(key) is not None:
                    chain.dirty = True
                    return
        chain.num_hashes = num_hashes

    def _republish(
        self, request: Request, block: KVCacheBlock, block_size: int
    ) -> None:
        chain, _, _ = self.dead_rows.pop(block.block_id)
        chain.dead.discard(block.block_id)
        self.block_pool.republish_block_hash(request, block, block_size)

    def _has_other_holder(self, key: BlockHashWithGroupId) -> bool:
        return self.block_pool.cached_block_hash_to_block.num_blocks(key) > 1

    def _owns_rows(self, chain: _Chain) -> bool:
        return any(
            block.block_hash == key and self.row_owner.get(block.block_id) is chain
            for rows in chain.rows.values()
            for block, key in rows
        )

    def _sweep(self) -> None:
        """Drop finished chains that own no cached row."""
        for request_id, chain in list(self.chains.items()):
            if not chain.running and not self._owns_rows(chain):
                del self.chains[request_id]
        live = {id(chain) for chain in self.chains.values()}
        self.state_owner = {k: c for k, c in self.state_owner.items() if id(c) in live}
