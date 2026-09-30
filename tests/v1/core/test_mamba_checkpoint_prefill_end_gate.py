# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefill checkpoints are reserved only on the prompt-end chunk
under sparse retention (vLLM #59175, fix 1, adapted to this branch).

Geometry mirrors Kimi-K3 production: Mamba (KDA) block 1536, hash block 128
(`--prefix-match-unit 128`), prefill budget 8192, one prefill checkpoint block.
With retention_interval=0 a mid-prompt checkpoint is released the step after it
is reserved, so reserving it only evicts a cached block. The prompt-end chunk
ends at the last hash boundary (the scheduler computes the sub-hash remainder
separately), so the gate is `prefill_end` rounded down to the hash block.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 1536
HASH_BLOCK_SIZE = 128
BUDGET = 8192
MAMBA_GROUP_ID = 1


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _make_manager(retention_interval: int | None) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=4096,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=0,
                    num_prefill_checkpoint_blocks=1,
                ),
            ),
        ],
    )
    config.prefix_cache_retention_interval = retention_interval
    return KVCacheManager(
        config,
        max_model_len=1 << 20,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=HASH_BLOCK_SIZE,
        enable_caching=True,
        use_eagle=False,
    )


def _make_request(request_id: str, prompt_len: int, tail: int = 0) -> Request:
    tokens = list(range(prompt_len)) + [7] * tail
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(HASH_BLOCK_SIZE, sha256),
    )


def _split(request: Request, num_new_tokens: int) -> int:
    """Call the real `Scheduler._mamba_block_aligned_split` on a stub self."""
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        use_eagle=False,
        use_eagle_block_drop=False,
        max_num_scheduled_tokens=BUDGET,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=True,
        hash_block_size=HASH_BLOCK_SIZE,
        mamba_has_prefill_checkpoint_blocks=True,
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new_tokens)


def _prefill(manager: KVCacheManager, request: Request) -> list[tuple[int, int, int]]:
    """Prefill one chunk per step; returns (start, end, checkpoint_blocks)."""
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    chunks = []
    while request.num_computed_tokens < request.num_tokens:
        start = request.num_computed_tokens
        num_new = _split(request, min(request.num_tokens - start, BUDGET))
        assert num_new > 0
        assert manager.allocate_slots(request, num_new) is not None
        chunks.append(
            (
                start,
                start + num_new,
                mamba_manager._num_checkpoint_blocks.get(request.request_id, 0),
            )
        )
        request.num_computed_tokens = start + num_new
        manager.new_step_starts()
    return chunks


@pytest.mark.parametrize(
    ("retention_interval", "mid_prompt_checkpoints"), [(0, 0), (None, 1)]
)
def test_mid_prompt_checkpoint_reserved_only_under_dense_retention(
    retention_interval: int | None, mid_prompt_checkpoints: int
) -> None:
    manager = _make_manager(retention_interval)
    chunks = _prefill(manager, _make_request("req", 40000))
    # Chunks that start block-aligned and end mid-block, before the prompt end.
    mid_prompt = [
        blocks
        for start, end, blocks in chunks
        if start % BLOCK_SIZE == 0 and end % BLOCK_SIZE and end < 39936
    ]
    assert mid_prompt, chunks
    assert all(blocks == mid_prompt_checkpoints for blocks in mid_prompt), chunks


@pytest.mark.parametrize("retention_interval", [0, None])
def test_prompt_end_checkpoint_kept_at_hash_boundary(
    retention_interval: int | None,
) -> None:
    """The prompt-end chunk stops at the last hash boundary (4992 < 5000); its
    checkpoint at 4608 must still be reserved and published."""
    manager = _make_manager(retention_interval)
    request = _make_request("producer", 5000)
    chunks = _prefill(manager, request)
    assert chunks[0][:2] == (0, 4992)
    assert chunks[0][2] == 1
    manager.free(request)

    for diverge_at, expected in ((4700, 4608), (4900, 4608), (5000, 4992)):
        follower = Request(
            request_id=f"follower-{diverge_at}",
            prompt_token_ids=list(range(diverge_at)) + [9] * 3000,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=get_request_block_hasher(HASH_BLOCK_SIZE, sha256),
        )
        _, num_hit, _ = manager.get_computed_blocks(follower)
        assert num_hit == expected, (diverge_at, num_hit)


@pytest.mark.parametrize("retention_interval", [0, None])
def test_prefix_hits_unchanged_by_gate(retention_interval: int | None) -> None:
    manager = _make_manager(retention_interval)
    request = _make_request("producer", 20000)
    _prefill(manager, request)
    manager.free(request)
    assert manager.get_computed_blocks(_make_request("replay", 20000))[1] == 19968
    assert (
        manager.get_computed_blocks(_make_request("extend", 20000, tail=3000))[1]
        == 19968
    )
