# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import msgspec
import pytest

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    TierBlocksCleared,
    isolate_tier_clear_batches,
)

# Minimal ExternalBlockHash for testing (bytes are a valid ExternalBlockHash).
_FAKE_HASH: bytes = b"\xab" * 32


class _LegacyBlockStored(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag="BlockStored",  # type: ignore[call-arg]
):
    """BlockStored wire schema before locality was added."""

    block_hashes: list[bytes]
    parent_block_hash: bytes | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None


class _LegacyBlockRemoved(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag="BlockRemoved",  # type: ignore[call-arg]
):
    """BlockRemoved wire schema before locality was added."""

    block_hashes: list[bytes]
    medium: str | None
    group_idx: int | None = None


class _LegacyAllBlocksCleared(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag="AllBlocksCleared",  # type: ignore[call-arg]
):
    pass


def test_tier_clear_has_distinct_wire_tag_and_legacy_clear_still_decodes():
    decoder = msgspec.msgpack.Decoder(type=AllBlocksCleared)
    legacy = decoder.decode(msgspec.msgpack.encode(_LegacyAllBlocksCleared()))
    assert isinstance(legacy, AllBlocksCleared)

    gpu = TierBlocksCleared(medium="GPU")
    tier_decoder = msgspec.msgpack.Decoder(type=TierBlocksCleared)
    assert tier_decoder.decode(msgspec.msgpack.encode(gpu)) == gpu
    assert len({gpu, TierBlocksCleared(medium="CPU")}) == 2
    with pytest.raises(msgspec.ValidationError):
        decoder.decode(msgspec.msgpack.encode(gpu))

    batch = KVEventBatch(ts=1.0, events=[gpu], data_parallel_rank=0)
    decoded_batch = msgspec.msgpack.decode(
        msgspec.msgpack.encode(batch), type=KVEventBatch
    )
    assert decoded_batch.events == [gpu]


def test_tier_clear_is_published_as_a_singleton_between_ordinary_events():
    before = BlockRemoved(block_hashes=[_FAKE_HASH], medium="GPU")
    clear = TierBlocksCleared(medium="GPU")
    after = BlockRemoved(block_hashes=[b"\xcd" * 32], medium="CPU")
    batches = list(isolate_tier_clear_batches([before, clear, after]))
    assert batches == [[before], [clear], [after]]


def test_tier_clear_carries_an_ownership_domain_without_breaking_the_default():
    framework = TierBlocksCleared(medium="CPU")
    assert framework.ownership is None

    owned = TierBlocksCleared(medium="CPU", ownership="kvcr")
    decoder = msgspec.msgpack.Decoder(type=TierBlocksCleared)
    assert decoder.decode(msgspec.msgpack.encode(owned)) == owned
    assert owned != framework
    assert len({framework, owned}) == 2

    # `omit_defaults` keeps the framework's own resets byte-identical to the
    # pre-ownership wire, so the field costs nothing on the hot path.
    assert b"kvcr" not in msgspec.msgpack.encode(framework)
    assert b"kvcr" in msgspec.msgpack.encode(owned)


def test_isolating_tier_clears_preserves_stream_order_exactly():
    first = BlockStored(
        block_hashes=[_FAKE_HASH],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
        lora_id=None,
        medium="GPU",
        lora_name=None,
    )
    gpu_clear = TierBlocksCleared(medium="GPU")
    cpu_clear = TierBlocksCleared(medium="CPU")
    second = BlockRemoved(block_hashes=[_FAKE_HASH], medium="CPU")

    batches = list(isolate_tier_clear_batches([first, gpu_clear, cpu_clear, second]))
    assert batches == [[first], [gpu_clear], [cpu_clear], [second]]
    # Flattening the batches must reproduce the input stream: a scoped clear is
    # separated from neighbours, never reordered or dropped.
    assert [event for batch in batches for event in batch] == [
        first,
        gpu_clear,
        cpu_clear,
        second,
    ]


def test_isolating_leaves_a_clear_free_stream_as_one_batch():
    events = [
        BlockRemoved(block_hashes=[_FAKE_HASH], medium="GPU"),
        BlockRemoved(block_hashes=[b"\xcd" * 32], medium="GPU"),
    ]
    assert list(isolate_tier_clear_batches(events)) == [events]
    assert list(isolate_tier_clear_batches([])) == []


def test_legacy_all_clear_and_tier_clear_are_not_interchangeable_on_the_wire():
    # A legacy-only consumer must reject a scoped clear outright rather than
    # decode it as the all-tier event it is not.
    legacy_decoder = msgspec.msgpack.Decoder(
        type=BlockStored | BlockRemoved | AllBlocksCleared
    )
    with pytest.raises(msgspec.ValidationError):
        legacy_decoder.decode(msgspec.msgpack.encode(TierBlocksCleared(medium="GPU")))

    # And the new union still decodes every legacy event.
    new_decoder = msgspec.msgpack.Decoder(
        type=BlockStored | BlockRemoved | AllBlocksCleared | TierBlocksCleared
    )
    assert isinstance(
        new_decoder.decode(msgspec.msgpack.encode(AllBlocksCleared())),
        AllBlocksCleared,
    )


def _make_block_stored(
    group_idx: int | None = None,
    kv_cache_spec_sliding_window: int | None = None,
    locality: str | None = None,
) -> BlockStored:
    return BlockStored(
        block_hashes=[_FAKE_HASH],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
        lora_id=None,
        medium="GPU",
        lora_name=None,
        group_idx=group_idx,
        kv_cache_spec_sliding_window=kv_cache_spec_sliding_window,
        locality=locality,
    )


def _make_block_removed(
    group_idx: int | None = None,
    locality: str | None = None,
) -> BlockRemoved:
    return BlockRemoved(
        block_hashes=[_FAKE_HASH],
        medium="GPU",
        group_idx=group_idx,
        locality=locality,
    )


def test_block_stored_default_group_idx_is_none():
    """group_idx defaults to None when not provided."""
    event = _make_block_stored()
    assert event.group_idx is None


def test_block_removed_default_group_idx_is_none():
    """group_idx defaults to None when not provided."""
    event = _make_block_removed()
    assert event.group_idx is None


@pytest.mark.parametrize("group_idx", [1, 2, 3])
def test_block_stored_hash_differs_by_group_idx(group_idx: int):
    """BlockStored events that differ only in group_idx must hash differently."""
    other_group_idx = group_idx + 1
    event_a = _make_block_stored(group_idx=group_idx)
    event_b = _make_block_stored(group_idx=other_group_idx)
    assert hash(event_a) != hash(event_b)


def test_block_stored_hash_same_for_equal_group_idx():
    """Two BlockStored events with identical fields produce the same hash."""
    event_a = _make_block_stored(group_idx=1)
    event_b = _make_block_stored(group_idx=1)
    assert hash(event_a) == hash(event_b)


@pytest.mark.parametrize("group_idx", [1, 2, 3])
def test_block_removed_hash_differs_by_group_idx(group_idx: int):
    """BlockRemoved events that differ only in group_idx must hash differently."""
    other_group_idx = group_idx + 1
    event_a = _make_block_removed(group_idx=group_idx)
    event_b = _make_block_removed(group_idx=other_group_idx)
    assert hash(event_a) != hash(event_b)


def test_block_removed_hash_same_for_equal_group_idx():
    """Two BlockRemoved events with identical fields produce the same hash."""
    event_a = _make_block_removed(group_idx=1)
    event_b = _make_block_removed(group_idx=1)
    assert hash(event_a) == hash(event_b)


def test_block_stored_hash_differs_by_sliding_window():
    event_a = _make_block_stored(group_idx=1, kv_cache_spec_sliding_window=128)
    event_b = _make_block_stored(group_idx=1, kv_cache_spec_sliding_window=256)
    assert hash(event_a) != hash(event_b)


@pytest.mark.parametrize(
    ("event_a", "event_b"),
    [
        (
            _make_block_stored(locality="LOCAL"),
            _make_block_stored(locality="REMOTE"),
        ),
        (
            _make_block_removed(locality="LOCAL"),
            _make_block_removed(locality="REMOTE"),
        ),
    ],
)
def test_event_hash_differs_by_locality(
    event_a: BlockStored | BlockRemoved,
    event_b: BlockStored | BlockRemoved,
):
    assert hash(event_a) != hash(event_b)


def test_block_stored_locality_is_wire_compatible():
    legacy = _LegacyBlockStored(
        block_hashes=[_FAKE_HASH],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
        lora_id=None,
        medium="GPU",
        lora_name=None,
        group_idx=2,
        kv_cache_spec_sliding_window=128,
    )
    legacy_payload = msgspec.msgpack.encode(legacy)
    assert (
        msgspec.msgpack.encode(
            _make_block_stored(
                group_idx=2,
                kv_cache_spec_sliding_window=128,
            )
        )
        == legacy_payload
    )
    assert msgspec.msgpack.decode(legacy_payload, type=BlockStored).locality is None
    new_payload = msgspec.msgpack.encode(_make_block_stored(locality="LOCAL"))
    assert msgspec.msgpack.decode(new_payload)["locality"] == "LOCAL"
    assert msgspec.msgpack.decode(new_payload, type=_LegacyBlockStored).medium == "GPU"


def test_block_removed_locality_is_wire_compatible():
    legacy = _LegacyBlockRemoved(block_hashes=[_FAKE_HASH], medium="GPU")
    legacy_payload = msgspec.msgpack.encode(legacy)
    assert msgspec.msgpack.encode(_make_block_removed()) == legacy_payload
    assert msgspec.msgpack.decode(legacy_payload, type=BlockRemoved).locality is None
    new_payload = msgspec.msgpack.encode(_make_block_removed(locality="REMOTE"))
    assert msgspec.msgpack.decode(new_payload)["locality"] == "REMOTE"
    assert msgspec.msgpack.decode(new_payload, type=_LegacyBlockRemoved).medium == "GPU"
