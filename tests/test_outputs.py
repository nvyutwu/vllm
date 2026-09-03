# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.outputs import RequestOutput

pytestmark = pytest.mark.cpu_test


def test_request_output_forward_compatible():
    output = RequestOutput(
        request_id="test_request_id",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
        example_arg_added_in_new_version="some_value",
    )
    assert output is not None


def test_request_output_carries_cache_loss_counters():
    output = RequestOutput(
        request_id="test_request_id",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=True,
        num_local_cached_tokens=16,
        num_external_cached_tokens=32,
        num_external_lookup_tokens=32,
    )
    assert output.num_local_cached_tokens == 16
    assert output.num_external_cached_tokens == 32
    assert output.num_external_lookup_tokens == 32
