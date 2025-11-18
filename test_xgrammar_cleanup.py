#!/usr/bin/env python3
"""
Test script to verify xgrammar memory cleanup.
This script helps verify that the nanobind memory leak fix is working.

Run this to test the fix:
1. Start a vllm server with structured outputs
2. Watch for nanobind warnings when you stop the server
"""

import gc
import sys


def test_xgrammar_cleanup():
    """Test that xgrammar objects are properly cleaned up."""
    try:
        import xgrammar as xgr
        from vllm.config import (
            ModelConfig,
            SchedulerConfig,
            StructuredOutputsConfig,
            VllmConfig,
        )
        from vllm.sampling_params import SamplingParams, StructuredOutputsParams
        from vllm.transformers_utils.tokenizer import get_tokenizer
        from vllm.v1.structured_output.backend_xgrammar import (
            XgrammarBackend,
            XgrammarGrammar,
        )

        print("Testing xgrammar cleanup...")

        # Create a minimal config
        model_config = ModelConfig(
            model="gpt2",
            task="generate",
            tokenizer="gpt2",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float16",
            seed=0,
        )

        vllm_config = VllmConfig(
            model_config=model_config,
            scheduler_config=SchedulerConfig(),
            structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
        )

        # Get tokenizer
        tokenizer = get_tokenizer("gpt2")
        vocab_size = len(tokenizer.get_vocab())

        # Create backend
        print(f"Creating XgrammarBackend with vocab_size={vocab_size}...")
        backend = XgrammarBackend(vllm_config, tokenizer=tokenizer, vocab_size=vocab_size)

        # Compile a simple JSON schema
        schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        print(f"Compiling JSON schema: {schema}")
        grammar = backend.compile_grammar(
            backend.vllm_config.structured_outputs_config.backend, schema
        )

        print(f"Grammar created: {type(grammar)}")
        print(f"  - matcher: {type(grammar.matcher)}")
        print(f"  - ctx: {type(grammar.ctx)}")

        # Clean up
        print("\nCleaning up...")
        del grammar
        gc.collect()
        print("Deleted grammar object")

        backend.destroy()
        print("Destroyed backend")

        del backend
        gc.collect()
        print("Deleted backend object and ran garbage collection")

        print("\n✅ Cleanup test completed successfully!")
        print("If you see nanobind leak warnings above, the fix may need adjustment.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_xgrammar_cleanup()
    sys.exit(0 if success else 1)

