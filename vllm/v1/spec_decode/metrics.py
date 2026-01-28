# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass, field

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
        )

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1


class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        draft_throughput = 0
        accepted_throughput = 0

        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
            draft_acceptance_rate,
        )
        self.reset()


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]
    """

    _counter_cls = prometheus_client.Counter
    _gauge_cls = prometheus_client.Gauge

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.spec_decoding_enabled = speculative_config is not None
        if not self.spec_decoding_enabled:
            return

        counter_drafts = self._counter_cls(
            name="spec_decode_num_drafts",
            documentation="Number of spec decoding drafts.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_drafts = make_per_engine(
            counter_drafts, per_engine_labelvalues
        )

        counter_draft_tokens = self._counter_cls(
            name="spec_decode_num_draft_tokens",
            documentation="Number of draft tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_draft_tokens = make_per_engine(
            counter_draft_tokens, per_engine_labelvalues
        )

        counter_accepted_tokens = self._counter_cls(
            name="spec_decode_num_accepted_tokens",
            documentation="Number of accepted tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens = make_per_engine(
            counter_accepted_tokens, per_engine_labelvalues
        )

        assert speculative_config is not None
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if self.spec_decoding_enabled
            else 0
        )
        pos_labelnames = labelnames + ["position"]
        base_counter = self._counter_cls(
            name="spec_decode_num_accepted_tokens_per_pos",
            documentation="Accepted tokens per draft position.",
            labelnames=pos_labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)]
            for idx, lv in per_engine_labelvalues.items()
        }

        # SGLang-compatible: Mean acceptance length gauge
        gauge_spec_accept_length = self._gauge_cls(
            name="spec_accept_length",
            documentation=(
                "Average accepted sequence length in speculative decoding "
                "(including bonus token)."
            ),
            labelnames=labelnames,
            multiprocess_mode="mostrecent",
        )
        self.gauge_spec_accept_length = {
            idx: gauge_spec_accept_length.labels(*lv)
            for idx, lv in per_engine_labelvalues.items()
        }

        # SGLang-compatible: Draft acceptance rate gauge
        gauge_spec_accept_rate = self._gauge_cls(
            name="spec_accept_rate",
            documentation=(
                "Draft acceptance rate in speculative decoding "
                "(accepted_tokens / draft_tokens, 0-1)."
            ),
            labelnames=labelnames,
            multiprocess_mode="mostrecent",
        )
        self.gauge_spec_accept_rate = {
            idx: gauge_spec_accept_rate.labels(*lv)
            for idx, lv in per_engine_labelvalues.items()
        }

        # Per-interval accumulators for gauge computation.
        # Accumulated across observe() calls and flushed to gauges
        # periodically via flush_gauges(). Cumulative values are
        # available from the Prometheus counters above via PromQL.
        self._interval_drafts: dict[int, int] = {
            idx: 0 for idx in per_engine_labelvalues
        }
        self._interval_draft_tokens: dict[int, int] = {
            idx: 0 for idx in per_engine_labelvalues
        }
        self._interval_accepted_tokens: dict[int, int] = {
            idx: 0 for idx in per_engine_labelvalues
        }

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        if not self.spec_decoding_enabled:
            return
        # Increment monotonic counters (cumulative, for PromQL rate())
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])

        # Accumulate into per-interval buffers for gauges
        self._interval_drafts[engine_idx] += spec_decoding_stats.num_drafts
        self._interval_draft_tokens[engine_idx] += (
            spec_decoding_stats.num_draft_tokens
        )
        self._interval_accepted_tokens[engine_idx] += (
            spec_decoding_stats.num_accepted_tokens
        )

    def flush_gauges(self):
        """Compute gauges from per-interval accumulators and reset.

        Called periodically by PrometheusStatLogger.log() to update
        spec_accept_length and spec_accept_rate gauges with values
        from the current logging interval, matching SGLang's behavior.
        """
        if not self.spec_decoding_enabled:
            return
        for engine_idx in self._interval_drafts:
            drafts = self._interval_drafts[engine_idx]
            draft_tokens = self._interval_draft_tokens[engine_idx]
            accepted_tokens = self._interval_accepted_tokens[engine_idx]

            if drafts > 0:
                mean_accept_length = 1 + (accepted_tokens / drafts)
                self.gauge_spec_accept_length[engine_idx].set(
                    mean_accept_length
                )

            if draft_tokens > 0:
                accept_rate = accepted_tokens / draft_tokens
                self.gauge_spec_accept_rate[engine_idx].set(accept_rate)

        # Reset interval accumulators
        for engine_idx in self._interval_drafts:
            self._interval_drafts[engine_idx] = 0
            self._interval_draft_tokens[engine_idx] = 0
            self._interval_accepted_tokens[engine_idx] = 0


def make_per_engine(
    counter: prometheus_client.Counter,
    per_engine_labelvalues: dict[int, list[object]],
):
    """Create a counter for each label value."""
    return {
        idx: counter.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }
