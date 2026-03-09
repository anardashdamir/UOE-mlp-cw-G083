"""Latency metric: inference time per sample in milliseconds.

The orchestrator passes `latency_ms` via SampleContext for each sample.
This metric simply collects and aggregates those values.
"""

from .base import BaseMetric, SampleContext


class LatencyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "latency_ms"

    @property
    def description(self) -> str:
        return "Average inference latency per sample in milliseconds"

    def compute_sample(self, predicted: str, expected: str, ctx: SampleContext = None):
        return ctx.latency_ms if ctx else 0.0

    def aggregate(self, sample_results: list[float]) -> dict[str, float]:
        if not sample_results:
            return {"latency_ms_avg": 0.0, "latency_ms_p50": 0.0, "latency_ms_p95": 0.0}

        sorted_vals = sorted(sample_results)
        n = len(sorted_vals)
        return {
            "latency_ms_avg": sum(sorted_vals) / n,
            "latency_ms_p50": sorted_vals[n // 2],
            "latency_ms_p95": sorted_vals[int(n * 0.95)],
        }
