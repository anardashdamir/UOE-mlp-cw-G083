"""Exact Match metric: whether the predicted filter is identical to the expected."""

from .base import BaseMetric, SampleContext
from .parsing import parse_filters


class ExactMatchMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "exact_match"

    @property
    def description(self) -> str:
        return "1.0 if all predicted clauses exactly match all ground truth clauses, else 0.0"

    def compute_sample(self, predicted: str, expected: str, ctx: SampleContext = None):
        pred_set = parse_filters(predicted)
        gold_set = parse_filters(expected)
        return 1.0 if pred_set == gold_set else 0.0
