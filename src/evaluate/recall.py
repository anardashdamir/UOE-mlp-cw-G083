"""Recall metric: fraction of ground truth clauses that were predicted."""

from .base import BaseMetric, SampleContext
from .parsing import parse_filters


class RecallMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "recall"

    @property
    def description(self) -> str:
        return "Fraction of ground truth filter clauses found in the prediction"

    def compute_sample(self, predicted: str, expected: str, ctx: SampleContext = None):
        pred_set = parse_filters(predicted)
        gold_set = parse_filters(expected)

        if not pred_set and not gold_set:
            return 1.0
        if not gold_set:
            return 0.0
        if not pred_set:
            return 0.0

        tp = len(pred_set & gold_set)
        return tp / len(gold_set)
