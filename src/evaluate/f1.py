"""F1 metric: harmonic mean of precision and recall."""

from .base import BaseMetric, SampleContext
from .parsing import parse_filters


class F1Metric(BaseMetric):
    @property
    def name(self) -> str:
        return "f1"

    @property
    def description(self) -> str:
        return "Harmonic mean of precision and recall at clause level"

    def compute_sample(self, predicted: str, expected: str, ctx: SampleContext = None):
        pred_set = parse_filters(predicted)
        gold_set = parse_filters(expected)

        if not pred_set and not gold_set:
            return 1.0
        if not pred_set or not gold_set:
            return 0.0

        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set)
        recall = tp / len(gold_set)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
