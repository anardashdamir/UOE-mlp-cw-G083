"""Hallucination metric: number of predicted clauses not in the ground truth."""

from .base import BaseMetric, SampleContext
from .parsing import parse_filters


class HallucinationMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "hallucination_rate"

    @property
    def description(self) -> str:
        return "Average number of predicted clauses that do not appear in the ground truth"

    def compute_sample(self, predicted: str, expected: str, ctx: SampleContext = None):
        pred_set = parse_filters(predicted)
        gold_set = parse_filters(expected)
        return float(len(pred_set - gold_set))
