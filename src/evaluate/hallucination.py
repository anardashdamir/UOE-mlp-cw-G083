"""Hallucination metric: number of predicted clauses not in the ground truth."""

from .base import BaseMetric, SampleContext
from .parsing import parse_filters


class HallucinationMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "hallucination_rate"

    @property
    def description(self) -> str:
        return "Fraction of predicted clauses that do not appear in the ground truth"

    def compute_sample(self, predicted: str, expected: str, ctx: SampleContext = None):
        pred_set = parse_filters(predicted)
        gold_set = parse_filters(expected)
        
        extra = len(pred_set - gold_set)

        if not pred_set:
            return 0.0
        # Normalized: what fraction of predictions are hallucinated
        return extra / len(pred_set)

