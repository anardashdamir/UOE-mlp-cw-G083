"""Complexity Accuracy metric: whether the model produces the right number of clauses.

Compares the number of top-level AND clauses in the prediction vs the ground truth.
A score of 1.0 means the model generated the exact same number of clauses.
"""

from .base import BaseMetric
from .parsing import count_clauses


class ComplexityAccuracyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "complexity_accuracy"

    @property
    def description(self) -> str:
        return "Fraction of predictions with the correct number of filter clauses"

    def compute_sample(self, predicted, expected, ctx=None):
        pred_count = count_clauses(predicted)
        gold_count = count_clauses(expected)

        if pred_count == gold_count:
            return 1.0
        return 0.0
