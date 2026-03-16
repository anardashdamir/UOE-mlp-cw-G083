"""Operator Accuracy metric: whether the prediction uses the correct operators.

Extracts the set of operators (==, !=, >, >=, <, <=, IN, NOT IN) from both
predicted and expected filters, then computes F1 over the operator multisets.
"""

import re
from collections import Counter

from .base import BaseMetric


_OPERATOR_PATTERN = re.compile(
    r"(==|!=|>=|<=|>|<|\bNOT\s+IN\b|\bIN\b)", re.IGNORECASE
)


def _extract_operators(filter_str: str) -> list[str]:
    """Extract all operators from a filter expression."""
    return [m.upper() for m in _OPERATOR_PATTERN.findall(filter_str)]


class OperatorAccuracyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "operator_accuracy"

    @property
    def description(self) -> str:
        return "F1 score over the multiset of operators used in predicted vs expected"

    def compute_sample(self, predicted, expected, ctx=None):
        pred_ops = Counter(_extract_operators(predicted))
        gold_ops = Counter(_extract_operators(expected))

        if not pred_ops and not gold_ops:
            return 1.0
        if not pred_ops or not gold_ops:
            return 0.0

        # Multiset intersection
        common = sum((pred_ops & gold_ops).values())
        precision = common / sum(pred_ops.values())
        recall = common / sum(gold_ops.values())

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
