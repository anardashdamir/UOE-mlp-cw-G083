"""Value Accuracy metric: whether the prediction uses the correct values.

Extracts literal values (strings in quotes, numbers, booleans) from both
predicted and expected filters, then computes recall over the value sets.
"""

import re

from .base import BaseMetric


def _extract_values(filter_str: str) -> set[str]:
    """Extract all literal values from a filter expression."""
    values = set()
    # Quoted strings: 'value' or "value"
    for m in re.finditer(r"'([^']*)'|\"([^\"]*)\"", filter_str):
        val = m.group(1) if m.group(1) is not None else m.group(2)
        values.add(val.lower())
    # Numbers (int and float)
    for m in re.finditer(r"(?<!['\"\w])(-?\d+\.?\d*)(?!['\"\w])", filter_str):
        val = m.group(1)
        # Normalize 4.0 -> 4
        try:
            f = float(val)
            val = str(int(f)) if f == int(f) else str(f)
        except ValueError:
            pass
        values.add(val)
    # Booleans
    for m in re.finditer(r"\b(true|false)\b", filter_str, re.IGNORECASE):
        values.add(m.group(1).lower())
    return values


class ValueAccuracyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "value_accuracy"

    @property
    def description(self) -> str:
        return "F1 score over the set of literal values (strings, numbers, booleans)"

    def compute_sample(self, predicted, expected, ctx=None):
        pred_vals = _extract_values(predicted)
        gold_vals = _extract_values(expected)

        if not pred_vals and not gold_vals:
            return 1.0
        if not pred_vals or not gold_vals:
            return 0.0

        tp = len(pred_vals & gold_vals)
        precision = tp / len(pred_vals)
        recall = tp / len(gold_vals)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
