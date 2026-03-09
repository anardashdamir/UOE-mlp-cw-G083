"""Structural Validity metric: whether the prediction is a syntactically valid filter.

Checks:
- Non-empty output
- Balanced parentheses
- Contains at least one valid operator
- No garbage text (model rambling instead of outputting filters)
"""

from .base import BaseMetric
from .parsing import is_valid_syntax


class StructuralValidityMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "structural_validity"

    @property
    def description(self) -> str:
        return "Fraction of predictions that are syntactically valid filter expressions"

    def compute_sample(self, predicted, expected, ctx=None):
        return 1.0 if is_valid_syntax(predicted) else 0.0
