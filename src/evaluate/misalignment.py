"""Misalignment metric: fraction of predicted fields that do NOT exist in the schema.

Measures how often the model invents column names that aren't in the schema.
Returns a normalized rate (0.0 = all fields valid, 1.0 = all fields invalid).
"""

from .base import BaseMetric, SampleContext
from .parsing import extract_fields


class MisalignmentMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "misaligned_fields"

    @property
    def description(self) -> str:
        return "Fraction of predicted fields that do not exist in the schema"

    def compute_sample(self, predicted: str, expected: str, ctx: SampleContext = None):
        pred_fields = list(set(extract_fields(predicted)))

        if not pred_fields or not ctx or not ctx.schema_columns:
            return 0.0

        invalid = sum(1 for f in pred_fields if f not in ctx.schema_columns)
        return invalid / len(pred_fields)
