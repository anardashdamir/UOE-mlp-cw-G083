"""Misalignment metric: fields in the prediction that do NOT exist in the schema.

Unlike field_accuracy (which is a ratio), this counts the raw number of
invalid fields per sample — useful for understanding how many hallucinated
columns the model invents.
"""

from .base import BaseMetric, SampleContext
from .parsing import extract_fields


class MisalignmentMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "misaligned_fields"

    @property
    def description(self) -> str:
        return "Average number of predicted fields that do not exist in the schema"

    def compute_sample(self, predicted: str, expected: str, ctx: SampleContext = None):
        pred_fields = list(set(extract_fields(predicted)))

        if not pred_fields or not ctx or not ctx.schema_columns:
            return 0.0

        invalid = sum(1 for f in pred_fields if f not in ctx.schema_columns)
        return float(invalid)
