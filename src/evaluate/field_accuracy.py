"""Field Accuracy metric: fraction of predicted fields that exist in the schema."""

from .base import BaseMetric, SampleContext
from .parsing import extract_fields


class FieldAccuracyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "field_accuracy"

    @property
    def description(self) -> str:
        return "Fraction of predicted field names that are valid schema columns"

    def compute_sample(self, predicted: str, expected: str, ctx: SampleContext = None):
        pred_fields = extract_fields(predicted)

        if not pred_fields:
            return 1.0 if not predicted.strip() else 0.0
        if not ctx or not ctx.schema_columns:
            return 0.0

        valid = sum(1 for f in pred_fields if f in ctx.schema_columns)
        return valid / len(pred_fields)
