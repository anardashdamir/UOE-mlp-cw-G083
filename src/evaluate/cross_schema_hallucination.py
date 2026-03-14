"""Cross-Schema Hallucination: predicted fields that belong to other schemas."""

from .base import BaseMetric, SampleContext
from .parsing import extract_fields


class CrossSchemaMetric(BaseMetric):
    def __init__(self, all_schema_fields: dict[str, set[str]]):
        """
        Args:
            all_schema_fields: dict mapping schema name -> set of column names
                               e.g. {"used_cars": {"brand", "price"}, "hotel_bookings": {"rooms", "city"}}
        """
        self._all_fields = all_schema_fields

    @property
    def name(self) -> str:
        return "cross_schema_hallucination"

    @property
    def description(self) -> str:
        return "Fraction of invalid predicted fields that exist in other training schemas"

    def compute_sample(self, predicted, expected, ctx=None):
        if not ctx or not ctx.schema_columns or not ctx.schema_name:
            return 0.0

        pred_fields = set(extract_fields(predicted))
        # Find fields not in the current schema
        invalid = {f for f in pred_fields if f not in ctx.schema_columns}

        if not invalid:
            return 0.0

        # Check how many invalid fields exist in OTHER schemas
        other_fields = set()
        for name, cols in self._all_fields.items():
            if name != ctx.schema_name:
                other_fields.update(cols)

        leaked = sum(1 for f in invalid if f in other_fields)
        return leaked / len(invalid)