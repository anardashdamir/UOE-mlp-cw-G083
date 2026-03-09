"""Abstract base class for all evaluation metrics.

To create a new metric:
1. Create a new file in src/evaluate/ (e.g., my_metric.py)
2. Subclass BaseMetric
3. Implement the `name` property and `compute_sample()` method
4. Optionally override `aggregate()` for custom aggregation
5. Register your metric in orchestrator.py's METRICS list

Example:
    from src.evaluate.base import BaseMetric, SampleContext

    class MyMetric(BaseMetric):
        @property
        def name(self) -> str:
            return "my_metric"

        @property
        def description(self) -> str:
            return "Measures something useful about filter predictions"

        def compute_sample(self, predicted, expected, ctx):
            # Your logic here — use ctx.schema_columns, ctx.latency_ms, etc.
            return 0.95  # Return a float or a dict of floats
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class SampleContext(BaseModel):
    """Context passed to each metric's compute_sample() for a single eval sample.

    Provides all metadata about the sample beyond predicted/expected strings.
    """
    schema_columns: set[str] = set()
    schema_name: str = "unknown"
    difficulty: str = "unknown"
    latency_ms: float = 0.0

    class Config:
        frozen = True


class EvaluationResult(BaseModel):
    """Structured result from a single evaluation run."""
    quantization: str = "fp16"
    overall: dict[str, float] = {}
    per_schema: dict[str, dict[str, list[float]]] = {}
    per_difficulty: dict[str, dict[str, list[float]]] = {}
    predictions: list[str] = []


class BaseMetric(ABC):
    """Abstract base class that all evaluation metrics must implement.

    Attributes:
        name: Short identifier for the metric (e.g., 'precision', 'f1').
        description: Human-readable explanation of what the metric measures.

    Methods:
        compute_sample: Compute the metric for a single (predicted, expected) pair.
        aggregate: Aggregate per-sample results into summary statistics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this metric (used in output tables)."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this metric measures."""
        return ""

    @abstractmethod
    def compute_sample(
        self,
        predicted: str,
        expected: str,
        ctx: SampleContext | None = None,
    ) -> float | dict[str, float]:
        """Compute metric for a single sample.

        Args:
            predicted: Model's predicted filter expression.
            expected: Ground truth filter expression.
            ctx: SampleContext with schema_columns, schema_name, difficulty, latency_ms.

        Returns:
            A single float score (0.0 to 1.0 for most metrics)
            OR a dict of named float scores (e.g., {"precision": 0.8, "valid": 5}).
        """
        ...

    def aggregate(self, sample_results: list[float] | list[dict[str, float]]) -> dict[str, float]:
        """Aggregate per-sample results into summary statistics.

        Default implementation computes the mean across all samples.
        Override this for custom aggregation (e.g., micro-averaging).

        Args:
            sample_results: List of values returned by compute_sample().

        Returns:
            Dict mapping metric names to aggregated values.
        """
        if not sample_results:
            return {self.name: 0.0}

        if isinstance(sample_results[0], dict):
            keys = sample_results[0].keys()
            return {
                k: sum(r[k] for r in sample_results) / len(sample_results)
                for k in keys
            }

        return {self.name: sum(sample_results) / len(sample_results)}
