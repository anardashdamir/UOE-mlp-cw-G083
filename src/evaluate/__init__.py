"""Evaluation package for AutoFilter.

Provides a modular, metric-based evaluation framework.
Each metric is a separate file implementing BaseMetric.
The orchestrator runs batched inference and computes all metrics.
"""

from .orchestrator import main

__all__ = ["main"]
