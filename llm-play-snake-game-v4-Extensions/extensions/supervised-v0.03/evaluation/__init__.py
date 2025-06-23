"""
Supervised Learning v0.03 - Evaluation Module
============================================

Evaluation utilities and metrics for model assessment.
Clean API for model evaluation and comparison.

Design Pattern: Strategy Pattern
- Multiple evaluation strategies
- Extensible metric computation
- Clean separation of concerns
"""

from .metrics import (
    MetricStrategy,
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1Metric,
    ConfusionMatrixMetric,
    MetricsCalculator,
    evaluate_predictions,
    format_metrics
)

__all__ = [
    "MetricStrategy",
    "AccuracyMetric",
    "PrecisionMetric", 
    "RecallMetric",
    "F1Metric",
    "ConfusionMatrixMetric",
    "MetricsCalculator",
    "evaluate_predictions",
    "format_metrics"
] 