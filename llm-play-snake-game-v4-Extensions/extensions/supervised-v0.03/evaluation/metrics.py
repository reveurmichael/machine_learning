"""
Evaluation metrics for supervised learning v0.03.

Design Pattern: Strategy Pattern
- Multiple evaluation strategies
- Extensible metric computation
- Clean separation of concerns
"""

from typing import Dict, Any, List, Optional
import numpy as np
from abc import ABC, abstractmethod


class MetricStrategy(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric value."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return metric name."""
        pass


class AccuracyMetric(MetricStrategy):
    """Accuracy metric implementation."""
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(y_true == y_pred)
    
    @property
    def name(self) -> str:
        return "accuracy"


class PrecisionMetric(MetricStrategy):
    """Precision metric implementation."""
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute precision for each class and return macro average."""
        from sklearn.metrics import precision_score
        return precision_score(y_true, y_pred, average='macro', zero_division=0)
    
    @property
    def name(self) -> str:
        return "precision"


class RecallMetric(MetricStrategy):
    """Recall metric implementation."""
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute recall for each class and return macro average."""
        from sklearn.metrics import recall_score
        return recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    @property
    def name(self) -> str:
        return "recall"


class F1Metric(MetricStrategy):
    """F1 score metric implementation."""
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute F1 score for each class and return macro average."""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    @property
    def name(self) -> str:
        return "f1_score"


class ConfusionMatrixMetric(MetricStrategy):
    """Confusion matrix metric implementation."""
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)
    
    @property
    def name(self) -> str:
        return "confusion_matrix"


class MetricsCalculator:
    """Calculator for multiple evaluation metrics."""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """Initialize with specified metrics."""
        self.metrics = metrics or ["accuracy", "precision", "recall", "f1_score"]
        self.strategies = self._create_strategies()
    
    def _create_strategies(self) -> Dict[str, MetricStrategy]:
        """Create metric strategy instances."""
        strategy_map = {
            "accuracy": AccuracyMetric(),
            "precision": PrecisionMetric(),
            "recall": RecallMetric(),
            "f1_score": F1Metric(),
            "confusion_matrix": ConfusionMatrixMetric()
        }
        
        return {name: strategy_map[name] for name in self.metrics if name in strategy_map}
    
    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute all specified metrics."""
        results = {}
        
        for metric_name, strategy in self.strategies.items():
            try:
                value = strategy.compute(y_true, y_pred)
                results[metric_name] = value
            except Exception as e:
                results[metric_name] = f"Error: {str(e)}"
        
        return results
    
    def compute_single(self, metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Any:
        """Compute a single metric."""
        if metric_name not in self.strategies:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        return self.strategies[metric_name].compute(y_true, y_pred)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function for evaluating predictions."""
    calculator = MetricsCalculator(metrics)
    return calculator.compute_all(y_true, y_pred)


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics for display."""
    lines = []
    lines.append("Evaluation Metrics:")
    lines.append("-" * 30)
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{metric_name}: {value:.4f}")
        elif isinstance(value, np.ndarray):
            lines.append(f"{metric_name}:")
            lines.append(str(value))
        else:
            lines.append(f"{metric_name}: {value}")
    
    return "\n".join(lines) 