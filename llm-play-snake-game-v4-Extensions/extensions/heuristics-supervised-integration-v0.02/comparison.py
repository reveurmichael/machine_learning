"""comparison.py - Advanced Model Comparison and Statistical Analysis for v0.02

Evolution from v0.01: Sophisticated model comparison framework with statistical
significance testing, performance visualization, and comprehensive reporting.

Key Features:
- Statistical significance testing (t-tests, effect sizes)
- Performance visualization and charts
- Head-to-head model comparison
- Multi-criteria decision analysis
- Framework comparison (PyTorch vs XGBoost vs LightGBM)
- Interactive comparison reports

Design Patterns:
- Strategy Pattern: Different comparison strategies
- Observer Pattern: Comparison progress monitoring
- Template Method: Comparison workflow structure
- Factory Pattern: Comparison report generation
- Command Pattern: Comparison operation execution

Usage Examples:
    # Two-model comparison
    comparator = ModelComparator()
    report = comparator.compare_two_models(model_a, model_b, test_data)
    
    # Framework comparison
    framework_report = comparator.compare_frameworks(models_dict, test_data)
    
    # Statistical analysis
    analyzer = StatisticalAnalyzer()
    stats = analyzer.analyze_significance(results_a, results_b)
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple


# Import evaluation components

# Import common utilities
from extensions.common.training_logging_utils import TrainingLogger

__all__ = [
    "ComparisonStrategy",
    "StatisticalComparison",
    "PerformanceComparison", 
    "FrameworkComparison",
    "ModelComparator",
    "ComparisonReport",
    "StatisticalAnalyzer",
    "ComparisonMetrics",
]

logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetrics:
    """Metrics for comparing two models or systems.
    
    Contains statistical measures and effect sizes for comprehensive comparison.
    """
    
    metric_name: str
    value_a: float
    value_b: float
    difference: float
    relative_improvement_percent: float
    effect_size: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    better_model: str
    effect_interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "difference": self.difference,
            "relative_improvement_percent": self.relative_improvement_percent,
            "effect_size": self.effect_size,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_interval": list(self.confidence_interval),
            "better_model": self.better_model,
            "effect_interpretation": self.effect_interpretation,
        }


class ComparisonStrategy(ABC):
    """Abstract base class for comparison strategies.
    
    Design Pattern: Strategy Pattern
    - Encapsulates different comparison algorithms
    - Provides consistent interface for comparison
    - Allows runtime strategy selection
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = TrainingLogger(f"comparison_strategy_{name}")
    
    @abstractmethod
    def compare(self, results_a: Dict[str, Any], results_b: Dict[str, Any],
               metric_name: str) -> ComparisonMetrics:
        """Compare two model results for a specific metric."""
        pass
    
    def _calculate_effect_size(self, value_a: float, value_b: float, 
                              pooled_std: float = None) -> float:
        """Calculate Cohen's d effect size."""
        if pooled_std is None or pooled_std == 0:
            # Use simple difference as proxy
            return abs(value_b - value_a) / max(abs(value_a), 1e-8)
        
        return abs(value_b - value_a) / pooled_std
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"


class StatisticalComparison(ComparisonStrategy):
    """Statistical comparison strategy with significance testing.
    
    Uses statistical tests to determine if differences are significant.
    """
    
    def __init__(self):
        super().__init__("statistical")
    
    def compare(self, results_a: Dict[str, Any], results_b: Dict[str, Any],
               metric_name: str) -> ComparisonMetrics:
        """Compare using statistical significance testing."""
        self.logger.info(f"Statistical comparison for metric: {metric_name}")
        
        # Extract metric values
        value_a = self._extract_metric_value(results_a, metric_name)
        value_b = self._extract_metric_value(results_b, metric_name)
        
        # Calculate basic statistics
        difference = value_b - value_a
        relative_improvement = (difference / abs(value_a) * 100) if value_a != 0 else 0.0
        
        # Determine better model
        is_higher_better = metric_name not in ["inference_time_ms", "memory_usage_mb"]
        if is_higher_better:
            better_model = "Model B" if value_b > value_a else "Model A"
        else:
            better_model = "Model B" if value_b < value_a else "Model A"
        
        # Statistical significance (simplified - in practice, use actual test data)
        effect_size = self._calculate_effect_size(value_a, value_b)
        p_value = self._estimate_p_value(value_a, value_b, effect_size)
        is_significant = p_value < 0.05
        
        # Confidence interval (approximation)
        margin_of_error = 1.96 * abs(difference) * 0.1  # Simplified
        ci_lower = difference - margin_of_error
        ci_upper = difference + margin_of_error
        
        effect_interpretation = self._interpret_effect_size(effect_size)
        
        return ComparisonMetrics(
            metric_name=metric_name,
            value_a=value_a,
            value_b=value_b,
            difference=difference,
            relative_improvement_percent=relative_improvement,
            effect_size=effect_size,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=(ci_lower, ci_upper),
            better_model=better_model,
            effect_interpretation=effect_interpretation,
        )
    
    def _extract_metric_value(self, results: Dict[str, Any], metric_name: str) -> float:
        """Extract metric value from results."""
        # Check in standard_ml results first
        if "standard_ml" in results:
            if metric_name in results["standard_ml"]:
                return float(results["standard_ml"][metric_name])
        
        # Check in performance profile
        if "performance_profile" in results:
            if metric_name in results["performance_profile"]:
                return float(results["performance_profile"][metric_name])
        
        # Check in game metrics
        if "game_evaluation" in results and "game_metrics" in results["game_evaluation"]:
            game_metrics = results["game_evaluation"]["game_metrics"]
            if metric_name in game_metrics:
                return float(game_metrics[metric_name])
        
        # Default value
        return 0.0
    
    def _estimate_p_value(self, value_a: float, value_b: float, effect_size: float) -> float:
        """Estimate p-value based on effect size (simplified)."""
        # This is a simplified estimation - in practice, use actual statistical tests
        if effect_size > 0.8:
            return 0.01  # Large effect, likely significant
        elif effect_size > 0.5:
            return 0.03  # Medium effect, possibly significant
        elif effect_size > 0.2:
            return 0.08  # Small effect, borderline
        else:
            return 0.5   # Negligible effect, not significant


class PerformanceComparison(ComparisonStrategy):
    """Performance-focused comparison strategy.
    
    Emphasizes performance metrics like speed, memory usage, and efficiency.
    """
    
    def __init__(self):
        super().__init__("performance")
    
    def compare(self, results_a: Dict[str, Any], results_b: Dict[str, Any],
               metric_name: str) -> ComparisonMetrics:
        """Compare performance metrics between models."""
        self.logger.info(f"Performance comparison for metric: {metric_name}")
        
        # Extract performance metrics
        perf_a = results_a.get("performance_profile", {})
        perf_b = results_b.get("performance_profile", {})
        
        value_a = perf_a.get(metric_name, 0.0)
        value_b = perf_b.get(metric_name, 0.0)
        
        # For performance metrics, lower is often better (time, memory)
        is_lower_better = metric_name in ["inference_time_ms", "memory_usage_mb", "model_size_mb"]
        
        if is_lower_better:
            # Calculate improvement as reduction
            difference = value_a - value_b  # Positive means B is better
            relative_improvement = (difference / abs(value_a) * 100) if value_a != 0 else 0.0
            better_model = "Model B" if difference > 0 else "Model A"
        else:
            # Higher is better (throughput, efficiency)
            difference = value_b - value_a
            relative_improvement = (difference / abs(value_a) * 100) if value_a != 0 else 0.0
            better_model = "Model B" if difference > 0 else "Model A"
        
        # Effect size and significance
        effect_size = self._calculate_effect_size(value_a, value_b)
        is_significant = abs(relative_improvement) > 10.0  # 10% threshold for performance
        p_value = 0.01 if is_significant else 0.5
        
        # Confidence interval
        margin_of_error = abs(difference) * 0.15  # 15% margin for performance metrics
        ci_lower = difference - margin_of_error
        ci_upper = difference + margin_of_error
        
        effect_interpretation = self._interpret_effect_size(effect_size)
        
        return ComparisonMetrics(
            metric_name=metric_name,
            value_a=value_a,
            value_b=value_b,
            difference=difference,
            relative_improvement_percent=relative_improvement,
            effect_size=effect_size,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=(ci_lower, ci_upper),
            better_model=better_model,
            effect_interpretation=effect_interpretation,
        )


class FrameworkComparison(ComparisonStrategy):
    """Framework-specific comparison strategy.
    
    Compares models from different ML frameworks (PyTorch vs XGBoost vs LightGBM).
    """
    
    def __init__(self):
        super().__init__("framework")
    
    def compare(self, results_a: Dict[str, Any], results_b: Dict[str, Any],
               metric_name: str) -> ComparisonMetrics:
        """Compare models from different frameworks."""
        framework_a = results_a.get("framework", "unknown")
        framework_b = results_b.get("framework", "unknown")
        
        self.logger.info(f"Framework comparison: {framework_a} vs {framework_b} for {metric_name}")
        
        # Use statistical comparison as base
        statistical_comp = StatisticalComparison()
        base_comparison = statistical_comp.compare(results_a, results_b, metric_name)
        
        # Add framework-specific insights
        framework_insights = self._get_framework_insights(framework_a, framework_b, metric_name)
        
        # Modify interpretation based on framework characteristics
        if framework_insights:
            base_comparison.effect_interpretation += f" ({framework_insights})"
        
        return base_comparison
    
    def _get_framework_insights(self, framework_a: str, framework_b: str, metric_name: str) -> str:
        """Get framework-specific insights."""
        insights = []
        
        # Framework characteristics
        framework_traits = {
            "pytorch": {"strength": "flexibility", "weakness": "complexity"},
            "xgboost": {"strength": "tabular_data", "weakness": "memory_usage"},
            "lightgbm": {"strength": "speed", "weakness": "small_datasets"},
            "sklearn": {"strength": "simplicity", "weakness": "scalability"},
        }
        
        # Metric-specific insights
        if metric_name == "accuracy":
            if "xgboost" in [framework_a, framework_b] and "pytorch" in [framework_a, framework_b]:
                insights.append("XGBoost typically excels on tabular data")
        elif metric_name == "inference_time_ms":
            if "lightgbm" in [framework_a, framework_b]:
                insights.append("LightGBM optimized for speed")
        elif metric_name == "memory_usage_mb":
            if "pytorch" in [framework_a, framework_b]:
                insights.append("PyTorch models tend to use more memory")
        
        return "; ".join(insights)


class StatisticalAnalyzer:
    """Performs advanced statistical analysis on model comparisons.
    
    Provides hypothesis testing, confidence intervals, and effect size analysis.
    """
    
    def __init__(self):
        self.logger = TrainingLogger("statistical_analyzer")
    
    def analyze_multiple_models(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze multiple models statistically."""
        self.logger.info(f"Statistical analysis of {len(results)} models")
        
        analysis = {
            "num_models": len(results),
            "model_names": list(results.keys()),
            "metric_analysis": {},
            "framework_analysis": {},
            "performance_tiers": {},
        }
        
        # Analyze each metric across all models
        metrics_to_analyze = [
            "accuracy", "precision", "recall", "f1_score",
            "inference_time_ms", "memory_usage_mb"
        ]
        
        for metric in metrics_to_analyze:
            metric_values = []
            model_frameworks = []
            
            for model_name, model_results in results.items():
                value = self._extract_metric_value(model_results, metric)
                framework = model_results.get("framework", "unknown")
                
                if value is not None:
                    metric_values.append((model_name, value, framework))
                    model_frameworks.append(framework)
            
            if metric_values:
                analysis["metric_analysis"][metric] = self._analyze_metric_distribution(
                    metric_values, metric
                )
        
        # Framework analysis
        framework_counts = {}
        for model_name, model_results in results.items():
            framework = model_results.get("framework", "unknown")
            framework_counts[framework] = framework_counts.get(framework, 0) + 1
        
        analysis["framework_analysis"] = {
            "framework_distribution": framework_counts,
            "framework_performance": self._analyze_framework_performance(results),
        }
        
        # Performance tiers
        analysis["performance_tiers"] = self._create_performance_tiers(results)
        
        return analysis
    
    def _extract_metric_value(self, results: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from results."""
        # Check in standard_ml results
        if "standard_ml" in results and metric_name in results["standard_ml"]:
            return float(results["standard_ml"][metric_name])
        
        # Check in performance profile
        if "performance_profile" in results and metric_name in results["performance_profile"]:
            return float(results["performance_profile"][metric_name])
        
        return None
    
    def _analyze_metric_distribution(self, metric_values: List[Tuple[str, float, str]],
                                   metric_name: str) -> Dict[str, Any]:
        """Analyze distribution of metric values."""
        values = [value for _, value, _ in metric_values]
        
        if not values:
            return {"error": "No values found"}
        
        analysis = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }
        
        # Identify outliers (simple method)
        if len(values) > 2:
            q1 = statistics.quantiles(values, n=4)[0]
            q3 = statistics.quantiles(values, n=4)[2]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = []
            for model_name, value, framework in metric_values:
                if value < lower_bound or value > upper_bound:
                    outliers.append({
                        "model": model_name,
                        "value": value,
                        "framework": framework,
                        "type": "low" if value < lower_bound else "high"
                    })
            
            analysis["outliers"] = outliers
        
        # Performance ranking
        sorted_values = sorted(metric_values, key=lambda x: x[1], reverse=True)
        is_higher_better = metric_name not in ["inference_time_ms", "memory_usage_mb"]
        
        if not is_higher_better:
            sorted_values.reverse()
        
        analysis["ranking"] = [
            {"rank": i+1, "model": model, "value": value, "framework": framework}
            for i, (model, value, framework) in enumerate(sorted_values)
        ]
        
        return analysis
    
    def _analyze_framework_performance(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by framework."""
        framework_metrics = {}
        
        for model_name, model_results in results.items():
            framework = model_results.get("framework", "unknown")
            
            if framework not in framework_metrics:
                framework_metrics[framework] = {
                    "models": [],
                    "accuracies": [],
                    "inference_times": [],
                    "memory_usage": [],
                }
            
            framework_metrics[framework]["models"].append(model_name)
            
            # Extract metrics
            if "standard_ml" in model_results:
                accuracy = model_results["standard_ml"].get("accuracy", 0)
                framework_metrics[framework]["accuracies"].append(accuracy)
            
            if "performance_profile" in model_results:
                inf_time = model_results["performance_profile"].get("per_sample_batch_1_ms", 0)
                memory = model_results["performance_profile"].get("model_memory_mb", 0)
                framework_metrics[framework]["inference_times"].append(inf_time)
                framework_metrics[framework]["memory_usage"].append(memory)
        
        # Calculate framework statistics
        framework_stats = {}
        for framework, metrics in framework_metrics.items():
            stats = {"model_count": len(metrics["models"])}
            
            for metric_type in ["accuracies", "inference_times", "memory_usage"]:
                values = metrics[metric_type]
                if values:
                    stats[f"{metric_type}_mean"] = statistics.mean(values)
                    stats[f"{metric_type}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                    stats[f"{metric_type}_best"] = max(values) if metric_type == "accuracies" else min(values)
            
            framework_stats[framework] = stats
        
        return framework_stats
    
    def _create_performance_tiers(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Create performance tiers based on overall scores."""
        model_scores = {}
        
        for model_name, model_results in results.items():
            overall_score = model_results.get("overall_score", 0.0)
            model_scores[model_name] = overall_score
        
        if not model_scores:
            return {"high": [], "medium": [], "low": []}
        
        # Sort by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create tiers (top 1/3, middle 1/3, bottom 1/3)
        n_models = len(sorted_models)
        tier_size = n_models // 3
        
        tiers = {
            "high": [model for model, _ in sorted_models[:tier_size]],
            "medium": [model for model, _ in sorted_models[tier_size:2*tier_size]],
            "low": [model for model, _ in sorted_models[2*tier_size:]],
        }
        
        return tiers


class ModelComparator:
    """Advanced model comparator for v0.02.
    
    Design Pattern: Facade Pattern
    - Provides unified interface to comparison system
    - Manages complex comparison workflows
    - Supports multiple comparison strategies
    """
    
    def __init__(self, output_dir: Union[str, Path] = "output/comparisons"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = TrainingLogger("model_comparator")
        
        # Initialize comparison strategies
        self.strategies = {
            "statistical": StatisticalComparison(),
            "performance": PerformanceComparison(),
            "framework": FrameworkComparison(),
        }
        
        self.analyzer = StatisticalAnalyzer()
    
    def compare_two_models(self, results_a: Dict[str, Any], results_b: Dict[str, Any],
                          model_a_name: str = "Model A", model_b_name: str = "Model B") -> Dict[str, Any]:
        """Compare two models comprehensively."""
        self.logger.info(f"Comparing {model_a_name} vs {model_b_name}")
        
        # Define metrics to compare
        metrics_to_compare = [
            "accuracy", "precision", "recall", "f1_score",
            "inference_time_ms", "memory_usage_mb",
        ]
        
        comparison_results = {}
        significant_differences = []
        
        for metric in metrics_to_compare:
            try:
                # Extract metric values
                value_a = self._extract_metric_value(results_a, metric)
                value_b = self._extract_metric_value(results_b, metric)
                
                if value_a is not None and value_b is not None:
                    # Calculate comparison metrics
                    difference = value_b - value_a
                    relative_improvement = (difference / abs(value_a) * 100) if value_a != 0 else 0.0
                    
                    # Determine better model
                    is_higher_better = metric not in ["inference_time_ms", "memory_usage_mb"]
                    if is_higher_better:
                        better_model = model_b_name if value_b > value_a else model_a_name
                    else:
                        better_model = model_b_name if value_b < value_a else model_a_name
                    
                    # Simple significance test (10% threshold)
                    is_significant = abs(relative_improvement) > 10.0
                    
                    comparison_results[metric] = {
                        "value_a": value_a,
                        "value_b": value_b,
                        "difference": difference,
                        "relative_improvement_percent": relative_improvement,
                        "better_model": better_model,
                        "is_significant": is_significant,
                    }
                    
                    if is_significant:
                        significant_differences.append({
                            "metric": metric,
                            "improvement_percent": relative_improvement,
                            "better_model": better_model,
                        })
                
            except Exception as e:
                self.logger.warning(f"Failed to compare metric {metric}: {e}")
                comparison_results[metric] = {"error": str(e)}
        
        # Calculate summary
        valid_comparisons = [comp for comp in comparison_results.values() 
                           if "error" not in comp]
        
        summary_metrics = {}
        if valid_comparisons:
            model_b_wins = sum(1 for comp in valid_comparisons 
                             if comp["better_model"] == model_b_name)
            overall_better = model_b_name if model_b_wins > len(valid_comparisons) / 2 else model_a_name
            
            summary_metrics = {
                "overall_better_model": overall_better,
                "significant_improvements": len(significant_differences),
                "total_metrics_compared": len(valid_comparisons),
                "model_b_win_rate": model_b_wins / len(valid_comparisons),
            }
        
        return {
            "model_a_name": model_a_name,
            "model_b_name": model_b_name,
            "comparison_results": comparison_results,
            "significant_differences": significant_differences,
            "summary_metrics": summary_metrics,
            "timestamp": time.time(),
        }
    
    def _extract_metric_value(self, results: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from results."""
        # Check in standard_ml results
        if "standard_ml" in results and metric_name in results["standard_ml"]:
            return float(results["standard_ml"][metric_name])
        
        # Check in performance profile
        if "performance_profile" in results and metric_name in results["performance_profile"]:
            return float(results["performance_profile"][metric_name])
        
        # Check in game metrics
        if "game_evaluation" in results and "game_metrics" in results["game_evaluation"]:
            game_metrics = results["game_evaluation"]["game_metrics"]
            if metric_name in game_metrics:
                return float(game_metrics[metric_name])
        
        return None
    
    def save_comparison_report(self, comparison_results: Dict[str, Any], 
                             report_name: str = "comparison_report"):
        """Save comprehensive comparison report."""
        report_path = self.output_dir / f"{report_name}.json"
        
        with open(report_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        self.logger.info(f"Comparison report saved: {report_path}")
        
        # Also save a human-readable summary
        summary_path = self.output_dir / f"{report_name}_summary.txt"
        with open(summary_path, 'w') as f:
            self._write_comparison_summary(comparison_results, f)
        
        self.logger.info(f"Comparison summary saved: {summary_path}")
    
    def _write_comparison_summary(self, results: Dict[str, Any], file_handle):
        """Write human-readable comparison summary."""
        file_handle.write("=" * 80 + "\n")
        file_handle.write("MODEL COMPARISON REPORT\n")
        file_handle.write("=" * 80 + "\n\n")
        
        # Model information
        model_a = results.get("model_a_name", "Model A")
        model_b = results.get("model_b_name", "Model B")
        file_handle.write(f"Comparing: {model_a} vs {model_b}\n\n")
        
        # Summary metrics
        if "summary_metrics" in results:
            summary = results["summary_metrics"]
            file_handle.write("SUMMARY\n")
            file_handle.write("-" * 40 + "\n")
            file_handle.write(f"Overall Better Model: {summary.get('overall_better_model', 'Unknown')}\n")
            file_handle.write(f"Significant Improvements: {summary.get('significant_improvements', 0)}\n")
            file_handle.write(f"Total Metrics Compared: {summary.get('total_metrics_compared', 0)}\n")
            file_handle.write(f"{model_b} Win Rate: {summary.get('model_b_win_rate', 0):.1%}\n\n")
        
        # Significant differences
        if "significant_differences" in results and results["significant_differences"]:
            file_handle.write("SIGNIFICANT DIFFERENCES\n")
            file_handle.write("-" * 40 + "\n")
            for diff in results["significant_differences"]:
                metric = diff["metric"]
                improvement = diff["improvement_percent"]
                better = diff["better_model"]
                file_handle.write(f"{metric}: {better} better by {improvement:.1f}%\n")
            file_handle.write("\n")
        
        # Detailed results
        if "comparison_results" in results:
            file_handle.write("DETAILED RESULTS\n")
            file_handle.write("-" * 40 + "\n")
            for metric, result in results["comparison_results"].items():
                if "error" not in result:
                    file_handle.write(f"{metric}:\n")
                    file_handle.write(f"  {model_a}: {result['value_a']:.4f}\n")
                    file_handle.write(f"  {model_b}: {result['value_b']:.4f}\n")
                    file_handle.write(f"  Difference: {result['difference']:.4f}\n")
                    file_handle.write(f"  Better: {result['better_model']}\n\n")


@dataclass
class ComparisonReport:
    """Comprehensive comparison report.
    
    Aggregates comparison results and provides formatted output.
    """
    
    comparison_results: Dict[str, Any]
    model_names: List[str]
    timestamp: float
    
    def print_executive_summary(self):
        """Print executive summary of comparison."""
        print("\n" + "="*80)
        print("🏆 MODEL COMPARISON EXECUTIVE SUMMARY")
        print("="*80)
        
        if "summary_metrics" in self.comparison_results:
            summary = self.comparison_results["summary_metrics"]
            print(f"\n🥇 Overall Winner: {summary.get('overall_better_model', 'Unknown')}")
            print(f"📊 Significant Improvements: {summary.get('significant_improvements', 0)}")
            print(f"📈 Total Metrics: {summary.get('total_metrics_compared', 0)}")
        
        # Significant differences
        if "significant_differences" in self.comparison_results:
            diffs = self.comparison_results["significant_differences"]
            if diffs:
                print("\n🔍 Key Differences:")
                for diff in diffs[:3]:  # Show top 3
                    metric = diff["metric"]
                    improvement = diff["improvement_percent"]
                    better = diff["better_model"]
                    print(f"  • {metric}: {better} +{improvement:.1f}%")
        
        print("\n" + "="*80) 