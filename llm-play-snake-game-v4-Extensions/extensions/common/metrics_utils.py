"""
Performance Metrics Utilities for Snake Game AI Extensions

This module provides standardized performance measurement and analysis tools
for comparing algorithms across different extensions.

Design Patterns Used:
- Strategy Pattern: Different metric calculation strategies for different algorithm types
- Observer Pattern: Real-time metric collection during algorithm execution
- Template Method Pattern: Standard metric collection workflow
- Factory Pattern: Create appropriate metric collectors for different contexts

Educational Value:
Demonstrates how to design a comprehensive performance analysis system that
works across different algorithm types while maintaining statistical rigor.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime


class MetricType(Enum):
    """
    Types of metrics that can be collected
    
    Educational Note:
    Categorizing metrics helps ensure we collect the right data
    for meaningful comparisons between different algorithm types.
    """
    PERFORMANCE = "performance"  # Game scores, efficiency
    COMPUTATIONAL = "computational"  # Time, memory usage
    STATISTICAL = "statistical"  # Variance, distributions
    BEHAVIORAL = "behavioral"  # Decision patterns, strategies


@dataclass
class GameMetrics:
    """
    Comprehensive metrics for a single game session
    
    Design Pattern: Value Object Pattern
    Immutable container for game performance data that can be safely
    shared and aggregated across different analysis contexts.
    """
    # Basic game outcomes
    score: int
    steps: int
    game_duration_seconds: float
    game_won: bool
    death_reason: str
    
    # Efficiency metrics
    efficiency_ratio: float  # score / steps
    steps_per_apple: float
    average_move_time: float
    
    # Algorithm-specific data
    algorithm_name: str
    grid_size: int
    timestamp: str
    
    # Optional detailed metrics
    move_history: Optional[List[str]] = None
    score_progression: Optional[List[int]] = None
    computational_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameMetrics':
        """Create from dictionary"""
        return cls(**data)


@dataclass 
class AggregateMetrics:
    """
    Aggregated metrics across multiple games
    
    Educational Value:
    Shows how to properly aggregate performance data with statistical measures
    that account for variance and distribution characteristics.
    """
    # Basic statistics
    total_games: int
    mean_score: float
    median_score: float
    std_score: float
    min_score: int
    max_score: int
    
    # Win rate and efficiency
    win_rate: float
    mean_efficiency: float
    mean_steps_per_apple: float
    
    # Performance consistency
    score_variance: float
    coefficient_of_variation: float  # std/mean
    
    # Computational performance
    mean_move_time: float
    total_computation_time: float
    
    # Algorithm and context
    algorithm_name: str
    grid_size: int
    timestamp_range: Tuple[str, str]
    
    def performance_score(self) -> float:
        """
        Calculate overall performance score (0-1 scale)
        
        Combines multiple metrics into a single performance indicator
        for easy comparison between algorithms.
        """
        # Normalize components (all between 0-1)
        score_component = min(self.mean_score / 100.0, 1.0)  # Cap at 100
        efficiency_component = min(self.mean_efficiency / 10.0, 1.0)  # Cap at 10
        win_rate_component = self.win_rate
        consistency_component = max(0, 1.0 - self.coefficient_of_variation)  # Lower is better
        speed_component = max(0, min(1.0, 1.0 - self.mean_move_time))  # Faster is better
        
        # Weighted combination
        return (
            0.4 * score_component +      # 40% weight on score
            0.2 * efficiency_component +  # 20% weight on efficiency  
            0.2 * win_rate_component +    # 20% weight on win rate
            0.1 * consistency_component + # 10% weight on consistency
            0.1 * speed_component         # 10% weight on speed
        )


class MetricsCollector:
    """
    Collects and manages performance metrics during algorithm execution
    
    Design Pattern: Observer Pattern
    The collector observes game events and automatically records relevant metrics.
    This enables transparent performance monitoring without modifying algorithm code.
    
    Design Pattern: Strategy Pattern
    Different collection strategies for different algorithm types (heuristics vs ML vs RL).
    """
    
    def __init__(self, algorithm_name: str, grid_size: int):
        self.algorithm_name = algorithm_name
        self.grid_size = grid_size
        self.games: List[GameMetrics] = []
        self.current_game_start = None
        self.current_game_data = {}
        
    def start_game(self) -> None:
        """Start collecting metrics for a new game"""
        self.current_game_start = datetime.now()
        self.current_game_data = {
            'move_times': [],
            'score_progression': [],
            'move_history': []
        }
    
    def record_move(self, move: str, move_time: float, current_score: int) -> None:
        """Record a single move during gameplay"""
        if self.current_game_start is None:
            raise ValueError("Must call start_game() before recording moves")
            
        self.current_game_data['move_times'].append(move_time)
        self.current_game_data['score_progression'].append(current_score)
        self.current_game_data['move_history'].append(move)
    
    def end_game(self, final_score: int, steps: int, game_won: bool, 
                 death_reason: str = "unknown") -> GameMetrics:
        """End game and calculate final metrics"""
        if self.current_game_start is None:
            raise ValueError("Must call start_game() before ending game")
            
        # Calculate game duration
        game_duration = (datetime.now() - self.current_game_start).total_seconds()
        
        # Calculate derived metrics
        efficiency_ratio = final_score / max(steps, 1)
        steps_per_apple = steps / max(final_score, 1) if final_score > 0 else steps
        average_move_time = statistics.mean(self.current_game_data['move_times']) if self.current_game_data['move_times'] else 0.0
        
        # Create game metrics
        game_metrics = GameMetrics(
            score=final_score,
            steps=steps,
            game_duration_seconds=game_duration,
            game_won=game_won,
            death_reason=death_reason,
            efficiency_ratio=efficiency_ratio,
            steps_per_apple=steps_per_apple,
            average_move_time=average_move_time,
            algorithm_name=self.algorithm_name,
            grid_size=self.grid_size,
            timestamp=datetime.now().isoformat(),
            move_history=self.current_game_data['move_history'].copy(),
            score_progression=self.current_game_data['score_progression'].copy()
        )
        
        # Store the game metrics
        self.games.append(game_metrics)
        
        # Reset for next game
        self.current_game_start = None
        self.current_game_data = {}
        
        return game_metrics
    
    def get_aggregate_metrics(self) -> Optional[AggregateMetrics]:
        """Calculate aggregate metrics across all collected games"""
        if not self.games:
            return None
            
        scores = [game.score for game in self.games]
        efficiencies = [game.efficiency_ratio for game in self.games]
        steps_per_apple = [game.steps_per_apple for game in self.games]
        move_times = [game.average_move_time for game in self.games]
        game_durations = [game.game_duration_seconds for game in self.games]
        
        wins = sum(1 for game in self.games if game.game_won)
        
        # Calculate statistics
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        
        return AggregateMetrics(
            total_games=len(self.games),
            mean_score=mean_score,
            median_score=statistics.median(scores),
            std_score=std_score,
            min_score=min(scores),
            max_score=max(scores),
            win_rate=wins / len(self.games),
            mean_efficiency=statistics.mean(efficiencies),
            mean_steps_per_apple=statistics.mean(steps_per_apple),
            score_variance=statistics.variance(scores) if len(scores) > 1 else 0.0,
            coefficient_of_variation=std_score / mean_score if mean_score > 0 else 0.0,
            mean_move_time=statistics.mean(move_times),
            total_computation_time=sum(game_durations),
            algorithm_name=self.algorithm_name,
            grid_size=self.grid_size,
            timestamp_range=(self.games[0].timestamp, self.games[-1].timestamp)
        )
    
    def save_metrics(self, output_path: Path) -> None:
        """Save collected metrics to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'algorithm_name': self.algorithm_name,
            'grid_size': self.grid_size,
            'total_games': len(self.games),
            'aggregate_metrics': self.get_aggregate_metrics().to_dict() if self.games else None,
            'individual_games': [game.to_dict() for game in self.games]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


class MetricsAnalyzer:
    """
    Advanced analysis of collected metrics across algorithms and extensions
    
    Design Pattern: Template Method Pattern
    Provides standard analysis workflow while allowing customization
    of specific analysis steps for different algorithm types.
    """
    
    def __init__(self):
        self.metrics_data: List[AggregateMetrics] = []
    
    def load_metrics(self, metrics_path: Path) -> None:
        """Load metrics from JSON file"""
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            
        if data['aggregate_metrics']:
            aggregate = AggregateMetrics(**data['aggregate_metrics'])
            self.metrics_data.append(aggregate)
    
    def load_multiple_metrics(self, metrics_directory: Path) -> None:
        """Load metrics from multiple files in a directory"""
        for metrics_file in metrics_directory.glob("*.json"):
            try:
                self.load_metrics(metrics_file)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load metrics from {metrics_file}: {e}")
    
    def compare_algorithms(self, grid_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Compare performance across different algorithms
        
        Args:
            grid_size: Filter by specific grid size (None for all sizes)
            
        Returns:
            Dictionary with comparison results
        """
        # Filter data if grid size specified
        data = self.metrics_data
        if grid_size is not None:
            data = [m for m in data if m.grid_size == grid_size]
        
        if not data:
            return {"error": "No metrics data available for comparison"}
        
        # Sort by performance score
        sorted_data = sorted(data, key=lambda x: x.performance_score(), reverse=True)
        
        # Calculate comparative statistics
        performance_scores = [m.performance_score() for m in sorted_data]
        mean_scores = [m.mean_score for m in sorted_data]
        win_rates = [m.win_rate for m in sorted_data]
        
        return {
            'ranking': [
                {
                    'algorithm': m.algorithm_name,
                    'grid_size': m.grid_size,
                    'performance_score': m.performance_score(),
                    'mean_score': m.mean_score,
                    'win_rate': m.win_rate,
                    'efficiency': m.mean_efficiency,
                    'consistency': 1.0 - m.coefficient_of_variation
                }
                for m in sorted_data
            ],
            'statistics': {
                'best_algorithm': sorted_data[0].algorithm_name,
                'performance_range': (min(performance_scores), max(performance_scores)),
                'score_range': (min(mean_scores), max(mean_scores)),
                'win_rate_range': (min(win_rates), max(win_rates))
            }
        }
    
    def analyze_grid_size_impact(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Analyze how grid size affects algorithm performance
        
        Args:
            algorithm_name: Name of algorithm to analyze
            
        Returns:
            Analysis of grid size impact
        """
        # Filter data for specific algorithm
        algorithm_data = [m for m in self.metrics_data if m.algorithm_name == algorithm_name]
        
        if not algorithm_data:
            return {"error": f"No data found for algorithm: {algorithm_name}"}
        
        # Group by grid size
        grid_analysis = {}
        for metrics in algorithm_data:
            grid_size = metrics.grid_size
            if grid_size not in grid_analysis:
                grid_analysis[grid_size] = []
            grid_analysis[grid_size].append(metrics)
        
        # Calculate trends
        grid_sizes = sorted(grid_analysis.keys())
        performance_trend = []
        score_trend = []
        
        for grid_size in grid_sizes:
            grid_metrics = grid_analysis[grid_size]
            avg_performance = statistics.mean([m.performance_score() for m in grid_metrics])
            avg_score = statistics.mean([m.mean_score for m in grid_metrics])
            
            performance_trend.append(avg_performance)
            score_trend.append(avg_score)
        
        return {
            'algorithm': algorithm_name,
            'grid_sizes_tested': grid_sizes,
            'performance_trend': performance_trend,
            'score_trend': score_trend,
            'scaling_analysis': {
                'performance_correlation': np.corrcoef(grid_sizes, performance_trend)[0, 1] if len(grid_sizes) > 1 else 0,
                'score_correlation': np.corrcoef(grid_sizes, score_trend)[0, 1] if len(grid_sizes) > 1 else 0,
                'optimal_grid_size': grid_sizes[performance_trend.index(max(performance_trend))]
            }
        }
    
    def generate_performance_report(self, output_path: Path) -> None:
        """Generate comprehensive performance analysis report"""
        if not self.metrics_data:
            print("No metrics data available for report generation")
            return
        
        # Overall comparison
        overall_comparison = self.compare_algorithms()
        
        # Grid size analysis for each algorithm
        algorithms = list(set(m.algorithm_name for m in self.metrics_data))
        grid_analyses = {}
        for algorithm in algorithms:
            grid_analyses[algorithm] = self.analyze_grid_size_impact(algorithm)
        
        # Generate report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'total_algorithms_analyzed': len(algorithms),
            'total_metrics_records': len(self.metrics_data),
            'overall_algorithm_comparison': overall_comparison,
            'grid_size_analyses': grid_analyses,
            'summary': {
                'best_overall_algorithm': overall_comparison['statistics']['best_algorithm'],
                'performance_insights': self._generate_insights()
            }
        }
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Performance report saved to: {output_path}")
    
    def _generate_insights(self) -> List[str]:
        """Generate human-readable insights from the analysis"""
        insights = []
        
        if not self.metrics_data:
            return ["No data available for insights"]
        
        # Best performing algorithm
        best_metrics = max(self.metrics_data, key=lambda x: x.performance_score())
        insights.append(f"Best performing algorithm: {best_metrics.algorithm_name} with performance score {best_metrics.performance_score():.3f}")
        
        # Win rate analysis
        high_win_rate = [m for m in self.metrics_data if m.win_rate > 0.8]
        if high_win_rate:
            insights.append(f"{len(high_win_rate)} algorithms achieve >80% win rate")
        
        # Consistency analysis
        consistent_algorithms = [m for m in self.metrics_data if m.coefficient_of_variation < 0.3]
        if consistent_algorithms:
            insights.append(f"{len(consistent_algorithms)} algorithms show high consistency (CV < 0.3)")
        
        # Speed analysis
        fast_algorithms = [m for m in self.metrics_data if m.mean_move_time < 0.1]
        if fast_algorithms:
            insights.append(f"{len(fast_algorithms)} algorithms achieve <100ms average move time")
        
        return insights


def create_metrics_collector(algorithm_name: str, grid_size: int) -> MetricsCollector:
    """
    Factory function for creating metrics collectors
    
    Args:
        algorithm_name: Name of the algorithm being tested
        grid_size: Size of the game grid
        
    Returns:
        Configured MetricsCollector instance
        
    Example:
        collector = create_metrics_collector("BFS", 10)
        collector.start_game()
        # ... play game ...
        collector.end_game(score, steps, won, reason)
    """
    return MetricsCollector(algorithm_name, grid_size)


def analyze_metrics_directory(metrics_dir: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Analyze all metrics files in a directory and generate comparison report
    
    Args:
        metrics_dir: Directory containing metrics JSON files
        output_path: Optional path to save detailed report
        
    Returns:
        Summary analysis results
        
    Educational Value:
    Shows how to build a comprehensive analysis pipeline that can process
    multiple algorithm results and generate meaningful comparisons.
    """
    analyzer = MetricsAnalyzer()
    analyzer.load_multiple_metrics(metrics_dir)
    
    # Generate comparison
    comparison = analyzer.compare_algorithms()
    
    # Generate full report if requested
    if output_path:
        analyzer.generate_performance_report(output_path)
    
    return comparison 