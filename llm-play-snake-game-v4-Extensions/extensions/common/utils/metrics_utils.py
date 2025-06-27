"""
Performance Metrics Utilities for Snake Game AI Extensions

This module provides standardized performance measurement and analysis tools
for comparing algorithms across different extensions.

INTEGRATION WITH CORE TASK-0 ARCHITECTURE:
This module INHERITS from the established core classes from Task-0:
- ExtensionGameData(BaseGameData): Extension-specific game data tracking
- ExtensionGameStatistics(BaseGameStatistics): Extension-specific statistics
- ExtensionStepStats(BaseStepStats): Extension-specific step tracking

Following the exact pattern established by Task-0:
- BaseGameData → GameData (Task-0) | ExtensionGameData (Extensions)
- BaseGameStatistics → GameStatistics (Task-0) | ExtensionGameStatistics (Extensions)
- BaseStepStats → StepStats (Task-0) | ExtensionStepStats (Extensions)

Design Patterns Used:
- Inheritance Pattern: Direct inheritance from core Task-0 base classes
- Template Method Pattern: Standard metric collection workflow
- Strategy Pattern: Different metric calculation strategies for different algorithm types
- Observer Pattern: Real-time metric collection during algorithm execution
- Factory Pattern: Create appropriate metric collectors for different contexts

Educational Value:
Demonstrates proper inheritance patterns that follow established Task-0 architecture
while adding extension-specific capabilities for algorithm performance analysis.

SUPREME_RULE NO.4 Implementation:
Base classes provide complete metrics functionality with protected extension
points that specialized extensions can override for algorithm-specific needs.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from datetime import datetime
import time

# Import core Task-0 classes for proper inheritance
from core.game_data import BaseGameData
from core.game_stats import BaseGameStatistics, BaseStepStats, TimeStats


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
class ExtensionStepStats(BaseStepStats):
    """
    Extension-specific step statistics (inherits from BaseStepStats)
    
    Educational Note:
    Following the exact pattern from Task-0:
    - BaseStepStats: Universal counters (valid, invalid_reversals, no_path_found)
    - StepStats: Task-0 LLM-specific (adds empty, something_wrong)
    - ExtensionStepStats: Extension-specific (adds algorithm_performance_metrics)
    
    This ensures consistent JSON schema while adding extension capabilities.
    """
    
    # Extension-specific metrics
    algorithm_performance_steps: int = 0  # Algorithm-specific performance tracking
    optimization_steps: int = 0  # Algorithm optimization events
    
    def asdict(self) -> dict:
        """JSON-friendly view maintaining compatibility with Task-0 schema"""
        base = super().asdict()
        base.update({
            "algorithm_performance_steps": self.algorithm_performance_steps,
            "optimization_steps": self.optimization_steps,
        })
        return base


@dataclass
class ExtensionGameStatistics(BaseGameStatistics):
    """
    Extension-specific game statistics (inherits from BaseGameStatistics)
    
    Educational Note:
    Following the exact pattern from Task-0:
    - BaseGameStatistics: Universal statistics (time_stats, step_stats)
    - GameStatistics: Task-0 LLM-specific (adds token stats, response times)
    - ExtensionGameStatistics: Extension-specific (adds algorithm metrics)
    
    This ensures JSON compatibility while adding extension-specific capabilities.
    """
    
    # Override with extension-specific step stats
    step_stats: ExtensionStepStats = field(default_factory=ExtensionStepStats)
    
    # Extension-specific metrics
    algorithm_metrics: Dict[str, float] = field(default_factory=dict)
    performance_scores: List[float] = field(default_factory=list)
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    
    def record_algorithm_metric(self, metric_name: str, value: float) -> None:
        """Record an algorithm-specific metric"""
        self.algorithm_metrics[metric_name] = value
    
    def record_performance_score(self, score: float) -> None:
        """Record a performance score"""
        self.performance_scores.append(score)
    
    def record_efficiency_metric(self, metric_name: str, value: float) -> None:
        """Record an efficiency metric"""
        self.efficiency_metrics[metric_name] = value
    
    def asdict(self) -> dict:
        """JSON-friendly view maintaining compatibility with Task-0 schema"""
        base = super().asdict()
        base.update({
            "algorithm_metrics": self.algorithm_metrics,
            "performance_scores": self.performance_scores,
            "efficiency_metrics": self.efficiency_metrics,
        })
        return base


class ExtensionGameData(BaseGameData):
    """
    Extension-specific game data tracking (inherits from BaseGameData)
    
    Educational Note:
    Following the exact pattern from Task-0:
    - BaseGameData: Universal game state (score, steps, snake_positions, etc.)
    - GameData: Task-0 LLM-specific (adds LLM counters, token tracking)
    - ExtensionGameData: Extension-specific (adds algorithm performance tracking)
    
    This ensures perfect compatibility with Task-0 architecture while enabling
    extension-specific capabilities for algorithm performance analysis.
    
    SUPREME_RULE NO.4 Implementation:
    This class provides complete game data functionality with protected extension
    points that specialized algorithm extensions can override for their specific needs.
    """
    
    def __init__(self, algorithm_name: str, grid_size: int) -> None:
        """Initialize extension-specific game data tracking."""
        super().__init__()
        
        # Extension-specific attributes
        self.algorithm_name = algorithm_name
        self.grid_size = grid_size
        
        # Override with extension-specific statistics
        self.stats = ExtensionGameStatistics()
        
        # Algorithm-specific tracking
        self._initialize_algorithm_specific_tracking()
    
    def _initialize_algorithm_specific_tracking(self) -> None:
        """
        Initialize algorithm-specific tracking (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to set up specialized
        tracking for specific algorithm types.
        
        Example:
            class HeuristicGameData(ExtensionGameData):
                def _initialize_algorithm_specific_tracking(self):
                    self.pathfinding_metrics = PathfindingTracker()
                    self.search_efficiency = SearchEfficiencyTracker()
        """
        pass
    
    def reset(self) -> None:
        """Reset all tracking data to initial state."""
        super().reset()
        
        # Reset extension-specific statistics
        self.stats = ExtensionGameStatistics()
        
        # Reset algorithm-specific tracking
        self._initialize_algorithm_specific_tracking()
    
    def record_move(self, move: str, apple_eaten: bool = False) -> None:
        """Record a move and update relevant statistics."""
        # Call base class method
        super().record_move(move, apple_eaten)
        
        # Extension-specific tracking
        self.stats.step_stats.valid += 1
        
        # Algorithm-specific tracking
        self._record_algorithm_specific_move(move, apple_eaten)
    
    def _record_algorithm_specific_move(self, move: str, apple_eaten: bool) -> None:
        """
        Record algorithm-specific move data (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to track algorithm-specific
        move patterns, efficiency metrics, or decision-making characteristics.
        
        Example:
            class PathfindingGameData(ExtensionGameData):
                def _record_algorithm_specific_move(self, move, apple_eaten):
                    if apple_eaten:
                        self.pathfinding_efficiency.record_success()
                    self.search_patterns.record_move(move)
        """
        pass
    
    def record_algorithm_performance(self, metric_name: str, value: float) -> None:
        """Record algorithm-specific performance metric."""
        self.stats.record_algorithm_metric(metric_name, value)
    
    def record_optimization_event(self) -> None:
        """Record an algorithm optimization event."""
        self.stats.step_stats.optimization_steps += 1
    
    def calculate_efficiency_ratio(self) -> float:
        """Calculate algorithm efficiency ratio."""
        return self.score / max(self.steps, 1)
    
    def calculate_steps_per_apple(self) -> float:
        """Calculate average steps per apple collected."""
        return self.steps / max(self.score, 1)
    
    def generate_extension_summary(self, **kwargs) -> Dict[str, Any]:
        """
        Generate extension-compatible summary with Task-0 JSON schema compatibility.
        
        This method generates a summary that maintains perfect compatibility with
        Task-0 replay tools while adding extension-specific metrics.
        
        Returns:
            JSON-compatible dictionary with all game data and extension metrics
        """
        # Generate base summary (compatible with Task-0)
        summary = {
            # Core game data (from BaseGameData)
            "score": self.score,
            "steps": self.steps,
            "moves": self.moves,
            "snake_positions": self.snake_positions,
            "apple_positions": self.apple_positions,
            "game_over_reason": getattr(self, 'game_over_reason', 'completed'),
            "timestamp": datetime.now().isoformat(),
            
            # Extension-specific metadata
            "algorithm_name": self.algorithm_name,
            "grid_size": self.grid_size,
            
            # Performance metrics
            "efficiency_ratio": self.calculate_efficiency_ratio(),
            "steps_per_apple": self.calculate_steps_per_apple(),
            
            # Statistics (from ExtensionGameStatistics)
            "stats": self.stats.asdict(),
            
            # Extension metadata
            "extension_type": "algorithm_performance",
            "schema_version": "extension_v1.0",
        }
        
        # Add algorithm-specific summary data
        algorithm_specific = self._generate_algorithm_specific_summary()
        if algorithm_specific:
            summary["algorithm_specific"] = algorithm_specific
        
        # Add any additional kwargs
        summary.update(kwargs)
        
        return summary
    
    def _generate_algorithm_specific_summary(self) -> Optional[Dict[str, Any]]:
        """
        Generate algorithm-specific summary data (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to include specialized
        summary data specific to their algorithm type.
        
        Example:
            class RLGameData(ExtensionGameData):
                def _generate_algorithm_specific_summary(self):
                    return {
                        "episode_reward": self.total_reward,
                        "q_values": self.q_value_history[-10:],  # Last 10 Q-values
                        "exploration_rate": self.epsilon,
                        "model_updates": self.model_update_count
                    }
        """
        return None


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

    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MetricsCollector:
    """
    Comprehensive metrics collection system for algorithm performance analysis
    
    Design Pattern: Observer Pattern + Template Method Pattern
    Purpose: Collect detailed performance metrics during algorithm execution
    
    Educational Note (SUPREME_RULE NO.4):
    This collector is designed to work with any algorithm type while allowing
    specialized metric collection for specific algorithm families through
    inheritance and composition patterns.
    
    SUPREME_RULE NO.4 Implementation:
    - Base collector provides complete metrics functionality
    - Protected methods allow specialized metric collection
    - Virtual methods enable custom metric calculation
    - Algorithm-specific collectors can inherit and extend
    """
    
    def __init__(self, algorithm_name: str, grid_size: int):
        self.algorithm_name = algorithm_name
        self.grid_size = grid_size
        
        # Game tracking
        self.current_game: Optional[ExtensionGameData] = None
        self.completed_games: List[ExtensionGameData] = []
        
        # Performance tracking
        self.move_times: List[float] = []
        self.game_start_time: Optional[float] = None
        
        # Initialize collector-specific settings
        self._initialize_collector_specific_settings()
    
    def _initialize_collector_specific_settings(self) -> None:
        """
        Initialize collector-specific settings (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to set up specialized
        metric collection configurations for specific algorithm types.
        
        Example:
            class RLMetricsCollector(MetricsCollector):
                def _initialize_collector_specific_settings(self):
                    self.reward_tracker = RewardTracker()
                    self.q_value_tracker = QValueTracker()
                    self.exploration_tracker = ExplorationTracker()
        """
        pass
    
    def start_game(self) -> ExtensionGameData:
        """
        Start tracking a new game
        
        Returns:
            ExtensionGameData object for the new game
        """
        # End current game if one is in progress
        if self.current_game is not None:
            self.end_game("new_game_started")
        
        # Start new game
        self.current_game = ExtensionGameData(self.algorithm_name, self.grid_size)
        self.game_start_time = time.time()
        self.move_times = []
        
        # Initialize game-specific metrics
        self._initialize_game_specific_metrics()
        
        return self.current_game
    
    def _initialize_game_specific_metrics(self) -> None:
        """
        Initialize game-specific metrics (SUPREME_RULE NO.4 Extension Point).
        
        Override this method in subclasses to initialize specialized
        per-game tracking for specific algorithm types.
        """
        pass
    
    def record_move(self, move: str, apple_eaten: bool = False, move_time: float = 0.0) -> None:
        """Record a move and associated metrics."""
        if self.current_game is None:
            raise RuntimeError("No active game. Call start_game() first.")
        
        # Record move in game data
        self.current_game.record_move(move, apple_eaten)
        
        # Record timing
        if move_time > 0:
            self.move_times.append(move_time)
            self.current_game.stats.time_stats.total_move_time += move_time
        
        # Record algorithm-specific metrics
        self._record_algorithm_specific_metrics(move, move_time)
    
    def _record_algorithm_specific_metrics(self, move: str, move_time: float) -> None:
        """
        Record algorithm-specific metrics (SUPREME_RULE NO.4 Extension Point).
        
        Override this method in subclasses to record specialized metrics
        for specific algorithm types during move execution.
        
        Example:
            class PathfindingMetricsCollector(MetricsCollector):
                def _record_algorithm_specific_metrics(self, move, move_time):
                    path_length = self.pathfinder.get_last_path_length()
                    nodes_explored = self.pathfinder.get_nodes_explored()
                    
                    self.current_game.record_algorithm_performance("path_length", path_length)
                    self.current_game.record_algorithm_performance("nodes_explored", nodes_explored)
        """
        pass
    
    def record_apple_position(self, position: Tuple[int, int]) -> None:
        """Record apple position for analysis."""
        if self.current_game is not None:
            self.current_game.apple_positions.append(position)
    
    def record_algorithm_performance(self, metric_name: str, value: float) -> None:
        """Record custom algorithm performance metric."""
        if self.current_game is not None:
            self.current_game.record_algorithm_performance(metric_name, value)
    
    def end_game(self, reason: str) -> Optional[ExtensionGameData]:
        """
        End current game and finalize metrics
        
        Args:
            reason: Reason for game ending
            
        Returns:
            Completed game data
        """
        if self.current_game is None:
            return None
        
        # Set game over reason
        self.current_game.game_over_reason = reason
        
        # Calculate final metrics
        if self.game_start_time is not None:
            total_time = time.time() - self.game_start_time
            self.current_game.stats.time_stats.total_game_time = total_time
        
        # Calculate move time statistics
        if self.move_times:
            self.current_game.stats.time_stats.mean_move_time = statistics.mean(self.move_times)
            self.current_game.stats.time_stats.max_move_time = max(self.move_times)
        
        # Finalize algorithm-specific metrics
        self._finalize_algorithm_specific_metrics()
        
        # Store completed game
        self.completed_games.append(self.current_game)
        completed_game = self.current_game
        
        # Reset for next game
        self.current_game = None
        self.game_start_time = None
        self.move_times = []
        
        return completed_game
    
    def _finalize_algorithm_specific_metrics(self) -> None:
        """
        Finalize algorithm-specific metrics (SUPREME_RULE NO.4 Extension Point).
        
        Override this method in subclasses to perform final calculations
        or cleanup for algorithm-specific metrics at game end.
        """
        pass
    
    def get_aggregate_metrics(self) -> Optional[AggregateMetrics]:
        """Calculate aggregate metrics across all completed games."""
        if not self.completed_games:
            return None
        
        scores = [game.score for game in self.completed_games]
        efficiencies = [game.calculate_efficiency_ratio() for game in self.completed_games]
        steps_per_apple = [game.calculate_steps_per_apple() for game in self.completed_games]
        
        # Calculate win rate (assuming games with score > 0 are wins)
        wins = sum(1 for score in scores if score > 0)
        win_rate = wins / len(scores)
        
        # Time statistics
        all_move_times = []
        total_computation_time = 0.0
        for game in self.completed_games:
            if hasattr(game.stats.time_stats, 'total_move_time'):
                all_move_times.extend(getattr(game.stats.time_stats, 'move_times', []))
                total_computation_time += getattr(game.stats.time_stats, 'total_game_time', 0.0)
        
        mean_move_time = statistics.mean(all_move_times) if all_move_times else 0.0
        
        # Timestamp range
        first_game_time = getattr(self.completed_games[0], 'timestamp', datetime.now().isoformat())
        last_game_time = getattr(self.completed_games[-1], 'timestamp', datetime.now().isoformat())
        
        return AggregateMetrics(
            total_games=len(self.completed_games),
            mean_score=statistics.mean(scores),
            median_score=statistics.median(scores),
            std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            min_score=min(scores),
            max_score=max(scores),
            win_rate=win_rate,
            mean_efficiency=statistics.mean(efficiencies),
            mean_steps_per_apple=statistics.mean(steps_per_apple),
            score_variance=statistics.variance(scores) if len(scores) > 1 else 0.0,
            coefficient_of_variation=statistics.stdev(scores) / statistics.mean(scores) if len(scores) > 1 and statistics.mean(scores) > 0 else 0.0,
            mean_move_time=mean_move_time,
            total_computation_time=total_computation_time,
            algorithm_name=self.algorithm_name,
            grid_size=self.grid_size,
            timestamp_range=(first_game_time, last_game_time)
        )
    
    def save_metrics(self, output_path: Path) -> None:
        """Save all collected metrics to JSON file."""
        data = {
            "algorithm_name": self.algorithm_name,
            "grid_size": self.grid_size,
            "total_games": len(self.completed_games),
            "games": [game.generate_extension_summary() for game in self.completed_games],
            "aggregate_metrics": self.get_aggregate_metrics().asdict() if self.get_aggregate_metrics() else None,
            "collection_timestamp": datetime.now().isoformat()
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


class MetricsAnalyzer:
    """
    Analysis and comparison of metrics across different algorithms
    
    Design Pattern: Strategy Pattern + Template Method Pattern
    Purpose: Analyze and compare performance metrics across algorithms
    
    Educational Note (SUPREME_RULE NO.4):
    This analyzer is designed to handle any algorithm type while allowing
    specialized analysis strategies for specific algorithm families.
    
    SUPREME_RULE NO.4 Implementation:
    - Base analyzer provides complete analysis functionality
    - Protected methods allow specialized analysis strategies
    - Virtual methods enable custom comparison metrics
    - Algorithm-specific analyzers can inherit and extend
    """
    
    def __init__(self):
        self.metrics_data: Dict[str, List[Dict[str, Any]]] = {}
        self._initialize_analyzer_specific_settings()
    
    def _initialize_analyzer_specific_settings(self) -> None:
        """
        Initialize analyzer-specific settings (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to set up specialized
        analysis configurations for specific algorithm families.
        
        Example:
            class MLAnalyzer(MetricsAnalyzer):
                def _initialize_analyzer_specific_settings(self):
                    self.statistical_tests = StatisticalTestSuite()
                    self.learning_curve_analyzer = LearningCurveAnalyzer()
                    self.convergence_detector = ConvergenceDetector()
        """
        pass
    
    def load_metrics(self, metrics_path: Path) -> None:
        """Load metrics from a JSON file."""
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        algorithm_name = data['algorithm_name']
        if algorithm_name not in self.metrics_data:
            self.metrics_data[algorithm_name] = []
        
        self.metrics_data[algorithm_name].extend(data['games'])
    
    def load_multiple_metrics(self, metrics_directory: Path) -> None:
        """Load metrics from all JSON files in a directory."""
        for metrics_file in metrics_directory.glob("*.json"):
            self.load_metrics(metrics_file)
    
    def compare_algorithms(self, grid_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Compare performance across all loaded algorithms
        
        Args:
            grid_size: Filter by specific grid size (optional)
            
        Returns:
            Comparison analysis results
        """
        if not self.metrics_data:
            return {"error": "No metrics data loaded"}
        
        comparison = {
            "algorithms": {},
            "rankings": {},
            "statistical_summary": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Filter by grid size if specified
        filtered_data = {}
        for algorithm, games in self.metrics_data.items():
            if grid_size is not None:
                games = [game for game in games if game.get('grid_size') == grid_size]
            if games:  # Only include algorithms with data
                filtered_data[algorithm] = games
        
        if not filtered_data:
            return {"error": f"No data found for grid size {grid_size}"}
        
        # Analyze each algorithm
        for algorithm, games in filtered_data.items():
            scores = [game['score'] for game in games]
            
            algorithm_analysis = {
                "total_games": len(games),
                "mean_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "min_score": min(scores),
                "max_score": max(scores),
                "win_rate": sum(1 for score in scores if score > 0) / len(scores),
                "performance_score": self._calculate_performance_score(games)
            }
            
            # Add algorithm-specific analysis
            algorithm_specific = self._generate_algorithm_specific_comparison(algorithm, games)
            if algorithm_specific:
                algorithm_analysis["algorithm_specific"] = algorithm_specific
            
            comparison["algorithms"][algorithm] = algorithm_analysis
        
        # Generate rankings
        algorithms_by_performance = sorted(
            comparison["algorithms"].items(),
            key=lambda x: x[1]["performance_score"],
            reverse=True
        )
        
        comparison["rankings"] = {
            "by_performance_score": [algo for algo, _ in algorithms_by_performance],
            "by_mean_score": sorted(
                comparison["algorithms"].keys(),
                key=lambda x: comparison["algorithms"][x]["mean_score"],
                reverse=True
            ),
            "by_win_rate": sorted(
                comparison["algorithms"].keys(),
                key=lambda x: comparison["algorithms"][x]["win_rate"],
                reverse=True
            )
        }
        
        return comparison
    
    def _generate_algorithm_specific_comparison(self, algorithm: str, games: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Generate algorithm-specific comparison metrics (SUPREME_RULE NO.4 Extension Point).
        
        Override this method in subclasses to provide specialized comparison
        metrics for specific algorithm families.
        
        Example:
            class PathfindingAnalyzer(MetricsAnalyzer):
                def _generate_algorithm_specific_comparison(self, algorithm, games):
                    if "pathfinding" in algorithm.lower():
                        return {
                            "average_path_optimality": self._calculate_path_optimality(games),
                            "search_efficiency": self._calculate_search_efficiency(games),
                            "memory_usage": self._calculate_memory_usage(games)
                        }
        """
        return None
    
    def _calculate_performance_score(self, games: List[Dict[str, Any]]) -> float:
        """Calculate overall performance score for an algorithm."""
        if not games:
            return 0.0
        
        scores = [game['score'] for game in games]
        mean_score = statistics.mean(scores)
        win_rate = sum(1 for score in scores if score > 0) / len(scores)
        consistency = 1.0 - (statistics.stdev(scores) / max(mean_score, 1)) if len(scores) > 1 else 1.0
        
        # Normalize and combine (basic implementation)
        score_component = min(mean_score / 100.0, 1.0)
        win_rate_component = win_rate
        consistency_component = max(0, consistency)
        
        return (0.5 * score_component + 0.3 * win_rate_component + 0.2 * consistency_component)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_metrics_collector(algorithm_name: str, grid_size: int) -> MetricsCollector:
    """
    Create a metrics collector for an algorithm
    
    Args:
        algorithm_name: Name of the algorithm
        grid_size: Grid size for the games
        
    Returns:
        Configured MetricsCollector instance
        
    Example:
        >>> collector = create_metrics_collector("BFS", 10)
        >>> game_data = collector.start_game()
        >>> collector.record_move("UP", apple_eaten=True, move_time=0.001)
        >>> collector.end_game("completed")
    """
    return MetricsCollector(algorithm_name, grid_size)

def analyze_metrics_directory(metrics_dir: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Analyze all metrics in a directory and optionally save results
    
    Args:
        metrics_dir: Directory containing metrics JSON files
        output_path: Optional path to save analysis results
        
    Returns:
        Analysis results
        
    Example:
        >>> results = analyze_metrics_directory(
        ...     Path("logs/extensions/datasets/grid-size-10"),
        ...     Path("analysis_results.json")
        ... )
        >>> print(f"Best algorithm: {results['rankings']['by_performance_score'][0]}")
    """
    analyzer = MetricsAnalyzer()
    analyzer.load_multiple_metrics(metrics_dir)
    results = analyzer.compare_algorithms()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results 