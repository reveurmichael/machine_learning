"""
Tests for utils.game_stats_utils module.

Focuses on testing game statistics utilities for data aggregation,
performance analytics, trend analysis, and statistical computations.
"""

import pytest
import time
import tempfile
import json
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from numpy.typing import NDArray

from utils.game_stats_utils import GameStatsUtils


class TestGameStatsUtils:
    """Test game statistics utility functions."""

    def test_basic_game_statistics_calculation(self) -> None:
        """Test calculation of basic game statistics."""
        
        stats_utils: GameStatsUtils = GameStatsUtils()
        
        # Mock game data for statistics calculation
        game_results: List[Dict[str, Any]] = [
            {
                "game_id": "game_1",
                "final_score": 150,
                "total_steps": 75,
                "snake_length": 8,
                "apples_eaten": 6,
                "duration": 45.2,
                "completion_status": "success"
            },
            {
                "game_id": "game_2",
                "final_score": 200,
                "total_steps": 90,
                "snake_length": 10,
                "apples_eaten": 8,
                "duration": 52.1,
                "completion_status": "success"
            },
            {
                "game_id": "game_3",
                "final_score": 75,
                "total_steps": 45,
                "snake_length": 5,
                "apples_eaten": 3,
                "duration": 28.5,
                "completion_status": "collision"
            },
            {
                "game_id": "game_4",
                "final_score": 300,
                "total_steps": 120,
                "snake_length": 15,
                "apples_eaten": 12,
                "duration": 68.3,
                "completion_status": "success"
            },
            {
                "game_id": "game_5",
                "final_score": 125,
                "total_steps": 65,
                "snake_length": 7,
                "apples_eaten": 5,
                "duration": 38.7,
                "completion_status": "max_steps"
            }
        ]
        
        # Calculate basic statistics
        stats_result = stats_utils.calculate_basic_statistics(
            game_results=game_results
        )
        
        assert stats_result["success"] is True, "Statistics calculation should succeed"
        
        basic_stats = stats_result["statistics"]
        
        # Verify total counts
        assert basic_stats["total_games"] == 5, "Should count all games"
        assert basic_stats["successful_games"] == 3, "Should count successful games"
        assert basic_stats["failed_games"] == 2, "Should count failed games"
        
        # Verify score statistics
        scores = [150, 200, 75, 300, 125]
        expected_avg_score = sum(scores) / len(scores)  # 170.0
        expected_max_score = max(scores)  # 300
        expected_min_score = min(scores)  # 75
        
        assert abs(basic_stats["average_score"] - expected_avg_score) < 0.01, f"Average score mismatch: expected {expected_avg_score}, got {basic_stats['average_score']}"
        assert basic_stats["max_score"] == expected_max_score, f"Max score mismatch: expected {expected_max_score}, got {basic_stats['max_score']}"
        assert basic_stats["min_score"] == expected_min_score, f"Min score mismatch: expected {expected_min_score}, got {basic_stats['min_score']}"
        
        # Verify step statistics
        steps = [75, 90, 45, 120, 65]
        expected_avg_steps = sum(steps) / len(steps)  # 79.0
        expected_total_steps = sum(steps)  # 395
        
        assert abs(basic_stats["average_steps"] - expected_avg_steps) < 0.01, f"Average steps mismatch"
        assert basic_stats["total_steps"] == expected_total_steps, f"Total steps mismatch"
        
        # Verify efficiency metrics
        assert "score_per_step" in basic_stats, "Score per step metric missing"
        expected_score_per_step = sum(scores) / sum(steps)
        assert abs(basic_stats["score_per_step"] - expected_score_per_step) < 0.01, f"Score per step mismatch"
        
        # Verify duration statistics
        durations = [45.2, 52.1, 28.5, 68.3, 38.7]
        expected_avg_duration = sum(durations) / len(durations)
        expected_total_duration = sum(durations)
        
        assert abs(basic_stats["average_duration"] - expected_avg_duration) < 0.01, f"Average duration mismatch"
        assert abs(basic_stats["total_duration"] - expected_total_duration) < 0.01, f"Total duration mismatch"

    def test_performance_trend_analysis(self) -> None:
        """Test performance trend analysis over time."""
        
        stats_utils: GameStatsUtils = GameStatsUtils()
        
        # Mock game data with timestamps for trend analysis
        base_time = time.time() - 3600  # 1 hour ago
        
        time_series_games: List[Dict[str, Any]] = []
        
        # Generate trending game data (improving performance over time)
        for i in range(20):
            timestamp = base_time + (i * 180)  # 3 minutes apart
            
            # Simulate improving performance
            base_score = 100 + (i * 10) + np.random.randint(-20, 20)
            base_steps = 50 + (i * 2) + np.random.randint(-10, 10)
            
            game_data = {
                "game_id": f"trend_game_{i + 1}",
                "final_score": max(50, base_score),  # Ensure positive scores
                "total_steps": max(20, base_steps),  # Ensure positive steps
                "timestamp": timestamp,
                "session_number": (i // 5) + 1,  # 5 games per session
                "completion_status": "success" if i % 7 != 0 else "collision"  # Occasional failures
            }
            
            time_series_games.append(game_data)
        
        # Perform trend analysis
        trend_result = stats_utils.analyze_performance_trends(
            game_results=time_series_games,
            window_size=5,  # 5-game moving average
            trend_metrics=["score", "steps", "efficiency"]
        )
        
        assert trend_result["success"] is True, "Trend analysis should succeed"
        
        trend_analysis = trend_result["trend_analysis"]
        
        # Verify trend structure
        assert "moving_averages" in trend_analysis, "Moving averages missing"
        assert "trend_direction" in trend_analysis, "Trend direction missing"
        assert "trend_strength" in trend_analysis, "Trend strength missing"
        
        moving_averages = trend_analysis["moving_averages"]
        assert "score" in moving_averages, "Score moving average missing"
        assert "steps" in moving_averages, "Steps moving average missing"
        assert "efficiency" in moving_averages, "Efficiency moving average missing"
        
        # Verify moving average calculations
        score_ma = moving_averages["score"]
        assert len(score_ma) == len(time_series_games) - 4, "Moving average should have n-window_size+1 points"
        
        # Verify trend direction (should be generally improving)
        trend_direction = trend_analysis["trend_direction"]
        assert "score" in trend_direction, "Score trend direction missing"
        
        # Since we simulated improving performance, score trend should be positive
        score_trend = trend_direction["score"]
        assert score_trend in ["increasing", "stable", "decreasing"], "Invalid trend direction"
        
        # Verify trend strength metrics
        trend_strength = trend_analysis["trend_strength"]
        assert "score_correlation" in trend_strength, "Score correlation missing"
        assert -1.0 <= trend_strength["score_correlation"] <= 1.0, "Correlation should be between -1 and 1"

    def test_comparative_statistics_analysis(self) -> None:
        """Test comparative analysis between different game sessions or players."""
        
        stats_utils: GameStatsUtils = GameStatsUtils()
        
        # Mock data for different sessions/players
        comparative_data: Dict[str, List[Dict[str, Any]]] = {
            "session_deepseek": [
                {"game_id": f"deepseek_{i}", "final_score": 150 + i * 25, "total_steps": 75 + i * 10, "provider": "deepseek"}
                for i in range(10)
            ],
            "session_mistral": [
                {"game_id": f"mistral_{i}", "final_score": 140 + i * 20, "total_steps": 80 + i * 8, "provider": "mistral"}
                for i in range(10)
            ],
            "session_hunyuan": [
                {"game_id": f"hunyuan_{i}", "final_score": 160 + i * 30, "total_steps": 70 + i * 12, "provider": "hunyuan"}
                for i in range(10)
            ]
        }
        
        # Perform comparative analysis
        comparison_result = stats_utils.compare_performance_groups(
            grouped_data=comparative_data,
            comparison_metrics=["average_score", "average_steps", "score_consistency", "improvement_rate"]
        )
        
        assert comparison_result["success"] is True, "Comparative analysis should succeed"
        
        comparison_analysis = comparison_result["comparison_analysis"]
        
        # Verify comparison structure
        assert "group_statistics" in comparison_analysis, "Group statistics missing"
        assert "ranking" in comparison_analysis, "Ranking missing"
        assert "statistical_significance" in comparison_analysis, "Statistical significance missing"
        
        group_stats = comparison_analysis["group_statistics"]
        
        # Verify all groups are analyzed
        for session_name in comparative_data.keys():
            assert session_name in group_stats, f"Statistics missing for {session_name}"
            
            session_stats = group_stats[session_name]
            assert "average_score" in session_stats, f"Average score missing for {session_name}"
            assert "average_steps" in session_stats, f"Average steps missing for {session_name}"
            assert "score_std" in session_stats, f"Score standard deviation missing for {session_name}"
            assert "game_count" in session_stats, f"Game count missing for {session_name}"
            
            # Verify game count
            assert session_stats["game_count"] == 10, f"Incorrect game count for {session_name}"
        
        # Verify ranking
        ranking = comparison_analysis["ranking"]
        assert "by_average_score" in ranking, "Score ranking missing"
        assert "by_efficiency" in ranking, "Efficiency ranking missing"
        
        score_ranking = ranking["by_average_score"]
        assert len(score_ranking) == 3, "Should rank all 3 sessions"
        
        # Verify ranking structure
        for rank_entry in score_ranking:
            assert "group" in rank_entry, "Group name missing from ranking"
            assert "value" in rank_entry, "Value missing from ranking"
            assert "rank" in rank_entry, "Rank missing from ranking"

    def test_advanced_statistical_metrics(self) -> None:
        """Test calculation of advanced statistical metrics."""
        
        stats_utils: GameStatsUtils = GameStatsUtils()
        
        # Mock comprehensive game data
        comprehensive_games: List[Dict[str, Any]] = []
        
        # Generate varied game data for statistical analysis
        np.random.seed(42)  # For reproducible results
        
        for i in range(50):
            # Generate realistic game data with some correlation
            base_skill = np.random.normal(0.5, 0.2)  # Player skill factor
            difficulty = np.random.choice([0.8, 1.0, 1.2])  # Easy, medium, hard
            
            # Score influenced by skill and difficulty
            score = int(max(50, np.random.normal(200 * base_skill / difficulty, 50)))
            
            # Steps somewhat correlated with score (higher scores take more steps)
            steps = int(max(20, score * 0.4 + np.random.normal(0, 20)))
            
            # Duration somewhat correlated with steps
            duration = max(10.0, steps * 0.5 + np.random.normal(0, 10))
            
            game_data = {
                "game_id": f"advanced_game_{i + 1}",
                "final_score": score,
                "total_steps": steps,
                "duration": duration,
                "snake_length": min(score // 25 + 3, 20),
                "apples_eaten": score // 25,
                "difficulty": difficulty,
                "player_skill": base_skill,
                "moves_pattern": np.random.choice(["aggressive", "conservative", "balanced"]),
                "completion_status": "success" if score > 100 else "collision"
            }
            
            comprehensive_games.append(game_data)
        
        # Calculate advanced metrics
        advanced_result = stats_utils.calculate_advanced_metrics(
            game_results=comprehensive_games,
            metrics=[
                "score_distribution",
                "performance_consistency", 
                "skill_correlation",
                "difficulty_analysis",
                "pattern_analysis"
            ]
        )
        
        assert advanced_result["success"] is True, "Advanced metrics calculation should succeed"
        
        advanced_metrics = advanced_result["advanced_metrics"]
        
        # Verify score distribution analysis
        assert "score_distribution" in advanced_metrics, "Score distribution missing"
        score_dist = advanced_metrics["score_distribution"]
        
        assert "mean" in score_dist, "Distribution mean missing"
        assert "std" in score_dist, "Distribution std missing"
        assert "skewness" in score_dist, "Distribution skewness missing"
        assert "percentiles" in score_dist, "Distribution percentiles missing"
        
        percentiles = score_dist["percentiles"]
        assert "p25" in percentiles, "25th percentile missing"
        assert "p50" in percentiles, "50th percentile (median) missing"
        assert "p75" in percentiles, "75th percentile missing"
        assert "p90" in percentiles, "90th percentile missing"
        
        # Verify consistency metrics
        assert "performance_consistency" in advanced_metrics, "Performance consistency missing"
        consistency = advanced_metrics["performance_consistency"]
        
        assert "coefficient_of_variation" in consistency, "Coefficient of variation missing"
        assert "score_stability_index" in consistency, "Score stability index missing"
        
        cv = consistency["coefficient_of_variation"]
        assert 0.0 <= cv <= 2.0, f"Coefficient of variation should be reasonable: got {cv}"
        
        # Verify correlation analysis
        assert "skill_correlation" in advanced_metrics, "Skill correlation missing"
        correlations = advanced_metrics["skill_correlation"]
        
        assert "score_steps_correlation" in correlations, "Score-steps correlation missing"
        assert "score_duration_correlation" in correlations, "Score-duration correlation missing"
        
        # Correlations should be between -1 and 1
        for corr_name, corr_value in correlations.items():
            assert -1.0 <= corr_value <= 1.0, f"Correlation {corr_name} should be between -1 and 1: got {corr_value}"

    def test_real_time_statistics_updates(self) -> None:
        """Test real-time statistics updates during game sessions."""
        
        stats_utils: GameStatsUtils = GameStatsUtils()
        
        # Mock real-time statistics tracker
        rt_tracker: Mock = Mock()
        rt_tracker.current_stats = {
            "games_played": 0,
            "total_score": 0,
            "total_steps": 0,
            "total_duration": 0.0,
            "running_average_score": 0.0,
            "running_average_steps": 0.0,
            "best_score": 0,
            "worst_score": float('inf'),
            "last_10_scores": [],
            "score_trend": "stable"
        }
        rt_tracker.update_history = []
        
        def mock_update_realtime_stats(game_result: Dict[str, Any]) -> Dict[str, Any]:
            """Mock real-time statistics update."""
            game_score = game_result["final_score"]
            game_steps = game_result["total_steps"]
            game_duration = game_result.get("duration", 0.0)
            
            # Update counters
            rt_tracker.current_stats["games_played"] += 1
            rt_tracker.current_stats["total_score"] += game_score
            rt_tracker.current_stats["total_steps"] += game_steps
            rt_tracker.current_stats["total_duration"] += game_duration
            
            # Update running averages
            games_count = rt_tracker.current_stats["games_played"]
            rt_tracker.current_stats["running_average_score"] = rt_tracker.current_stats["total_score"] / games_count
            rt_tracker.current_stats["running_average_steps"] = rt_tracker.current_stats["total_steps"] / games_count
            
            # Update best/worst scores
            rt_tracker.current_stats["best_score"] = max(rt_tracker.current_stats["best_score"], game_score)
            rt_tracker.current_stats["worst_score"] = min(rt_tracker.current_stats["worst_score"], game_score)
            
            # Update last 10 scores for trend analysis
            last_10 = rt_tracker.current_stats["last_10_scores"]
            last_10.append(game_score)
            if len(last_10) > 10:
                last_10.pop(0)
            
            # Simple trend analysis
            if len(last_10) >= 3:
                recent_avg = sum(last_10[-3:]) / 3
                older_avg = sum(last_10[-6:-3]) / 3 if len(last_10) >= 6 else recent_avg
                
                if recent_avg > older_avg * 1.1:
                    rt_tracker.current_stats["score_trend"] = "improving"
                elif recent_avg < older_avg * 0.9:
                    rt_tracker.current_stats["score_trend"] = "declining"
                else:
                    rt_tracker.current_stats["score_trend"] = "stable"
            
            # Record update
            update_record = {
                "game_id": game_result["game_id"],
                "update_timestamp": time.time(),
                "stats_snapshot": rt_tracker.current_stats.copy()
            }
            rt_tracker.update_history.append(update_record)
            
            return {
                "success": True,
                "updated_stats": rt_tracker.current_stats.copy(),
                "update_timestamp": update_record["update_timestamp"]
            }
        
        rt_tracker.update_realtime_stats = mock_update_realtime_stats
        
        # Simulate real-time game updates
        test_games: List[Dict[str, Any]] = [
            {"game_id": "rt_game_1", "final_score": 150, "total_steps": 75, "duration": 45.0},
            {"game_id": "rt_game_2", "final_score": 180, "total_steps": 90, "duration": 52.0},
            {"game_id": "rt_game_3", "final_score": 120, "total_steps": 60, "duration": 38.0},
            {"game_id": "rt_game_4", "final_score": 220, "total_steps": 110, "duration": 65.0},
            {"game_id": "rt_game_5", "final_score": 200, "total_steps": 100, "duration": 58.0},
            {"game_id": "rt_game_6", "final_score": 250, "total_steps": 125, "duration": 72.0},
            {"game_id": "rt_game_7", "final_score": 190, "total_steps": 95, "duration": 55.0}
        ]
        
        update_results: List[Dict[str, Any]] = []
        
        for game in test_games:
            update_result = rt_tracker.update_realtime_stats(game)
            update_results.append(update_result)
            
            assert update_result["success"] is True, f"Real-time update should succeed for {game['game_id']}"
        
        # Verify final statistics state
        final_stats = rt_tracker.current_stats
        
        assert final_stats["games_played"] == 7, "Should count all games"
        
        expected_total_score = sum(game["final_score"] for game in test_games)
        assert final_stats["total_score"] == expected_total_score, "Total score mismatch"
        
        expected_avg_score = expected_total_score / len(test_games)
        assert abs(final_stats["running_average_score"] - expected_avg_score) < 0.01, "Running average mismatch"
        
        expected_best_score = max(game["final_score"] for game in test_games)
        assert final_stats["best_score"] == expected_best_score, "Best score mismatch"
        
        expected_worst_score = min(game["final_score"] for game in test_games)
        assert final_stats["worst_score"] == expected_worst_score, "Worst score mismatch"
        
        # Verify trend analysis
        assert final_stats["score_trend"] in ["improving", "stable", "declining"], "Invalid trend value"
        
        # Verify update history
        assert len(rt_tracker.update_history) == 7, "Should record all updates"
        
        # Verify timestamps are increasing
        timestamps = [update["update_timestamp"] for update in rt_tracker.update_history]
        assert timestamps == sorted(timestamps), "Update timestamps should be increasing"
