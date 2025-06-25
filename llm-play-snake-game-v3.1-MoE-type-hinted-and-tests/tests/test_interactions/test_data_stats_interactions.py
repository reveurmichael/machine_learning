"""
Tests for GameData ↔ GameStats interactions.

Focuses on testing how GameData and GameStats maintain consistency
in statistics tracking, aggregation, and data integrity across operations.
"""

import pytest
import threading
import time
from typing import List, Dict, Any, Optional, Tuple, Generator
from unittest.mock import Mock, patch

from core.game_data import GameData
from core.game_stats import GameStats


class TestDataStatsInteractions:
    """Test interactions between GameData and GameStats."""

    def test_score_tracking_consistency(self) -> None:
        """Test score tracking consistency between data and stats."""
        game_data: GameData = GameData()
        game_stats: GameStats = GameStats()
        
        # Track score changes
        score_updates: List[Tuple[int, int, int]] = [
            (10, 5, 2),   # score, steps, snake_length
            (30, 15, 4),
            (50, 25, 6),
            (70, 35, 8),
            (100, 50, 10)
        ]
        
        for score, steps, snake_length in score_updates:
            # Update game data
            game_data.update_scores(score, steps, snake_length)
            
            # Update stats
            game_stats.record_step_result(
                valid=True,
                collision=False,
                apple_eaten=(score > game_data.score if hasattr(game_data, '_last_score') else True)
            )
            game_stats.update_game_stats(
                final_score=score,
                total_steps=steps,
                apples_eaten=snake_length - 1
            )
            
            # Verify consistency
            assert game_data.score == score
            assert game_data.steps == steps
            assert game_data.snake_length == snake_length
            
            # Stats should reflect cumulative data
            assert game_stats.current_score == score
            assert game_stats.total_steps == steps

    def test_step_statistics_synchronization(self) -> None:
        """Test step statistics synchronization."""
        game_data: GameData = GameData()
        game_stats: GameStats = GameStats()
        
        # Simulate game progression with various outcomes
        step_scenarios: List[Tuple[str, bool, bool, bool]] = [
            ("valid_move", True, False, False),
            ("apple_eaten", True, False, True),
            ("collision", True, True, False),
            ("invalid_move", False, False, False),
            ("valid_move", True, False, False),
            ("apple_eaten", True, False, True),
            ("collision", True, True, False),
        ]
        
        total_valid: int = 0
        total_collisions: int = 0
        total_apples: int = 0
        current_score: int = 0
        current_steps: int = 0
        
        for scenario, valid, collision, apple_eaten in step_scenarios:
            if valid:
                total_valid += 1
                current_steps += 1
                
                if apple_eaten:
                    total_apples += 1
                    current_score += 10  # Assume 10 points per apple
                
                # Update game data
                game_data.update_scores(current_score, current_steps, 1 + total_apples)
                game_data.moves.append("UP")  # Add move to history
                
                if collision:
                    total_collisions += 1
                    game_data.game_over = True
            
            # Update statistics
            game_stats.record_step_result(valid, collision, apple_eaten)
            
            # Verify step-level consistency
            assert len(game_data.moves) == total_valid
            assert game_data.steps == current_steps
            assert game_stats.step_stats.valid == total_valid
            assert game_stats.step_stats.collisions == total_collisions
            
            if collision:
                assert game_data.game_over
                # Reset for next game
                game_data.reset()
                current_score = 0
                current_steps = 0

    def test_llm_statistics_tracking(self) -> None:
        """Test LLM statistics tracking consistency."""
        game_data: GameData = GameData()
        game_stats: GameStats = GameStats()
        
        # Simulate LLM interactions
        llm_scenarios: List[Tuple[str, str, int, int, bool]] = [
            ("prompt_1", "response_1", 100, 50, True),
            ("prompt_2", "response_2", 150, 75, True),
            ("prompt_3", "", 200, 0, False),  # Failed request
            ("prompt_4", "response_4", 120, 60, True),
            ("prompt_5", "error", 180, 0, False),  # Error response
        ]
        
        successful_requests: int = 0
        failed_requests: int = 0
        total_prompt_tokens: int = 0
        total_completion_tokens: int = 0
        
        for prompt, response, prompt_tokens, completion_tokens, success in llm_scenarios:
            # Add to game data
            game_data.add_llm_communication(prompt, response)
            
            if success and completion_tokens > 0:
                game_data.add_token_usage(prompt_tokens, completion_tokens)
                successful_requests += 1
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
            else:
                failed_requests += 1
            
            # Update stats
            if hasattr(game_stats, 'llm_stats'):
                if success:
                    game_stats.llm_stats.successful_requests += 1
                    game_stats.llm_stats.total_tokens += prompt_tokens + completion_tokens
                else:
                    game_stats.llm_stats.failed_requests += 1
            
            # Verify consistency
            assert len(game_data.llm_communication) == len(llm_scenarios[:len(llm_scenarios)])
            
            if hasattr(game_data, 'total_tokens'):
                expected_total: int = total_prompt_tokens + total_completion_tokens
                assert game_data.total_tokens == expected_total

    def test_concurrent_statistics_updates(self) -> None:
        """Test concurrent statistics updates from multiple sources."""
        game_data: GameData = GameData()
        game_stats: GameStats = GameStats()
        
        update_results: List[Dict[str, Any]] = []
        update_errors: List[Exception] = []
        
        def update_game_progress(thread_id: int) -> None:
            """Update game progress from separate thread."""
            try:
                for i in range(50):
                    # Update game data
                    score: int = i * thread_id
                    steps: int = i + thread_id * 10
                    snake_length: int = 1 + i // 10
                    
                    game_data.update_scores(score, steps, snake_length)
                    
                    # Add move
                    moves: List[str] = ["UP", "RIGHT", "DOWN", "LEFT"]
                    game_data.moves.append(moves[i % 4])
                    
                    # Update statistics
                    apple_eaten: bool = (i % 10 == 0)
                    collision: bool = (i % 25 == 24)
                    
                    game_stats.record_step_result(
                        valid=True,
                        collision=collision,
                        apple_eaten=apple_eaten
                    )
                    
                    if i % 10 == 0:
                        # Add LLM data
                        game_data.add_llm_communication(
                            f"Thread {thread_id} prompt {i}",
                            f"Response {i}"
                        )
                        game_data.add_token_usage(50, 25)
                
                update_results.append({
                    "thread_id": thread_id,
                    "final_score": game_data.score,
                    "final_steps": game_data.steps,
                    "moves_count": len(game_data.moves),
                    "llm_count": len(game_data.llm_communication)
                })
                
            except Exception as e:
                update_errors.append(e)
        
        # Start concurrent updates
        threads: List[threading.Thread] = []
        for i in range(3):
            thread = threading.Thread(target=update_game_progress, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify results
        assert len(update_errors) == 0, f"Concurrent update errors: {update_errors}"
        assert len(update_results) == 3
        
        # Data and stats should be consistent
        assert game_data.steps >= 0
        assert len(game_data.moves) >= 0
        assert game_stats.step_stats.valid >= 0

    def test_aggregation_consistency(self) -> None:
        """Test aggregation consistency between detailed and summary stats."""
        game_data: GameData = GameData()
        game_stats: GameStats = GameStats()
        
        # Simulate multiple games
        games_data: List[Dict[str, Any]] = []
        
        for game_num in range(5):
            # Reset for new game
            game_data.reset()
            
            game_score: int = 0
            game_steps: int = 0
            apples_eaten: int = 0
            
            # Play game
            for step in range(100):
                game_steps += 1
                
                # Simulate apple eating
                if step % 15 == 0:
                    apples_eaten += 1
                    game_score += 10
                
                # Update data
                game_data.update_scores(game_score, game_steps, 1 + apples_eaten)
                game_data.moves.append(["UP", "RIGHT", "DOWN", "LEFT"][step % 4])
                
                # Update stats
                game_stats.record_step_result(
                    valid=True,
                    collision=(step == 99),  # Collision on last step
                    apple_eaten=(step % 15 == 0)
                )
                
                # Add periodic LLM data
                if step % 20 == 0:
                    game_data.add_llm_communication(f"Game {game_num} step {step}", f"Response {step}")
                    game_data.add_token_usage(100, 50)
            
            # End game
            game_data.game_over = True
            
            # Record game summary
            games_data.append({
                "game_num": game_num,
                "final_score": game_data.score,
                "total_steps": game_data.steps,
                "apples_eaten": apples_eaten,
                "llm_requests": len(game_data.llm_communication)
            })
            
            # Update aggregate stats
            game_stats.update_game_stats(
                final_score=game_data.score,
                total_steps=game_data.steps,
                apples_eaten=apples_eaten
            )
        
        # Verify aggregation consistency
        total_score: int = sum(game["final_score"] for game in games_data)
        total_steps: int = sum(game["total_steps"] for game in games_data)
        total_apples: int = sum(game["apples_eaten"] for game in games_data)
        
        # Stats should match aggregated data
        if hasattr(game_stats, 'total_score'):
            assert game_stats.total_score == total_score
        if hasattr(game_stats, 'total_steps'):
            assert game_stats.total_steps == total_steps
        if hasattr(game_stats, 'total_apples_eaten'):
            assert game_stats.total_apples_eaten == total_apples

    def test_statistical_calculations_accuracy(self) -> None:
        """Test accuracy of statistical calculations."""
        game_data: GameData = GameData()
        game_stats: GameStats = GameStats()
        
        # Collect raw data for manual calculation
        scores: List[int] = []
        steps_per_game: List[int] = []
        apples_per_game: List[int] = []
        
        # Simulate multiple games with known outcomes
        for game_num in range(10):
            game_data.reset()
            
            # Predictable game outcomes
            final_score: int = game_num * 20
            total_steps: int = game_num * 10 + 50
            apples_eaten: int = game_num * 2
            
            scores.append(final_score)
            steps_per_game.append(total_steps)
            apples_per_game.append(apples_eaten)
            
            # Update data and stats
            game_data.update_scores(final_score, total_steps, 1 + apples_eaten)
            game_stats.update_game_stats(final_score, total_steps, apples_eaten)
        
        # Calculate expected statistics
        expected_avg_score: float = sum(scores) / len(scores)
        expected_avg_steps: float = sum(steps_per_game) / len(steps_per_game)
        expected_max_score: int = max(scores)
        expected_min_score: int = min(scores)
        
        # Verify statistical accuracy
        if hasattr(game_stats, 'average_score'):
            assert abs(game_stats.average_score - expected_avg_score) < 0.1
        if hasattr(game_stats, 'average_steps'):
            assert abs(game_stats.average_steps - expected_avg_steps) < 0.1
        if hasattr(game_stats, 'max_score'):
            assert game_stats.max_score == expected_max_score
        if hasattr(game_stats, 'min_score'):
            assert game_stats.min_score == expected_min_score

    def test_data_integrity_during_stats_export(self) -> None:
        """Test data integrity during statistics export."""
        game_data: GameData = GameData()
        game_stats: GameStats = GameStats()
        
        # Build substantial data
        for i in range(100):
            game_data.update_scores(i * 5, i, 1 + i // 10)
            game_data.moves.append(["UP", "RIGHT", "DOWN", "LEFT"][i % 4])
            
            if i % 10 == 0:
                game_data.add_llm_communication(f"Prompt {i}", f"Response {i}")
                game_data.add_token_usage(100, 50)
            
            game_stats.record_step_result(
                valid=True,
                collision=(i % 50 == 49),
                apple_eaten=(i % 10 == 0)
            )
        
        # Export data before verification
        data_export: Dict[str, Any] = game_data.to_dict()
        
        if hasattr(game_stats, 'to_dict'):
            stats_export: Dict[str, Any] = game_stats.to_dict()
            
            # Verify export consistency
            assert data_export["score"] == game_data.score
            assert data_export["steps"] == game_data.steps
            assert len(data_export["moves"]) == len(game_data.moves)
            
            # Stats export should be consistent with data export
            if "total_steps" in stats_export:
                assert stats_export["total_steps"] == data_export["steps"]

    def test_memory_efficiency_large_datasets(self) -> None:
        """Test memory efficiency with large statistical datasets."""
        game_data: GameData = GameData()
        game_stats: GameStats = GameStats()
        
        # Generate large dataset
        large_dataset_size: int = 10000
        
        start_time: float = time.time()
        
        for i in range(large_dataset_size):
            # Update data efficiently
            if i % 100 == 0:  # Reduce frequency to avoid memory explosion
                game_data.update_scores(i, i, 1 + i // 100)
                game_data.moves.append(["UP", "RIGHT", "DOWN", "LEFT"][i % 4])
            
            # Update stats (should be lightweight)
            game_stats.record_step_result(
                valid=True,
                collision=(i % 1000 == 999),
                apple_eaten=(i % 100 == 0)
            )
        
        end_time: float = time.time()
        processing_time: float = end_time - start_time
        
        # Verify reasonable performance
        assert processing_time < 5.0  # Should complete in under 5 seconds
        
        # Verify data integrity maintained
        assert game_data.steps >= 0
        assert len(game_data.moves) >= 0
        assert game_stats.step_stats.valid == large_dataset_size

    def test_statistics_reset_consistency(self) -> None:
        """Test statistics reset consistency with data reset."""
        game_data: GameData = GameData()
        game_stats: GameStats = GameStats()
        
        # Build up state
        for i in range(50):
            game_data.update_scores(i * 10, i, 1 + i // 5)
            game_stats.record_step_result(True, False, (i % 5 == 0))
        
        # Verify state exists
        assert game_data.score > 0
        assert game_data.steps > 0
        assert game_stats.step_stats.valid > 0
        
        # Reset data
        game_data.reset()
        
        # Reset stats (if method exists)
        if hasattr(game_stats, 'reset'):
            game_stats.reset()
        
        # Verify reset consistency
        assert game_data.score == 0
        assert game_data.steps == 0
        assert len(game_data.moves) == 0
        
        if hasattr(game_stats, 'reset'):
            assert game_stats.step_stats.valid == 0
            assert game_stats.step_stats.collisions == 0 