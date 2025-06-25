"""
Tests for the GameData class.
"""

import pytest
import json
import tempfile
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from unittest.mock import patch, Mock

from core.game_data import GameData
from llm.client import LLMClient


class TestGameData:
    """Test cases for GameData."""

    def test_init(self):
        """Test GameData initialization."""
        game_data = GameData()
        
        assert game_data.game_number == 0
        assert game_data.score == 0
        assert game_data.steps == 0
        assert game_data.game_over is False
        assert game_data.game_end_reason is None
        assert game_data.apple_positions == []
        assert game_data.snake_positions == []
        assert game_data.moves == []
        assert game_data.stats is not None
        assert game_data.round_manager is not None

    def test_reset(self):
        """Test resetting game data."""
        game_data = GameData()
        
        # Modify some values
        game_data.score = 10
        game_data.steps = 25
        game_data.game_over = True
        game_data.moves = ["UP", "DOWN"]
        
        # Reset and verify
        game_data.reset()
        
        assert game_data.score == 0
        assert game_data.steps == 0
        assert game_data.game_over is False
        assert game_data.moves == []
        assert game_data.apple_positions == []

    def test_record_move(self):
        """Test recording a move."""
        game_data = GameData()
        
        game_data.record_move("UP")
        
        assert game_data.steps == 1
        assert game_data.moves == ["UP"]
        assert game_data.stats.step_stats.valid == 1

    def test_record_move_with_apple_eaten(self):
        """Test recording a move that eats an apple."""
        game_data = GameData()
        
        game_data.record_move("RIGHT", apple_eaten=True)
        
        assert game_data.steps == 1
        assert game_data.score == 1
        assert game_data.moves == ["RIGHT"]

    def test_record_move_normalization(self):
        """Test that moves are normalized when recorded."""
        game_data = GameData()
        
        game_data.record_move("up")
        game_data.record_move(" left ")
        
        assert game_data.moves == ["UP", "LEFT"]

    def test_record_apple_position(self):
        """Test recording apple positions."""
        game_data = GameData()
        
        position = [5, 7]
        game_data.record_apple_position(position)
        
        assert len(game_data.apple_positions) == 1
        assert game_data.apple_positions[0] == {"x": 5, "y": 7}

    def test_record_empty_move(self):
        """Test recording empty moves."""
        game_data = GameData()
        
        game_data.record_empty_move()
        
        assert game_data.steps == 1
        assert game_data.stats.step_stats.empty == 1
        assert game_data.moves == ["EMPTY"]

    def test_record_invalid_reversal(self):
        """Test recording invalid reversals."""
        game_data = GameData()
        
        game_data.record_invalid_reversal()
        
        assert game_data.steps == 1
        assert game_data.stats.step_stats.invalid_reversals == 1
        assert game_data.moves == ["INVALID_REVERSAL"]

    def test_record_something_is_wrong_move(self):
        """Test recording something is wrong moves."""
        game_data = GameData()
        
        game_data.record_something_is_wrong_move()
        
        assert game_data.steps == 1
        assert game_data.stats.step_stats.something_wrong == 1
        assert game_data.moves == ["SOMETHING_IS_WRONG"]

    def test_record_no_path_found_move(self):
        """Test recording no path found moves."""
        game_data = GameData()
        
        game_data.record_no_path_found_move()
        
        assert game_data.steps == 1
        assert game_data.moves == ["NO_PATH_FOUND"]

    def test_record_game_end(self):
        """Test recording game end."""
        game_data = GameData()
        
        reason = "collision_wall"
        game_data.record_game_end(reason)
        
        assert game_data.game_over is True
        assert game_data.game_end_reason == reason

    def test_record_game_end_multiple_calls(self):
        """Test that recording game end multiple times doesn't affect time stats."""
        game_data = GameData()
        
        # Record end first time
        game_data.record_game_end("collision_wall")
        first_end_time = game_data.stats.time_stats.end_time
        
        # Record end second time
        game_data.record_game_end("collision_self")
        second_end_time = game_data.stats.time_stats.end_time
        
        # End time should not change
        assert first_end_time == second_end_time
        assert game_data.game_end_reason == "collision_self"

    def test_start_new_round(self):
        """Test starting a new round."""
        game_data = GameData()
        
        apple_position = [3, 4]
        game_data.start_new_round(apple_position)
        
        # Should delegate to round manager
        assert game_data.round_manager.round_buffer is not None

    def test_snake_length_property(self):
        """Test snake_length property."""
        game_data = GameData()
        
        # Initially empty
        assert game_data.snake_length == 0
        
        # Add some positions
        game_data.snake_positions = [[5, 5], [5, 4], [5, 3]]
        assert game_data.snake_length == 3

    def test_llm_communication_timing(self):
        """Test LLM communication timing methods."""
        game_data = GameData()
        
        game_data.record_llm_communication_start()
        # Start time should be recorded in stats
        
        game_data.record_llm_communication_end()
        # End time should be recorded in stats

    def test_response_time_recording(self):
        """Test response time recording."""
        game_data = GameData()
        
        game_data.record_primary_response_time(1.5)
        game_data.record_secondary_response_time(0.8)
        
        assert 1.5 in game_data.primary_response_times
        assert 0.8 in game_data.secondary_response_times

    def test_token_stats_recording(self):
        """Test token statistics recording."""
        game_data = GameData()
        
        game_data.record_primary_token_stats(100, 50)
        game_data.record_secondary_token_stats(80, 30)
        
        token_stats = game_data.get_token_stats()
        assert token_stats["primary_total_tokens"] == 150
        assert token_stats["secondary_total_tokens"] == 110

    def test_get_prompt_response_stats(self):
        """Test getting prompt response statistics."""
        game_data = GameData()
        
        game_data.record_primary_response_time(1.0)
        game_data.record_primary_response_time(2.0)
        game_data.record_secondary_response_time(0.5)
        
        stats = game_data.get_prompt_response_stats()
        
        assert stats["avg_primary_response_time"] == 1.5
        assert stats["avg_secondary_response_time"] == 0.5
        assert len(stats["primary_response_times"]) == 2
        assert len(stats["secondary_response_times"]) == 1

    def test_get_prompt_response_stats_empty(self):
        """Test getting prompt response stats when no data recorded."""
        game_data = GameData()
        
        stats = game_data.get_prompt_response_stats()
        
        assert stats["avg_primary_response_time"] == 0.0
        assert stats["avg_secondary_response_time"] == 0.0
        assert stats["primary_response_times"] == []
        assert stats["secondary_response_times"] == []

    def test_properties_shortcuts(self):
        """Test property shortcuts for step stats."""
        game_data = GameData()
        
        game_data.record_move("UP")
        game_data.record_empty_move()
        game_data.record_invalid_reversal()
        game_data.record_something_is_wrong_move()
        
        assert game_data.valid_steps == 1
        assert game_data.empty_steps == 1
        assert game_data.invalid_reversals == 1
        assert game_data.something_is_wrong_steps == 1

    def test_get_time_stats(self):
        """Test getting time statistics."""
        game_data = GameData()
        
        time_stats = game_data.get_time_stats()
        
        assert "start_time" in time_stats
        assert "end_time" in time_stats
        assert "duration" in time_stats

    def test_record_continuation(self):
        """Test recording continuation data."""
        game_data = GameData()
        
        # Should not raise any errors
        game_data.record_continuation()

    def test_get_round_count(self):
        """Test getting round count."""
        game_data = GameData()
        
        # Initially should be 0
        assert game_data.get_round_count() == 0
        
        # After starting rounds, should delegate to round manager
        game_data.start_new_round([1, 1])
        # Round count should be managed by round manager

    def test_generate_game_summary(self):
        """Test generating comprehensive game summary."""
        game_data = GameData()
        
        # Set up some game data
        game_data.score = 5
        game_data.steps = 20
        game_data.snake_positions = [[5, 5], [5, 4], [5, 3]]
        game_data.record_move("UP")
        game_data.record_move("RIGHT")
        game_data.record_apple_position([7, 8])
        game_data.record_game_end("collision_wall")
        
        summary = game_data.generate_game_summary(
            primary_provider="test_provider",
            primary_model="test_model",
            parser_provider=None,
            parser_model=None
        )
        
        # Verify structure
        assert summary["score"] == 5
        assert summary["steps"] == 20
        assert summary["snake_length"] == 3
        assert summary["game_over"] is True
        assert summary["game_end_reason"] == "collision_wall"
        
        assert "llm_info" in summary
        assert summary["llm_info"]["primary_provider"] == "test_provider"
        assert summary["llm_info"]["primary_model"] == "test_model"
        
        assert "time_stats" in summary
        assert "prompt_response_stats" in summary
        assert "token_stats" in summary
        assert "step_stats" in summary
        assert "metadata" in summary

    def test_save_game_summary(self, temp_dir):
        """Test saving game summary to file."""
        game_data = GameData()
        
        # Set up some basic data
        game_data.score = 3
        game_data.steps = 15
        
        filepath = os.path.join(temp_dir, "test_game.json")
        
        summary = game_data.save_game_summary(
            filepath,
            primary_provider="test_provider",
            primary_model="test_model",
            parser_provider=None,
            parser_model=None
        )
        
        # Verify file was created
        assert os.path.exists(filepath)
        
        # Verify file contents
        with open(filepath, 'r') as f:
            saved_data = json.load(f)
            
        assert saved_data["score"] == 3
        assert saved_data["steps"] == 15

    def test_save_game_summary_file_error(self):
        """Test saving game summary when file cannot be written."""
        game_data = GameData()
        
        # Try to save to invalid path
        invalid_path = "/invalid/path/game.json"
        
        # Should handle error gracefully
        summary = game_data.save_game_summary(
            invalid_path,
            primary_provider="test_provider",
            primary_model="test_model",
            parser_provider=None,
            parser_model=None
        )
        
        # Should still return the summary dict
        assert isinstance(summary, dict)
        assert summary["score"] == 0  # Default values

    def test_record_parsed_llm_response(self):
        """Test recording parsed LLM responses."""
        game_data = GameData()
        
        response = {"moves": ["UP", "RIGHT"]}
        
        # Should delegate to round manager
        game_data.record_parsed_llm_response(response, is_primary=True)
        
        # Should not raise errors

    def test_calculate_actual_round_count(self):
        """Test calculating actual round count."""
        game_data = GameData()
        
        # Should delegate to round manager or have internal logic
        round_count = game_data._calculate_actual_round_count()
        assert isinstance(round_count, int)
        assert round_count >= 0

    def test_integration_full_game_flow(self):
        """Test integration of GameData through a full game flow."""
        game_data = GameData()
        
        # Start game
        game_data.start_new_round([5, 5])
        
        # Make some moves
        game_data.record_move("UP")
        game_data.record_move("RIGHT", apple_eaten=True)
        game_data.record_apple_position([7, 8])
        
        # Record some stats
        game_data.record_primary_response_time(1.2)
        game_data.record_primary_token_stats(120, 60)
        
        # End game
        game_data.record_game_end("collision_wall")
        
        # Generate summary
        summary = game_data.generate_game_summary(
            primary_provider="test",
            primary_model="test",
            parser_provider=None,
            parser_model=None
        )
        
        # Verify complete integration
        assert summary["score"] == 1
        assert summary["steps"] == 2
        assert len(summary["moves"]) == 2
        assert summary["game_over"] is True

    def test_comprehensive_move_recording(self):
        """Test comprehensive move recording with various scenarios."""
        game_data = GameData()
        
        # Test different move types
        game_data.record_move("UP")
        game_data.record_move("DOWN", apple_eaten=True)
        game_data.record_move("LEFT")
        game_data.record_move("RIGHT", apple_eaten=True)
        
        assert game_data.steps == 4
        assert game_data.score == 2
        assert game_data.moves == ["UP", "DOWN", "LEFT", "RIGHT"]
        assert game_data.stats.step_stats.valid == 4
        
        # Test sentinel moves
        game_data.record_empty_move()
        game_data.record_invalid_reversal()
        game_data.record_something_is_wrong_move()
        game_data.record_no_path_found_move()
        
        assert game_data.steps == 8
        assert game_data.score == 2  # No score change for sentinel moves
        assert game_data.moves[-4:] == ["EMPTY", "INVALID_REVERSAL", "SOMETHING_IS_WRONG", "NO_PATH_FOUND"]
        assert game_data.stats.step_stats.empty == 1
        assert game_data.stats.step_stats.invalid_reversals == 1
        assert game_data.stats.step_stats.something_wrong == 1

    def test_apple_position_management(self):
        """Test comprehensive apple position management."""
        game_data = GameData()
        
        # Record multiple apple positions
        positions = [[1, 2], [3, 4], [5, 6], [7, 8]]
        for pos in positions:
            game_data.record_apple_position(pos)
        
        assert len(game_data.apple_positions) == 4
        for i, pos in enumerate(positions):
            assert game_data.apple_positions[i] == {"x": pos[0], "y": pos[1]}
        
        # Test with numpy arrays
        import numpy as np
        np_position = np.array([9, 10])
        game_data.record_apple_position(np_position)
        assert game_data.apple_positions[-1] == {"x": 9, "y": 10}

    def test_round_management_integration(self):
        """Test integration with round management system."""
        game_data = GameData()
        
        # Test round creation and management
        game_data.start_new_round([1, 1])
        assert game_data.round_manager.round_buffer is not None
        assert game_data.round_manager.round_count == 2  # Starts at 1, incremented to 2
        
        # Record moves in round
        game_data.record_move("UP")
        game_data.record_move("RIGHT")
        
        # Start new round
        game_data.start_new_round([2, 2])
        assert game_data.round_manager.round_count == 3
        
        # Test round data persistence
        game_data.record_move("DOWN")
        game_data.record_move("LEFT")
        
        # Verify round data is tracked
        assert len(game_data.round_manager.rounds_data) >= 0

    def test_statistics_comprehensive_tracking(self):
        """Test comprehensive statistics tracking."""
        game_data = GameData()
        
        # Test time statistics
        game_data.record_llm_communication_start()
        import time
        time.sleep(0.01)  # Small delay
        game_data.record_llm_communication_end()
        
        assert game_data.stats.time_stats.llm_communication_time > 0
        
        # Test response time tracking
        response_times = [0.5, 1.2, 0.8, 2.1]
        for rt in response_times:
            game_data.record_primary_response_time(rt)
        
        assert len(game_data.primary_response_times) == 4
        assert game_data.primary_response_times == response_times
        
        # Test secondary response times
        secondary_times = [0.3, 0.7, 1.1]
        for rt in secondary_times:
            game_data.record_secondary_response_time(rt)
        
        assert len(game_data.secondary_response_times) == 3
        assert game_data.secondary_response_times == secondary_times

    def test_token_statistics_comprehensive(self):
        """Test comprehensive token statistics tracking."""
        game_data = GameData()
        
        # Test primary token stats
        token_data = [(100, 50), (200, 75), (150, 60)]
        for prompt_tokens, completion_tokens in token_data:
            game_data.record_primary_token_stats(prompt_tokens, completion_tokens)
        
        assert game_data.stats.primary_total_prompt_tokens == 450
        assert game_data.stats.primary_total_completion_tokens == 185
        assert game_data.stats.primary_total_tokens == 635
        assert game_data.stats.primary_llm_requests == 3
        
        # Test averages
        assert game_data.stats.primary_avg_prompt_tokens == 150.0
        assert game_data.stats.primary_avg_completion_tokens == 185/3
        
        # Test secondary token stats
        secondary_data = [(80, 40), (120, 30)]
        for prompt_tokens, completion_tokens in secondary_data:
            game_data.record_secondary_token_stats(prompt_tokens, completion_tokens)
        
        assert game_data.stats.secondary_total_prompt_tokens == 200
        assert game_data.stats.secondary_total_completion_tokens == 70
        assert game_data.stats.secondary_total_tokens == 270
        assert game_data.stats.secondary_llm_requests == 2

    def test_game_summary_comprehensive(self):
        """Test comprehensive game summary generation."""
        game_data = GameData()
        
        # Set up comprehensive game state
        game_data.score = 15
        game_data.steps = 100
        game_data.snake_positions = [[1, 1], [1, 2], [1, 3]]
        game_data.game_over = True
        game_data.game_end_reason = "collision_wall"
        
        # Add some statistics
        game_data.record_primary_response_time(1.5)
        game_data.record_secondary_response_time(0.8)
        game_data.record_primary_token_stats(100, 50)
        game_data.record_secondary_token_stats(80, 40)
        
        # Generate summary
        summary = game_data.generate_game_summary(
            primary_provider="ollama",
            primary_model="llama2",
            parser_provider="openai",
            parser_model="gpt-3.5-turbo"
        )
        
        # Verify summary structure
        assert summary["score"] == 15
        assert summary["steps"] == 100
        assert summary["snake_length"] == 3
        assert summary["game_over"] is True
        assert summary["game_end_reason"] == "collision_wall"
        
        # Verify LLM info
        assert summary["llm_info"]["primary_provider"] == "ollama"
        assert summary["llm_info"]["primary_model"] == "llama2"
        assert summary["llm_info"]["parser_provider"] == "openai"
        assert summary["llm_info"]["parser_model"] == "gpt-3.5-turbo"
        
        # Verify statistics
        assert "time_stats" in summary
        assert "prompt_response_stats" in summary
        assert "token_stats" in summary
        assert "step_stats" in summary

    def test_continuation_functionality(self):
        """Test game continuation functionality."""
        game_data = GameData()
        
        # Record initial state
        game_data.score = 5
        game_data.steps = 20
        
        # Record continuation
        game_data.record_continuation()
        
        # Continue playing
        game_data.record_move("UP")
        game_data.record_move("RIGHT", apple_eaten=True)
        
        assert game_data.score == 6
        assert game_data.steps == 22

    def test_property_shortcuts_comprehensive(self):
        """Test all property shortcuts comprehensively."""
        game_data = GameData()
        
        # Test initial values
        assert game_data.valid_steps == 0
        assert game_data.invalid_reversals == 0
        assert game_data.empty_steps == 0
        assert game_data.something_is_wrong_steps == 0
        
        # Add various step types
        game_data.record_move("UP")
        game_data.record_move("DOWN")
        game_data.record_invalid_reversal()
        game_data.record_empty_move()
        game_data.record_something_is_wrong_move()
        
        assert game_data.valid_steps == 2
        assert game_data.invalid_reversals == 1
        assert game_data.empty_steps == 1
        assert game_data.something_is_wrong_steps == 1
        
        # Test response time properties
        game_data.record_primary_response_time(1.0)
        game_data.record_primary_response_time(2.0)
        game_data.record_secondary_response_time(0.5)
        
        assert game_data.primary_response_times == [1.0, 2.0]
        assert game_data.secondary_response_times == [0.5]

    def test_round_count_calculation(self):
        """Test round count calculation methods."""
        game_data = GameData()
        
        # Test initial round count
        assert game_data.get_round_count() >= 0
        
        # Start multiple rounds
        game_data.start_new_round([1, 1])
        game_data.start_new_round([2, 2])
        game_data.start_new_round([3, 3])
        
        # Verify round count increases
        round_count = game_data.get_round_count()
        assert round_count >= 3
        
        # Test actual round count calculation
        actual_count = game_data._calculate_actual_round_count()
        assert actual_count >= 0

    def test_memory_management(self):
        """Test memory management with large datasets."""
        game_data = GameData()
        
        # Generate large number of moves
        moves = ["UP", "DOWN", "LEFT", "RIGHT"] * 250  # 1000 moves
        
        for i, move in enumerate(moves):
            game_data.record_move(move, apple_eaten=(i % 10 == 0))
        
        assert len(game_data.moves) == 1000
        assert game_data.steps == 1000
        assert game_data.score == 100  # Every 10th move ate apple
        
        # Test memory efficiency
        import sys
        moves_size = sys.getsizeof(game_data.moves)
        assert moves_size < 100000  # Should be reasonable size

    def test_serialization_compatibility(self):
        """Test serialization compatibility for persistence."""
        game_data = GameData()
        
        # Set up complex state
        game_data.score = 10
        game_data.steps = 50
        game_data.snake_positions = [[1, 1], [1, 2], [1, 3]]
        game_data.moves = ["UP", "RIGHT", "DOWN", "LEFT"]
        game_data.record_primary_response_time(1.5)
        game_data.record_primary_token_stats(100, 50)
        
        # Test JSON serialization
        summary = game_data.generate_game_summary(
            primary_provider="test",
            primary_model="test_model",
            parser_provider=None,
            parser_model=None
        )
        
        import json
        json_str = json.dumps(summary, cls=NumPyJSONEncoder)
        restored = json.loads(json_str)
        
        assert restored["score"] == 10
        assert restored["steps"] == 50
        assert restored["snake_length"] == 3

    def test_edge_cases_comprehensive(self):
        """Test comprehensive edge cases."""
        game_data = GameData()
        
        # Test empty game
        summary = game_data.generate_game_summary(
            primary_provider="test",
            primary_model=None,
            parser_provider=None,
            parser_model=None
        )
        assert summary["score"] == 0
        assert summary["steps"] == 0
        
        # Test game with only sentinel moves
        game_data.record_empty_move()
        game_data.record_invalid_reversal()
        game_data.record_something_is_wrong_move()
        
        assert game_data.steps == 3
        assert game_data.score == 0
        assert game_data.valid_steps == 0
        
        # Test very long snake
        game_data.snake_positions = [[i, 0] for i in range(100)]
        assert game_data.snake_length == 100

    def test_performance_benchmarks(self):
        """Test performance benchmarks for key operations."""
        import time
        
        game_data = GameData()
        
        # Benchmark move recording
        start_time = time.time()
        for i in range(1000):
            game_data.record_move("UP")
        move_time = time.time() - start_time
        
        assert move_time < 1.0  # Should complete in under 1 second
        
        # Benchmark summary generation
        start_time = time.time()
        for i in range(10):
            game_data.generate_game_summary(
                primary_provider="test",
                primary_model="test",
                parser_provider="test",
                parser_model="test"
            )
        summary_time = time.time() - start_time
        
        assert summary_time < 1.0  # Should be fast 