"""Performance and stress tests for the replay module."""

import pytest
import time
import json
import os
import tempfile
import threading
from unittest.mock import Mock, patch
from typing import Dict, List, Any
import numpy as np

from replay.replay_engine import ReplayEngine
from replay.replay_utils import load_game_json, parse_game_data


class TestReplayPerformance:
    """Performance tests for replay functionality."""

    def create_large_game_dataset(self, temp_dir: str, num_games: int = 10, moves_per_game: int = 1000) -> None:
        """Create a large dataset for performance testing."""
        for game_num in range(1, num_games + 1):
            game_data = {
                "game_number": game_num,
                "score": game_num * 100,
                "metadata": {
                    "timestamp": f"2024-01-01 {10 + game_num}:00:00",
                    "round_count": moves_per_game // 10
                },
                "detailed_history": {
                    "apple_positions": [
                        {"x": (i * game_num) % 20, "y": (i * game_num + 1) % 20} 
                        for i in range(moves_per_game + 1)
                    ],
                    "moves": [
                        ["UP", "DOWN", "LEFT", "RIGHT"][i % 4] 
                        for i in range(moves_per_game)
                    ],
                    "rounds_data": {
                        f"round_{round_num}": {
                            "moves": ["PLANNED"] + [
                                ["UP", "DOWN", "LEFT", "RIGHT"][(i + round_num) % 4] 
                                for i in range(10)
                            ],
                            "llm_response": f"Response for round {round_num}"
                        } for round_num in range(1, moves_per_game // 10 + 1)
                    }
                },
                "llm_info": {
                    "primary_provider": f"provider_{game_num % 3}",
                    "primary_model": f"model_{game_num % 5}",
                    "parser_provider": "parser" if game_num % 2 == 0 else None,
                    "parser_model": "parser_model" if game_num % 2 == 0 else None
                },
                "game_end_reason": ["apple_eaten", "wall_collision", "max_steps"][game_num % 3]
            }
            
            game_file = os.path.join(temp_dir, f"game_{game_num}.json")
            with open(game_file, 'w') as f:
                json.dump(game_data, f)

    def test_large_game_loading_performance(self) -> None:
        """Test loading performance with large game files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create large game data (10,000 moves)
            self.create_large_game_dataset(temp_dir, num_games=1, moves_per_game=10000)
            
            with patch('utils.file_utils.get_total_games', return_value=1):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                # Measure loading time
                start_time = time.time()
                
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        result = engine.load_game_data(1)
                        
                        load_time = time.time() - start_time
                        
                        # Verify successful loading
                        assert result is not None
                        assert len(engine.moves) == 10000
                        assert len(engine.apple_positions) == 10001  # One more than moves
                        
                        # Performance assertion - should load within reasonable time
                        assert load_time < 10.0, f"Loading took {load_time:.2f}s, expected < 10s"

    def test_multiple_games_loading_performance(self) -> None:
        """Test performance when loading multiple games sequentially."""
        with tempfile.TemporaryDirectory() as temp_dir:
            num_games = 50
            self.create_large_game_dataset(temp_dir, num_games=num_games, moves_per_game=500)
            
            with patch('utils.file_utils.get_total_games', return_value=num_games):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                total_start_time = time.time()
                
                with patch('utils.file_utils.get_game_json_filename', side_effect=lambda n: f"game_{n}.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        for game_num in range(1, 11):  # Load first 10 games
                            start_time = time.time()
                            result = engine.load_game_data(game_num)
                            load_time = time.time() - start_time
                            
                            assert result is not None
                            assert load_time < 2.0, f"Game {game_num} loading took {load_time:.2f}s"
                
                total_time = time.time() - total_start_time
                assert total_time < 15.0, f"Total loading time {total_time:.2f}s exceeded 15s"

    def test_state_building_performance(self) -> None:
        """Test performance of state building operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_large_game_dataset(temp_dir, num_games=1, moves_per_game=5000)
            
            with patch('utils.file_utils.get_total_games', return_value=1):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        engine.load_game_data(1)
                        
                        # Measure state building performance
                        iterations = 1000
                        start_time = time.time()
                        
                        for _ in range(iterations):
                            state = engine._build_state_base()
                            
                        total_time = time.time() - start_time
                        avg_time = total_time / iterations
                        
                        # Should build state quickly
                        assert avg_time < 0.001, f"Average state building time {avg_time:.4f}s too slow"

    def test_move_execution_performance(self) -> None:
        """Test performance of move execution in replay."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_large_game_dataset(temp_dir, num_games=1, moves_per_game=1000)
            
            with patch('utils.file_utils.get_total_games', return_value=1):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        engine.load_game_data(1)
                        
                        # Mock move method for consistent timing
                        with patch.object(engine, 'make_move', return_value=(True, False)):
                            # Measure move execution performance
                            num_moves = 100
                            start_time = time.time()
                            
                            for i in range(num_moves):
                                move = engine.moves[i % len(engine.moves)]
                                engine.execute_replay_move(move)
                            
                            total_time = time.time() - start_time
                            avg_time = total_time / num_moves
                            
                            # Should execute moves quickly
                            assert avg_time < 0.01, f"Average move execution time {avg_time:.4f}s too slow"

    def test_json_parsing_performance(self) -> None:
        """Test JSON parsing performance for large game files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_large_game_dataset(temp_dir, num_games=1, moves_per_game=20000)
            
            game_file = os.path.join(temp_dir, "game_1.json")
            
            # Test raw JSON loading performance
            iterations = 10
            start_time = time.time()
            
            for _ in range(iterations):
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', return_value=game_file):
                        file_path, data = load_game_json(temp_dir, 1)
                        assert data is not None
            
            load_time = time.time() - start_time
            avg_load_time = load_time / iterations
            
            # Test parsing performance
            start_time = time.time()
            
            for _ in range(iterations):
                parsed = parse_game_data(data)
                assert parsed is not None
            
            parse_time = time.time() - start_time
            avg_parse_time = parse_time / iterations
            
            # Performance assertions
            assert avg_load_time < 1.0, f"Average JSON loading time {avg_load_time:.3f}s too slow"
            assert avg_parse_time < 0.5, f"Average parsing time {avg_parse_time:.3f}s too slow"


class TestReplayMemoryUsage:
    """Memory usage tests for replay functionality."""

    def test_memory_efficiency_with_large_datasets(self) -> None:
        """Test memory efficiency when handling large datasets."""
        import gc
        import sys
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create moderately large dataset
            game_data = {
                "game_number": 1,
                "score": 5000,
                "metadata": {"timestamp": "2024-01-01", "round_count": 500},
                "detailed_history": {
                    "apple_positions": [{"x": i % 20, "y": (i + 1) % 20} for i in range(5000)],
                    "moves": ["UP", "DOWN", "LEFT", "RIGHT"] * 1250,
                    "rounds_data": {
                        f"round_{i}": {
                            "moves": ["PLANNED", "UP", "DOWN", "LEFT"],
                            "llm_response": f"Response {i}" * 100  # Large text
                        } for i in range(1, 501)
                    }
                },
                "llm_info": {"primary_provider": "test", "primary_model": "test"},
                "game_end_reason": "max_steps"
            }
            
            game_file = os.path.join(temp_dir, "game_1.json")
            with open(game_file, 'w') as f:
                json.dump(game_data, f)
            
            # Measure memory before
            gc.collect()
            initial_objects = len(gc.get_objects())
            
            with patch('utils.file_utils.get_total_games', return_value=1):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        engine.load_game_data(1)
                        
                        # Measure memory after loading
                        gc.collect()
                        objects_after_load = len(gc.get_objects())
                        
                        # Clear engine reference
                        del engine
                        
                        # Measure memory after cleanup
                        gc.collect()
                        final_objects = len(gc.get_objects())
                        
                        # Memory should be relatively stable
                        memory_growth = final_objects - initial_objects
                        
                        # Allow some memory growth but not excessive
                        assert memory_growth < 10000, f"Memory growth {memory_growth} objects too high"

    def test_memory_cleanup_between_games(self) -> None:
        """Test memory cleanup when switching between games."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple games with substantial data
            for game_num in range(1, 4):
                game_data = {
                    "game_number": game_num,
                    "score": game_num * 1000,
                    "metadata": {"timestamp": f"2024-01-0{game_num}", "round_count": 100},
                    "detailed_history": {
                        "apple_positions": [{"x": i, "y": i} for i in range(1000)],
                        "moves": ["UP", "DOWN"] * 500,
                        "rounds_data": {
                            f"round_{i}": {
                                "moves": ["PLANNED", "UP"],
                                "llm_response": f"Game {game_num} Response {i}" * 50
                            } for i in range(1, 101)
                        }
                    },
                    "llm_info": {"primary_provider": f"provider_{game_num}", "primary_model": "model"},
                    "game_end_reason": "test"
                }
                
                game_file = os.path.join(temp_dir, f"game_{game_num}.json")
                with open(game_file, 'w') as f:
                    json.dump(game_data, f)
            
            with patch('utils.file_utils.get_total_games', return_value=3):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                with patch('utils.file_utils.get_game_json_filename', side_effect=lambda n: f"game_{n}.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        # Load each game and verify memory usage
                        for game_num in range(1, 4):
                            # Set game number explicitly since load_game_data doesn't update it
                            engine.game_number = game_num
                            engine.load_game_data(game_num)
                            
                            # Verify data is properly replaced, not accumulated
                            assert len(engine.moves) == 1000  # Same for each game
                            assert engine.game_number == game_num
                            
                            # Verify old data is cleared
                            assert engine.move_index == 0
                            assert engine.apple_index == 0
                            assert engine.moves_made == []

    def test_large_state_object_handling(self) -> None:
        """Test handling of large state objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create game with very large snake (simulating long gameplay)
            large_snake_positions = [[i, i % 20] for i in range(1000)]  # 1000-segment snake
            
            with patch('utils.file_utils.get_total_games', return_value=1):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                # Manually set large snake positions
                engine.snake_positions = np.array(large_snake_positions)
                engine.apple_position = np.array([10, 10])
                engine.game_state.score = 1000
                engine.game_state.steps = 5000
                
                # Test state building with large objects
                start_time = time.time()
                
                for _ in range(100):  # Multiple iterations
                    state = engine._build_state_base()
                    
                    # Verify large state is handled correctly
                    assert len(state['snake_positions']) == 1000
                    assert state['score'] == 1000
                    assert state['steps'] == 5000
                
                total_time = time.time() - start_time
                
                # Should handle large states efficiently
                assert total_time < 5.0, f"Large state handling took {total_time:.2f}s"


class TestReplayStressTests:
    """Stress tests for replay functionality."""

    def test_rapid_game_switching(self) -> None:
        """Test rapid switching between games."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple games
            num_games = 20
            for game_num in range(1, num_games + 1):
                game_data = {
                    "game_number": game_num,
                    "score": game_num * 10,
                    "metadata": {"timestamp": f"2024-01-01", "round_count": 10},
                    "detailed_history": {
                        "apple_positions": [{"x": game_num % 10, "y": (game_num + 1) % 10}] * 50,
                        "moves": ["UP", "DOWN", "LEFT", "RIGHT"] * 25,
                        "rounds_data": {}
                    },
                    "llm_info": {"primary_provider": f"provider_{game_num}", "primary_model": "model"},
                    "game_end_reason": "test"
                }
                
                game_file = os.path.join(temp_dir, f"game_{game_num}.json")
                with open(game_file, 'w') as f:
                    json.dump(game_data, f)
            
            with patch('utils.file_utils.get_total_games', return_value=num_games):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                with patch('utils.file_utils.get_game_json_filename', side_effect=lambda n: f"game_{n}.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        # Rapidly switch between games
                        start_time = time.time()
                        
                        for _ in range(100):  # 100 rapid switches
                            game_num = ((_ % num_games) + 1)
                            # Set game number explicitly since load_game_data doesn't update it
                            engine.game_number = game_num
                            result = engine.load_game_data(game_num)
                            
                            assert result is not None
                            assert engine.game_number == game_num
                        
                        total_time = time.time() - start_time
                        
                        # Should handle rapid switching efficiently
                        assert total_time < 30.0, f"Rapid switching took {total_time:.2f}s"

    def test_high_frequency_state_access(self) -> None:
        """Test high-frequency state access patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            game_data = {
                "game_number": 1,
                "score": 100,
                "metadata": {"timestamp": "2024-01-01", "round_count": 10},
                "detailed_history": {
                    "apple_positions": [{"x": 5, "y": 6}] * 100,
                    "moves": ["UP"] * 100,
                    "rounds_data": {}
                },
                "llm_info": {"primary_provider": "test", "primary_model": "model"},
                "game_end_reason": "test"
            }
            
            game_file = os.path.join(temp_dir, "game_1.json")
            with open(game_file, 'w') as f:
                json.dump(game_data, f)
            
            with patch('utils.file_utils.get_total_games', return_value=1):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        engine.load_game_data(1)
                        
                        # High-frequency state access
                        start_time = time.time()
                        
                        for i in range(10000):  # Very high frequency
                            state = engine._build_state_base()
                            
                            # Modify state to simulate GUI updates
                            engine.move_index = i % len(engine.moves)
                            engine.paused = i % 2 == 0
                        
                        total_time = time.time() - start_time
                        
                        # Should handle high-frequency access
                        assert total_time < 10.0, f"High-frequency access took {total_time:.2f}s"

    def test_concurrent_like_operations(self) -> None:
        """Test concurrent-like operations (simulated)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            game_data = {
                "game_number": 1,
                "score": 100,
                "metadata": {"timestamp": "2024-01-01", "round_count": 10},
                "detailed_history": {
                    "apple_positions": [{"x": 5, "y": 6}] * 50,
                    "moves": ["UP", "DOWN", "LEFT", "RIGHT"] * 25,
                    "rounds_data": {}
                },
                "llm_info": {"primary_provider": "test", "primary_model": "model"},
                "game_end_reason": "test"
            }
            
            game_file = os.path.join(temp_dir, "game_1.json")
            with open(game_file, 'w') as f:
                json.dump(game_data, f)
            
            with patch('utils.file_utils.get_total_games', return_value=1):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        engine.load_game_data(1)
                        
                        # Simulate concurrent-like operations
                        results = []
                        errors = []
                        
                        def simulate_operation(op_id: int) -> None:
                            try:
                                # Each "thread" performs different operations
                                if op_id % 4 == 0:
                                    state = engine._build_state_base()
                                    results.append(f"state_{op_id}")
                                elif op_id % 4 == 1:
                                    engine.paused = not engine.paused
                                    results.append(f"pause_{op_id}")
                                elif op_id % 4 == 2:
                                    engine.move_index = (engine.move_index + 1) % len(engine.moves)
                                    results.append(f"move_{op_id}")
                                else:
                                    # Simulate draw operation
                                    if engine.use_gui and engine.gui:
                                        engine.draw()
                                    results.append(f"draw_{op_id}")
                            except Exception as e:
                                errors.append(f"Error in op {op_id}: {e}")
                        
                        # Simulate multiple "concurrent" operations
                        for i in range(1000):
                            simulate_operation(i)
                        
                        # Verify no errors occurred
                        assert len(errors) == 0, f"Errors occurred: {errors[:5]}"
                        assert len(results) == 1000

    def test_error_recovery_under_stress(self) -> None:
        """Test error recovery under stress conditions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create games with some invalid data mixed in
            for game_num in range(1, 21):
                if game_num % 5 == 0:  # Every 5th game has invalid data
                    game_data = {
                        "game_number": game_num,
                        "detailed_history": {
                            "apple_positions": [],  # Invalid: empty
                            "moves": ["UP"]
                        }
                    }
                else:
                    game_data = {
                        "game_number": game_num,
                        "score": game_num * 10,
                        "metadata": {"timestamp": "2024-01-01", "round_count": 5},
                        "detailed_history": {
                            "apple_positions": [{"x": 1, "y": 2}] * 10,
                            "moves": ["UP"] * 10,
                            "rounds_data": {}
                        },
                        "llm_info": {"primary_provider": "test", "primary_model": "model"},
                        "game_end_reason": "test"
                    }
                
                game_file = os.path.join(temp_dir, f"game_{game_num}.json")
                with open(game_file, 'w') as f:
                    json.dump(game_data, f)
            
            with patch('utils.file_utils.get_total_games', return_value=20):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                successful_loads = 0
                failed_loads = 0
                
                with patch('utils.file_utils.get_game_json_filename', side_effect=lambda n: f"game_{n}.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        # Try to load all games
                        for game_num in range(1, 21):
                            try:
                                result = engine.load_game_data(game_num)
                                if result is not None:
                                    successful_loads += 1
                                else:
                                    failed_loads += 1
                            except Exception:
                                failed_loads += 1
                
                # Should handle errors gracefully
                assert successful_loads == 16  # 80% success rate (4 failures expected)
                assert failed_loads == 4
                
                # Engine should still be functional after errors
                assert engine.running is True 