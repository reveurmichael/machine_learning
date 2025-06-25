"""Integration tests for the replay module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
import time
import numpy as np
from typing import Dict, List, Any, Optional

from replay.replay_engine import ReplayEngine
from replay.replay_utils import load_game_json, parse_game_data
from replay import ReplayEngine as ImportedReplayEngine, load_game_json as ImportedLoadGameJson


class TestReplayModuleIntegration:
    """Test integration of the entire replay module."""

    def create_comprehensive_test_session(self, temp_dir: str) -> None:
        """Create a comprehensive test session with multiple games."""
        games_data = [
            {
                "game_number": 1,
                "score": 100,
                "metadata": {
                    "timestamp": "2024-01-01 10:00:00",
                    "round_count": 3
                },
                "detailed_history": {
                    "apple_positions": [
                        {"x": 5, "y": 6},
                        {"x": 7, "y": 8},
                        {"x": 3, "y": 4}
                    ],
                    "moves": ["UP", "RIGHT", "DOWN", "LEFT", "UP"],
                    "rounds_data": {
                        "round_1": {
                            "moves": ["PLANNED", "UP", "RIGHT"],
                            "llm_response": "First move response"
                        },
                        "round_2": {
                            "moves": ["PLANNED", "DOWN"],
                            "llm_response": "Second move response"
                        }
                    }
                },
                "llm_info": {
                    "primary_provider": "deepseek",
                    "primary_model": "deepseek-reasoner",
                    "parser_provider": "mistral",
                    "parser_model": "mistral-7b"
                },
                "game_end_reason": "apple_eaten"
            },
            {
                "game_number": 2,
                "score": 250,
                "metadata": {
                    "timestamp": "2024-01-01 10:05:00",
                    "round_count": 5
                },
                "detailed_history": {
                    "apple_positions": [
                        {"x": 2, "y": 3},
                        {"x": 8, "y": 9},
                        {"x": 1, "y": 1},
                        {"x": 6, "y": 7}
                    ],
                    "moves": ["DOWN", "LEFT", "UP", "RIGHT", "DOWN", "UP"],
                    "rounds_data": {
                        "round_1": {
                            "moves": ["PLANNED", "DOWN", "LEFT", "UP"],
                            "llm_response": "Complex strategy response"
                        }
                    }
                },
                "llm_info": {
                    "primary_provider": "ollama",
                    "primary_model": "llama3.1-8b"
                },
                "game_end_reason": "wall_collision"
            },
            {
                "game_number": 3,
                "score": 50,
                "metadata": {
                    "timestamp": "2024-01-01 10:10:00",
                    "round_count": 1
                },
                "detailed_history": {
                    "apple_positions": [
                        {"x": 9, "y": 9}
                    ],
                    "moves": ["UP", "UP", "UP"],
                    "rounds_data": {}
                },
                "llm_info": {
                    "primary_provider": "hunyuan",
                    "primary_model": "hunyuan-lite"
                },
                "game_end_reason": "max_steps"
            }
        ]
        
        # Create game files
        for i, game_data in enumerate(games_data, 1):
            game_file = os.path.join(temp_dir, f"game_{i}.json")
            with open(game_file, 'w') as f:
                json.dump(game_data, f)
        
        # Create summary file
        summary_data = {
            "total_games": 3,
            "session_timestamp": "2024-01-01 10:00:00",
            "configuration": {
                "primary_llm": "deepseek",
                "max_games": 3
            }
        }
        summary_file = os.path.join(temp_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f)

    def test_module_imports(self) -> None:
        """Test that module imports work correctly."""
        # Test direct imports
        assert ReplayEngine is not None
        assert load_game_json is not None
        assert parse_game_data is not None
        
        # Test package imports
        assert ImportedReplayEngine is not None
        assert ImportedLoadGameJson is not None
        
        # Verify they're the same classes/functions
        assert ReplayEngine == ImportedReplayEngine
        assert load_game_json == ImportedLoadGameJson

    def test_complete_replay_session_workflow(self) -> None:
        """Test complete replay session workflow with multiple games."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_comprehensive_test_session(temp_dir)
            
            with patch('utils.file_utils.get_total_games', return_value=3):
                # Initialize replay engine
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False, auto_advance=True)
                
                assert engine.total_games == 3
                assert engine.game_number == 1
                
                # Load first game
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        result = engine.load_game_data(1)
                        
                        assert result is not None
                        assert engine.apple_positions == [[5, 6], [7, 8], [3, 4]]
                        assert engine.moves == ["UP", "RIGHT", "DOWN", "LEFT", "UP"]
                        assert engine.planned_moves == ["UP", "RIGHT"]
                        assert engine.primary_llm == "deepseek/deepseek-reasoner"
                        assert engine.secondary_llm == "mistral/mistral-7b"

    def test_replay_state_persistence_across_games(self) -> None:
        """Test state persistence and reset when switching between games."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_comprehensive_test_session(temp_dir)
            
            with patch('utils.file_utils.get_total_games', return_value=3):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                # Load game 1
                with patch('utils.file_utils.get_game_json_filename', side_effect=lambda n: f"game_{n}.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        # Load first game
                        engine.load_game_data(1)
                        
                        # Simulate some gameplay
                        engine.move_index = 2
                        engine.apple_index = 1
                        engine.moves_made = ["UP", "RIGHT"]
                        
                        # Load second game
                        engine.load_game_data(2)
                        
                        # Verify state was reset
                        assert engine.move_index == 0
                        assert engine.apple_index == 0
                        assert engine.moves_made == []
                        assert engine.moves == ["DOWN", "LEFT", "UP", "RIGHT", "DOWN", "UP"]
                        assert engine.primary_llm == "ollama/llama3.1-8b"
                        assert engine.secondary_llm == "None/None"

    def test_replay_engine_with_gui_integration(self) -> None:
        """Test replay engine integration with GUI components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_comprehensive_test_session(temp_dir)
            
            with patch('utils.file_utils.get_total_games', return_value=3):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=True)
                
                # Mock GUI
                mock_gui = Mock()
                mock_gui.move_history = ["OLD", "MOVES"]
                mock_gui.set_paused = Mock()
                
                engine.set_gui(mock_gui)
                
                # Load game data
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        engine.load_game_data(1)
                        
                        # Verify GUI integration
                        assert mock_gui.move_history == []  # Should be reset
                        
                        # Test drawing
                        engine.draw()
                        mock_gui.draw.assert_called_once()
                        
                        # Verify replay data structure
                        call_args = mock_gui.draw.call_args[0][0]
                        assert 'snake_positions' in call_args
                        assert 'apple_position' in call_args
                        assert 'game_number' in call_args
                        assert 'primary_llm' in call_args

    def test_replay_move_execution_integration(self) -> None:
        """Test complete move execution integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_comprehensive_test_session(temp_dir)
            
            with patch('utils.file_utils.get_total_games', return_value=3):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                # Load game data
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        engine.load_game_data(1)
                        
                        # Mock move execution
                        with patch.object(engine, 'move', return_value=True) as mock_move:
                            # Execute first move
                            result = engine.execute_replay_move("UP")
                            
                            assert result is True
                            assert "UP" in engine.moves_made
                            mock_move.assert_called_once_with("UP")
                            
                            # Simulate apple eaten
                            engine.apple_eaten = True
                            initial_apple_index = engine.apple_index
                            
                            result = engine.execute_replay_move("RIGHT")
                            
                            assert result is True
                            assert engine.apple_index == initial_apple_index + 1

    def test_replay_timing_and_control_integration(self) -> None:
        """Test replay timing and control integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_comprehensive_test_session(temp_dir)
            
            with patch('utils.file_utils.get_total_games', return_value=3):
                engine = ReplayEngine(log_dir=temp_dir, move_pause=0.1, use_gui=False)
                
                # Load game data
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        engine.load_game_data(1)
                        
                        # Test pause/unpause
                        assert engine.paused is False
                        
                        # Simulate pause event
                        engine.paused = True
                        
                        # Update should not execute moves when paused
                        with patch.object(engine, 'execute_replay_move') as mock_execute:
                            engine.update()
                            mock_execute.assert_not_called()
                        
                        # Unpause and test timing
                        engine.paused = False
                        engine.last_move_time = time.time() - 1.0  # Force move execution
                        
                        with patch.object(engine, 'execute_replay_move', return_value=True) as mock_execute:
                            engine.update()
                            mock_execute.assert_called_once()

    def test_replay_error_handling_integration(self) -> None:
        """Test error handling integration across replay components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid game data
            invalid_game_data = {
                "game_number": 1,
                "detailed_history": {
                    "apple_positions": [],  # Invalid: empty
                    "moves": ["UP"]
                }
            }
            
            game_file = os.path.join(temp_dir, "game_1.json")
            with open(game_file, 'w') as f:
                json.dump(invalid_game_data, f)
            
            with patch('utils.file_utils.get_total_games', return_value=1):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                # Load should fail gracefully
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        result = engine.load_game_data(1)
                        
                        assert result is None  # Should handle invalid data gracefully

    def test_replay_multi_game_navigation(self) -> None:
        """Test navigation between multiple games in replay."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_comprehensive_test_session(temp_dir)
            
            with patch('utils.file_utils.get_total_games', return_value=3):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                # Start with game 1
                with patch('utils.file_utils.get_game_json_filename', side_effect=lambda n: f"game_{n}.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        engine.load_game_data(1)
                        assert engine.primary_llm == "deepseek/deepseek-reasoner"
                        
                        # Navigate to next game
                        engine.load_next_game()
                        assert engine.game_number == 2
                        
                        # Load game 2 data
                        engine.load_game_data(2)
                        assert engine.primary_llm == "ollama/llama3.1-8b"
                        
                        # Navigate to game 3
                        engine.load_next_game()
                        assert engine.game_number == 3
                        
                        engine.load_game_data(3)
                        assert engine.primary_llm == "hunyuan/hunyuan-lite"

    def test_replay_data_consistency_validation(self) -> None:
        """Test data consistency validation throughout replay workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_comprehensive_test_session(temp_dir)
            
            with patch('utils.file_utils.get_total_games', return_value=3):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                # Load game and verify data consistency
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        engine.load_game_data(1)
                        
                        # Verify data consistency
                        state = engine._build_state_base()
                        
                        assert state['total_moves'] == len(engine.moves)
                        assert state['game_number'] == engine.game_number
                        assert state['move_index'] == engine.move_index
                        assert state['primary_llm'] == engine.primary_llm
                        assert state['secondary_llm'] == engine.secondary_llm
                        
                        # Verify apple positions and moves are consistent
                        assert len(engine.apple_positions) >= 1
                        assert len(engine.moves) >= 1
                        assert engine.apple_index == 0
                        assert engine.move_index == 0

    def test_replay_performance_with_large_dataset(self) -> None:
        """Test replay performance with larger datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create game with many moves
            large_game_data = {
                "game_number": 1,
                "score": 1000,
                "metadata": {
                    "timestamp": "2024-01-01 10:00:00",
                    "round_count": 50
                },
                "detailed_history": {
                    "apple_positions": [{"x": i % 10, "y": (i + 1) % 10} for i in range(100)],
                    "moves": ["UP", "DOWN", "LEFT", "RIGHT"] * 250,  # 1000 moves
                    "rounds_data": {
                        f"round_{i}": {
                            "moves": ["PLANNED"] + ["UP", "DOWN"] * 5
                        } for i in range(1, 51)
                    }
                },
                "llm_info": {
                    "primary_provider": "test_provider",
                    "primary_model": "test_model"
                },
                "game_end_reason": "max_steps"
            }
            
            game_file = os.path.join(temp_dir, "game_1.json")
            with open(game_file, 'w') as f:
                json.dump(large_game_data, f)
            
            with patch('utils.file_utils.get_total_games', return_value=1):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                # Time the loading operation
                start_time = time.time()
                
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        result = engine.load_game_data(1)
                        
                        load_time = time.time() - start_time
                        
                        # Verify loading was successful and reasonably fast
                        assert result is not None
                        assert len(engine.moves) == 1000
                        assert len(engine.apple_positions) == 100
                        assert load_time < 5.0  # Should load within 5 seconds

    def test_replay_memory_management(self) -> None:
        """Test memory management during replay operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_comprehensive_test_session(temp_dir)
            
            with patch('utils.file_utils.get_total_games', return_value=3):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                # Load multiple games to test memory usage
                with patch('utils.file_utils.get_game_json_filename', side_effect=lambda n: f"game_{n}.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        for game_num in range(1, 4):
                            # Set game number explicitly since load_game_data doesn't update it
                            engine.game_number = game_num
                            engine.load_game_data(game_num)
                            
                            # Verify data is properly replaced, not accumulated
                            state = engine._build_state_base()
                            
                            # Should have data for current game only
                            assert state['game_number'] == game_num
                            assert engine.move_index == 0  # Reset for each game
                            assert engine.apple_index == 0  # Reset for each game

    def test_replay_concurrent_operations(self) -> None:
        """Test replay engine behavior with concurrent-like operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_comprehensive_test_session(temp_dir)
            
            with patch('utils.file_utils.get_total_games', return_value=3):
                engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
                
                # Load game data
                with patch('utils.file_utils.get_game_json_filename', return_value="game_1.json"):
                    with patch('utils.file_utils.join_log_path', side_effect=lambda d, f: os.path.join(d, f)):
                        engine.load_game_data(1)
                        
                        # Simulate rapid state changes
                        original_state = engine._build_state_base()
                        
                        # Pause/unpause rapidly
                        engine.paused = True
                        paused_state = engine._build_state_base()
                        
                        engine.paused = False
                        unpaused_state = engine._build_state_base()
                        
                        # Verify state consistency
                        assert paused_state['paused'] is True
                        assert unpaused_state['paused'] is False
                        
                        # Other state should remain consistent
                        for key in ['game_number', 'total_moves', 'primary_llm']:
                            assert original_state[key] == paused_state[key] == unpaused_state[key] 