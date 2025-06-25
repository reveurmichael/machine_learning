"""Tests for replay workflow interactions.

This module tests the complete replay system component interaction chains,
ensuring that replay functionality works seamlessly across all components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import json
import tempfile
import os
from typing import Dict, List, Any, Optional

from replay.replay_engine import ReplayEngine
from replay.replay_utils import ReplayUtils
from core.game_data import GameData
from core.game_controller import GameController
from gui.replay_gui import ReplayGUI
from utils.file_utils import FileUtils
from utils.json_utils import JsonUtils


class TestCompleteReplayWorkflow:
    """Test suite for complete replay workflow with all components."""

    def setup_test_data(self, temp_dir: str) -> Dict[str, Any]:
        """Set up test data for replay testing."""
        game_data = {
            "game_number": 1,
            "score": 150,
            "steps": 45,
            "final_length": 8,
            "game_over_reason": "apple_eaten",
            "rounds": {
                1: {
                    "round": 1,
                    "apple_position": [5, 6],
                    "planned_moves": ["UP", "RIGHT", "DOWN"],
                    "moves": ["UP", "RIGHT", "DOWN"]
                },
                2: {
                    "round": 2,
                    "apple_position": [3, 4],
                    "planned_moves": ["LEFT", "UP"],
                    "moves": ["LEFT", "UP"]
                },
                3: {
                    "round": 3,
                    "apple_position": [7, 8],
                    "planned_moves": ["RIGHT", "DOWN", "RIGHT"],
                    "moves": ["RIGHT", "DOWN"]
                }
            },
            "statistics": {
                "start_time": "2024-01-01 10:00:00",
                "end_time": "2024-01-01 10:05:00",
                "total_duration_seconds": 300,
                "llm_communication_time": 45.5
            }
        }
        
        session_data = {
            "session_config": {
                "provider": "test_provider",
                "model": "test_model",
                "max_games": 2,
                "grid_size": 10
            },
            "games": [game_data],
            "summary": {
                "total_games": 1,
                "total_score": 150,
                "average_score": 150.0,
                "total_steps": 45
            }
        }
        
        # Save test data files
        game_file = os.path.join(temp_dir, "game_1.json")
        session_file = os.path.join(temp_dir, "summary.json")
        
        with open(game_file, 'w') as f:
            json.dump(game_data, f)
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f)
        
        return {
            "game_data": game_data,
            "session_data": session_data,
            "game_file": game_file,
            "session_file": session_file,
            "temp_dir": temp_dir
        }

    def test_complete_replay_initialization_workflow(self) -> None:
        """Test complete replay initialization workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = self.setup_test_data(temp_dir)
            
            # Initialize replay components
            replay_engine = ReplayEngine()
            replay_utils = ReplayUtils()
            
            # Mock file operations
            with patch.object(FileUtils, 'load_json_file') as mock_load:
                mock_load.return_value = test_data["game_data"]
                
                # Test initialization workflow
                init_result = replay_engine.initialize(test_data["game_file"])
                
                # Verify initialization
                assert init_result["success"] is True
                assert replay_engine.game_data is not None
                mock_load.assert_called_once_with(test_data["game_file"])

    def test_replay_data_loading_workflow(self) -> None:
        """Test replay data loading workflow with all components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = self.setup_test_data(temp_dir)
            
            replay_engine = ReplayEngine()
            
            # Mock components
            with patch.object(JsonUtils, 'load_game_data') as mock_load_game:
                with patch.object(FileUtils, 'validate_file_exists') as mock_validate:
                    
                    mock_validate.return_value = True
                    mock_load_game.return_value = test_data["game_data"]
                    
                    # Load data workflow
                    load_result = replay_engine.load_game_data(test_data["game_file"])
                    
                    # Verify data loading workflow
                    assert load_result["success"] is True
                    assert load_result["game_data"] == test_data["game_data"]
                    mock_validate.assert_called_once_with(test_data["game_file"])
                    mock_load_game.assert_called_once_with(test_data["game_file"])

    @patch('gui.replay_gui.ReplayGUI')
    def test_replay_gui_integration_workflow(self, mock_gui_class: Mock) -> None:
        """Test replay GUI integration workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = self.setup_test_data(temp_dir)
            
            # Mock GUI
            mock_gui = Mock(spec=ReplayGUI)
            mock_gui_class.return_value = mock_gui
            
            replay_engine = ReplayEngine()
            replay_engine.game_data = test_data["game_data"]
            
            # Test GUI integration
            replay_engine.set_gui(mock_gui)
            
            # Simulate replay with GUI
            for round_num in [1, 2, 3]:
                round_data = test_data["game_data"]["rounds"][round_num]
                
                # Test round visualization
                replay_engine.visualize_round(round_num)
                
                # Verify GUI calls
                expected_calls = [
                    call.update_board(round_data["apple_position"]),
                    call.display_moves(round_data["moves"]),
                    call.show_round_info(round_num)
                ]
                
                # Check that GUI methods were called appropriately
                assert mock_gui.update_board.called
                assert mock_gui.display_moves.called

    def test_step_by_step_replay_workflow(self) -> None:
        """Test step-by-step replay workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = self.setup_test_data(temp_dir)
            
            replay_engine = ReplayEngine()
            replay_engine.game_data = test_data["game_data"]
            
            # Initialize step-by-step replay
            replay_state = replay_engine.initialize_step_replay()
            
            assert replay_state["current_round"] == 1
            assert replay_state["current_move"] == 0
            assert replay_state["total_rounds"] == 3
            
            # Step through each move
            all_moves = []
            for round_num in [1, 2, 3]:
                round_moves = test_data["game_data"]["rounds"][round_num]["moves"]
                all_moves.extend(round_moves)
            
            step_count = 0
            while replay_engine.has_next_step():
                step_result = replay_engine.next_step()
                
                assert step_result["success"] is True
                assert step_result["move"] == all_moves[step_count]
                assert step_result["step_number"] == step_count + 1
                
                step_count += 1
            
            # Verify all moves were replayed
            assert step_count == len(all_moves)

    def test_replay_speed_control_workflow(self) -> None:
        """Test replay speed control workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = self.setup_test_data(temp_dir)
            
            replay_engine = ReplayEngine()
            replay_engine.game_data = test_data["game_data"]
            
            # Test different speed settings
            speeds = [0.5, 1.0, 2.0, 4.0]
            
            for speed in speeds:
                replay_engine.set_replay_speed(speed)
                
                # Verify speed setting
                assert replay_engine.replay_speed == speed
                
                # Test that timing calculations use the speed
                base_delay = 1.0
                expected_delay = base_delay / speed
                actual_delay = replay_engine.calculate_move_delay(base_delay)
                
                assert abs(actual_delay - expected_delay) < 0.001

    def test_replay_navigation_workflow(self) -> None:
        """Test replay navigation workflow (play, pause, reset, seek)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = self.setup_test_data(temp_dir)
            
            replay_engine = ReplayEngine()
            replay_engine.game_data = test_data["game_data"]
            
            # Initialize replay
            replay_engine.initialize_step_replay()
            
            # Test play functionality
            play_result = replay_engine.play()
            assert play_result["success"] is True
            assert replay_engine.is_playing is True
            
            # Test pause functionality
            pause_result = replay_engine.pause()
            assert pause_result["success"] is True
            assert replay_engine.is_playing is False
            
            # Test seeking to specific position
            seek_position = 3
            seek_result = replay_engine.seek_to_step(seek_position)
            assert seek_result["success"] is True
            assert replay_engine.current_step == seek_position
            
            # Test reset functionality
            reset_result = replay_engine.reset()
            assert reset_result["success"] is True
            assert replay_engine.current_step == 0
            assert replay_engine.current_round == 1

    def test_replay_error_handling_workflow(self) -> None:
        """Test replay error handling workflow."""
        replay_engine = ReplayEngine()
        
        # Test error handling with no data loaded
        play_result = replay_engine.play()
        assert play_result["success"] is False
        assert "no_data" in play_result["error"]
        
        # Test error handling with invalid seek position
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = self.setup_test_data(temp_dir)
            replay_engine.game_data = test_data["game_data"]
            replay_engine.initialize_step_replay()
            
            # Try to seek beyond available steps
            invalid_seek = 999
            seek_result = replay_engine.seek_to_step(invalid_seek)
            assert seek_result["success"] is False
            assert "invalid_position" in seek_result["error"]

    def test_replay_statistics_workflow(self) -> None:
        """Test replay statistics calculation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = self.setup_test_data(temp_dir)
            
            replay_utils = ReplayUtils()
            
            # Calculate replay statistics
            stats_result = replay_utils.calculate_replay_statistics(test_data["game_data"])
            
            # Verify statistics calculation
            assert stats_result["success"] is True
            stats = stats_result["statistics"]
            
            assert stats["total_rounds"] == 3
            assert stats["total_moves"] == 7  # 3 + 2 + 2
            assert stats["average_moves_per_round"] == 7 / 3
            assert stats["score"] == 150
            assert stats["final_length"] == 8

    def test_replay_export_workflow(self) -> None:
        """Test replay export workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = self.setup_test_data(temp_dir)
            
            replay_utils = ReplayUtils()
            
            # Export replay data
            export_file = os.path.join(temp_dir, "exported_replay.json")
            
            export_result = replay_utils.export_replay_data(
                test_data["game_data"],
                export_file,
                format="detailed"
            )
            
            # Verify export
            assert export_result["success"] is True
            assert os.path.exists(export_file)
            
            # Verify exported content
            with open(export_file, 'r') as f:
                exported_data = json.load(f)
            
            assert exported_data["game_number"] == 1
            assert exported_data["score"] == 150
            assert len(exported_data["rounds"]) == 3

    @patch('time.sleep')
    def test_automated_replay_workflow(self, mock_sleep: Mock) -> None:
        """Test automated replay workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = self.setup_test_data(temp_dir)
            
            replay_engine = ReplayEngine()
            replay_engine.game_data = test_data["game_data"]
            
            # Mock GUI for automated replay
            mock_gui = Mock(spec=ReplayGUI)
            replay_engine.set_gui(mock_gui)
            
            # Run automated replay
            automation_result = replay_engine.run_automated_replay(
                speed=2.0,
                show_statistics=True,
                pause_between_rounds=True
            )
            
            # Verify automated replay completed
            assert automation_result["success"] is True
            assert automation_result["total_moves"] == 7
            assert automation_result["total_rounds"] == 3
            
            # Verify GUI was updated for each move
            assert mock_gui.update_board.call_count >= 7
            assert mock_gui.display_moves.call_count >= 3  # Once per round


class TestReplayComponentInteractions:
    """Test suite for replay component interactions."""

    def test_replay_engine_utils_interaction(self) -> None:
        """Test interaction between ReplayEngine and ReplayUtils."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_data = {
                "game_number": 1,
                "score": 100,
                "rounds": {
                    1: {"moves": ["UP", "RIGHT"]},
                    2: {"moves": ["DOWN", "LEFT"]}
                }
            }
            
            replay_engine = ReplayEngine()
            replay_utils = ReplayUtils()
            
            # Test data validation interaction
            validation_result = replay_utils.validate_replay_data(test_data)
            assert validation_result["valid"] is True
            
            # Load data into engine
            replay_engine.load_data_from_dict(test_data)
            
            # Test statistics calculation interaction
            stats = replay_utils.calculate_statistics_for_engine(replay_engine)
            assert stats["total_rounds"] == 2
            assert stats["total_moves"] == 4

    def test_replay_file_system_interaction(self) -> None:
        """Test replay system interaction with file system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock file operations
            with patch.object(FileUtils, 'list_game_files') as mock_list:
                with patch.object(JsonUtils, 'load_game_data') as mock_load:
                    
                    game_files = ["game_1.json", "game_2.json", "game_3.json"]
                    mock_list.return_value = game_files
                    
                    mock_load.side_effect = [
                        {"game_number": 1, "score": 50},
                        {"game_number": 2, "score": 75},
                        {"game_number": 3, "score": 100}
                    ]
                    
                    replay_utils = ReplayUtils()
                    
                    # Test batch loading interaction
                    batch_result = replay_utils.load_session_replays(temp_dir)
                    
                    assert batch_result["success"] is True
                    assert len(batch_result["games"]) == 3
                    assert batch_result["total_score"] == 225

    def test_replay_gui_controller_interaction(self) -> None:
        """Test interaction between replay GUI and game controller."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock components
            mock_gui = Mock(spec=ReplayGUI)
            mock_controller = Mock(spec=GameController)
            
            replay_engine = ReplayEngine()
            replay_engine.set_gui(mock_gui)
            replay_engine.set_controller(mock_controller)
            
            # Test state synchronization
            game_state = {
                "snake_positions": [[5, 5], [4, 5], [3, 5]],
                "apple_position": [7, 8],
                "score": 30
            }
            
            # Simulate state update
            replay_engine.update_replay_state(game_state)
            
            # Verify interactions
            mock_gui.update_snake_positions.assert_called_with([[5, 5], [4, 5], [3, 5]])
            mock_gui.update_apple_position.assert_called_with([7, 8])
            mock_gui.update_score.assert_called_with(30)


class TestReplayPerformanceWorkflow:
    """Test suite for replay performance and optimization."""

    def test_large_game_replay_performance(self) -> None:
        """Test replay performance with large game data."""
        # Create large test data
        large_game_data = {
            "game_number": 1,
            "score": 1000,
            "rounds": {}
        }
        
        # Simulate 100 rounds with many moves each
        for round_num in range(1, 101):
            moves = ["UP", "DOWN", "LEFT", "RIGHT"] * 25  # 100 moves per round
            large_game_data["rounds"][round_num] = {
                "round": round_num,
                "moves": moves,
                "apple_position": [round_num % 10, (round_num * 2) % 10]
            }
        
        replay_engine = ReplayEngine()
        replay_engine.game_data = large_game_data
        
        # Test loading performance
        import time
        start_time = time.time()
        
        replay_engine.initialize_step_replay()
        
        load_time = time.time() - start_time
        
        # Should load large data quickly (under 1 second)
        assert load_time < 1.0
        
        # Test seeking performance
        start_time = time.time()
        
        replay_engine.seek_to_step(5000)  # Seek to middle
        
        seek_time = time.time() - start_time
        
        # Seeking should be fast
        assert seek_time < 0.1

    def test_memory_efficient_replay(self) -> None:
        """Test memory-efficient replay workflow."""
        # Create test data
        game_data = {
            "game_number": 1,
            "rounds": {i: {"moves": ["UP"] * 100} for i in range(1, 51)}
        }
        
        replay_engine = ReplayEngine()
        replay_engine.enable_memory_optimization(True)
        replay_engine.game_data = game_data
        
        # Test that memory optimization is working
        replay_engine.initialize_step_replay()
        
        # Should not load all moves into memory at once
        assert replay_engine.memory_optimized is True
        assert hasattr(replay_engine, 'current_round_cache')
        
        # Test streaming replay
        move_count = 0
        while replay_engine.has_next_step() and move_count < 10:
            step_result = replay_engine.next_step()
            assert step_result["success"] is True
            move_count += 1


class TestReplayIntegrationScenarios:
    """Test suite for various replay integration scenarios."""

    def test_web_replay_integration(self) -> None:
        """Test replay integration with web interface."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock web interface components
            with patch('flask.Flask') as mock_flask:
                with patch('utils.web_utils.WebUtils') as mock_web_utils:
                    
                    mock_app = Mock()
                    mock_flask.return_value = mock_app
                    
                    # Create replay web handler
                    from replay.replay_utils import ReplayUtils
                    replay_utils = ReplayUtils()
                    
                    # Test web API integration
                    api_result = replay_utils.create_web_replay_api(
                        session_dir=temp_dir,
                        host="localhost",
                        port=5000
                    )
                    
                    assert api_result["success"] is True
                    assert api_result["endpoints"] is not None

    def test_cli_replay_integration(self) -> None:
        """Test replay integration with command line interface."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test session data
            test_data = {
                "game_number": 1,
                "score": 50,
                "rounds": {1: {"moves": ["UP", "DOWN"]}}
            }
            
            game_file = os.path.join(temp_dir, "game_1.json")
            with open(game_file, 'w') as f:
                json.dump(test_data, f)
            
            # Mock CLI components
            with patch('argparse.ArgumentParser') as mock_parser:
                mock_args = Mock()
                mock_args.session_dir = temp_dir
                mock_args.game_number = 1
                mock_args.speed = 1.0
                mock_args.auto_play = True
                
                mock_parser.return_value.parse_args.return_value = mock_args
                
                # Test CLI replay integration
                from replay.replay_utils import ReplayUtils
                replay_utils = ReplayUtils()
                
                cli_result = replay_utils.run_cli_replay(mock_args)
                
                assert cli_result["success"] is True
                assert cli_result["game_replayed"] == 1

    def test_replay_data_validation_workflow(self) -> None:
        """Test comprehensive replay data validation workflow."""
        replay_utils = ReplayUtils()
        
        # Test valid data
        valid_data = {
            "game_number": 1,
            "score": 100,
            "steps": 25,
            "rounds": {
                1: {
                    "round": 1,
                    "moves": ["UP", "DOWN"],
                    "apple_position": [5, 6]
                }
            }
        }
        
        validation_result = replay_utils.validate_replay_data(valid_data)
        assert validation_result["valid"] is True
        assert validation_result["errors"] == []
        
        # Test invalid data
        invalid_data = {
            "game_number": "not_a_number",
            "score": -10,  # Invalid negative score
            "rounds": {
                1: {
                    "moves": ["INVALID_MOVE"],  # Invalid move
                    "apple_position": [15, 20]  # Out of bounds
                }
            }
        }
        
        validation_result = replay_utils.validate_replay_data(invalid_data)
        assert validation_result["valid"] is False
        assert len(validation_result["errors"]) > 0 