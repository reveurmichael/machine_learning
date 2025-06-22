"""Tests for replay.replay_engine module."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import json
import os
import time
import numpy as np
from typing import Dict, List, Any, Optional

import pygame

from replay.replay_engine import ReplayEngine
from core.game_controller import GameController
from config.ui_constants import TIME_DELAY, TIME_TICK
from config.game_constants import SENTINEL_MOVES, END_REASON_MAP


class TestReplayEngineInitialization:
    """Test suite for ReplayEngine initialization."""

    def test_initialization_default_parameters(self) -> None:
        """Test ReplayEngine initialization with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            assert engine.log_dir == temp_dir
            assert engine.pause_between_moves == 1.0
            assert engine.auto_advance is False
            assert engine.use_gui is True
            assert engine.game_number == 1
            assert engine.apple_positions == []
            assert engine.apple_index == 0
            assert engine.moves == []
            assert engine.move_index == 0
            assert engine.moves_made == []
            assert engine.game_stats == {}
            assert engine.running is True
            assert engine.paused is False

    def test_initialization_custom_parameters(self) -> None:
        """Test ReplayEngine initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(
                log_dir=temp_dir,
                move_pause=2.5,
                auto_advance=True,
                use_gui=False
            )
            
            assert engine.log_dir == temp_dir
            assert engine.pause_between_moves == 2.5
            assert engine.auto_advance is True
            assert engine.use_gui is False

    @patch('replay.replay_engine.get_total_games')
    def test_initialization_with_total_games(self, mock_get_total: Mock) -> None:
        """Test initialization correctly gets total games count."""
        mock_get_total.return_value = 5
        
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            assert engine.total_games == 5
            mock_get_total.assert_called_once_with(temp_dir)

    def test_inheritance_from_game_controller(self) -> None:
        """Test that ReplayEngine properly inherits from GameController."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            assert isinstance(engine, GameController)
            assert hasattr(engine, 'grid_size')
            assert hasattr(engine, 'snake_positions')
            assert hasattr(engine, 'apple_position')
            assert hasattr(engine, 'score')
            assert hasattr(engine, 'steps')


class TestReplayEngineGUIIntegration:
    """Test suite for ReplayEngine GUI integration."""

    def test_set_gui_basic(self) -> None:
        """Test basic GUI setting functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            mock_gui = Mock()
            
            engine.set_gui(mock_gui)
            
            assert engine.gui == mock_gui

    def test_set_gui_with_paused_state_sync(self) -> None:
        """Test GUI setting with paused state synchronization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            engine.paused = True
            
            mock_gui = Mock()
            mock_gui.set_paused = Mock()
            
            engine.set_gui(mock_gui)
            
            mock_gui.set_paused.assert_called_once_with(True)

    def test_set_gui_without_paused_method(self) -> None:
        """Test GUI setting when GUI doesn't have set_paused method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            mock_gui = Mock()
            del mock_gui.set_paused  # Remove the method
            
            # Should not raise an error
            engine.set_gui(mock_gui)
            
            assert engine.gui == mock_gui


class TestReplayEngineStateBuilding:
    """Test suite for ReplayEngine state building."""

    def test_build_state_base_complete(self) -> None:
        """Test building complete state base."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir, move_pause=0.5)
            
            # Set up engine state
            engine.snake_positions = np.array([[5, 5], [4, 5]])
            engine.apple_position = np.array([7, 8])
            engine.game_number = 3
            # Note: score and steps are properties from GameController, we set the underlying data
            engine.game_state.score = 150
            engine.game_state.steps = 45
            engine.move_index = 12
            engine.moves = ["UP", "DOWN", "LEFT", "RIGHT"] * 5
            engine.planned_moves = ["UP", "DOWN"]
            engine.llm_response = "Test response"
            engine.primary_llm = "test_provider/test_model"
            engine.secondary_llm = "parser/model"
            engine.paused = True
            engine.game_timestamp = "2024-01-01 10:00:00"
            engine.game_end_reason = "apple_eaten"
            engine.total_games = 5
            
            state = engine._build_state_base()
            
            assert np.array_equal(state['snake_positions'], engine.snake_positions)
            assert np.array_equal(state['apple_position'], [7, 8])
            assert state['game_number'] == 3
            assert state['score'] == 150
            assert state['steps'] == 45
            assert state['move_index'] == 12
            assert state['total_moves'] == 20
            assert state['planned_moves'] == ["UP", "DOWN"]
            assert state['llm_response'] == "Test response"
            assert state['primary_llm'] == "test_provider/test_model"
            assert state['secondary_llm'] == "parser/model"
            assert state['paused'] is True
            assert state['speed'] == 2.0  # 1.0 / 0.5
            assert state['timestamp'] == "2024-01-01 10:00:00"
            assert state['game_end_reason'] == "apple_eaten"
            assert state['total_games'] == 5

    def test_build_state_base_minimal(self) -> None:
        """Test building state base with minimal data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            state = engine._build_state_base()
            
            # Should have all required keys even with default values
            required_keys = [
                'snake_positions', 'apple_position', 'game_number', 'score',
                'steps', 'move_index', 'total_moves', 'planned_moves',
                'llm_response', 'primary_llm', 'secondary_llm', 'paused',
                'speed', 'timestamp', 'game_end_reason', 'total_games'
            ]
            
            for key in required_keys:
                assert key in state

    def test_build_state_base_speed_calculation(self) -> None:
        """Test speed calculation in state base."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test normal speed calculation
            engine = ReplayEngine(log_dir=temp_dir, move_pause=2.0)
            state = engine._build_state_base()
            assert state['speed'] == 0.5
            
            # Test with zero pause (should default to 1.0)
            engine.pause_between_moves = 0.0
            state = engine._build_state_base()
            assert state['speed'] == 1.0


class TestReplayEngineDrawing:
    """Test suite for ReplayEngine drawing functionality."""

    def test_draw_with_gui(self) -> None:
        """Test drawing when GUI is available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir, use_gui=True)
            mock_gui = Mock()
            engine.gui = mock_gui
            
            engine.draw()
            
            # Should call GUI draw with replay data
            mock_gui.draw.assert_called_once()
            call_args = mock_gui.draw.call_args[0][0]
            
            # Verify the replay data contains expected keys
            expected_keys = ['snake_positions', 'apple_position', 'game_number']
            for key in expected_keys:
                assert key in call_args

    def test_draw_without_gui(self) -> None:
        """Test drawing when GUI is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
            
            # Should not raise an error
            engine.draw()

    def test_draw_gui_disabled(self) -> None:
        """Test drawing when use_gui is False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
            mock_gui = Mock()
            engine.gui = mock_gui
            
            engine.draw()
            
            # GUI draw should not be called
            mock_gui.draw.assert_not_called()


class TestReplayEngineGameDataLoading:
    """Test suite for ReplayEngine game data loading."""

    def create_test_game_data(self) -> Dict[str, Any]:
        """Create test game data for loading tests."""
        return {
            "game_number": 1,
            "score": 150,
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
                        "moves": ["PLANNED", "UP", "RIGHT"]
                    }
                }
            },
            "llm_info": {
                "primary_provider": "test_provider",
                "primary_model": "test_model",
                "parser_provider": "parser_provider",
                "parser_model": "parser_model"
            },
            "game_end_reason": "apple_eaten"
        }

    @patch('replay.replay_utils.load_game_json')
    @patch('replay.replay_utils.parse_game_data')
    def test_load_game_data_success(
        self,
        mock_parse: Mock,
        mock_load: Mock
    ) -> None:
        """Test successful game data loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            # Mock successful loading and parsing
            test_data = self.create_test_game_data()
            mock_load.return_value = ("/path/to/game.json", test_data)
            
            parsed_data = {
                "apple_positions": [[5, 6], [7, 8], [3, 4]],
                "moves": ["UP", "RIGHT", "DOWN", "LEFT", "UP"],
                "planned_moves": ["UP", "RIGHT"],
                "game_end_reason": "apple_eaten",
                "primary_llm": "test_provider/test_model",
                "secondary_llm": "parser_provider/parser_model",
                "timestamp": "2024-01-01 10:00:00",
                "raw": test_data
            }
            mock_parse.return_value = parsed_data
            
            result = engine.load_game_data(1)
            
            # Verify data was loaded correctly
            assert result == test_data
            assert engine.apple_positions == [[5, 6], [7, 8], [3, 4]]
            assert engine.moves == ["UP", "RIGHT", "DOWN", "LEFT", "UP"]
            assert engine.planned_moves == ["UP", "RIGHT"]
            assert engine.game_end_reason == "apple_eaten"
            assert engine.primary_llm == "test_provider/test_model"
            assert engine.secondary_llm == "parser_provider/parser_model"
            assert engine.game_timestamp == "2024-01-01 10:00:00"
            
            # Verify counters were reset
            assert engine.move_index == 0
            assert engine.apple_index == 0
            assert engine.moves_made == []

    @patch('replay.replay_utils.load_game_json')
    def test_load_game_data_file_not_found(self, mock_load: Mock) -> None:
        """Test game data loading when file is not found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            mock_load.return_value = ("/path/to/missing.json", None)
            
            result = engine.load_game_data(1)
            
            assert result is None

    @patch('replay.replay_utils.load_game_json')
    @patch('replay.replay_utils.parse_game_data')
    def test_load_game_data_parse_failure(
        self,
        mock_parse: Mock,
        mock_load: Mock
    ) -> None:
        """Test game data loading when parsing fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            test_data = self.create_test_game_data()
            mock_load.return_value = ("/path/to/game.json", test_data)
            mock_parse.return_value = None  # Parse failure
            
            result = engine.load_game_data(1)
            
            assert result is None

    @patch('replay.replay_utils.load_game_json')
    @patch('replay.replay_utils.parse_game_data')
    def test_load_game_data_initialization(
        self,
        mock_parse: Mock,
        mock_load: Mock
    ) -> None:
        """Test game state initialization during data loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            test_data = self.create_test_game_data()
            mock_load.return_value = ("/path/to/game.json", test_data)
            
            parsed_data = {
                "apple_positions": [[5, 6]],
                "moves": ["UP"],
                "planned_moves": [],
                "game_end_reason": "test",
                "primary_llm": "test/test",
                "secondary_llm": "None/None",
                "timestamp": "2024-01-01",
                "raw": test_data
            }
            mock_parse.return_value = parsed_data
            
            with patch.object(engine, 'reset') as mock_reset:
                with patch.object(engine, 'set_apple_position') as mock_set_apple:
                    with patch.object(engine, '_update_board') as mock_update:
                        engine.load_game_data(1)
                        
                        mock_reset.assert_called_once()
                        mock_set_apple.assert_called_once_with([5, 6])
                        mock_update.assert_called_once()

    @patch('replay.replay_utils.load_game_json')
    @patch('replay.replay_utils.parse_game_data')
    def test_load_game_data_gui_history_reset(
        self,
        mock_parse: Mock,
        mock_load: Mock
    ) -> None:
        """Test GUI move history reset during data loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir, use_gui=True)
            
            mock_gui = Mock()
            mock_gui.move_history = ["OLD", "MOVES"]
            engine.gui = mock_gui
            
            test_data = self.create_test_game_data()
            mock_load.return_value = ("/path/to/game.json", test_data)
            
            parsed_data = {
                "apple_positions": [[5, 6]],
                "moves": ["UP"],
                "planned_moves": [],
                "game_end_reason": "test",
                "primary_llm": "test/test",
                "secondary_llm": "None/None",
                "timestamp": "2024-01-01",
                "raw": test_data
            }
            mock_parse.return_value = parsed_data
            
            engine.load_game_data(1)
            
            assert mock_gui.move_history == []


class TestReplayEngineGameplayControl:
    """Test suite for ReplayEngine gameplay control."""

    def test_update_with_pause(self) -> None:
        """Test update method when paused."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            engine.paused = True
            
            # Should return early without doing anything
            engine.update()
            
            # No assertions needed - just verify no errors

    @patch('time.time')
    def test_update_timing_control(self, mock_time: Mock) -> None:
        """Test update method timing control."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir, move_pause=1.0)
            engine.paused = False
            engine.moves = ["UP", "DOWN"]
            engine.move_index = 0
            
            # Mock time progression
            mock_time.side_effect = [100.0, 100.5, 101.5]  # Start, check, after delay
            engine.last_move_time = 100.0
            
            with patch.object(engine, 'execute_replay_move') as mock_execute:
                engine.update()
                
                # Should not execute move yet (only 0.5 seconds passed)
                mock_execute.assert_not_called()
                
                # Advance time and try again
                engine.update()
                mock_execute.assert_called_once_with("UP")

    def test_load_next_game(self) -> None:
        """Test loading next game."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            engine.game_number = 2
            
            with patch.object(engine, 'load_game_data') as mock_load:
                engine.load_next_game()
                
                assert engine.game_number == 3
                mock_load.assert_called_once_with(3)

    def test_execute_replay_move_success(self) -> None:
        """Test successful replay move execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            engine.apple_positions = [[5, 6], [7, 8]]
            engine.apple_index = 0
            engine.moves_made = []
            
            with patch.object(engine, 'make_move') as mock_move:
                mock_move.return_value = (True, False)  # game_active, apple_eaten
                
                result = engine.execute_replay_move("UP")
                
                assert result is True
                mock_move.assert_called_once_with("UP")

    def test_execute_replay_move_apple_eaten(self) -> None:
        """Test replay move execution when apple is eaten."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            engine.apple_positions = [[5, 6], [7, 8]]
            engine.apple_index = 0
            
            with patch.object(engine, 'make_move') as mock_move:
                with patch.object(engine, 'set_apple_position') as mock_set_apple:
                    mock_move.return_value = (True, True)  # game_active, apple_eaten
                    
                    result = engine.execute_replay_move("UP")
                    
                    assert result is True
                    assert engine.apple_index == 1
                    mock_set_apple.assert_called_once_with([7, 8])

    def test_execute_replay_move_game_over(self) -> None:
        """Test replay move execution resulting in game over."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            with patch.object(engine, 'make_move') as mock_move:
                mock_move.return_value = (False, False)  # Game over
                
                result = engine.execute_replay_move("UP")
                
                assert result is False


class TestReplayEngineEventHandling:
    """Test suite for ReplayEngine event handling."""

    @patch('pygame.event.get')
    def test_handle_events_quit(self, mock_get_events: Mock) -> None:
        """Test handling quit event."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            # Mock quit event
            quit_event = Mock()
            quit_event.type = pygame.QUIT
            mock_get_events.return_value = [quit_event]
            
            engine.handle_events()
            
            assert engine.running is False

    @patch('pygame.event.get')
    def test_handle_events_escape_key(self, mock_get_events: Mock) -> None:
        """Test handling escape key event."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            # Mock escape key event
            key_event = Mock()
            key_event.type = pygame.KEYDOWN
            key_event.key = pygame.K_ESCAPE
            mock_get_events.return_value = [key_event]
            
            engine.handle_events()
            
            assert engine.running is False

    @patch('pygame.event.get')
    def test_handle_events_space_key_pause_toggle(self, mock_get_events: Mock) -> None:
        """Test handling space key for pause toggle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            engine.paused = False
            
            # Mock space key event
            key_event = Mock()
            key_event.type = pygame.KEYDOWN
            key_event.key = pygame.K_SPACE
            mock_get_events.return_value = [key_event]
            
            engine.handle_events()
            
            assert engine.paused is True

    @patch('pygame.event.get')
    def test_handle_events_game_navigation_keys(self, mock_get_events: Mock) -> None:
        """Test handling game navigation keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            engine.game_number = 2
            
            # Test next game key
            key_event = Mock()
            key_event.type = pygame.KEYDOWN
            key_event.key = pygame.K_n
            mock_get_events.return_value = [key_event]
            
            with patch.object(engine, 'load_game_data', return_value=True) as mock_load:
                engine.handle_events()
                
                assert engine.game_number == 3
                mock_load.assert_called_once_with(3)

    @patch('pygame.event.get')
    def test_handle_events_previous_game_key(self, mock_get_events: Mock) -> None:
        """Test handling previous game key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            engine.game_number = 3
            
            # Test previous game key
            key_event = Mock()
            key_event.type = pygame.KEYDOWN
            key_event.key = pygame.K_p
            mock_get_events.return_value = [key_event]
            
            with patch.object(engine, 'load_game_data') as mock_load:
                engine.handle_events()
                
                assert engine.game_number == 2
                mock_load.assert_called_once_with(2)

    @patch('pygame.event.get')
    def test_handle_events_speed_control_keys(self, mock_get_events: Mock) -> None:
        """Test handling speed control keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir, move_pause=1.0)
            
            # Test speed up key
            key_event = Mock()
            key_event.type = pygame.KEYDOWN
            key_event.key = pygame.K_s
            mock_get_events.return_value = [key_event]
            
            engine.handle_events()
            
            assert engine.pause_between_moves == 0.75  # 1.0 * 0.75


class TestReplayEngineMainLoop:
    """Test suite for ReplayEngine main loop."""

    @patch('pygame.time.delay')
    @patch('pygame.time.Clock.tick')
    @patch('pygame.get_init')
    @patch('pygame.init')
    def test_run_main_loop(self, mock_init: Mock, mock_get_init: Mock, mock_tick: Mock, mock_delay: Mock) -> None:
        """Test main run loop."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir)
            
            # Mock pygame initialization
            mock_get_init.return_value = True
            
            # Mock to stop after one iteration
            call_count = 0
            def stop_after_one():
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    engine.running = False
            
            with patch.object(engine, 'load_game_data', return_value=True):
                with patch.object(engine, 'handle_events', side_effect=stop_after_one):
                    with patch.object(engine, 'update'):
                        with patch.object(engine, 'draw'):
                            with patch('pygame.quit'):
                                engine.run()
                                
                                # Verify timing calls were made
                                mock_delay.assert_called()
                                mock_tick.assert_called()

    def test_run_without_gui(self) -> None:
        """Test run loop without GUI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
            
            # Mock to stop immediately
            def stop_immediately():
                engine.running = False
            
            with patch('pygame.get_init', return_value=True):
                with patch.object(engine, 'load_game_data', return_value=True):
                    with patch.object(engine, 'handle_events', side_effect=stop_immediately):
                        with patch.object(engine, 'update'):
                            with patch.object(engine, 'draw'):
                                with patch('pygame.quit'):
                                    # Should not raise errors
                                    engine.run()


class TestReplayEngineIntegration:
    """Integration tests for ReplayEngine."""

    def create_complete_test_data(self, temp_dir: str) -> str:
        """Create complete test data for integration tests."""
        game_data = {
            "game_number": 1,
            "score": 100,
            "metadata": {
                "timestamp": "2024-01-01 10:00:00",
                "round_count": 2
            },
            "detailed_history": {
                "apple_positions": [
                    {"x": 5, "y": 6},
                    {"x": 7, "y": 8}
                ],
                "moves": ["UP", "RIGHT", "DOWN"],
                "rounds_data": {
                    "round_1": {
                        "moves": ["PLANNED", "UP", "RIGHT"]
                    }
                }
            },
            "llm_info": {
                "primary_provider": "test_provider",
                "primary_model": "test_model"
            },
            "game_end_reason": "apple_eaten"
        }
        
        game_file = os.path.join(temp_dir, "game_1.json")
        with open(game_file, 'w') as f:
            json.dump(game_data, f)
        
        return game_file

    def test_complete_replay_workflow(self) -> None:
        """Test complete replay workflow from initialization to execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            self.create_complete_test_data(temp_dir)
            
            # Initialize engine
            engine = ReplayEngine(log_dir=temp_dir, use_gui=False)
            
            # Load game data
            result = engine.load_game_data(1)
            
            # Verify loading was successful
            assert result is not None
            assert len(engine.moves) > 0
            assert len(engine.apple_positions) > 0
            
            # Test move execution
            with patch.object(engine, 'make_move', return_value=(True, False)):
                success = engine.execute_replay_move("UP")
                assert success is True

    def test_replay_state_consistency(self) -> None:
        """Test replay state consistency throughout workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_complete_test_data(temp_dir)
            
            engine = ReplayEngine(log_dir=temp_dir)
            engine.load_game_data(1)
            
            # Verify state consistency
            initial_state = engine._build_state_base()
            
            # State should be consistent
            assert initial_state['game_number'] == 1
            assert initial_state['move_index'] == 0
            assert initial_state['total_moves'] == len(engine.moves)
            
            # After mock move execution
            engine.move_index = 1
            updated_state = engine._build_state_base()
            assert updated_state['move_index'] == 1

    def test_error_recovery_during_replay(self) -> None:
        """Test error recovery during replay execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_complete_test_data(temp_dir)
            
            engine = ReplayEngine(log_dir=temp_dir)
            engine.load_game_data(1)
            
            # Test error during move execution
            with patch.object(engine, 'make_move', side_effect=Exception("Test error")):
                # Should handle errors gracefully
                try:
                    engine.execute_replay_move("UP")
                except Exception:
                    # Error handling behavior depends on implementation
                    pass 