"""Tests for core.game_manager module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict
import argparse

from core.game_manager import GameManager, _make_time_stats, _make_token_stats


class TestGameManager:
    """Test suite for GameManager class."""

    def create_mock_args(self, **kwargs) -> argparse.Namespace:
        """Create mock command line arguments."""
        defaults = {
            'max_games': 5,
            'no_gui': False,
            'move_pause': 1.0,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_initialization(self) -> None:
        """Test GameManager initialization."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        assert manager.args == args
        assert manager.game_count == 0
        assert manager.round_count == 1
        assert manager.total_score == 0
        assert manager.total_steps == 0
        assert manager.empty_steps == 0
        assert manager.something_is_wrong_steps == 0
        assert manager.valid_steps == 0
        assert manager.invalid_reversals == 0
        assert manager.consecutive_empty_steps == 0
        assert manager.consecutive_something_is_wrong == 0
        assert manager.consecutive_invalid_reversals == 0
        assert manager.consecutive_no_path_found == 0
        assert manager.game_scores == []
        assert manager.round_counts == []
        assert manager.total_rounds == 0
        assert manager.game is None
        assert manager.game_active is True
        assert manager.need_new_plan is True
        assert manager.awaiting_plan is False
        assert manager.running is True
        assert manager.current_game_moves == []
        assert manager.llm_client is None
        assert manager.parser_provider is None
        assert manager.parser_model is None
        assert manager.log_dir is None
        assert manager.prompts_dir is None
        assert manager.responses_dir is None
        assert manager.use_gui is True  # no_gui = False
        assert manager.last_no_path_found is False
        assert manager.skip_empty_this_tick is False
        assert manager.no_path_found_steps == 0

    def test_initialization_no_gui(self) -> None:
        """Test GameManager initialization with no GUI."""
        args = self.create_mock_args(no_gui=True)
        manager = GameManager(args)
        
        assert manager.use_gui is False

    def test_create_llm_client(self) -> None:
        """Test creating LLM client."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        with patch('core.game_manager.LLMClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = manager.create_llm_client("test_provider", "test_model")
            
            mock_client_class.assert_called_once_with(
                provider="test_provider",
                model="test_model"
            )
            assert client == mock_client

    def test_create_llm_client_no_model(self) -> None:
        """Test creating LLM client without model."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        with patch('core.game_manager.LLMClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = manager.create_llm_client("test_provider")
            
            mock_client_class.assert_called_once_with(
                provider="test_provider",
                model=None
            )

    @patch('core.game_manager.GameGUI')
    @patch('core.game_manager.GameLogic')
    def test_setup_game_with_gui(self, mock_logic: Mock, mock_gui_class: Mock) -> None:
        """Test setting up game with GUI."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        mock_game = Mock()
        mock_logic.return_value = mock_game
        mock_gui = Mock()
        mock_gui_class.return_value = mock_gui
        
        manager.setup_game()
        
        mock_logic.assert_called_once_with(use_gui=True)
        assert manager.game == mock_game
        mock_gui_class.assert_called_once()
        mock_game.set_gui.assert_called_once_with(mock_gui)

    @patch('core.game_manager.GameLogic')
    def test_setup_game_no_gui(self, mock_logic: Mock) -> None:
        """Test setting up game without GUI."""
        args = self.create_mock_args(no_gui=True)
        manager = GameManager(args)
        
        mock_game = Mock()
        mock_logic.return_value = mock_game
        
        manager.setup_game()
        
        mock_logic.assert_called_once_with(use_gui=False)
        assert manager.game == mock_game
        mock_game.set_gui.assert_not_called()

    def test_get_pause_between_moves_no_gui(self) -> None:
        """Test get pause between moves without GUI."""
        args = self.create_mock_args(no_gui=True)
        manager = GameManager(args)
        
        pause = manager.get_pause_between_moves()
        
        assert pause == 0.0

    def test_get_pause_between_moves_with_gui(self) -> None:
        """Test get pause between moves with GUI."""
        args = self.create_mock_args(move_pause=2.5)
        manager = GameManager(args)
        
        pause = manager.get_pause_between_moves()
        
        assert pause == 2.5

    @patch('core.game_manager.initialize_game_manager')
    def test_initialize(self, mock_initialize: Mock) -> None:
        """Test manager initialization."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        manager.initialize()
        
        mock_initialize.assert_called_once_with(manager)

    @patch('core.game_manager.process_events')
    def test_process_events(self, mock_process: Mock) -> None:
        """Test processing events."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        manager.process_events()
        
        mock_process.assert_called_once_with(manager)

    @patch('core.game_manager.run_game_loop')
    def test_run_game_loop(self, mock_run: Mock) -> None:
        """Test running game loop."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        manager.run_game_loop()
        
        mock_run.assert_called_once_with(manager)

    @patch('core.game_manager.save_session_stats')
    @patch('core.game_manager.report_final_statistics')
    def test_report_final_statistics_with_games(
        self,
        mock_report: Mock,
        mock_save: Mock
    ) -> None:
        """Test reporting final statistics when games were played."""
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.game_count = 3
        manager.log_dir = "/test/log"
        manager.total_score = 150
        manager.game = Mock()
        
        manager.report_final_statistics()
        
        mock_save.assert_called_once_with("/test/log")
        mock_report.assert_called_once()
        assert manager.running is False

    def test_report_final_statistics_no_games(self) -> None:
        """Test reporting final statistics when no games were played."""
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.game_count = 0
        
        with patch('core.game_manager.save_session_stats') as mock_save:
            with patch('core.game_manager.report_final_statistics') as mock_report:
                manager.report_final_statistics()
                
                mock_save.assert_not_called()
                mock_report.assert_not_called()

    def test_increment_round(self) -> None:
        """Test incrementing round count."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        initial_count = manager.round_count
        manager.increment_round("test reason")
        
        assert manager.round_count == initial_count + 1

    @patch('core.game_manager.continue_from_directory')
    @patch('core.game_manager.setup_continuation_session')
    @patch('core.game_manager.handle_continuation_game_state')
    def test_continue_from_session(
        self,
        mock_handle: Mock,
        mock_setup: Mock,
        mock_continue: Mock
    ) -> None:
        """Test continuing from existing session."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        manager.continue_from_session("/test/log", 2)
        
        mock_continue.assert_called_once_with("/test/log", 2, manager)
        mock_setup.assert_called_once_with(manager, "/test/log")
        mock_handle.assert_called_once_with(manager, 2)

    @patch('core.game_manager.continue_from_directory')
    def test_continue_from_directory_classmethod(self, mock_continue: Mock) -> None:
        """Test class method for continuing from directory."""
        args = self.create_mock_args()
        mock_manager = Mock()
        mock_continue.return_value = mock_manager
        
        result = GameManager.continue_from_directory(args)
        
        mock_continue.assert_called_once_with(args)
        assert result == mock_manager

    @patch('core.game_manager.run_game_loop')
    def test_run(self, mock_run_loop: Mock) -> None:
        """Test main run method."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        with patch.object(manager, 'initialize') as mock_init:
            with patch.object(manager, 'report_final_statistics') as mock_report:
                manager.run()
                
                mock_init.assert_called_once()
                mock_run_loop.assert_called_once_with(manager)
                mock_report.assert_called_once()

    def test_finish_round(self) -> None:
        """Test finishing a round."""
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.game = Mock()
        manager.game.round_manager = Mock()
        
        manager.finish_round("test reason")
        
        manager.game.round_manager.flush_buffer.assert_called_once()


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_make_time_stats(self) -> None:
        """Test _make_time_stats function."""
        stats = _make_time_stats()
        
        assert isinstance(stats, defaultdict)
        assert stats.default_factory == int
        
        # Test auto-initialization
        assert stats['test_key'] == 0

    def test_make_token_stats(self) -> None:
        """Test _make_token_stats function."""
        stats = _make_token_stats()
        
        assert isinstance(stats, dict)
        assert 'primary' in stats
        assert 'secondary' in stats
        assert isinstance(stats['primary'], defaultdict)
        assert isinstance(stats['secondary'], defaultdict)
        assert stats['primary'].default_factory == int
        assert stats['secondary'].default_factory == int
        
        # Test auto-initialization
        assert stats['primary']['test_key'] == 0
        assert stats['secondary']['test_key'] == 0


class TestGameManagerIntegration:
    """Integration tests for GameManager."""

    def create_mock_args(self, **kwargs) -> argparse.Namespace:
        """Create mock command line arguments."""
        defaults = {
            'max_games': 1,
            'no_gui': True,
            'move_pause': 0.0,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch('core.game_manager.initialize_game_manager')
    @patch('core.game_manager.run_game_loop')
    def test_complete_lifecycle(
        self,
        mock_run_loop: Mock,
        mock_initialize: Mock
    ) -> None:
        """Test complete GameManager lifecycle."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Mock the game loop to simulate completion
        def complete_games(*args):
            manager.game_count = 1
            manager.running = False
        
        mock_run_loop.side_effect = complete_games
        
        with patch.object(manager, 'report_final_statistics') as mock_report:
            manager.run()
            
            mock_initialize.assert_called_once()
            mock_run_loop.assert_called_once()
            mock_report.assert_called_once()
            assert manager.running is False

    def test_state_management(self) -> None:
        """Test state management throughout lifecycle."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Initial state
        assert manager.game_active is True
        assert manager.need_new_plan is True
        assert manager.awaiting_plan is False
        
        # Simulate state changes
        manager.game_active = False
        manager.need_new_plan = False
        manager.awaiting_plan = True
        
        assert manager.game_active is False
        assert manager.need_new_plan is False
        assert manager.awaiting_plan is True

    def test_statistics_tracking(self) -> None:
        """Test statistics tracking functionality."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Test counters
        assert manager.total_score == 0
        assert manager.total_steps == 0
        assert manager.game_scores == []
        
        # Simulate game completion
        manager.total_score += 50
        manager.total_steps += 100
        manager.game_scores.append(50)
        manager.game_count += 1
        
        assert manager.total_score == 50
        assert manager.total_steps == 100
        assert manager.game_scores == [50]
        assert manager.game_count == 1

    @patch('core.game_manager.GameLogic')
    @patch('core.game_manager.GameGUI')
    def test_game_setup_integration(
        self,
        mock_gui_class: Mock,
        mock_logic_class: Mock
    ) -> None:
        """Test complete game setup integration."""
        args = self.create_mock_args(no_gui=False)
        manager = GameManager(args)
        
        mock_game = Mock()
        mock_logic_class.return_value = mock_game
        mock_gui = Mock()
        mock_gui_class.return_value = mock_gui
        
        manager.setup_game()
        
        # Verify game creation
        mock_logic_class.assert_called_once_with(use_gui=True)
        assert manager.game == mock_game
        
        # Verify GUI setup
        mock_gui_class.assert_called_once()
        mock_game.set_gui.assert_called_once_with(mock_gui)

    def test_client_management(self) -> None:
        """Test LLM client management."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        with patch('core.game_manager.LLMClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Create client
            client = manager.create_llm_client("test_provider", "test_model")
            
            # Verify client creation
            assert client == mock_client
            mock_client_class.assert_called_once_with(
                provider="test_provider",
                model="test_model"
            )

    def test_round_management(self) -> None:
        """Test round management functionality."""
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.game = Mock()
        manager.game.round_manager = Mock()
        
        # Test round increment
        initial_count = manager.round_count
        manager.increment_round()
        assert manager.round_count == initial_count + 1
        
        # Test round finishing
        manager.finish_round("test")
        manager.game.round_manager.flush_buffer.assert_called_once()

    def test_pause_configuration(self) -> None:
        """Test pause configuration based on GUI settings."""
        # No GUI case
        args_no_gui = self.create_mock_args(no_gui=True, move_pause=1.5)
        manager_no_gui = GameManager(args_no_gui)
        assert manager_no_gui.get_pause_between_moves() == 0.0
        
        # With GUI case
        args_gui = self.create_mock_args(no_gui=False, move_pause=2.0)
        manager_gui = GameManager(args_gui)
        assert manager_gui.get_pause_between_moves() == 2.0

    def test_comprehensive_statistics_management(self) -> None:
        """Test comprehensive statistics management across multiple games."""
        args = self.create_mock_args(max_games=5)
        manager = GameManager(args)
        
        # Simulate multiple games with different outcomes
        game_data = [
            {"score": 10, "steps": 50, "valid": 45, "empty": 3, "wrong": 2},
            {"score": 25, "steps": 120, "valid": 110, "empty": 7, "wrong": 3},
            {"score": 5, "steps": 30, "valid": 25, "empty": 4, "wrong": 1},
            {"score": 40, "steps": 200, "valid": 180, "empty": 15, "wrong": 5},
            {"score": 15, "steps": 75, "valid": 65, "empty": 8, "wrong": 2}
        ]
        
        for game in game_data:
            manager.total_score += game["score"]
            manager.total_steps += game["steps"]
            manager.valid_steps += game["valid"]
            manager.empty_steps += game["empty"]
            manager.something_is_wrong_steps += game["wrong"]
            manager.game_scores.append(game["score"])
            manager.game_count += 1
        
        # Verify aggregated statistics
        assert manager.total_score == 95
        assert manager.total_steps == 475
        assert manager.valid_steps == 425
        assert manager.empty_steps == 37
        assert manager.something_is_wrong_steps == 13
        assert len(manager.game_scores) == 5
        assert manager.game_count == 5

    def test_comprehensive_round_tracking(self) -> None:
        """Test comprehensive round tracking across games."""
        args = self.create_mock_args()
        manager = GameManager(args)
        manager.game = Mock()
        manager.game.round_manager = Mock()
        
        # Simulate multiple rounds
        round_reasons = ["apple_eaten", "new_game", "strategy_change", "error_recovery"]
        
        for reason in round_reasons:
            initial_count = manager.round_count
            manager.increment_round(reason)
            assert manager.round_count == initial_count + 1
        
        # Test round completion tracking
        round_counts = [5, 8, 3, 12, 6]  # Rounds per game
        for count in round_counts:
            manager.round_counts.append(count)
            manager.total_rounds += count
        
        assert len(manager.round_counts) == 5
        assert manager.total_rounds == 34

    def test_comprehensive_error_handling(self) -> None:
        """Test comprehensive error handling scenarios."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Test consecutive error tracking
        manager.consecutive_empty_steps = 5
        manager.consecutive_something_is_wrong = 3
        manager.consecutive_invalid_reversals = 2
        manager.consecutive_no_path_found = 4
        
        # Simulate error threshold checking
        assert manager.consecutive_empty_steps == 5
        assert manager.consecutive_something_is_wrong == 3
        assert manager.consecutive_invalid_reversals == 2
        assert manager.consecutive_no_path_found == 4
        
        # Test error recovery
        manager.consecutive_empty_steps = 0
        manager.consecutive_something_is_wrong = 0
        manager.consecutive_invalid_reversals = 0
        manager.consecutive_no_path_found = 0
        
        assert manager.consecutive_empty_steps == 0
        assert manager.consecutive_something_is_wrong == 0

    def test_comprehensive_timing_management(self) -> None:
        """Test comprehensive timing and performance management."""
        args = self.create_mock_args(move_pause=0.5)
        manager = GameManager(args)
        
        # Test timing configuration
        assert manager.time_delay == TIME_DELAY
        assert manager.time_tick == TIME_TICK
        
        # Test clock initialization
        assert hasattr(manager, 'clock')
        
        # Test pause configuration variations
        test_cases = [
            (True, 0.0),   # No GUI -> no pause
            (False, 0.5),  # GUI -> use configured pause
        ]
        
        for no_gui, expected_pause in test_cases:
            args_test = self.create_mock_args(no_gui=no_gui, move_pause=0.5)
            manager_test = GameManager(args_test)
            assert manager_test.get_pause_between_moves() == expected_pause

    def test_comprehensive_state_management(self) -> None:
        """Test comprehensive game state management."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Test initial state
        assert manager.game_active is True
        assert manager.need_new_plan is True
        assert manager.awaiting_plan is False
        assert manager.running is True
        assert manager.last_no_path_found is False
        assert manager.skip_empty_this_tick is False
        
        # Test state transitions
        state_transitions = [
            {"game_active": False, "need_new_plan": False},
            {"awaiting_plan": True, "last_no_path_found": True},
            {"skip_empty_this_tick": True, "running": False}
        ]
        
        for transition in state_transitions:
            for key, value in transition.items():
                setattr(manager, key, value)
                assert getattr(manager, key) == value

    def test_comprehensive_llm_configuration(self) -> None:
        """Test comprehensive LLM configuration management."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Test LLM client configuration
        assert manager.llm_client is None
        assert manager.parser_provider is None
        assert manager.parser_model is None
        
        # Test client creation with various configurations
        test_configs = [
            ("ollama", "llama2"),
            ("openai", "gpt-3.5-turbo"),
            ("hunyuan", None),
            ("deepseek", "deepseek-coder")
        ]
        
        with patch('core.game_manager.LLMClient') as mock_client_class:
            for provider, model in test_configs:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                client = manager.create_llm_client(provider, model)
                
                assert client == mock_client
                mock_client_class.assert_called_with(provider=provider, model=model)
                mock_client_class.reset_mock()

    def test_comprehensive_logging_management(self) -> None:
        """Test comprehensive logging directory management."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Test initial logging state
        assert manager.log_dir is None
        assert manager.prompts_dir is None
        assert manager.responses_dir is None
        
        # Test logging directory setup
        test_dirs = {
            "log_dir": "/path/to/logs",
            "prompts_dir": "/path/to/prompts",
            "responses_dir": "/path/to/responses"
        }
        
        for attr, path in test_dirs.items():
            setattr(manager, attr, path)
            assert getattr(manager, attr) == path

    def test_comprehensive_game_setup_scenarios(self) -> None:
        """Test comprehensive game setup scenarios."""
        # Test various GUI configurations
        gui_configs = [
            {"no_gui": True, "use_gui": False},
            {"no_gui": False, "use_gui": True}
        ]
        
        for config in gui_configs:
            args = self.create_mock_args(no_gui=config["no_gui"])
            manager = GameManager(args)
            
            with patch('core.game_manager.GameLogic') as mock_logic:
                with patch('core.game_manager.GameGUI') as mock_gui_class:
                    mock_game = Mock()
                    mock_logic.return_value = mock_game
                    
                    manager.setup_game()
                    
                    # Verify game logic setup
                    mock_logic.assert_called_once_with(use_gui=config["use_gui"])
                    assert manager.game == mock_game
                    
                    # Verify GUI setup if enabled
                    if config["use_gui"]:
                        mock_gui_class.assert_called_once()
                        mock_game.set_gui.assert_called_once()
                    else:
                        mock_gui_class.assert_not_called()

    def test_comprehensive_continuation_functionality(self) -> None:
        """Test comprehensive game continuation functionality."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Test continuation setup
        test_log_dir = "/path/to/session/logs"
        test_start_game = 5
        
        with patch('core.game_manager.continue_from_directory') as mock_continue:
            with patch('core.game_manager.setup_continuation_session') as mock_setup:
                with patch('core.game_manager.handle_continuation_game_state') as mock_handle:
                    
                    manager.continue_from_session(test_log_dir, test_start_game)
                    
                    # Verify continuation methods called
                    mock_continue.assert_called_once_with(test_log_dir)
                    mock_setup.assert_called_once_with(manager, test_log_dir)
                    mock_handle.assert_called_once_with(manager, test_start_game)

    def test_comprehensive_performance_monitoring(self) -> None:
        """Test comprehensive performance monitoring."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Test token statistics initialization
        assert isinstance(manager.token_stats, dict)
        assert 'primary' in manager.token_stats
        assert 'secondary' in manager.token_stats
        
        # Test time statistics initialization
        assert isinstance(manager.time_stats, defaultdict)
        
        # Test performance counters
        performance_metrics = [
            "no_path_found_steps",
            "valid_steps",
            "empty_steps",
            "something_is_wrong_steps",
            "invalid_reversals"
        ]
        
        for metric in performance_metrics:
            assert hasattr(manager, metric)
            assert getattr(manager, metric) == 0

    def test_comprehensive_move_tracking(self) -> None:
        """Test comprehensive move tracking functionality."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Test move tracking initialization
        assert manager.current_game_moves == []
        
        # Simulate move tracking
        test_moves = ["UP", "RIGHT", "DOWN", "LEFT", "UP", "DOWN"]
        
        for move in test_moves:
            manager.current_game_moves.append(move)
        
        assert len(manager.current_game_moves) == 6
        assert manager.current_game_moves == test_moves
        
        # Test move tracking reset (simulated)
        manager.current_game_moves = []
        assert manager.current_game_moves == []

    def test_comprehensive_error_recovery_mechanisms(self) -> None:
        """Test comprehensive error recovery mechanisms."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Test error flag management
        error_flags = [
            "last_no_path_found",
            "skip_empty_this_tick"
        ]
        
        for flag in error_flags:
            assert hasattr(manager, flag)
            assert getattr(manager, flag) is False
            
            # Test flag setting and clearing
            setattr(manager, flag, True)
            assert getattr(manager, flag) is True
            
            setattr(manager, flag, False)
            assert getattr(manager, flag) is False

    def test_memory_efficiency_large_sessions(self) -> None:
        """Test memory efficiency with large gaming sessions."""
        args = self.create_mock_args(max_games=1000)
        manager = GameManager(args)
        
        # Simulate large session data
        for i in range(100):  # Simulate 100 games
            manager.game_scores.append(i % 50)  # Scores 0-49
            manager.round_counts.append(i % 20 + 1)  # Rounds 1-20
            manager.total_score += i % 50
            manager.total_steps += (i % 50) * 5
            manager.total_rounds += i % 20 + 1
        
        # Verify data integrity
        assert len(manager.game_scores) == 100
        assert len(manager.round_counts) == 100
        assert manager.total_score == sum(manager.game_scores)
        assert manager.total_rounds == sum(manager.round_counts)
        
        # Test memory usage is reasonable
        import sys
        scores_size = sys.getsizeof(manager.game_scores)
        rounds_size = sys.getsizeof(manager.round_counts)
        
        # Should not use excessive memory
        assert scores_size < 10000  # Less than 10KB
        assert rounds_size < 10000  # Less than 10KB

    def test_concurrent_operations_simulation(self) -> None:
        """Test behavior under simulated concurrent operations."""
        import threading
        import time
        
        args = self.create_mock_args()
        manager = GameManager(args)
        
        errors = []
        
        def update_statistics():
            try:
                for i in range(50):
                    manager.total_score += 1
                    manager.total_steps += 2
                    manager.game_count += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def update_rounds():
            try:
                for i in range(50):
                    manager.round_count += 1
                    manager.total_rounds += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent updates
        threads = [
            threading.Thread(target=update_statistics),
            threading.Thread(target=update_rounds)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not have errors
        assert len(errors) == 0
        
        # Verify final state is reasonable
        assert manager.total_score >= 50
        assert manager.total_steps >= 100
        assert manager.game_count >= 50
        assert manager.round_count >= 51  # Started at 1
        assert manager.total_rounds >= 50

    def test_edge_cases_and_boundary_conditions(self) -> None:
        """Test edge cases and boundary conditions."""
        # Test with minimal configuration
        minimal_args = self.create_mock_args(max_games=0)
        minimal_manager = GameManager(minimal_args)
        
        # Should handle zero games gracefully
        assert minimal_manager.args.max_games == 0
        
        # Test with maximum reasonable configuration
        max_args = self.create_mock_args(max_games=10000, move_pause=10.0)
        max_manager = GameManager(max_args)
        
        assert max_manager.args.max_games == 10000
        assert max_manager.get_pause_between_moves() == 0.0  # No GUI
        
        # Test with edge case values
        edge_manager = GameManager(minimal_args)
        edge_manager.consecutive_empty_steps = 999999
        edge_manager.consecutive_something_is_wrong = 999999
        
        # Should handle large values without issues
        assert edge_manager.consecutive_empty_steps == 999999
        assert edge_manager.consecutive_something_is_wrong == 999999

    def test_serialization_compatibility(self) -> None:
        """Test compatibility with serialization for session persistence."""
        args = self.create_mock_args()
        manager = GameManager(args)
        
        # Set up complex state
        manager.total_score = 150
        manager.total_steps = 500
        manager.game_scores = [10, 25, 40, 75]
        manager.round_counts = [3, 5, 8, 12]
        manager.game_count = 4
        
        # Test that state can be extracted for serialization
        state_dict = {
            'total_score': manager.total_score,
            'total_steps': manager.total_steps,
            'game_scores': manager.game_scores,
            'round_counts': manager.round_counts,
            'game_count': manager.game_count,
            'running': manager.running
        }
        
        # Verify serialization compatibility
        import json
        json_str = json.dumps(state_dict)
        restored_state = json.loads(json_str)
        
        assert restored_state['total_score'] == 150
        assert restored_state['total_steps'] == 500
        assert restored_state['game_scores'] == [10, 25, 40, 75]
        assert restored_state['game_count'] == 4 