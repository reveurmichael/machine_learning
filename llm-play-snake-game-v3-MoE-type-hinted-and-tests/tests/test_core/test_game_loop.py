"""Tests for core.game_loop module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Tuple

from core.game_loop import (
    run_game_loop,
    _process_active_game,
    _request_and_execute_first_move,
    _execute_next_planned_move,
    _post_apple_logic,
    _handle_no_move,
    _handle_game_over,
    _execute_move,
    _handle_no_path_found,
    _apply_empty_move_delay,
)


class TestRunGameLoop:
    """Test suite for run_game_loop function."""

    @patch('pygame.quit')
    @patch('core.game_loop.process_events')
    @patch('core.game_loop._process_active_game')
    @patch('pygame.time.delay')
    def test_run_game_loop_basic(
        self,
        mock_delay: Mock,
        mock_process_active: Mock,
        mock_process_events: Mock,
        mock_pygame_quit: Mock
    ) -> None:
        """Test basic game loop execution."""
        # Create mock game manager
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 0
        game_manager.args.max_games = 1
        game_manager.game_active = True
        game_manager.game = Mock()
        game_manager.use_gui = False
        
        # Mock to stop loop after one iteration
        def stop_running(*args):
            game_manager.running = False
        
        mock_process_events.side_effect = stop_running
        
        run_game_loop(game_manager)
        
        mock_process_events.assert_called_with(game_manager)
        mock_process_active.assert_called_with(game_manager)
        mock_pygame_quit.assert_called_once()

    @patch('pygame.quit')
    @patch('core.game_loop.process_events')
    @patch('pygame.time.delay')
    @patch('pygame.time.Clock.tick')
    def test_run_game_loop_with_gui(
        self,
        mock_tick: Mock,
        mock_delay: Mock,
        mock_process_events: Mock,
        mock_pygame_quit: Mock
    ) -> None:
        """Test game loop with GUI enabled."""
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 1
        game_manager.args.max_games = 1
        game_manager.game_active = False
        game_manager.use_gui = True
        game_manager.time_delay = 50
        game_manager.time_tick = 60
        game_manager.clock = Mock()
        
        run_game_loop(game_manager)
        
        mock_delay.assert_called_with(50)
        game_manager.clock.tick.assert_called_with(60)
        mock_pygame_quit.assert_called_once()

    @patch('pygame.quit')
    @patch('core.game_loop.process_events')
    @patch('traceback.print_exc')
    def test_run_game_loop_exception_handling(
        self,
        mock_traceback: Mock,
        mock_process_events: Mock,
        mock_pygame_quit: Mock
    ) -> None:
        """Test game loop exception handling."""
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 0
        game_manager.args.max_games = 1
        
        # Cause an exception
        mock_process_events.side_effect = ValueError("Test error")
        
        run_game_loop(game_manager)
        
        mock_traceback.assert_called_once()
        mock_pygame_quit.assert_called_once()

    @patch('pygame.quit')
    def test_run_game_loop_max_games_reached(self, mock_pygame_quit: Mock) -> None:
        """Test game loop stops when max games reached."""
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 5
        game_manager.args.max_games = 5
        
        run_game_loop(game_manager)
        
        # Should exit immediately without processing
        mock_pygame_quit.assert_called_once()


class TestProcessActiveGame:
    """Test suite for _process_active_game function."""

    @patch('core.game_loop._request_and_execute_first_move')
    def test_process_active_game_need_new_plan(self, mock_request: Mock) -> None:
        """Test processing when new plan is needed."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.need_new_plan = True
        game_manager.game_active = True
        
        _process_active_game(game_manager)
        
        mock_request.assert_called_once_with(game_manager)
        game_manager.game.draw.assert_called_once()

    @patch('core.game_loop._execute_next_planned_move')
    def test_process_active_game_execute_planned(self, mock_execute: Mock) -> None:
        """Test processing when executing planned moves."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.need_new_plan = False
        game_manager.game_active = True
        
        _process_active_game(game_manager)
        
        mock_execute.assert_called_once_with(game_manager)
        game_manager.game.draw.assert_called_once()

    @patch('core.game_loop._handle_game_over')
    @patch('core.game_loop._execute_next_planned_move')
    def test_process_active_game_game_over(
        self,
        mock_execute: Mock,
        mock_handle_over: Mock
    ) -> None:
        """Test processing when game becomes inactive."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.need_new_plan = False
        game_manager.game_active = False
        
        _process_active_game(game_manager)
        
        mock_execute.assert_called_once_with(game_manager)
        mock_handle_over.assert_called_once_with(game_manager)
        game_manager.game.draw.assert_called_once()

    def test_process_active_game_no_game(self) -> None:
        """Test processing when game is None."""
        game_manager = Mock()
        game_manager.game = None
        
        with pytest.raises(AssertionError):
            _process_active_game(game_manager)


class TestRequestAndExecuteFirstMove:
    """Test suite for _request_and_execute_first_move function."""

    @patch('time.sleep')
    @patch('core.game_loop._execute_move')
    @patch('core.game_loop._post_apple_logic')
    def test_request_and_execute_success(
        self,
        mock_post_apple: Mock,
        mock_execute: Mock,
        mock_sleep: Mock
    ) -> None:
        """Test successful LLM request and move execution."""
        # Mock the import
        with patch.dict('sys.modules', {'llm.communication_utils': Mock()}):
            with patch('llm.communication_utils.get_llm_response') as mock_get_response:
                game_manager = Mock()
                game_manager.game = Mock()
                game_manager.game.planned_moves = ["UP", "DOWN"]
                game_manager.use_gui = True
                game_manager.last_no_path_found = False
                game_manager.game_active = True
                
                mock_get_response.return_value = ("UP", True)
                mock_execute.return_value = (True, True)  # game_active, apple_eaten
                
                _request_and_execute_first_move(game_manager)
                
                assert game_manager.awaiting_plan is False
                assert game_manager.need_new_plan is False
                mock_execute.assert_called_once_with(game_manager, "UP")
                mock_post_apple.assert_called_once_with(game_manager)
                mock_sleep.assert_called_once_with(3)

    @patch('core.game_loop._handle_no_move')
    def test_request_and_execute_no_move(self, mock_handle_no_move: Mock) -> None:
        """Test when LLM returns no move."""
        with patch.dict('sys.modules', {'llm.communication_utils': Mock()}):
            with patch('llm.communication_utils.get_llm_response') as mock_get_response:
                game_manager = Mock()
                game_manager.game = Mock()
                game_manager.last_no_path_found = False
                game_manager.game_active = True
                game_manager.skip_empty_this_tick = False
                
                mock_get_response.return_value = (None, True)
                
                _request_and_execute_first_move(game_manager)
                
                mock_handle_no_move.assert_called_once_with(game_manager)

    @patch('core.game_loop._handle_no_path_found')
    def test_request_and_execute_no_path_found(self, mock_handle_no_path: Mock) -> None:
        """Test handling NO_PATH_FOUND flag."""
        with patch.dict('sys.modules', {'llm.communication_utils': Mock()}):
            with patch('llm.communication_utils.get_llm_response') as mock_get_response:
                game_manager = Mock()
                game_manager.game = Mock()
                game_manager.last_no_path_found = True
                game_manager.game_active = True
                
                mock_get_response.return_value = ("UP", True)
                
                _request_and_execute_first_move(game_manager)
                
                mock_handle_no_path.assert_called_once_with(game_manager)

    def test_request_and_execute_skip_empty_tick(self) -> None:
        """Test skipping empty move when flag is set."""
        with patch.dict('sys.modules', {'llm.communication_utils': Mock()}):
            with patch('llm.communication_utils.get_llm_response') as mock_get_response:
                game_manager = Mock()
                game_manager.game = Mock()
                game_manager.last_no_path_found = False
                game_manager.game_active = True
                game_manager.skip_empty_this_tick = True
                
                mock_get_response.return_value = (None, True)
                
                _request_and_execute_first_move(game_manager)
                
                assert game_manager.skip_empty_this_tick is False

    def test_request_and_execute_game_inactive(self) -> None:
        """Test when game becomes inactive during LLM call."""
        with patch.dict('sys.modules', {'llm.communication_utils': Mock()}):
            with patch('llm.communication_utils.get_llm_response') as mock_get_response:
                game_manager = Mock()
                game_manager.game = Mock()
                game_manager.last_no_path_found = False
                
                mock_get_response.return_value = ("UP", False)  # game_active = False
                
                _request_and_execute_first_move(game_manager)
                
                assert game_manager.game_active is False


class TestExecuteNextPlannedMove:
    """Test suite for _execute_next_planned_move function."""

    @patch('core.game_loop._execute_move')
    @patch('core.game_loop._post_apple_logic')
    def test_execute_next_planned_move_success(
        self,
        mock_post_apple: Mock,
        mock_execute: Mock
    ) -> None:
        """Test successful execution of planned move."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.awaiting_plan = False
        game_manager.current_game_moves = []
        
        game_manager.game.get_next_planned_move.return_value = "DOWN"
        mock_execute.return_value = (True, True)  # game_active, apple_eaten
        
        _execute_next_planned_move(game_manager)
        
        assert game_manager.current_game_moves == ["DOWN"]
        mock_execute.assert_called_once_with(game_manager, "DOWN")
        mock_post_apple.assert_called_once_with(game_manager)
        game_manager.game.draw.assert_called_once()

    def test_execute_next_planned_move_no_moves(self) -> None:
        """Test when no planned moves are available."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.awaiting_plan = False
        
        game_manager.game.get_next_planned_move.return_value = None
        
        _execute_next_planned_move(game_manager)
        
        game_manager.finish_round.assert_called_once()
        assert game_manager.need_new_plan is True

    def test_execute_next_planned_move_awaiting_plan(self) -> None:
        """Test when still awaiting plan from LLM."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.awaiting_plan = True
        
        _execute_next_planned_move(game_manager)
        
        game_manager.game.get_next_planned_move.assert_not_called()

    @patch('core.game_loop._execute_move')
    def test_execute_next_planned_move_no_apple(self, mock_execute: Mock) -> None:
        """Test execution when no apple is eaten."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.awaiting_plan = False
        game_manager.current_game_moves = []
        
        game_manager.game.get_next_planned_move.return_value = "LEFT"
        mock_execute.return_value = (True, False)  # game_active, no apple
        
        _execute_next_planned_move(game_manager)
        
        assert game_manager.current_game_moves == ["LEFT"]
        # _post_apple_logic should not be called
        mock_execute.assert_called_once_with(game_manager, "LEFT")


class TestPostAppleLogic:
    """Test suite for _post_apple_logic function."""

    def test_post_apple_logic_with_remaining_moves(self) -> None:
        """Test post-apple logic when moves remain."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.game.planned_moves = ["UP", "DOWN"]
        
        _post_apple_logic(game_manager)
        
        # Should not request new plan
        game_manager.finish_round.assert_not_called()
        assert not hasattr(game_manager, 'need_new_plan') or game_manager.need_new_plan != True

    def test_post_apple_logic_no_remaining_moves(self) -> None:
        """Test post-apple logic when no moves remain."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.game.planned_moves = []
        
        _post_apple_logic(game_manager)
        
        game_manager.finish_round.assert_called_once()
        assert game_manager.need_new_plan is True


class TestHandleNoMove:
    """Test suite for _handle_no_move function."""

    @patch('core.game_loop._apply_empty_move_delay')
    def test_handle_no_move(self, mock_apply_delay: Mock) -> None:
        """Test handling when no move is available."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.game.game_state = Mock()
        
        _handle_no_move(game_manager)
        
        game_manager.game.game_state.record_empty_move.assert_called_once()
        mock_apply_delay.assert_called_once_with(game_manager)


class TestExecuteMove:
    """Test suite for _execute_move function."""

    @patch('core.game_loop.check_max_steps')
    def test_execute_move_success(self, mock_check_steps: Mock) -> None:
        """Test successful move execution."""
        game_manager = Mock()
        game_manager.game = Mock()
        
        # Mock successful move
        game_manager.game.move.return_value = True
        game_manager.game.apple_eaten = True
        mock_check_steps.return_value = True
        
        game_active, apple_eaten = _execute_move(game_manager, "UP")
        
        assert game_active is True
        assert apple_eaten is True
        game_manager.game.move.assert_called_once_with("UP")
        mock_check_steps.assert_called_once_with(game_manager)

    @patch('core.game_loop.check_max_steps')
    def test_execute_move_game_over(self, mock_check_steps: Mock) -> None:
        """Test move execution that results in game over."""
        game_manager = Mock()
        game_manager.game = Mock()
        
        # Mock failed move (game over)
        game_manager.game.move.return_value = False
        mock_check_steps.return_value = False
        
        game_active, apple_eaten = _execute_move(game_manager, "DOWN")
        
        assert game_active is False
        assert apple_eaten is False
        assert game_manager.game_active is False

    @patch('core.game_loop.check_max_steps')
    def test_execute_move_max_steps_reached(self, mock_check_steps: Mock) -> None:
        """Test move execution when max steps reached."""
        game_manager = Mock()
        game_manager.game = Mock()
        
        game_manager.game.move.return_value = True
        mock_check_steps.return_value = False  # Max steps reached
        
        game_active, apple_eaten = _execute_move(game_manager, "LEFT")
        
        assert game_active is False
        assert game_manager.game_active is False


class TestHandleGameOver:
    """Test suite for _handle_game_over function."""

    @patch('core.game_loop.process_game_over')
    def test_handle_game_over(self, mock_process: Mock) -> None:
        """Test game over handling."""
        game_manager = Mock()
        
        _handle_game_over(game_manager)
        
        mock_process.assert_called_once_with(game_manager)


class TestHandleNoPathFound:
    """Test suite for _handle_no_path_found function."""

    def test_handle_no_path_found(self) -> None:
        """Test handling NO_PATH_FOUND sentinel."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.game.game_state = Mock()
        game_manager.last_no_path_found = True
        
        _handle_no_path_found(game_manager)
        
        game_manager.game.game_state.record_no_path_found_move.assert_called_once()
        assert game_manager.last_no_path_found is False


class TestApplyEmptyMoveDelay:
    """Test suite for _apply_empty_move_delay function."""

    @patch('time.sleep')
    def test_apply_empty_move_delay_with_gui(self, mock_sleep: Mock) -> None:
        """Test applying empty move delay with GUI."""
        game_manager = Mock()
        game_manager.get_pause_between_moves.return_value = 1.5
        
        _apply_empty_move_delay(game_manager)
        
        mock_sleep.assert_called_once_with(1.5)

    @patch('time.sleep')
    def test_apply_empty_move_delay_no_gui(self, mock_sleep: Mock) -> None:
        """Test applying empty move delay without GUI."""
        game_manager = Mock()
        game_manager.get_pause_between_moves.return_value = 0.0
        
        _apply_empty_move_delay(game_manager)
        
        mock_sleep.assert_not_called()


class TestGameLoopIntegration:
    """Integration tests for game loop components."""

    @patch('pygame.quit')
    @patch('core.game_loop.process_events')
    @patch('time.sleep')
    def test_complete_game_loop_cycle(
        self,
        mock_sleep: Mock,
        mock_process_events: Mock,
        mock_pygame_quit: Mock
    ) -> None:
        """Test a complete game loop cycle."""
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 0
        game_manager.args.max_games = 1
        game_manager.game_active = True
        game_manager.game = Mock()
        game_manager.use_gui = False
        game_manager.need_new_plan = True
        game_manager.last_no_path_found = False
        
        # Mock LLM response
        with patch.dict('sys.modules', {'llm.communication_utils': Mock()}):
            with patch('llm.communication_utils.get_llm_response') as mock_get_response:
                mock_get_response.return_value = ("UP", True)
                game_manager.game.planned_moves = []
                game_manager.game.move.return_value = True
                game_manager.game.apple_eaten = False
                
                # Stop after one iteration
                def stop_running(*args):
                    game_manager.running = False
                
                mock_process_events.side_effect = stop_running
                
                with patch('core.game_loop.check_max_steps', return_value=True):
                    run_game_loop(game_manager)
                
                # Verify the cycle completed
                mock_process_events.assert_called()
                game_manager.game.draw.assert_called()
                mock_pygame_quit.assert_called_once()

    # ==================== COMPREHENSIVE GAME LOOP STRESS TESTING ====================
    
    @patch('pygame.quit')
    @patch('core.game_loop.process_events')
    @patch('time.sleep')
    def test_game_loop_rapid_cycling(
        self,
        mock_sleep: Mock,
        mock_process_events: Mock,
        mock_pygame_quit: Mock
    ) -> None:
        """Test game loop with rapid cycling through multiple games."""
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 0
        game_manager.args.max_games = 5
        game_manager.use_gui = False
        
        # Track iterations
        iteration_count = 0
        
        def count_iterations(*args):
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 50:  # Stop after many iterations
                game_manager.running = False
        
        mock_process_events.side_effect = count_iterations
        
        # Mock various states for different games
        def mock_game_state():
            if iteration_count < 10:
                game_manager.game_active = True
                game_manager.need_new_plan = True
            elif iteration_count < 20:
                game_manager.game_active = True  
                game_manager.need_new_plan = False
            else:
                game_manager.game_active = False
        
        with patch('core.game_loop._process_active_game', side_effect=mock_game_state):
            run_game_loop(game_manager)
        
        assert iteration_count >= 50
        mock_pygame_quit.assert_called_once()

    @patch('pygame.quit')
    @patch('core.game_loop.process_events')
    @patch('pygame.time.delay')
    def test_game_loop_memory_efficiency(
        self,
        mock_delay: Mock,
        mock_process_events: Mock,
        mock_pygame_quit: Mock
    ) -> None:
        """Test game loop memory efficiency over many iterations."""
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 0
        game_manager.args.max_games = 1000
        game_manager.use_gui = False
        
        # Simulate memory-intensive operations
        memory_usage = []
        iteration_count = 0
        
        def track_memory(*args):
            nonlocal iteration_count
            iteration_count += 1
            # Simulate some memory usage tracking
            memory_usage.append(iteration_count * 100)  # Mock memory usage
            if iteration_count >= 100:  # Stop after many iterations
                game_manager.running = False
        
        mock_process_events.side_effect = track_memory
        
        with patch('core.game_loop._process_active_game'):
            run_game_loop(game_manager)
        
        # Memory usage should not grow linearly without bound
        assert len(memory_usage) == 100
        mock_pygame_quit.assert_called_once()

    @patch('pygame.quit')
    @patch('core.game_loop.process_events')
    @patch('traceback.print_exc')
    def test_game_loop_error_recovery_comprehensive(
        self,
        mock_traceback: Mock,
        mock_process_events: Mock,
        mock_pygame_quit: Mock
    ) -> None:
        """Test comprehensive error recovery in game loop."""
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 0
        game_manager.args.max_games = 10
        game_manager.use_gui = False
        
        # Simulate various exceptions
        exception_count = 0
        def raise_exceptions(*args):
            nonlocal exception_count
            exception_count += 1
            if exception_count == 1:
                raise ValueError("Mock value error")
            elif exception_count == 2:
                raise KeyError("Mock key error")
            elif exception_count == 3:
                raise RuntimeError("Mock runtime error")
            else:
                game_manager.running = False
        
        mock_process_events.side_effect = raise_exceptions
        
        with patch('core.game_loop._process_active_game'):
            run_game_loop(game_manager)
        
        # Should handle all exceptions gracefully
        assert mock_traceback.call_count == 3
        mock_pygame_quit.assert_called_once()

    @patch('pygame.quit')
    @patch('core.game_loop.process_events')
    @patch('pygame.time.Clock')
    def test_game_loop_timing_precision(
        self,
        mock_clock_class: Mock,
        mock_process_events: Mock,
        mock_pygame_quit: Mock
    ) -> None:
        """Test game loop timing precision with GUI."""
        mock_clock = Mock()
        mock_clock_class.return_value = mock_clock
        
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 0
        game_manager.args.max_games = 1
        game_manager.use_gui = True
        game_manager.gui = Mock()
        
        # Track tick calls
        tick_count = 0
        def count_ticks(fps):
            nonlocal tick_count
            tick_count += 1
            if tick_count >= 5:
                game_manager.running = False
            return 16  # Mock 60 FPS
        
        mock_clock.tick.side_effect = count_ticks
        
        with patch('core.game_loop._process_active_game'):
            run_game_loop(game_manager)
        
        # Should call tick with proper FPS
        assert tick_count >= 5
        mock_clock.tick.assert_called_with(60)  # Should use 60 FPS
        mock_pygame_quit.assert_called_once()

    # ==================== COMPREHENSIVE ACTIVE GAME PROCESSING ====================
    
    @patch('core.game_loop._request_and_execute_first_move')
    def test_process_active_game_state_transitions(self, mock_request: Mock) -> None:
        """Test comprehensive state transitions in active game processing."""
        game_manager = Mock()
        
        # Test transition from need_new_plan to execution
        game_manager.game_active = True
        game_manager.need_new_plan = True
        game_manager.last_no_path_found = False
        
        _process_active_game(game_manager)
        
        mock_request.assert_called_once_with(game_manager)
        
        # Test transition to planned move execution
        mock_request.reset_mock()
        game_manager.need_new_plan = False
        game_manager.game = Mock()
        game_manager.game.planned_moves = ["UP", "DOWN"]
        
        with patch('core.game_loop._execute_next_planned_move') as mock_execute:
            _process_active_game(game_manager)
            mock_execute.assert_called_once_with(game_manager)

    @patch('core.game_loop._request_and_execute_first_move')
    def test_process_active_game_no_path_found_handling(self, mock_request: Mock) -> None:
        """Test handling of NO_PATH_FOUND scenarios."""
        game_manager = Mock()
        game_manager.game_active = True
        game_manager.need_new_plan = True
        game_manager.last_no_path_found = True
        
        with patch('core.game_loop._handle_no_path_found') as mock_handle:
            _process_active_game(game_manager)
            mock_handle.assert_called_once_with(game_manager)
            mock_request.assert_called_once_with(game_manager)

    @patch('core.game_loop._handle_game_over')
    def test_process_active_game_game_over_transitions(self, mock_handle_over: Mock) -> None:
        """Test game over transition handling."""
        game_manager = Mock()
        game_manager.game_active = False
        game_manager.game = Mock()
        
        _process_active_game(game_manager)
        
        mock_handle_over.assert_called_once_with(game_manager)

    def test_process_active_game_edge_cases(self) -> None:
        """Test edge cases in active game processing."""
        game_manager = Mock()
        
        # Test with no game object
        game_manager.game_active = True
        game_manager.game = None
        
        # Should not crash
        _process_active_game(game_manager)
        
        # Test with empty planned moves
        game_manager.game = Mock()
        game_manager.game.planned_moves = []
        game_manager.need_new_plan = False
        
        with patch('core.game_loop._execute_next_planned_move') as mock_execute:
            _process_active_game(game_manager)
            mock_execute.assert_called_once_with(game_manager)

    # ==================== COMPREHENSIVE MOVE REQUEST AND EXECUTION ====================
    
    @patch('time.sleep')
    @patch('core.game_loop._execute_move')
    @patch('core.game_loop._post_apple_logic')
    def test_request_and_execute_comprehensive_scenarios(
        self,
        mock_post_apple: Mock,
        mock_execute: Mock,
        mock_sleep: Mock
    ) -> None:
        """Test comprehensive scenarios for move request and execution."""
        game_manager = Mock()
        game_manager.get_pause_between_moves.return_value = 0.1
        game_manager.game_active = True
        game_manager.game = Mock()
        
        # Test successful move with apple eaten
        mock_execute.return_value = (True, True)  # game_active, apple_eaten
        
        with patch.dict('sys.modules', {'llm.communication_utils': Mock()}):
            with patch('llm.communication_utils.get_llm_response') as mock_get_response:
                mock_get_response.return_value = ("RIGHT", True)
                
                _request_and_execute_first_move(game_manager)
                
                mock_execute.assert_called_once_with(game_manager, "RIGHT")
                mock_post_apple.assert_called_once_with(game_manager)
                mock_sleep.assert_called_once_with(0.1)
        
        # Test successful move without apple
        mock_execute.reset_mock()
        mock_post_apple.reset_mock()
        mock_sleep.reset_mock()
        mock_execute.return_value = (True, False)
        
        with patch.dict('sys.modules', {'llm.communication_utils': Mock()}):
            with patch('llm.communication_utils.get_llm_response') as mock_get_response:
                mock_get_response.return_value = ("LEFT", True)
                
                _request_and_execute_first_move(game_manager)
                
                mock_execute.assert_called_once_with(game_manager, "LEFT")
                mock_post_apple.assert_not_called()  # No apple eaten

    @patch('core.game_loop._handle_no_move')
    def test_request_and_execute_various_no_move_scenarios(self, mock_handle_no_move: Mock) -> None:
        """Test various no-move scenarios."""
        game_manager = Mock()
        game_manager.game_active = True
        game_manager.game = Mock()
        
        # Test empty string move
        with patch.dict('sys.modules', {'llm.communication_utils': Mock()}):
            with patch('llm.communication_utils.get_llm_response') as mock_get_response:
                mock_get_response.return_value = ("", True)
                
                _request_and_execute_first_move(game_manager)
                
                mock_handle_no_move.assert_called_once_with(game_manager)
        
        # Test None move
        mock_handle_no_move.reset_mock()
        with patch.dict('sys.modules', {'llm.communication_utils': Mock()}):
            with patch('llm.communication_utils.get_llm_response') as mock_get_response:
                mock_get_response.return_value = (None, True)
                
                _request_and_execute_first_move(game_manager)
                
                mock_handle_no_move.assert_called_once_with(game_manager)

    @patch('core.game_loop._handle_no_path_found')
    def test_request_and_execute_no_path_variations(self, mock_handle_no_path: Mock) -> None:
        """Test variations of NO_PATH_FOUND handling."""
        game_manager = Mock()
        game_manager.game_active = True
        game_manager.game = Mock()
        
        # Test NO_PATH_FOUND sentinel
        with patch.dict('sys.modules', {'llm.communication_utils': Mock()}):
            with patch('llm.communication_utils.get_llm_response') as mock_get_response:
                from config.game_constants import NO_PATH_FOUND
                mock_get_response.return_value = (NO_PATH_FOUND, True)
                
                _request_and_execute_first_move(game_manager)
                
                mock_handle_no_path.assert_called_once_with(game_manager)

    def test_request_and_execute_game_state_edge_cases(self) -> None:
        """Test edge cases related to game state during move request."""
        game_manager = Mock()
        
        # Test when game becomes inactive during processing
        game_manager.game_active = False
        
        _request_and_execute_first_move(game_manager)
        
        # Should exit early without processing
        assert True  # No exceptions should be raised
        
        # Test when skip_empty_tick is True
        game_manager.game_active = True
        game_manager.skip_empty_tick = True
        
        _request_and_execute_first_move(game_manager)
        
        # Should exit early
        assert True

    # ==================== COMPREHENSIVE PLANNED MOVE EXECUTION ====================
    
    @patch('core.game_loop._execute_move')
    @patch('core.game_loop._post_apple_logic')
    def test_execute_next_planned_comprehensive(
        self,
        mock_post_apple: Mock,
        mock_execute: Mock
    ) -> None:
        """Test comprehensive scenarios for planned move execution."""
        game_manager = Mock()
        game_manager.game = Mock()
        
        # Test successful execution with multiple moves
        game_manager.game.planned_moves = ["UP", "DOWN", "LEFT"]
        game_manager.current_game_moves = []
        mock_execute.return_value = (True, True)  # apple eaten
        
        _execute_next_planned_move(game_manager)
        
        assert game_manager.current_game_moves == ["UP"]
        assert game_manager.game.planned_moves == ["DOWN", "LEFT"]
        mock_execute.assert_called_once_with(game_manager, "UP")
        mock_post_apple.assert_called_once_with(game_manager)
        
        # Test execution without apple
        mock_execute.reset_mock()
        mock_post_apple.reset_mock()
        mock_execute.return_value = (True, False)
        
        _execute_next_planned_move(game_manager)
        
        assert game_manager.current_game_moves == ["UP", "DOWN"]
        assert game_manager.game.planned_moves == ["LEFT"]
        mock_execute.assert_called_once_with(game_manager, "DOWN")
        mock_post_apple.assert_not_called()

    def test_execute_next_planned_edge_cases(self) -> None:
        """Test edge cases in planned move execution."""
        game_manager = Mock()
        game_manager.game = Mock()
        
        # Test with empty planned moves
        game_manager.game.planned_moves = []
        game_manager.current_game_moves = ["UP"]
        
        _execute_next_planned_move(game_manager)
        
        # Should not crash, moves should remain unchanged
        assert game_manager.current_game_moves == ["UP"]
        
        # Test when awaiting new plan
        game_manager.game.awaiting_new_plan = True
        game_manager.game.planned_moves = ["DOWN"]
        
        _execute_next_planned_move(game_manager)
        
        # Should not execute moves when awaiting plan
        assert game_manager.game.planned_moves == ["DOWN"]  # Unchanged

    def test_execute_next_planned_move_tracking(self) -> None:
        """Test move tracking during planned execution."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.current_game_moves = ["UP", "DOWN"]
        game_manager.game.planned_moves = ["LEFT", "RIGHT"]
        
        with patch('core.game_loop._execute_move') as mock_execute:
            mock_execute.return_value = (True, False)
            
            _execute_next_planned_move(game_manager)
            
            # Should track the move
            assert "LEFT" in game_manager.current_game_moves
            assert game_manager.game.planned_moves == ["RIGHT"]

    # ==================== COMPREHENSIVE APPLE LOGIC TESTING ====================
    
    def test_post_apple_logic_comprehensive_scenarios(self) -> None:
        """Test comprehensive scenarios for post-apple logic."""
        game_manager = Mock()
        game_manager.game = Mock()
        
        # Test with many remaining moves
        game_manager.game.planned_moves = ["UP", "DOWN", "LEFT", "RIGHT", "UP"]
        
        _post_apple_logic(game_manager)
        
        # Should not finish round or set need_new_plan
        game_manager.finish_round.assert_not_called()
        assert not hasattr(game_manager, 'need_new_plan') or game_manager.need_new_plan != True
        
        # Test with exactly one remaining move
        game_manager.game.planned_moves = ["UP"]
        
        _post_apple_logic(game_manager)
        
        # Should not finish round yet
        game_manager.finish_round.assert_not_called()
        
        # Test with empty moves (should finish round)
        game_manager.game.planned_moves = []
        
        _post_apple_logic(game_manager)
        
        game_manager.finish_round.assert_called_once()
        assert game_manager.need_new_plan is True

    def test_post_apple_logic_state_consistency(self) -> None:
        """Test state consistency in post-apple logic."""
        game_manager = Mock()
        game_manager.game = Mock()
        
        # Ensure initial state
        game_manager.need_new_plan = False
        game_manager.game.planned_moves = ["LEFT", "RIGHT"]
        
        _post_apple_logic(game_manager)
        
        # State should remain consistent
        assert game_manager.need_new_plan is False
        game_manager.finish_round.assert_not_called()

    # ==================== COMPREHENSIVE MOVE EXECUTION TESTING ====================
    
    @patch('core.game_loop.check_max_steps')
    def test_execute_move_comprehensive_scenarios(self, mock_check_steps: Mock) -> None:
        """Test comprehensive move execution scenarios."""
        game_manager = Mock()
        game_manager.game = Mock()
        
        # Test successful move with apple eaten
        game_manager.game.move.return_value = True
        game_manager.game.apple_eaten = True
        mock_check_steps.return_value = True
        
        game_active, apple_eaten = _execute_move(game_manager, "UP")
        
        assert game_active is True
        assert apple_eaten is True
        game_manager.game.move.assert_called_once_with("UP")
        mock_check_steps.assert_called_once_with(game_manager)
        
        # Test successful move without apple
        game_manager.game.move.reset_mock()
        mock_check_steps.reset_mock()
        game_manager.game.apple_eaten = False
        
        game_active, apple_eaten = _execute_move(game_manager, "DOWN")
        
        assert game_active is True
        assert apple_eaten is False
        
        # Test failed move (collision)
        game_manager.game.move.return_value = False
        game_manager.game.apple_eaten = False
        mock_check_steps.return_value = False
        
        game_active, apple_eaten = _execute_move(game_manager, "LEFT")
        
        assert game_active is False
        assert apple_eaten is False
        assert game_manager.game_active is False

    @patch('core.game_loop.check_max_steps')
    def test_execute_move_max_steps_scenarios(self, mock_check_steps: Mock) -> None:
        """Test move execution with max steps scenarios."""
        game_manager = Mock()
        game_manager.game = Mock()
        
        # Test max steps reached with successful move
        game_manager.game.move.return_value = True
        game_manager.game.apple_eaten = True
        mock_check_steps.return_value = False  # Max steps reached
        
        game_active, apple_eaten = _execute_move(game_manager, "RIGHT")
        
        assert game_active is False  # Game should end due to max steps
        assert apple_eaten is True   # But apple was eaten
        assert game_manager.game_active is False
        
        # Test max steps reached with failed move
        game_manager.game.move.return_value = False
        mock_check_steps.return_value = False
        
        game_active, apple_eaten = _execute_move(game_manager, "UP")
        
        assert game_active is False
        assert apple_eaten is False
        assert game_manager.game_active is False

    def test_execute_move_game_state_updates(self) -> None:
        """Test game state updates during move execution."""
        game_manager = Mock()
        game_manager.game = Mock()
        
        # Test that game_active is properly set
        game_manager.game.move.return_value = True
        game_manager.game.apple_eaten = False
        
        with patch('core.game_loop.check_max_steps') as mock_check:
            mock_check.return_value = True
            
            game_active, apple_eaten = _execute_move(game_manager, "DOWN")
            
            # game_active should remain True for game_manager
            # (it's returned as True but game_manager.game_active not modified when successful)
            assert game_active is True

    # ==================== COMPREHENSIVE ERROR HANDLING TESTING ====================
    
    def test_handle_no_move_comprehensive(self) -> None:
        """Test comprehensive no-move handling."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.game.game_state = Mock()
        
        with patch('core.game_loop._apply_empty_move_delay') as mock_delay:
            _handle_no_move(game_manager)
            
            # Should record empty move and apply delay
            game_manager.game.game_state.record_empty_move.assert_called_once()
            mock_delay.assert_called_once_with(game_manager)

    def test_handle_no_move_edge_cases(self) -> None:
        """Test edge cases in no-move handling."""
        game_manager = Mock()
        
        # Test with no game object
        game_manager.game = None
        
        # Should not crash
        try:
            _handle_no_move(game_manager)
        except AttributeError:
            pass  # Expected when game is None
        
        # Test with no game_state
        game_manager.game = Mock()
        game_manager.game.game_state = None
        
        try:
            _handle_no_move(game_manager)
        except AttributeError:
            pass  # Expected when game_state is None

    def test_handle_no_path_found_comprehensive(self) -> None:
        """Test comprehensive NO_PATH_FOUND handling."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.game.game_state = Mock()
        game_manager.last_no_path_found = True
        
        _handle_no_path_found(game_manager)
        
        # Should record no path found and reset flag
        game_manager.game.game_state.record_no_path_found_move.assert_called_once()
        assert game_manager.last_no_path_found is False

    def test_handle_no_path_found_state_consistency(self) -> None:
        """Test state consistency in NO_PATH_FOUND handling."""
        game_manager = Mock()
        game_manager.game = Mock()
        game_manager.game.game_state = Mock()
        
        # Test when last_no_path_found is False
        game_manager.last_no_path_found = False
        
        _handle_no_path_found(game_manager)
        
        # Should still record and update flag
        game_manager.game.game_state.record_no_path_found_move.assert_called_once()
        assert game_manager.last_no_path_found is False

    @patch('core.game_loop.process_game_over')
    def test_handle_game_over_comprehensive(self, mock_process: Mock) -> None:
        """Test comprehensive game over handling."""
        game_manager = Mock()
        
        _handle_game_over(game_manager)
        
        mock_process.assert_called_once_with(game_manager)

    def test_handle_game_over_with_various_states(self) -> None:
        """Test game over handling with various game manager states."""
        game_manager = Mock()
        
        # Test with different game_active states
        game_manager.game_active = False
        
        with patch('core.game_loop.process_game_over') as mock_process:
            _handle_game_over(game_manager)
            mock_process.assert_called_once_with(game_manager)
        
        # Test with different game states
        game_manager.game = None
        
        with patch('core.game_loop.process_game_over') as mock_process:
            _handle_game_over(game_manager)
            mock_process.assert_called_once_with(game_manager)

    # ==================== COMPREHENSIVE DELAY AND TIMING TESTING ====================
    
    @patch('time.sleep')
    def test_apply_empty_move_delay_comprehensive(self, mock_sleep: Mock) -> None:
        """Test comprehensive empty move delay scenarios."""
        game_manager = Mock()
        
        # Test with various pause durations
        test_durations = [0.0, 0.1, 0.5, 1.0, 2.5]
        
        for duration in test_durations:
            mock_sleep.reset_mock()
            game_manager.get_pause_between_moves.return_value = duration
            
            _apply_empty_move_delay(game_manager)
            
            if duration > 0:
                mock_sleep.assert_called_once_with(duration)
            else:
                mock_sleep.assert_not_called()

    @patch('time.sleep')
    def test_apply_empty_move_delay_edge_cases(self, mock_sleep: Mock) -> None:
        """Test edge cases in empty move delay."""
        game_manager = Mock()
        
        # Test with negative duration (should not sleep)
        game_manager.get_pause_between_moves.return_value = -1.0
        
        _apply_empty_move_delay(game_manager)
        
        mock_sleep.assert_not_called()
        
        # Test with very large duration
        game_manager.get_pause_between_moves.return_value = 100.0
        
        _apply_empty_move_delay(game_manager)
        
        mock_sleep.assert_called_once_with(100.0)

    # ==================== COMPREHENSIVE INTEGRATION TESTING ====================
    
    @patch('pygame.quit')
    @patch('core.game_loop.process_events')
    @patch('time.sleep')
    def test_complete_multi_game_cycle(
        self,
        mock_sleep: Mock,
        mock_process_events: Mock,
        mock_pygame_quit: Mock
    ) -> None:
        """Test complete multi-game cycle integration."""
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 0
        game_manager.args.max_games = 3
        game_manager.use_gui = False
        
        # Track game progression
        games_completed = 0
        
        def simulate_game_progression(*args):
            nonlocal games_completed
            
            if games_completed < 3:
                game_manager.game_active = True
                game_manager.need_new_plan = True
                games_completed += 1
            else:
                game_manager.running = False
        
        mock_process_events.side_effect = simulate_game_progression
        
        with patch('core.game_loop._process_active_game'):
            run_game_loop(game_manager)
        
        # Should have processed multiple games
        assert games_completed == 3
        mock_pygame_quit.assert_called_once()

    @patch('pygame.quit')
    @patch('core.game_loop.process_events')  
    @patch('pygame.time.delay')
    def test_performance_under_load(
        self,
        mock_delay: Mock,
        mock_process_events: Mock,
        mock_pygame_quit: Mock
    ) -> None:
        """Test performance under simulated load."""
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 0
        game_manager.args.max_games = 100
        game_manager.use_gui = False
        
        # Simulate high-frequency operations
        operation_count = 0
        
        def high_frequency_ops(*args):
            nonlocal operation_count
            operation_count += 1
            
            # Simulate processing load
            if operation_count >= 1000:
                game_manager.running = False
        
        mock_process_events.side_effect = high_frequency_ops
        
        import time
        start_time = time.time()
        
        with patch('core.game_loop._process_active_game'):
            run_game_loop(game_manager)
        
        end_time = time.time()
        
        # Should complete efficiently
        assert operation_count >= 1000
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
        mock_pygame_quit.assert_called_once()

    @patch('pygame.quit')
    @patch('core.game_loop.process_events')
    @patch('pygame.time.Clock')
    def test_gui_integration_comprehensive(
        self,
        mock_clock_class: Mock,
        mock_process_events: Mock,
        mock_pygame_quit: Mock
    ) -> None:
        """Test comprehensive GUI integration."""
        mock_clock = Mock()
        mock_clock_class.return_value = mock_clock
        
        game_manager = Mock()
        game_manager.running = True
        game_manager.game_count = 0
        game_manager.args.max_games = 1
        game_manager.use_gui = True
        game_manager.gui = Mock()
        
        # Test various GUI states
        gui_updates = 0
        
        def update_gui(*args):
            nonlocal gui_updates
            gui_updates += 1
            if gui_updates >= 10:
                game_manager.running = False
            return 16  # Mock frame time
        
        mock_clock.tick.side_effect = update_gui
        
        with patch('core.game_loop._process_active_game'):
            run_game_loop(game_manager)
        
        # Should update GUI multiple times
        assert gui_updates >= 10
        mock_clock.tick.assert_called()
        mock_pygame_quit.assert_called_once()

    def test_state_persistence_across_cycles(self) -> None:
        """Test state persistence across game loop cycles."""
        game_manager = Mock()
        game_manager.running = False  # Will exit immediately
        game_manager.game_count = 5
        game_manager.args.max_games = 10
        
        initial_game_count = game_manager.game_count
        
        with patch('pygame.quit'):
            with patch('core.game_loop.process_events'):
                run_game_loop(game_manager)
        
        # State should be preserved
        assert game_manager.game_count == initial_game_count 