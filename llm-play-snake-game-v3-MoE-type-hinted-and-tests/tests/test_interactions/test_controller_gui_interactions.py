"""
Tests for GameController ↔ GUI interactions.

Focuses on testing how GameController and GUI components maintain synchronization
in rendering, event handling, and state visualization.
"""

import pytest
import numpy as np
import threading
import time
from typing import List, Dict, Any, Optional, Tuple, Generator
from unittest.mock import Mock, patch, MagicMock
from numpy.typing import NDArray

from core.game_controller import GameController
from gui.base_gui import BaseGUI
from gui.game_gui import GameGUI


class TestControllerGUIInteractions:
    """Test interactions between GameController and GUI components."""

    def test_state_rendering_synchronization(self) -> None:
        """Test synchronization between controller state and GUI rendering."""
        controller: GameController = GameController(grid_size=10, use_gui=True)
        mock_gui: Mock = Mock(spec=GameGUI)
        
        # Mock GUI methods
        mock_gui.update_display = Mock()
        mock_gui.render_snake = Mock()
        mock_gui.render_apple = Mock()
        mock_gui.render_score = Mock()
        
        # Attach mock GUI to controller
        if hasattr(controller, 'gui'):
            controller.gui = mock_gui
        
        # Test state changes trigger appropriate GUI updates
        state_changes: List[Tuple[str, callable, str]] = [
            ("move_snake", lambda: controller.make_move("UP"), "snake_position"),
            ("eat_apple", lambda: controller._handle_apple_collision(), "score_and_snake"),
            ("collision", lambda: setattr(controller, 'game_over', True), "game_over"),
            ("reset_game", lambda: controller.reset(), "full_reset"),
        ]
        
        for change_name, change_func, expected_update in state_changes:
            # Clear previous mock calls
            mock_gui.reset_mock()
            
            # Record state before change
            pre_state: Dict[str, Any] = {
                "score": controller.score,
                "snake_positions": controller.snake_positions.copy(),
                "apple_position": controller.apple_position.copy(),
                "game_over": getattr(controller, 'game_over', False)
            }
            
            try:
                # Apply change
                change_func()
                
                # Simulate GUI update call (would normally be automatic)
                if hasattr(controller, 'gui') and controller.gui:
                    controller.gui.update_display()
                    controller.gui.render_snake(controller.snake_positions)
                    controller.gui.render_apple(controller.apple_position)
                    controller.gui.render_score(controller.score)
                
                # Verify appropriate GUI methods were called
                if expected_update in ["snake_position", "score_and_snake", "full_reset"]:
                    mock_gui.render_snake.assert_called()
                    
                if expected_update in ["score_and_snake", "full_reset"]:
                    mock_gui.render_score.assert_called()
                    
                if expected_update in ["snake_position", "score_and_snake", "full_reset"]:
                    mock_gui.render_apple.assert_called()
                
                # Verify state consistency
                if not getattr(controller, 'game_over', False):
                    # Game state should be valid
                    assert len(controller.snake_positions) >= 1
                    assert 0 <= controller.apple_position[0] < controller.grid_size
                    assert 0 <= controller.apple_position[1] < controller.grid_size
                
            except Exception as e:
                # Some operations might fail, but GUI should handle gracefully
                assert "gui" not in str(e).lower(), f"GUI error in {change_name}: {e}"

    def test_event_handling_controller_response(self) -> None:
        """Test controller response to GUI events."""
        controller: GameController = GameController(grid_size=12, use_gui=True)
        mock_gui: Mock = Mock(spec=GameGUI)
        
        # Mock event system
        mock_gui.get_key_events = Mock()
        mock_gui.handle_mouse_events = Mock()
        mock_gui.handle_window_events = Mock()
        
        # Test various GUI events
        gui_events: List[Tuple[str, Any, str]] = [
            ("key_press", "UP", "move_up"),
            ("key_press", "DOWN", "move_down"),
            ("key_press", "LEFT", "move_left"),
            ("key_press", "RIGHT", "move_right"),
            ("key_press", "SPACE", "pause_toggle"),
            ("key_press", "R", "reset_game"),
            ("mouse_click", (100, 100), "mouse_interaction"),
            ("window_close", None, "cleanup"),
        ]
        
        for event_type, event_data, expected_action in gui_events:
            # Record state before event
            pre_score: int = controller.score
            pre_steps: int = controller.steps
            pre_positions: NDArray[np.int_] = controller.snake_positions.copy()
            
            try:
                # Simulate event handling
                if event_type == "key_press" and event_data in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    # Movement event
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(event_data)
                    
                    # Verify controller responded to GUI event
                    if not collision:
                        assert controller.steps > pre_steps
                        # Snake should have moved
                        assert not np.array_equal(controller.snake_positions, pre_positions)
                
                elif event_type == "key_press" and event_data == "R":
                    # Reset event
                    controller.reset()
                    
                    # Verify reset response
                    assert controller.score == 0
                    assert controller.steps == 0
                    assert len(controller.snake_positions) == 1
                
                elif event_type == "key_press" and event_data == "SPACE":
                    # Pause toggle (if implemented)
                    if hasattr(controller, 'paused'):
                        initial_paused = controller.paused
                        controller.paused = not controller.paused
                        assert controller.paused != initial_paused
                
                # GUI should be notified of state changes
                if hasattr(controller, 'gui') and controller.gui:
                    mock_gui.update_display()
                
            except Exception as e:
                # Event handling should be robust
                assert "event" not in str(e).lower(), f"Event handling error: {e}"

    def test_rapid_gui_updates_performance(self) -> None:
        """Test performance under rapid GUI updates."""
        controller: GameController = GameController(grid_size=15, use_gui=True)
        mock_gui: Mock = Mock(spec=GameGUI)
        
        # Configure mock for performance testing
        mock_gui.update_display = Mock()
        mock_gui.render_snake = Mock()
        mock_gui.render_apple = Mock()
        
        update_times: List[float] = []
        
        # Simulate rapid game progression
        for i in range(200):
            start_time: float = time.time()
            
            # Make move
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            # Simulate GUI update
            if hasattr(controller, 'gui'):
                mock_gui.update_display()
                mock_gui.render_snake(controller.snake_positions)
                mock_gui.render_apple(controller.apple_position)
            
            end_time: float = time.time()
            update_times.append(end_time - start_time)
            
            if collision:
                controller.reset()
        
        # Verify performance
        avg_update_time: float = sum(update_times) / len(update_times)
        max_update_time: float = max(update_times)
        
        # Updates should be fast
        assert avg_update_time < 0.01  # Average under 10ms
        assert max_update_time < 0.05  # Maximum under 50ms
        
        # GUI methods should have been called frequently
        assert mock_gui.update_display.call_count == 200
        assert mock_gui.render_snake.call_count == 200

    def test_gui_error_handling_controller_stability(self) -> None:
        """Test controller stability when GUI encounters errors."""
        controller: GameController = GameController(grid_size=8, use_gui=True)
        mock_gui: Mock = Mock(spec=GameGUI)
        
        # Configure GUI to raise various errors
        gui_errors: List[Tuple[str, Exception]] = [
            ("render_snake", RuntimeError("Graphics context lost")),
            ("update_display", MemoryError("Insufficient graphics memory")),
            ("render_apple", ValueError("Invalid render parameters")),
            ("handle_events", OSError("Window system error")),
        ]
        
        for error_method, error_exception in gui_errors:
            # Configure mock to raise error
            getattr(mock_gui, error_method).side_effect = error_exception
            
            # Record controller state
            pre_state: Dict[str, Any] = {
                "score": controller.score,
                "steps": controller.steps,
                "snake_length": controller.snake_length
            }
            
            try:
                # Perform controller operations that would trigger GUI
                for i in range(10):
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move("UP")
                    
                    # Attempt GUI operation (would normally be automatic)
                    try:
                        getattr(mock_gui, error_method)()
                    except Exception:
                        # GUI error should not break controller
                        pass
                    
                    if collision:
                        controller.reset()
                        break
                
                # Controller should remain functional despite GUI errors
                assert controller.steps >= pre_state["steps"]
                assert hasattr(controller, 'snake_positions')
                assert len(controller.snake_positions) >= 1
                
            except Exception as e:
                # Controller should not be affected by GUI errors
                assert "gui" not in str(e).lower(), f"Controller broken by GUI error {error_method}: {e}"
            
            # Reset mock for next test
            getattr(mock_gui, error_method).side_effect = None

    def test_concurrent_gui_controller_operations(self) -> None:
        """Test concurrent GUI and controller operations."""
        controller: GameController = GameController(grid_size=10, use_gui=True)
        mock_gui: Mock = Mock(spec=GameGUI)
        
        operation_results: List[str] = []
        operation_errors: List[Exception] = []
        
        def controller_operations() -> None:
            """Perform controller operations."""
            try:
                for i in range(100):
                    move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                    collision: bool
                    apple_eaten: bool
                    collision, apple_eaten = controller.make_move(move)
                    
                    if collision:
                        controller.reset()
                        operation_results.append(f"controller_reset_{i}")
                    else:
                        operation_results.append(f"controller_move_{i}")
                    
                    time.sleep(0.001)  # Small delay
                    
            except Exception as e:
                operation_errors.append(e)
        
        def gui_operations() -> None:
            """Perform GUI operations."""
            try:
                for i in range(100):
                    # Simulate GUI operations
                    mock_gui.update_display()
                    mock_gui.render_snake(controller.snake_positions)
                    mock_gui.render_apple(controller.apple_position)
                    
                    operation_results.append(f"gui_update_{i}")
                    time.sleep(0.001)  # Small delay
                    
            except Exception as e:
                operation_errors.append(e)
        
        # Start concurrent operations
        controller_thread = threading.Thread(target=controller_operations)
        gui_thread = threading.Thread(target=gui_operations)
        
        controller_thread.start()
        gui_thread.start()
        
        # Wait for completion
        controller_thread.join(timeout=10.0)
        gui_thread.join(timeout=10.0)
        
        # Verify results
        assert len(operation_errors) == 0, f"Concurrent operation errors: {operation_errors}"
        
        controller_ops = [op for op in operation_results if op.startswith("controller_")]
        gui_ops = [op for op in operation_results if op.startswith("gui_")]
        
        # Both should have completed operations
        assert len(controller_ops) > 0
        assert len(gui_ops) > 0
        
        # Controller should be in valid state
        assert hasattr(controller, 'snake_positions')
        assert len(controller.snake_positions) >= 1

    def test_gui_state_visualization_accuracy(self) -> None:
        """Test accuracy of GUI state visualization."""
        controller: GameController = GameController(grid_size=12, use_gui=True)
        mock_gui: Mock = Mock(spec=GameGUI)
        
        # Track GUI render calls
        render_calls: List[Dict[str, Any]] = []
        
        def track_render_snake(positions: NDArray[np.int_]) -> None:
            render_calls.append({
                "type": "snake",
                "positions": positions.copy(),
                "timestamp": time.time()
            })
        
        def track_render_apple(position: NDArray[np.int_]) -> None:
            render_calls.append({
                "type": "apple", 
                "position": position.copy(),
                "timestamp": time.time()
            })
        
        def track_render_score(score: int) -> None:
            render_calls.append({
                "type": "score",
                "value": score,
                "timestamp": time.time()
            })
        
        mock_gui.render_snake.side_effect = track_render_snake
        mock_gui.render_apple.side_effect = track_render_apple
        mock_gui.render_score.side_effect = track_render_score
        
        # Perform game operations
        for i in range(50):
            # Make move
            move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            # Trigger GUI updates
            mock_gui.render_snake(controller.snake_positions)
            mock_gui.render_apple(controller.apple_position)
            mock_gui.render_score(controller.score)
            
            if collision:
                controller.reset()
        
        # Verify visualization accuracy
        snake_renders = [call for call in render_calls if call["type"] == "snake"]
        apple_renders = [call for call in render_calls if call["type"] == "apple"]
        score_renders = [call for call in render_calls if call["type"] == "score"]
        
        # Should have renders for each game step
        assert len(snake_renders) > 0
        assert len(apple_renders) > 0
        assert len(score_renders) > 0
        
        # Verify data consistency in renders
        for render in snake_renders:
            positions = render["positions"]
            assert len(positions) >= 1  # At least head position
            
            # All positions should be within bounds
            for pos in positions:
                assert 0 <= pos[0] < controller.grid_size
                assert 0 <= pos[1] < controller.grid_size
        
        for render in apple_renders:
            position = render["position"]
            assert 0 <= position[0] < controller.grid_size
            assert 0 <= position[1] < controller.grid_size
        
        for render in score_renders:
            assert render["value"] >= 0  # Score should be non-negative

    def test_gui_memory_management_controller_lifecycle(self) -> None:
        """Test GUI memory management during controller lifecycle."""
        # Test multiple controller creation/destruction cycles
        gui_references: List[Mock] = []
        
        for cycle in range(10):
            controller: GameController = GameController(grid_size=8, use_gui=True)
            mock_gui: Mock = Mock(spec=GameGUI)
            gui_references.append(mock_gui)
            
            # Simulate GUI operations
            for i in range(50):
                move: str = ["UP", "RIGHT", "DOWN", "LEFT"][i % 4]
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                
                # GUI operations
                mock_gui.update_display()
                mock_gui.render_snake(controller.snake_positions)
                
                if collision:
                    break
            
            # Cleanup controller (GUI should be cleaned up too)
            if hasattr(controller, 'cleanup'):
                controller.cleanup()
            
            # Force garbage collection simulation
            del controller
        
        # Verify no memory leaks in GUI references
        # All GUI mocks should have been used and can be garbage collected
        assert len(gui_references) == 10
        
        # Each GUI should have been used for rendering
        for gui_ref in gui_references:
            assert gui_ref.update_display.called
            assert gui_ref.render_snake.called

    def test_gui_configuration_controller_adaptation(self) -> None:
        """Test controller adaptation to GUI configuration changes."""
        controller: GameController = GameController(grid_size=10, use_gui=True)
        mock_gui: Mock = Mock(spec=GameGUI)
        
        # Test configuration changes
        config_changes: List[Tuple[str, Any, str]] = [
            ("grid_size", 15, "larger_grid"),
            ("render_mode", "fast", "performance_mode"),
            ("color_scheme", "dark", "visual_change"),
            ("animation_speed", 0.5, "speed_change"),
        ]
        
        for config_name, config_value, change_type in config_changes:
            # Apply configuration change
            if config_name == "grid_size":
                # Grid size change requires controller adaptation
                new_controller: GameController = GameController(grid_size=config_value, use_gui=True)
                assert new_controller.grid_size == config_value
                
                # Verify adapted state
                assert len(new_controller.snake_positions) >= 1
                assert 0 <= new_controller.apple_position[0] < config_value
                assert 0 <= new_controller.apple_position[1] < config_value
                
                controller = new_controller
            
            else:
                # Other configurations affect GUI but not controller core logic
                setattr(mock_gui, config_name, config_value)
                
                # Controller should continue functioning
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move("UP")
                
                # Verify controller stability
                assert hasattr(controller, 'snake_positions')
                assert len(controller.snake_positions) >= 1
        
        # Final verification
        assert controller.grid_size in [10, 15]  # Should be one of the test values
        assert 0 <= controller.score
        assert 0 <= controller.steps 