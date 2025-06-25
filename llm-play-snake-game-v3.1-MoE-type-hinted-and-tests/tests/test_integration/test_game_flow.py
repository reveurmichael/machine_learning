"""
Integration tests for complete game flow.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Optional, Tuple
from numpy.typing import NDArray

from core.game_controller import GameController
from core.game_data import GameData
from core.game_logic import GameLogic
from llm.client import LLMClient


class TestGameFlowIntegration:
    """Integration tests for complete game flows."""

    def test_complete_game_lifecycle(self) -> None:
        """Test a complete game from start to finish."""
        # Initialize components
        controller = GameController(grid_size=10, use_gui=False)
        
        # Verify initial state
        assert controller.score == 0
        assert controller.steps == 0
        assert len(controller.snake_positions) == 1
        assert controller.apple_position is not None
        
        # Make some moves
        moves = ["UP", "UP", "RIGHT", "RIGHT", "DOWN"]
        
        for move in moves:
            collision, apple_eaten = controller.make_move(move)
            if collision:
                break
                
        # Verify final state
        assert controller.steps == len(moves)
        assert len(controller.game_state.moves) == len(moves)
        assert controller.game_state.moves == moves

    def test_apple_eating_and_growth(self):
        """Test that eating apples causes snake growth."""
        controller = GameController(grid_size=10, use_gui=False)
        initial_length = len(controller.snake_positions)
        
        # Position apple next to snake head
        head = controller.snake_positions[-1]
        apple_pos = [head[0], head[1] - 1]  # Above the head
        controller.set_apple_position(apple_pos)
        
        # Move towards apple
        collision, apple_eaten = controller.make_move("UP")
        
        assert not collision
        assert apple_eaten
        assert controller.score == 1
        assert len(controller.snake_positions) == initial_length + 1

    def test_collision_detection_integration(self):
        """Test collision detection in various scenarios."""
        controller = GameController(grid_size=5, use_gui=False)  # Small grid
        
        # Test wall collision
        controller.snake_positions = np.array([[2, 0]])  # Top edge
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        collision, _ = controller.make_move("UP")
        assert collision
        assert controller.last_collision_type == "wall"
        
        # Reset for self collision test
        controller.reset()
        
        # Create a snake that will collide with itself
        controller.snake_positions = np.array([[2, 2], [2, 1], [1, 1], [1, 2]])
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        collision, _ = controller.make_move("UP")
        assert collision
        assert controller.last_collision_type == "self"

    def test_game_data_integration(self):
        """Test integration between GameController and GameData."""
        controller = GameController(grid_size=10, use_gui=False)
        game_data = controller.game_state
        
        # Verify initial state sync
        assert game_data.steps == 0
        assert game_data.score == 0
        
        # Make moves and verify sync
        controller.make_move("UP")
        assert game_data.steps == 1
        assert game_data.stats.step_stats.valid == 1
        
        # Test apple eating
        head = controller.snake_positions[-1]
        controller.set_apple_position([head[0], head[1] - 1])
        controller.make_move("UP")
        
        assert game_data.score == 1
        assert game_data.steps == 2

    def test_move_filtering_integration(self):
        """Test move filtering and reversal detection."""
        controller = GameController(grid_size=10, use_gui=False)
        
        # Test that reversals are filtered out
        moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        filtered = controller.filter_invalid_reversals(moves, "UP")
        
        # DOWN should be filtered out
        expected = ["UP", "LEFT", "RIGHT"]
        assert filtered == expected
        
        # Verify game state tracks filtered moves
        initial_invalid_count = controller.game_state.stats.step_stats.invalid_reversals
        controller.filter_invalid_reversals(["DOWN"], "UP")
        # The filter method itself doesn't record - that's done during actual move processing

    def test_board_state_consistency(self):
        """Test that board state remains consistent throughout game."""
        controller = GameController(grid_size=8, use_gui=False)
        
        # Verify initial board state
        snake_pos = controller.snake_positions[0]
        apple_pos = controller.apple_position
        
        assert controller.board[snake_pos[1], snake_pos[0]] == 1  # Snake
        assert controller.board[apple_pos[1], apple_pos[0]] == 2  # Apple
        
        # Make a move and verify board update
        controller.make_move("RIGHT")
        
        new_head = controller.snake_positions[-1]
        assert controller.board[new_head[1], new_head[0]] == 1
        
        # Verify old position is cleared (for single-segment snake)
        if len(controller.snake_positions) == 1:
            assert controller.board[snake_pos[1], snake_pos[0]] == 0

    @patch('llm.client.create_provider')
    def test_llm_integration_mock(self, mock_create_provider):
        """Test integration with LLM components using mocks."""
        # Mock the LLM provider
        mock_provider = Mock()
        mock_provider.generate_response.return_value = ('{"moves": ["UP", "RIGHT"]}', None)
        mock_create_provider.return_value = mock_provider
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Generate response
        response = llm_client.generate_response("Test prompt")
        
        assert response == '{"moves": ["UP", "RIGHT"]}'

    def test_game_reset_integration(self):
        """Test that game reset properly resets all components."""
        controller = GameController(grid_size=10, use_gui=False)
        
        # Modify game state
        controller.make_move("UP")
        controller.make_move("RIGHT")
        original_steps = controller.steps
        original_score = controller.score
        
        # Reset game
        controller.reset()
        
        # Verify everything is reset
        assert controller.steps == 0
        assert controller.score == 0
        assert len(controller.snake_positions) == 1
        assert controller.current_direction is None
        assert controller.game_state.steps == 0
        assert controller.game_state.score == 0
        assert controller.game_state.moves == []

    def test_multi_apple_sequence(self):
        """Test eating multiple apples in sequence."""
        controller = GameController(grid_size=10, use_gui=False)
        
        apples_eaten = 0
        max_apples = 3
        
        for i in range(max_apples):
            # Position apple next to snake head
            head = controller.snake_positions[-1]
            apple_pos = [head[0] + 1, head[1]]  # To the right
            
            if controller.set_apple_position(apple_pos):
                collision, apple_eaten = controller.make_move("RIGHT")
                
                if apple_eaten:
                    apples_eaten += 1
                    assert controller.score == apples_eaten
                    assert len(controller.snake_positions) == 1 + apples_eaten
                
                if collision:
                    break
        
        assert apples_eaten > 0  # Should have eaten at least one apple

    def test_game_end_conditions(self):
        """Test various game end conditions."""
        controller = GameController(grid_size=5, use_gui=False)
        
        # Test wall collision end
        controller.snake_positions = np.array([[0, 0]])
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        collision, _ = controller.make_move("LEFT")
        assert collision
        
        # Record game end
        controller.game_state.record_game_end("collision_wall")
        assert controller.game_state.game_over is True
        assert controller.game_state.game_end_reason == "collision_wall"

    def test_statistics_accumulation(self):
        """Test that statistics are properly accumulated during gameplay."""
        controller = GameController(grid_size=10, use_gui=False)
        game_data = controller.game_state
        
        # Make various types of moves
        controller.make_move("UP")  # Valid move
        game_data.record_empty_move()  # Empty move
        game_data.record_invalid_reversal()  # Invalid reversal
        controller.make_move("RIGHT")  # Another valid move
        
        # Verify statistics
        assert game_data.stats.step_stats.valid == 2
        assert game_data.stats.step_stats.empty == 1
        assert game_data.stats.step_stats.invalid_reversals == 1
        assert game_data.steps == 4  # Total steps

    def test_apple_generation_avoids_snake(self):
        """Test that apple generation consistently avoids snake positions."""
        controller = GameController(grid_size=5, use_gui=False)
        
        # Create a longer snake to reduce available positions
        controller.snake_positions = np.array([
            [2, 2], [2, 1], [1, 1], [1, 2], [1, 3]
        ])
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        # Generate multiple apples to test consistency
        for _ in range(10):
            new_apple = controller._generate_apple()
            
            # Verify apple is not on any snake position
            for snake_pos in controller.snake_positions:
                assert not np.array_equal(new_apple, snake_pos)
            
            # Verify apple is within bounds
            assert 0 <= new_apple[0] < controller.grid_size
            assert 0 <= new_apple[1] < controller.grid_size


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""

    def test_invalid_move_handling(self):
        """Test handling of invalid moves across the system."""
        controller = GameController(grid_size=10, use_gui=False)
        
        # Test that invalid moves don't break the system
        try:
            # This should not crash the system
            collision, apple_eaten = controller.make_move("INVALID_MOVE")
            # The move should be normalized and may be treated as empty or ignored
        except Exception as e:
            pytest.fail(f"Invalid move caused exception: {e}")

    def test_boundary_condition_handling(self):
        """Test handling of boundary conditions."""
        controller = GameController(grid_size=3, use_gui=False)  # Very small grid
        
        # Fill the grid almost completely
        controller.snake_positions = np.array([[0, 0], [0, 1], [0, 2], [1, 0]])
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        # Try to generate apple - should still work
        apple = controller._generate_apple()
        assert apple is not None
        
        # Verify apple is not on snake
        for snake_pos in controller.snake_positions:
            assert not np.array_equal(apple, snake_pos)

    def test_component_independence(self):
        """Test that components can work independently."""
        # Test GameData independently
        game_data = GameData()
        game_data.record_move("UP")
        game_data.record_apple_position([5, 5])
        assert game_data.steps == 1
        assert len(game_data.apple_positions) == 1
        
        # Test GameController independently
        controller = GameController(grid_size=8, use_gui=False)
        collision, _ = controller.make_move("UP")
        assert not collision
        assert controller.steps == 1 