"""
Tests for GameController ↔ GameLogic interactions.

Focuses on testing how GameController and GameLogic maintain consistency
in collision detection, move validation, and game state transitions.
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch
from numpy.typing import NDArray

from core.game_controller import GameController
from core.game_logic import GameLogic
from utils.moves_utils import calculate_next_position, normalize_direction


class TestControllerLogicInteractions:
    """Test interactions between GameController and GameLogic."""

    def test_collision_detection_consistency(self) -> None:
        """Test collision detection consistency between controller and logic."""
        grid_size: int = 10
        controller: GameController = GameController(grid_size=grid_size, use_gui=False)
        game_logic: GameLogic = GameLogic(grid_size=grid_size)
        
        # Test various collision scenarios
        collision_scenarios: List[Tuple[List[List[int]], str, bool]] = [
            ([[5, 5]], "UP", False),  # Normal move
            ([[0, 0]], "UP", True),   # Wall collision (top)
            ([[0, 0]], "LEFT", True), # Wall collision (left)
            ([[9, 9]], "DOWN", True), # Wall collision (bottom)
            ([[9, 9]], "RIGHT", True), # Wall collision (right)
            ([[5, 5], [5, 4], [4, 4]], "DOWN", True), # Self collision
            ([[2, 2], [2, 3], [3, 3], [3, 2]], "UP", True), # Complex self collision
        ]
        
        for snake_positions, move, expected_collision in collision_scenarios:
            # Set up controller state
            controller.snake_positions = np.array(snake_positions, dtype=np.int_)
            controller.head_position = controller.snake_positions[-1]
            controller._update_board()
            
            # Test GameLogic collision detection
            head_pos: List[int] = snake_positions[-1]
            next_pos: List[int] = calculate_next_position(head_pos, move)
            logic_collision: bool = game_logic.check_collision(next_pos, controller.snake_positions)
            
            # Test GameController collision detection
            controller_collision: bool
            _: bool
            controller_collision, _ = controller.make_move(move)
            
            # Both should agree on collision
            assert logic_collision == expected_collision, f"Logic collision mismatch for {snake_positions} moving {move}"
            assert controller_collision == expected_collision, f"Controller collision mismatch for {snake_positions} moving {move}"
            assert logic_collision == controller_collision, f"Logic and Controller disagree for {snake_positions} moving {move}"

    def test_move_validation_alignment(self) -> None:
        """Test move validation alignment between controller and logic."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_logic: GameLogic = GameLogic(grid_size=10)
        
        # Test various move validation scenarios
        validation_scenarios: List[Tuple[str, str, bool]] = [
            ("UP", "DOWN", False),    # Direct reversal
            ("DOWN", "UP", False),    # Direct reversal
            ("LEFT", "RIGHT", False), # Direct reversal
            ("RIGHT", "LEFT", False), # Direct reversal
            ("UP", "UP", True),       # Same direction
            ("UP", "LEFT", True),     # Valid turn
            ("UP", "RIGHT", True),    # Valid turn
            ("", "UP", True),         # No previous move
            ("INVALID", "UP", True),  # Invalid previous move
        ]
        
        for previous_move, current_move, should_be_valid in validation_scenarios:
            # Test controller's move filtering
            if previous_move:
                filtered_moves: List[str] = controller.filter_invalid_reversals([current_move], previous_move)
            else:
                filtered_moves = [current_move]
            
            controller_considers_valid: bool = len(filtered_moves) > 0
            
            # Test logic's move validation (if it has such a method)
            # Note: This assumes GameLogic has move validation - adjust if needed
            logic_considers_valid: bool = True  # Default assumption
            if hasattr(game_logic, 'is_valid_move'):
                logic_considers_valid = game_logic.is_valid_move(current_move, previous_move)
            elif hasattr(game_logic, 'validate_move'):
                logic_considers_valid = game_logic.validate_move(current_move, previous_move)
            
            # Both should agree on validity
            assert controller_considers_valid == should_be_valid, f"Controller validation failed for {previous_move} -> {current_move}"
            
            # If logic has validation, it should agree with controller
            if hasattr(game_logic, 'is_valid_move') or hasattr(game_logic, 'validate_move'):
                assert logic_considers_valid == controller_considers_valid, f"Logic and Controller disagree on {previous_move} -> {current_move}"

    def test_board_state_synchronization(self) -> None:
        """Test board state synchronization between controller and logic."""
        grid_size: int = 8
        controller: GameController = GameController(grid_size=grid_size, use_gui=False)
        game_logic: GameLogic = GameLogic(grid_size=grid_size)
        
        # Make several moves and check board consistency
        moves: List[str] = ["UP", "RIGHT", "RIGHT", "DOWN", "DOWN", "LEFT"]
        
        for i, move in enumerate(moves):
            # Capture state before move
            pre_snake_positions: NDArray[np.int_] = controller.snake_positions.copy()
            
            # Make move in controller
            collision: bool
            apple_eaten: bool
            collision, apple_eaten = controller.make_move(move)
            
            # Test board state consistency
            if not collision:
                # Controller's board should reflect snake positions
                for pos in controller.snake_positions:
                    assert controller.board[pos[0], pos[1]] == 1, f"Snake position {pos} not marked on board"
                
                # Apple position should be marked
                apple_pos: NDArray[np.int_] = controller.apple_position
                assert controller.board[apple_pos[0], apple_pos[1]] == 2, f"Apple position {apple_pos} not marked on board"
                
                # GameLogic should be able to verify this state
                if hasattr(game_logic, 'validate_board_state'):
                    is_valid: bool = game_logic.validate_board_state(controller.board, controller.snake_positions, controller.apple_position)
                    assert is_valid, f"GameLogic considers board state invalid after move {i}"
            
            if collision:
                break

    def test_apple_placement_logic_consistency(self) -> None:
        """Test apple placement consistency between controller and logic."""
        grid_size: int = 6
        controller: GameController = GameController(grid_size=grid_size, use_gui=False)
        game_logic: GameLogic = GameLogic(grid_size=grid_size)
        
        # Fill most of the grid with snake
        large_snake: List[List[int]] = []
        for x in range(grid_size):
            for y in range(grid_size):
                if len(large_snake) < grid_size * grid_size - 3:  # Leave 3 free spaces
                    large_snake.append([x, y])
        
        controller.snake_positions = np.array(large_snake, dtype=np.int_)
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        # Test apple generation multiple times
        for _ in range(20):
            apple: NDArray[np.int_] = controller._generate_apple()
            
            # Apple should not be on snake (controller logic)
            snake_collision: bool = False
            for snake_pos in controller.snake_positions:
                if np.array_equal(apple, snake_pos):
                    snake_collision = True
                    break
            assert not snake_collision, f"Apple {apple} placed on snake"
            
            # Apple should be within bounds (logic validation)
            assert 0 <= apple[0] < grid_size, f"Apple x-coordinate {apple[0]} out of bounds"
            assert 0 <= apple[1] < grid_size, f"Apple y-coordinate {apple[1]} out of bounds"
            
            # GameLogic should validate apple placement
            if hasattr(game_logic, 'is_valid_apple_position'):
                is_valid: bool = game_logic.is_valid_apple_position(apple, controller.snake_positions)
                assert is_valid, f"GameLogic considers apple position {apple} invalid"

    def test_game_over_condition_agreement(self) -> None:
        """Test game over condition agreement between controller and logic."""
        controller: GameController = GameController(grid_size=5, use_gui=False)
        game_logic: GameLogic = GameLogic(grid_size=5)
        
        # Test various game over scenarios
        game_over_scenarios: List[Tuple[List[List[int]], str, str]] = [
            ([[0, 0]], "UP", "wall_collision"),
            ([[4, 4]], "DOWN", "wall_collision"),
            ([[2, 2], [2, 1], [1, 1], [1, 2]], "UP", "self_collision"),
            ([[1, 1], [1, 2], [2, 2], [2, 1]], "DOWN", "self_collision"),
        ]
        
        for snake_positions, move, expected_reason in game_over_scenarios:
            # Reset and set up scenario
            controller.reset()
            controller.snake_positions = np.array(snake_positions, dtype=np.int_)
            controller.head_position = controller.snake_positions[-1]
            controller._update_board()
            
            # Test controller game over detection
            collision: bool
            _: bool
            collision, _ = controller.make_move(move)
            
            assert collision, f"Controller did not detect collision for {snake_positions} moving {move}"
            
            # Test logic game over detection
            head_pos: List[int] = snake_positions[-1]
            next_pos: List[int] = calculate_next_position(head_pos, move)
            logic_collision: bool = game_logic.check_collision(next_pos, controller.snake_positions)
            
            assert logic_collision, f"GameLogic did not detect collision for {snake_positions} moving {move}"
            
            # Both should agree
            assert collision == logic_collision, f"Controller and Logic disagree on game over for {snake_positions} moving {move}"

    def test_complex_snake_movement_patterns(self) -> None:
        """Test complex snake movement patterns between controller and logic."""
        controller: GameController = GameController(grid_size=12, use_gui=False)
        game_logic: GameLogic = GameLogic(grid_size=12)
        
        # Test complex movement patterns
        movement_patterns: List[Tuple[List[str], str]] = [
            (["UP", "RIGHT", "DOWN", "LEFT"] * 10, "square_pattern"),
            (["UP"] * 5 + ["RIGHT"] * 5 + ["DOWN"] * 5 + ["LEFT"] * 5, "large_square"),
            (["UP", "UP", "RIGHT", "DOWN", "DOWN", "LEFT"] * 8, "zigzag_pattern"),
            (["RIGHT"] * 8 + ["UP"] * 8 + ["LEFT"] * 8 + ["DOWN"] * 8, "clockwise_box"),
        ]
        
        for moves, pattern_name in movement_patterns:
            controller.reset()
            valid_moves: int = 0
            controller_states: List[NDArray[np.int_]] = []
            
            for move in moves:
                # Record state before move
                pre_positions: NDArray[np.int_] = controller.snake_positions.copy()
                controller_states.append(pre_positions)
                
                # Make move in controller
                collision: bool
                apple_eaten: bool
                collision, apple_eaten = controller.make_move(move)
                valid_moves += 1
                
                if not collision:
                    # Verify logic would agree with the move
                    head_pos: List[int] = pre_positions[-1].tolist()
                    next_pos: List[int] = calculate_next_position(head_pos, move)
                    
                    logic_collision: bool = game_logic.check_collision(next_pos, pre_positions)
                    assert not logic_collision, f"Logic thinks move {valid_moves} in {pattern_name} should collide"
                    
                    # Verify new position is correct
                    new_head: NDArray[np.int_] = controller.snake_positions[-1]
                    expected_head: NDArray[np.int_] = np.array(next_pos, dtype=np.int_)
                    assert np.array_equal(new_head, expected_head), f"Head position incorrect in {pattern_name} move {valid_moves}"
                
                if collision:
                    # Verify logic agrees on collision
                    head_pos = pre_positions[-1].tolist()
                    next_pos = calculate_next_position(head_pos, move)
                    logic_collision = game_logic.check_collision(next_pos, pre_positions)
                    assert logic_collision, f"Logic disagrees on collision in {pattern_name} move {valid_moves}"
                    break
            
            assert valid_moves > 0, f"No valid moves made in {pattern_name}"

    def test_boundary_condition_edge_cases(self) -> None:
        """Test boundary condition edge cases between controller and logic."""
        grid_size: int = 3  # Small grid for edge cases
        controller: GameController = GameController(grid_size=grid_size, use_gui=False)
        game_logic: GameLogic = GameLogic(grid_size=grid_size)
        
        # Test edge positions and moves
        edge_scenarios: List[Tuple[List[int], List[str], int]] = [
            ([0, 0], ["UP", "LEFT", "DOWN", "RIGHT"], 2),  # Corner position
            ([1, 0], ["UP", "LEFT", "RIGHT"], 1),          # Edge position
            ([2, 2], ["DOWN", "RIGHT", "UP", "LEFT"], 2),  # Opposite corner
            ([1, 1], ["UP", "DOWN", "LEFT", "RIGHT"], 0),  # Center position
        ]
        
        for start_pos, test_moves, expected_collisions in edge_scenarios:
            collision_count: int = 0
            
            for move in test_moves:
                controller.reset()
                controller.snake_positions = np.array([start_pos], dtype=np.int_)
                controller.head_position = controller.snake_positions[-1]
                controller._update_board()
                
                # Test controller collision detection
                collision: bool
                _: bool
                collision, _ = controller.make_move(move)
                
                # Test logic collision detection
                next_pos: List[int] = calculate_next_position(start_pos, move)
                logic_collision: bool = game_logic.check_collision(next_pos, controller.snake_positions)
                
                # Both should agree
                assert collision == logic_collision, f"Disagreement on {start_pos} moving {move}"
                
                if collision:
                    collision_count += 1
            
            assert collision_count == expected_collisions, f"Expected {expected_collisions} collisions from {start_pos}, got {collision_count}"

    def test_state_corruption_detection(self) -> None:
        """Test state corruption detection between controller and logic."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        game_logic: GameLogic = GameLogic(grid_size=10)
        
        # Test various state corruption scenarios
        corruption_scenarios: List[Tuple[str, callable]] = [
            ("duplicate_positions", lambda: controller.snake_positions.__setitem__(slice(None), 
                                                                                  np.array([[5, 5], [5, 5]], dtype=np.int_))),
            ("out_of_bounds", lambda: controller.snake_positions.__setitem__(slice(None), 
                                                                            np.array([[15, 15]], dtype=np.int_))),
            ("negative_position", lambda: controller.snake_positions.__setitem__(slice(None), 
                                                                                np.array([[-1, -1]], dtype=np.int_))),
            ("disconnected_snake", lambda: controller.snake_positions.__setitem__(slice(None), 
                                                                                  np.array([[5, 5], [7, 7]], dtype=np.int_))),
        ]
        
        for scenario_name, corruption_func in corruption_scenarios:
            try:
                # Start with valid state
                controller.reset()
                controller.make_move("UP")  # Establish some movement
                
                # Apply corruption
                corruption_func()
                controller._update_board()
                
                # Test if logic can detect corruption
                if hasattr(game_logic, 'validate_snake_positions'):
                    is_valid: bool = game_logic.validate_snake_positions(controller.snake_positions)
                    
                    if scenario_name in ["duplicate_positions", "disconnected_snake"]:
                        assert not is_valid, f"Logic should detect {scenario_name} as invalid"
                    elif scenario_name in ["out_of_bounds", "negative_position"]:
                        assert not is_valid, f"Logic should detect {scenario_name} as invalid"
                
                # Try to continue with corrupted state
                collision: bool
                _: bool
                collision, _ = controller.make_move("RIGHT")
                
                # System should handle corruption gracefully
                assert isinstance(collision, bool)
                
            except Exception as e:
                # Should be informative error about state corruption
                assert isinstance(e, (ValueError, IndexError, TypeError))
                assert len(str(e)) > 0
            
            # Reset should always restore valid state
            controller.reset()
            assert len(controller.snake_positions) == 1
            assert 0 <= controller.snake_positions[0][0] < controller.grid_size
            assert 0 <= controller.snake_positions[0][1] < controller.grid_size 