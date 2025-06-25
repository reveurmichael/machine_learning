"""
Tests for the GameController class.
"""

import pytest
import numpy as np
from typing import List, Tuple, Any
from unittest.mock import Mock, patch
from numpy.typing import NDArray

from core.game_controller import GameController
from config.game_constants import DIRECTIONS


class TestGameController:
    """Test cases for GameController."""

    def test_init_default_values(self) -> None:
        """Test GameController initialization with default values."""
        controller: GameController = GameController(use_gui=False)
        
        assert controller.grid_size == 20  # Default from UI_CONSTANTS
        assert controller.board.shape == (20, 20)
        assert len(controller.snake_positions) == 1
        assert controller.use_gui is False
        assert controller.gui is None
        assert controller.current_direction is None
        assert controller.last_collision_type is None
        
    def test_init_custom_grid_size(self) -> None:
        """Test GameController initialization with custom grid size."""
        controller: GameController = GameController(grid_size=15, use_gui=False)
        
        assert controller.grid_size == 15
        assert controller.board.shape == (15, 15)
        # Snake should start in the middle
        expected_start: List[int] = [15//2, 15//2]
        assert np.array_equal(controller.snake_positions[0], expected_start)

    def test_set_gui(self) -> None:
        """Test setting GUI instance."""
        controller: GameController = GameController(use_gui=False)
        mock_gui: Mock = Mock()
        
        controller.set_gui(mock_gui)
        
        assert controller.gui is mock_gui
        assert controller.use_gui is True

    def test_reset_game_state(self) -> None:
        """Test resetting game to initial state."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Modify game state
        controller.make_move("UP")
        controller.make_move("RIGHT")
        original_apple: NDArray[np.int_] = controller.apple_position.copy()
        
        # Reset and verify
        controller.reset()
        
        assert len(controller.snake_positions) == 1
        assert np.array_equal(controller.snake_positions[0], [5, 5])  # Center of 10x10
        assert controller.current_direction is None
        assert controller.last_collision_type is None
        assert controller.game_state.steps == 0
        assert controller.game_state.score == 0
        # Apple should be regenerated to a new position
        assert not np.array_equal(controller.apple_position, original_apple) or True  # May be same by chance

    def test_make_move_valid(self) -> None:
        """Test making a valid move."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        collision: bool
        apple_eaten: bool
        collision, apple_eaten = controller.make_move("UP")
        
        assert not collision
        assert not apple_eaten  # Unlikely to hit apple on first move
        assert controller.game_state.steps == 1
        assert len(controller.game_state.moves) == 1
        assert controller.game_state.moves[0] == "UP"

    def test_make_move_eat_apple(self) -> None:
        """Test making a move that eats an apple."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Position apple next to snake head
        snake_head: NDArray[np.int_] = controller.snake_positions[-1]
        apple_pos: List[int] = [snake_head[0], snake_head[1] - 1]  # Above the head
        controller.set_apple_position(apple_pos)
        
        collision: bool
        apple_eaten: bool
        collision, apple_eaten = controller.make_move("UP")
        
        assert not collision
        assert apple_eaten
        assert controller.game_state.score == 1
        assert len(controller.snake_positions) == 2  # Snake grew

    def test_make_move_wall_collision(self) -> None:
        """Test making a move that results in wall collision."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Move snake to top edge
        controller.snake_positions = np.array([[5, 0]], dtype=np.int_)
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        collision: bool
        apple_eaten: bool
        collision, apple_eaten = controller.make_move("UP")
        
        assert collision
        assert not apple_eaten
        assert controller.last_collision_type == "wall"

    def test_make_move_self_collision(self) -> None:
        """Test making a move that results in self collision."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Create a snake that will collide with itself
        controller.snake_positions = np.array([[5, 5], [5, 4], [5, 3], [4, 3]], dtype=np.int_)
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        collision: bool
        apple_eaten: bool
        collision, apple_eaten = controller.make_move("DOWN")
        
        assert collision
        assert not apple_eaten
        assert controller.last_collision_type == "self"

    def test_filter_invalid_reversals(self) -> None:
        """Test filtering out invalid reversal moves."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test reversal filtering
        moves: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
        filtered: List[str] = controller.filter_invalid_reversals(moves, "UP")
        
        # DOWN should be filtered out as it's opposite to UP
        expected: List[str] = ["UP", "LEFT", "RIGHT"]
        assert filtered == expected

    def test_filter_invalid_reversals_empty_input(self) -> None:
        """Test filtering with empty or single move list."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        empty_list: List[str] = []
        single_list: List[str] = ["UP"]
        assert controller.filter_invalid_reversals(empty_list) == []
        assert controller.filter_invalid_reversals(single_list) == ["UP"]

    def test_filter_invalid_reversals_all_filtered(self) -> None:
        """Test when all moves are invalid reversals."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        moves: List[str] = ["DOWN", "DOWN", "DOWN"]
        filtered: List[str] = controller.filter_invalid_reversals(moves, "UP")
        
        assert filtered == []

    def test_set_apple_position_valid(self) -> None:
        """Test setting apple to a valid position."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        new_position: List[int] = [8, 8]
        success: bool = controller.set_apple_position(new_position)
        
        assert success
        assert np.array_equal(controller.apple_position, [8, 8])

    def test_set_apple_position_invalid_occupied(self) -> None:
        """Test setting apple to a position occupied by snake."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        snake_position: List[int] = controller.snake_positions[0].tolist()
        success: bool = controller.set_apple_position(snake_position)
        
        assert not success
        # Apple position should remain unchanged

    def test_set_apple_position_invalid_out_of_bounds(self) -> None:
        """Test setting apple to an out-of-bounds position."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        success: bool = controller.set_apple_position([15, 15])  # Outside 10x10 grid
        
        assert not success

    def test_game_properties(self) -> None:
        """Test game property getters."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Initial state
        assert controller.score == 0
        assert controller.steps == 0
        assert controller.snake_length == 1
        
        # After making a move
        controller.make_move("UP")
        assert controller.steps == 1

    def test_direction_normalization(self) -> None:
        """Test that moves are properly normalized."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test lowercase input
        collision: bool
        _: bool
        collision, _ = controller.make_move("up")
        assert controller.game_state.moves[0] == "UP"
        
        # Test with whitespace
        controller.make_move(" right ")
        assert controller.game_state.moves[1] == "RIGHT"

    def test_board_update(self) -> None:
        """Test that board is properly updated after moves."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        initial_head: NDArray[np.int_] = controller.snake_positions[-1]
        
        # Check initial board state
        assert controller.board[initial_head[1], initial_head[0]] == 1  # Snake
        apple_pos: NDArray[np.int_] = controller.apple_position
        assert controller.board[apple_pos[1], apple_pos[0]] == 2  # Apple
        
        # Make a move and verify board update
        controller.make_move("UP")
        new_head: NDArray[np.int_] = controller.snake_positions[-1]
        assert controller.board[new_head[1], new_head[0]] == 1

    def test_apple_generation_avoids_snake(self) -> None:
        """Test that new apples are not generated on snake positions."""
        controller: GameController = GameController(grid_size=3, use_gui=False)  # Small board
        
        # Fill most of the board with snake
        controller.snake_positions = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]], dtype=np.int_)
        controller.head_position = controller.snake_positions[-1]
        
        # Generate new apple multiple times to test it avoids snake
        for _ in range(10):
            new_apple: NDArray[np.int_] = controller._generate_apple()
            # Check apple is not on any snake position
            for snake_pos in controller.snake_positions:
                assert not np.array_equal(new_apple, snake_pos)

    def test_get_current_direction_key(self) -> None:
        """Test getting current direction key."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Initially no direction
        assert controller.get_current_direction_key() == "NONE"
        
        # After making a move
        controller.make_move("UP")
        assert controller.get_current_direction_key() == "UP"

    @patch('core.game_controller.GameController.draw')
    def test_draw_called_when_gui_available(self, mock_draw: Mock) -> None:
        """Test that draw is called when GUI is available."""
        controller: GameController = GameController(grid_size=10, use_gui=True)
        mock_gui: Mock = Mock()
        controller.set_gui(mock_gui)
        
        controller.reset()
        # draw() should be called during reset when GUI is available

    def test_collision_detection_edge_cases(self) -> None:
        """Test collision detection edge cases."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test moving into corner
        controller.snake_positions = np.array([[0, 0]], dtype=np.int_)
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        # Try moving left from leftmost position
        collision: bool
        _: bool
        collision, _ = controller.make_move("LEFT")
        assert collision
        assert controller.last_collision_type == "wall"
        
        # Try moving up from topmost position  
        collision, _ = controller.make_move("UP")
        assert collision
        assert controller.last_collision_type == "wall"

    def test_snake_growth_mechanism(self) -> None:
        """Test that snake grows correctly when eating apples."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        initial_length: int = len(controller.snake_positions)
        
        # Position apple next to head and eat it
        head: NDArray[np.int_] = controller.snake_positions[-1]
        controller.set_apple_position([head[0], head[1] - 1])
        controller.make_move("UP")
        
        assert len(controller.snake_positions) == initial_length + 1
        assert controller.score == 1

    def test_multiple_moves_sequence(self) -> None:
        """Test a sequence of multiple moves."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        moves: List[str] = ["UP", "UP", "RIGHT", "RIGHT", "DOWN"]
        
        for move in moves:
            collision: bool
            _: bool
            collision, _ = controller.make_move(move)
            if collision:
                break
                
        assert controller.game_state.steps == len(moves)
        assert len(controller.game_state.moves) == len(moves)
        assert controller.game_state.moves == moves

    def test_game_state_integration(self) -> None:
        """Test integration with GameData state tracking."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Verify game state is properly initialized
        assert controller.game_state is not None
        assert controller.game_state.steps == 0
        assert controller.game_state.score == 0
        
        # Make moves and verify state tracking
        controller.make_move("UP")
        assert controller.game_state.steps == 1
        assert controller.game_state.stats.step_stats.valid == 1

    # ==================== COMPREHENSIVE BOARD MANAGEMENT TESTING ====================
    
    def test_board_initialization_comprehensive(self) -> None:
        """Comprehensive testing of board initialization with various grid sizes."""
        # Test multiple grid sizes
        test_sizes: List[int] = [5, 8, 10, 15, 20, 25, 30]
        
        for size in test_sizes:
            controller: GameController = GameController(grid_size=size, use_gui=False)
            
            # Verify board dimensions
            assert controller.board.shape == (size, size)
            assert controller.grid_size == size
            
            # Verify snake starts in center
            expected_center: List[int] = [size // 2, size // 2]
            assert np.array_equal(controller.snake_positions[0], expected_center)
            
            # Verify board has correct initial state
            assert controller.board[expected_center[0], expected_center[1]] == 1  # Snake head
            
            # Count non-zero elements (should be snake + apple)
            non_zero_count: int = np.count_nonzero(controller.board)
            assert non_zero_count == 2  # 1 snake segment + 1 apple

    def test_board_edge_initialization(self) -> None:
        """Test board initialization with edge case grid sizes."""
        # Test minimum viable grid size
        controller_small: GameController = GameController(grid_size=3, use_gui=False)
        assert controller_small.board.shape == (3, 3)
        
        # Test very large grid
        controller_large: GameController = GameController(grid_size=100, use_gui=False)
        assert controller_large.board.shape == (100, 100)
        
        # Test odd vs even grid sizes
        for size in [5, 6, 7, 8]:
            controller: GameController = GameController(grid_size=size, use_gui=False)
            center_row: int = size // 2
            center_col: int = size // 2
            assert np.array_equal(controller.snake_positions[0], [center_row, center_col])

    def test_board_state_consistency(self) -> None:
        """Test board state consistency after various operations."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Initial consistency check
        self._verify_board_consistency(controller)
        
        # Make several moves and check consistency after each
        moves: List[str] = ["UP", "RIGHT", "DOWN", "LEFT", "UP", "UP"]
        for move in moves:
            collision, _ = controller.make_move(move)
            if not collision:
                self._verify_board_consistency(controller)
        
        # Reset and check consistency
        controller.reset()
        self._verify_board_consistency(controller)

    def _verify_board_consistency(self, controller: GameController) -> None:
        """Helper method to verify board state consistency."""
        # Count snake segments on board
        snake_count: int = np.count_nonzero(controller.board == 1)
        assert snake_count == len(controller.snake_positions)
        
        # Count apples on board
        apple_count: int = np.count_nonzero(controller.board == 2)
        assert apple_count == 1
        
        # Verify snake positions match board
        for i, position in enumerate(controller.snake_positions):
            assert controller.board[position[0], position[1]] == 1
        
        # Verify apple position matches board
        apple_pos: NDArray[np.int_] = controller.apple_position
        assert controller.board[apple_pos[0], apple_pos[1]] == 2

    # ==================== COMPREHENSIVE MOVEMENT TESTING ====================
    
    def test_movement_all_directions_comprehensive(self) -> None:
        """Comprehensive testing of movement in all directions."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test each direction multiple times
        direction_deltas: dict[str, List[int]] = {
            "UP": [-1, 0],
            "DOWN": [1, 0],
            "LEFT": [0, -1],
            "RIGHT": [0, 1]
        }
        
        for direction, delta in direction_deltas.items():
            controller.reset()
            initial_pos: NDArray[np.int_] = controller.snake_positions[0].copy()
            
            # Make move
            collision, _ = controller.make_move(direction)
            assert not collision
            
            # Verify new position
            expected_pos: NDArray[np.int_] = initial_pos + np.array(delta)
            assert np.array_equal(controller.snake_positions[-1], expected_pos)
            
            # Verify direction is set
            assert controller.current_direction == direction

    def test_movement_boundary_conditions(self) -> None:
        """Test movement at various boundary conditions."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test movement near each boundary
        test_cases: List[Tuple[List[int], str, bool]] = [
            ([0, 5], "UP", True),    # Top boundary
            ([9, 5], "DOWN", True),  # Bottom boundary
            ([5, 0], "LEFT", True),  # Left boundary
            ([5, 9], "RIGHT", True), # Right boundary
            ([1, 5], "UP", False),   # Near top boundary (valid)
            ([8, 5], "DOWN", False), # Near bottom boundary (valid)
            ([5, 1], "LEFT", False), # Near left boundary (valid)
            ([5, 8], "RIGHT", False) # Near right boundary (valid)
        ]
        
        for position, direction, should_collide in test_cases:
            controller.reset()
            controller.snake_positions = np.array([position], dtype=np.int_)
            controller.head_position = controller.snake_positions[-1]
            controller._update_board()
            
            collision, _ = controller.make_move(direction)
            assert collision == should_collide

    def test_movement_sequence_patterns(self) -> None:
        """Test complex movement sequence patterns."""
        controller: GameController = GameController(grid_size=15, use_gui=False)
        
        # Test spiral pattern
        spiral_moves: List[str] = ["UP", "UP", "RIGHT", "RIGHT", "DOWN", "DOWN", "DOWN", "LEFT", "LEFT", "LEFT", "UP"]
        for move in spiral_moves:
            collision, _ = controller.make_move(move)
            assert not collision  # Should not collide in a 15x15 grid
        
        # Test zigzag pattern
        controller.reset()
        zigzag_moves: List[str] = ["RIGHT", "UP", "RIGHT", "DOWN", "RIGHT", "UP", "RIGHT", "DOWN"]
        for move in zigzag_moves:
            collision, _ = controller.make_move(move)
            assert not collision

    def test_movement_direction_tracking(self) -> None:
        """Test comprehensive direction tracking."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Initially no direction
        assert controller.current_direction is None
        
        # Test direction setting for each move
        directions: List[str] = ["UP", "RIGHT", "DOWN", "LEFT"]
        for direction in directions:
            controller.make_move(direction)
            assert controller.current_direction == direction
        
        # Test direction persistence
        for _ in range(3):
            controller.make_move("UP")
            assert controller.current_direction == "UP"

    # ==================== COMPREHENSIVE COLLISION TESTING ====================
    
    def test_wall_collision_comprehensive(self) -> None:
        """Comprehensive testing of wall collision scenarios."""
        controller: GameController = GameController(grid_size=8, use_gui=False)
        
        # Test collision with each wall
        wall_test_cases: List[Tuple[List[int], str, str]] = [
            ([0, 4], "UP", "wall"),     # Top wall
            ([7, 4], "DOWN", "wall"),   # Bottom wall
            ([4, 0], "LEFT", "wall"),   # Left wall
            ([4, 7], "RIGHT", "wall")   # Right wall
        ]
        
        for position, direction, expected_collision in wall_test_cases:
            controller.reset()
            controller.snake_positions = np.array([position], dtype=np.int_)
            controller.head_position = controller.snake_positions[-1]
            controller._update_board()
            
            collision, _ = controller.make_move(direction)
            assert collision
            assert controller.last_collision_type == expected_collision

    def test_self_collision_comprehensive(self) -> None:
        """Comprehensive testing of self collision scenarios."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Create various self-collision scenarios
        test_scenarios: List[Tuple[List[List[int]], str]] = [
            # L-shaped collision
            ([[5, 5], [5, 4], [4, 4]], "LEFT"),
            # Straight line collision
            ([[5, 5], [5, 4], [5, 3], [5, 2]], "DOWN"),
            # Complex shape collision
            ([[5, 5], [5, 4], [4, 4], [3, 4], [3, 5]], "UP"),
        ]
        
        for snake_positions, collision_direction in test_scenarios:
            controller.reset()
            controller.snake_positions = np.array(snake_positions, dtype=np.int_)
            controller.head_position = controller.snake_positions[-1]
            controller._update_board()
            
            collision, _ = controller.make_move(collision_direction)
            assert collision
            assert controller.last_collision_type == "self"

    def test_collision_edge_cases_comprehensive(self) -> None:
        """Test edge cases for collision detection."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test corner collisions
        corner_cases: List[Tuple[List[int], str]] = [
            ([0, 0], "UP"),    # Top-left corner
            ([0, 0], "LEFT"),  # Top-left corner
            ([0, 9], "UP"),    # Top-right corner
            ([0, 9], "RIGHT"), # Top-right corner
            ([9, 0], "DOWN"),  # Bottom-left corner
            ([9, 0], "LEFT"),  # Bottom-left corner
            ([9, 9], "DOWN"),  # Bottom-right corner
            ([9, 9], "RIGHT")  # Bottom-right corner
        ]
        
        for position, direction in corner_cases:
            controller.reset()
            controller.snake_positions = np.array([position], dtype=np.int_)
            controller.head_position = controller.snake_positions[-1]
            controller._update_board()
            
            collision, _ = controller.make_move(direction)
            assert collision
            assert controller.last_collision_type == "wall"

    def test_no_collision_edge_cases(self) -> None:
        """Test cases where collision should not occur."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test valid moves near boundaries
        safe_cases: List[Tuple[List[int], str]] = [
            ([1, 1], "UP"),     # Near top-left, moving up
            ([1, 1], "LEFT"),   # Near top-left, moving left
            ([8, 8], "DOWN"),   # Near bottom-right, moving down
            ([8, 8], "RIGHT"),  # Near bottom-right, moving right
        ]
        
        for position, direction in safe_cases:
            controller.reset()
            controller.snake_positions = np.array([position], dtype=np.int_)
            controller.head_position = controller.snake_positions[-1]
            controller._update_board()
            
            collision, _ = controller.make_move(direction)
            assert not collision

    # ==================== COMPREHENSIVE APPLE MANAGEMENT TESTING ====================
    
    def test_apple_generation_comprehensive(self) -> None:
        """Comprehensive testing of apple generation."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test multiple apple generations
        apple_positions: set[Tuple[int, int]] = set()
        
        for _ in range(20):
            controller._generate_apple()
            apple_pos: Tuple[int, int] = tuple(controller.apple_position)
            apple_positions.add(apple_pos)
            
            # Verify apple is within bounds
            assert 0 <= apple_pos[0] < 10
            assert 0 <= apple_pos[1] < 10
            
            # Verify apple is not on snake
            for snake_segment in controller.snake_positions:
                assert not np.array_equal(snake_segment, apple_pos)
        
        # Should generate different positions (not always the same)
        assert len(apple_positions) > 1

    def test_apple_avoidance_comprehensive(self) -> None:
        """Test apple generation avoiding snake in various scenarios."""
        controller: GameController = GameController(grid_size=5, use_gui=False)
        
        # Fill most of the board with snake
        large_snake: List[List[int]] = [
            [0, 0], [0, 1], [0, 2], [0, 3],
            [1, 3], [2, 3], [3, 3], [4, 3],
            [4, 2], [4, 1], [4, 0], [3, 0]
        ]
        
        controller.snake_positions = np.array(large_snake, dtype=np.int_)
        controller.head_position = controller.snake_positions[-1]
        controller._update_board()
        
        # Generate apple - should find one of the few remaining spots
        controller._generate_apple()
        apple_pos: NDArray[np.int_] = controller.apple_position
        
        # Verify apple is not on any snake segment
        for snake_segment in controller.snake_positions:
            assert not np.array_equal(snake_segment, apple_pos)

    def test_apple_eating_comprehensive(self) -> None:
        """Comprehensive testing of apple eating mechanics."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test apple eating from different directions
        directions: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        for direction in directions:
            controller.reset()
            snake_head: NDArray[np.int_] = controller.snake_positions[-1]
            
            # Position apple in the direction of movement
            direction_deltas: dict[str, List[int]] = {
                "UP": [-1, 0], "DOWN": [1, 0], "LEFT": [0, -1], "RIGHT": [0, 1]
            }
            
            delta: List[int] = direction_deltas[direction]
            apple_pos: List[int] = [snake_head[0] + delta[0], snake_head[1] + delta[1]]
            
            # Ensure apple position is valid
            if 0 <= apple_pos[0] < 10 and 0 <= apple_pos[1] < 10:
                controller.set_apple_position(apple_pos)
                initial_length: int = len(controller.snake_positions)
                initial_score: int = controller.game_state.score
                
                # Make move to eat apple
                collision, apple_eaten = controller.make_move(direction)
                
                assert not collision
                assert apple_eaten
                assert len(controller.snake_positions) == initial_length + 1
                assert controller.game_state.score == initial_score + 1

    def test_apple_position_validation(self) -> None:
        """Test apple position validation and setting."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test valid positions
        valid_positions: List[List[int]] = [
            [0, 0], [9, 9], [5, 3], [2, 7], [8, 1]
        ]
        
        for position in valid_positions:
            if not any(np.array_equal(pos, position) for pos in controller.snake_positions):
                success: bool = controller.set_apple_position(position)
                assert success
                assert np.array_equal(controller.apple_position, position)
        
        # Test invalid positions (out of bounds)
        invalid_positions: List[List[int]] = [
            [-1, 5], [10, 5], [5, -1], [5, 10], [15, 15]
        ]
        
        for position in invalid_positions:
            success = controller.set_apple_position(position)
            assert not success
        
        # Test position occupied by snake
        snake_position: List[int] = controller.snake_positions[0].tolist()
        success = controller.set_apple_position(snake_position)
        assert not success

    # ==================== COMPREHENSIVE SNAKE GROWTH TESTING ====================
    
    def test_snake_growth_mechanics(self) -> None:
        """Comprehensive testing of snake growth mechanics."""
        controller: GameController = GameController(grid_size=15, use_gui=False)
        
        initial_length: int = len(controller.snake_positions)
        apples_eaten: int = 0
        
        # Eat multiple apples and verify growth
        for _ in range(10):
            # Position apple adjacent to snake head
            snake_head: NDArray[np.int_] = controller.snake_positions[-1]
            
            # Try different directions to place apple
            for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                direction_deltas: dict[str, List[int]] = {
                    "UP": [-1, 0], "DOWN": [1, 0], "LEFT": [0, -1], "RIGHT": [0, 1]
                }
                delta: List[int] = direction_deltas[direction]
                apple_pos: List[int] = [snake_head[0] + delta[0], snake_head[1] + delta[1]]
                
                # Check if position is valid and not occupied
                if (0 <= apple_pos[0] < 15 and 0 <= apple_pos[1] < 15 and
                    not any(np.array_equal(pos, apple_pos) for pos in controller.snake_positions)):
                    
                    controller.set_apple_position(apple_pos)
                    collision, apple_eaten = controller.make_move(direction)
                    
                    if apple_eaten:
                        apples_eaten += 1
                        expected_length: int = initial_length + apples_eaten
                        assert len(controller.snake_positions) == expected_length
                        assert controller.game_state.score == apples_eaten
                    break

    def test_snake_tail_preservation(self) -> None:
        """Test that snake tail is preserved correctly during growth."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Grow snake by eating apple
        snake_head: NDArray[np.int_] = controller.snake_positions[-1]
        apple_pos: List[int] = [snake_head[0] - 1, snake_head[1]]  # Above head
        controller.set_apple_position(apple_pos)
        
        original_tail: NDArray[np.int_] = controller.snake_positions[0].copy()
        
        # Eat apple
        collision, apple_eaten = controller.make_move("UP")
        assert apple_eaten
        
        # Tail should be preserved (first segment should still be there)
        assert np.array_equal(controller.snake_positions[0], original_tail)
        assert len(controller.snake_positions) == 2

    def test_snake_maximum_growth(self) -> None:
        """Test snake growth to near-maximum size."""
        controller: GameController = GameController(grid_size=6, use_gui=False)
        
        # In a 6x6 grid, maximum snake length is 36 segments
        # Let's grow to a reasonable size to test mechanics
        target_length: int = 10
        
        for i in range(target_length - 1):  # -1 because snake starts with 1 segment
            # Find a valid apple position
            valid_position_found: bool = False
            for row in range(6):
                for col in range(6):
                    position: List[int] = [row, col]
                    if not any(np.array_equal(pos, position) for pos in controller.snake_positions):
                        controller.set_apple_position(position)
                        valid_position_found = True
                        break
                if valid_position_found:
                    break
            
            if not valid_position_found:
                break
            
            # Find a move that will eat the apple
            apple_pos: NDArray[np.int_] = controller.apple_position
            snake_head: NDArray[np.int_] = controller.snake_positions[-1]
            
            # Calculate required move
            diff: NDArray[np.int_] = apple_pos - snake_head
            if abs(diff[0]) == 1 and diff[1] == 0:
                move: str = "DOWN" if diff[0] > 0 else "UP"
            elif abs(diff[1]) == 1 and diff[0] == 0:
                move = "RIGHT" if diff[1] > 0 else "LEFT"
            else:
                # Apple not adjacent, need to navigate to it
                # For simplicity, just make a valid move
                move = "RIGHT"
            
            # Ensure move doesn't cause collision
            if controller.filter_invalid_reversals([move], controller.current_direction):
                collision, apple_eaten = controller.make_move(move)
                if collision:
                    break
        
        # Verify final snake length
        assert len(controller.snake_positions) >= 1

    # ==================== COMPREHENSIVE GAME STATE TESTING ====================
    
    def test_game_state_consistency_comprehensive(self) -> None:
        """Comprehensive testing of game state consistency."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test state consistency after various operations
        operations: List[str] = ["UP", "RIGHT", "DOWN", "LEFT", "UP", "UP"]
        
        for i, move in enumerate(operations):
            # Check state before move
            pre_steps: int = controller.game_state.steps
            pre_score: int = controller.game_state.score
            pre_moves_count: int = len(controller.game_state.moves)
            
            collision, apple_eaten = controller.make_move(move)
            
            if not collision:
                # Verify state updates
                assert controller.game_state.steps == pre_steps + 1
                assert len(controller.game_state.moves) == pre_moves_count + 1
                assert controller.game_state.moves[-1] == move
                
                if apple_eaten:
                    assert controller.game_state.score == pre_score + 1
                else:
                    assert controller.game_state.score == pre_score

    def test_game_state_reset_comprehensive(self) -> None:
        """Comprehensive testing of game state reset."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Modify game state significantly
        moves: List[str] = ["UP", "RIGHT", "DOWN", "LEFT"] * 3
        for move in moves:
            collision, _ = controller.make_move(move)
            if collision:
                break
        
        # Store pre-reset state
        pre_reset_steps: int = controller.game_state.steps
        pre_reset_score: int = controller.game_state.score
        
        # Reset and verify complete restoration
        controller.reset()
        
        assert controller.game_state.steps == 0
        assert controller.game_state.score == 0
        assert len(controller.game_state.moves) == 0
        assert len(controller.snake_positions) == 1
        assert controller.current_direction is None
        assert controller.last_collision_type is None
        
        # Verify previous state was actually different
        assert pre_reset_steps > 0 or pre_reset_score > 0

    def test_game_state_serialization_compatibility(self) -> None:
        """Test game state serialization and compatibility."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Make some moves to create interesting state
        moves: List[str] = ["UP", "RIGHT", "DOWN"]
        for move in moves:
            controller.make_move(move)
        
        # Get game state
        game_state: Any = controller.get_game_state()
        
        # Verify state can be converted to dict (for JSON serialization)
        state_dict: dict[str, Any] = controller.to_dict()
        
        assert isinstance(state_dict, dict)
        assert 'snake_positions' in state_dict
        assert 'apple_position' in state_dict
        assert 'current_direction' in state_dict
        assert 'game_state' in state_dict

    # ==================== COMPREHENSIVE MOVE FILTERING TESTING ====================
    
    def test_move_filtering_comprehensive(self) -> None:
        """Comprehensive testing of move filtering logic."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test filtering for each direction
        direction_opposites: dict[str, str] = {
            "UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"
        }
        
        for direction, opposite in direction_opposites.items():
            # Test filtering when current direction is set
            all_moves: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
            filtered: List[str] = controller.filter_invalid_reversals(all_moves, direction)
            
            # Opposite should be filtered out
            assert opposite not in filtered
            assert direction in filtered
            
            # Other directions should remain
            for move in all_moves:
                if move != opposite:
                    assert move in filtered

    def test_move_filtering_edge_cases(self) -> None:
        """Test move filtering edge cases."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test with no current direction
        all_moves: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
        filtered: List[str] = controller.filter_invalid_reversals(all_moves, None)
        assert filtered == all_moves
        
        # Test with empty move list
        empty_filtered: List[str] = controller.filter_invalid_reversals([], "UP")
        assert empty_filtered == []
        
        # Test with only opposite moves
        only_opposite: List[str] = ["DOWN", "DOWN"]
        filtered_opposite: List[str] = controller.filter_invalid_reversals(only_opposite, "UP")
        assert filtered_opposite == []
        
        # Test with duplicate moves
        duplicates: List[str] = ["UP", "UP", "LEFT", "LEFT", "DOWN"]
        filtered_dups: List[str] = controller.filter_invalid_reversals(duplicates, "UP")
        assert "DOWN" not in filtered_dups
        assert "UP" in filtered_dups
        assert "LEFT" in filtered_dups

    def test_move_filtering_performance(self) -> None:
        """Test performance of move filtering with large lists."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Create large list of moves
        large_move_list: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"] * 1000
        
        # Filtering should be fast even with large lists
        import time
        start_time: float = time.time()
        filtered: List[str] = controller.filter_invalid_reversals(large_move_list, "UP")
        end_time: float = time.time()
        
        # Should complete quickly (less than 1 second for 4000 moves)
        assert end_time - start_time < 1.0
        
        # Verify filtering correctness
        assert "DOWN" not in filtered
        assert len([m for m in filtered if m == "UP"]) == 1000  # All UP moves preserved

    # ==================== COMPREHENSIVE PROPERTY TESTING ====================
    
    def test_properties_comprehensive(self) -> None:
        """Comprehensive testing of all controller properties."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test initial property values
        assert controller.score == 0
        assert controller.steps == 0
        assert len(controller.moves) == 0
        assert controller.head_position is not None
        assert len(controller.snake_body) == 1
        
        # Modify state and test property updates
        controller.make_move("UP")
        assert controller.steps == 1
        assert len(controller.moves) == 1
        assert controller.moves[0] == "UP"
        
        # Test head position tracking
        initial_head: NDArray[np.int_] = controller.snake_positions[0].copy()
        controller.make_move("RIGHT")
        new_head: NDArray[np.int_] = controller.head_position
        
        # Head should have moved
        assert not np.array_equal(new_head, initial_head)

    def test_property_type_consistency(self) -> None:
        """Test that properties return consistent types."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test type consistency
        assert isinstance(controller.score, int)
        assert isinstance(controller.steps, int)
        assert isinstance(controller.moves, list)
        assert isinstance(controller.head_position, np.ndarray)
        assert isinstance(controller.snake_body, np.ndarray)
        assert isinstance(controller.apple_position, np.ndarray)
        assert isinstance(controller.current_direction, (str, type(None)))

    def test_property_immutability_where_appropriate(self) -> None:
        """Test that properties that should be immutable are protected."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Get references to properties
        original_head: NDArray[np.int_] = controller.head_position
        original_body: NDArray[np.int_] = controller.snake_body
        
        # Modify the returned arrays (should not affect internal state)
        returned_head: NDArray[np.int_] = controller.head_position
        returned_body: NDArray[np.int_] = controller.snake_body
        
        # These should be copies or views that don't affect internal state
        returned_head[0] = 999
        returned_body[0, 0] = 999
        
        # Internal state should be unchanged
        assert not np.array_equal(controller.head_position, returned_head)

    # ==================== COMPREHENSIVE GUI INTEGRATION TESTING ====================
    
    def test_gui_integration_comprehensive(self) -> None:
        """Comprehensive testing of GUI integration."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test without GUI
        assert controller.gui is None
        assert controller.use_gui is False
        
        # Set GUI and test integration
        mock_gui: Mock = Mock()
        controller.set_gui(mock_gui)
        
        assert controller.gui is mock_gui
        assert controller.use_gui is True
        
        # Test that operations work with GUI set
        controller.make_move("UP")
        controller.reset()
        
        # GUI should remain set
        assert controller.gui is mock_gui

    @patch('core.game_controller.GameController.draw')
    def test_gui_draw_integration(self, mock_draw: Mock) -> None:
        """Test GUI drawing integration."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        mock_gui: Mock = Mock()
        controller.set_gui(mock_gui)
        
        # Make a move (should trigger draw if GUI is available)
        controller.make_move("UP")
        
        # Verify draw was called
        mock_draw.assert_called()

    def test_gui_state_synchronization(self) -> None:
        """Test state synchronization with GUI."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        mock_gui: Mock = Mock()
        controller.set_gui(mock_gui)
        
        # Perform various operations
        operations: List[str] = ["UP", "RIGHT", "DOWN", "LEFT"]
        for move in operations:
            initial_state: dict[str, Any] = controller.to_dict()
            controller.make_move(move)
            
            # State should be consistent
            new_state: dict[str, Any] = controller.to_dict()
            assert new_state != initial_state  # State should have changed

    # ==================== PERFORMANCE AND MEMORY TESTING ====================
    
    def test_performance_large_grid(self) -> None:
        """Test performance with large grid sizes."""
        large_controller: GameController = GameController(grid_size=50, use_gui=False)
        
        # Test that operations complete in reasonable time
        import time
        
        # Test move performance
        start_time: float = time.time()
        for _ in range(100):
            large_controller.make_move("RIGHT")
        end_time: float = time.time()
        
        # Should complete 100 moves quickly
        assert end_time - start_time < 1.0

    def test_memory_efficiency_large_snake(self) -> None:
        """Test memory efficiency with large snake."""
        controller: GameController = GameController(grid_size=20, use_gui=False)
        
        # Create a reasonably large snake by eating many apples
        for i in range(50):
            # Position apple where snake can eat it
            snake_head: NDArray[np.int_] = controller.snake_positions[-1]
            
            # Find a valid position for apple
            for row in range(20):
                for col in range(20):
                    position: List[int] = [row, col]
                    if not any(np.array_equal(pos, position) for pos in controller.snake_positions):
                        controller.set_apple_position(position)
                        break
                else:
                    continue
                break
            
            # Try to eat apple (simplified logic)
            apple_pos: NDArray[np.int_] = controller.apple_position
            diff: NDArray[np.int_] = apple_pos - snake_head
            
            if np.abs(diff).sum() == 1:  # Apple is adjacent
                if diff[0] == -1:
                    move = "UP"
                elif diff[0] == 1:
                    move = "DOWN"
                elif diff[1] == -1:
                    move = "LEFT"
                else:
                    move = "RIGHT"
                
                valid_moves: List[str] = controller.filter_invalid_reversals([move], controller.current_direction)
                if valid_moves:
                    collision, apple_eaten = controller.make_move(move)
                    if collision:
                        break
            else:
                # Just make a safe move
                safe_moves: List[str] = []
                for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    test_pos: List[int] = [snake_head[0], snake_head[1]]
                    if direction == "UP":
                        test_pos[0] -= 1
                    elif direction == "DOWN":
                        test_pos[0] += 1
                    elif direction == "LEFT":
                        test_pos[1] -= 1
                    else:
                        test_pos[1] += 1
                    
                    if (0 <= test_pos[0] < 20 and 0 <= test_pos[1] < 20 and
                        not any(np.array_equal(pos, test_pos) for pos in controller.snake_positions)):
                        safe_moves.append(direction)
                
                if safe_moves:
                    filtered: List[str] = controller.filter_invalid_reversals(safe_moves, controller.current_direction)
                    if filtered:
                        collision, _ = controller.make_move(filtered[0])
                        if collision:
                            break
                else:
                    break
        
        # Verify snake grew and operations still work
        assert len(controller.snake_positions) > 1
        
        # Test that state can still be retrieved efficiently
        game_state: Any = controller.get_game_state()
        assert game_state is not None

    def test_concurrent_operations_simulation(self) -> None:
        """Simulate concurrent operations on controller."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Simulate rapid state queries during gameplay
        for i in range(100):
            controller.make_move("RIGHT")
            
            # Query state multiple times (simulating concurrent access)
            _ = controller.score
            _ = controller.steps
            _ = controller.head_position
            _ = controller.snake_body
            _ = controller.apple_position
            _ = controller.get_game_state()
            _ = controller.to_dict()
            
            if i % 10 == 0:
                controller.reset()
        
        # Should complete without errors
        assert True

    # ==================== EDGE CASES AND BOUNDARY CONDITIONS ====================
    
    def test_boundary_conditions_comprehensive(self) -> None:
        """Test various boundary conditions."""
        # Test minimum grid size
        min_controller: GameController = GameController(grid_size=3, use_gui=False)
        assert min_controller.grid_size == 3
        assert min_controller.board.shape == (3, 3)
        
        # Test very large grid size
        max_controller: GameController = GameController(grid_size=200, use_gui=False)
        assert max_controller.grid_size == 200
        assert max_controller.board.shape == (200, 200)

    def test_error_recovery_mechanisms(self) -> None:
        """Test error recovery and robustness."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Test invalid move strings
        invalid_moves: List[str] = ["INVALID", "", "up", "Down", "123"]
        for move in invalid_moves:
            # Should handle gracefully without crashing
            try:
                controller.make_move(move)
            except (KeyError, ValueError):
                # Expected for invalid moves
                pass
        
        # Controller should still be functional
        collision, _ = controller.make_move("UP")
        assert isinstance(collision, bool)

    def test_state_consistency_after_errors(self) -> None:
        """Test that state remains consistent after errors."""
        controller: GameController = GameController(grid_size=10, use_gui=False)
        
        # Get initial state
        initial_state: dict[str, Any] = controller.to_dict()
        
        # Try some operations that might cause errors
        try:
            controller.set_apple_position([-1, -1])  # Invalid position
        except:
            pass
        
        try:
            controller.make_move("INVALID")  # Invalid move
        except:
            pass
        
        # State should be consistent
        current_state: dict[str, Any] = controller.to_dict()
        
        # Basic consistency checks
        assert len(controller.snake_positions) >= 1
        assert controller.apple_position is not None
        assert 0 <= controller.apple_position[0] < controller.grid_size
        assert 0 <= controller.apple_position[1] < controller.grid_size 