"""
Tests for move utility functions.
"""

import pytest
import numpy as np
from typing import List, Tuple, Optional, Sequence, Union
from numpy.typing import NDArray

from utils.moves_utils import (
    normalize_direction,
    normalize_directions,
    is_reverse,
    calculate_move_differences,
    get_opposite_direction,
    is_valid_direction,
    calculate_next_position,
    detect_collision,
    filter_invalid_moves,
    get_direction_vector,
    manhattan_distance,
    get_adjacent_positions
)
from config.game_constants import DIRECTIONS


class TestNormalizeDirection:
    """Test cases for normalize_direction."""

    def test_uppercase_conversion(self):
        """Test conversion to uppercase."""
        assert normalize_direction("up") == "UP"
        assert normalize_direction("down") == "DOWN"
        assert normalize_direction("left") == "LEFT"
        assert normalize_direction("right") == "RIGHT"

    def test_whitespace_stripping(self):
        """Test removal of whitespace."""
        assert normalize_direction(" up ") == "UP"
        assert normalize_direction("down\n") == "DOWN"
        assert normalize_direction("\tleft") == "LEFT"
        assert normalize_direction("right ") == "RIGHT"

    def test_already_normalized(self):
        """Test already normalized directions."""
        assert normalize_direction("UP") == "UP"
        assert normalize_direction("DOWN") == "DOWN"
        assert normalize_direction("LEFT") == "LEFT"
        assert normalize_direction("RIGHT") == "RIGHT"

    def test_mixed_case(self):
        """Test mixed case directions."""
        assert normalize_direction("Up") == "UP"
        assert normalize_direction("DoWn") == "DOWN"
        assert normalize_direction("LeFt") == "LEFT"
        assert normalize_direction("RiGhT") == "RIGHT"

    def test_special_moves(self):
        """Test special move types."""
        assert normalize_direction("empty") == "EMPTY"
        assert normalize_direction("invalid_reversal") == "INVALID_REVERSAL"
        assert normalize_direction("something_is_wrong") == "SOMETHING_IS_WRONG"
        assert normalize_direction("no_path_found") == "NO_PATH_FOUND"


class TestNormalizeDirections:
    """Test cases for normalize_directions."""

    def test_list_normalization(self):
        """Test normalization of a list of directions."""
        moves = ["up", "DOWN", " left ", "Right\n"]
        expected = ["UP", "DOWN", "LEFT", "RIGHT"]
        assert normalize_directions(moves) == expected

    def test_empty_list(self):
        """Test normalization of empty list."""
        assert normalize_directions([]) == []

    def test_single_element(self):
        """Test normalization of single element list."""
        assert normalize_directions(["up"]) == ["UP"]

    def test_mixed_moves(self):
        """Test normalization of mixed regular and special moves."""
        moves = ["up", "empty", "down", "invalid_reversal"]
        expected = ["UP", "EMPTY", "DOWN", "INVALID_REVERSAL"]
        assert normalize_directions(moves) == expected


class TestIsReverse:
    """Test cases for is_reverse."""

    def test_up_down_reversal(self):
        """Test UP/DOWN reversal detection."""
        assert is_reverse("UP", "DOWN") is True
        assert is_reverse("DOWN", "UP") is True
        assert is_reverse("up", "down") is True
        assert is_reverse("down", "up") is True

    def test_left_right_reversal(self):
        """Test LEFT/RIGHT reversal detection."""
        assert is_reverse("LEFT", "RIGHT") is True
        assert is_reverse("RIGHT", "LEFT") is True
        assert is_reverse("left", "right") is True
        assert is_reverse("right", "left") is True

    def test_non_reversal_same_direction(self):
        """Test that same directions are not reversals."""
        assert is_reverse("UP", "UP") is False
        assert is_reverse("DOWN", "DOWN") is False
        assert is_reverse("LEFT", "LEFT") is False
        assert is_reverse("RIGHT", "RIGHT") is False

    def test_non_reversal_perpendicular(self):
        """Test that perpendicular directions are not reversals."""
        assert is_reverse("UP", "LEFT") is False
        assert is_reverse("UP", "RIGHT") is False
        assert is_reverse("DOWN", "LEFT") is False
        assert is_reverse("DOWN", "RIGHT") is False
        assert is_reverse("LEFT", "UP") is False
        assert is_reverse("LEFT", "DOWN") is False
        assert is_reverse("RIGHT", "UP") is False
        assert is_reverse("RIGHT", "DOWN") is False

    def test_case_insensitive(self):
        """Test that reversal detection is case insensitive."""
        assert is_reverse("up", "DOWN") is True
        assert is_reverse("Left", "right") is True
        assert is_reverse("DOWN", "up") is True
        assert is_reverse("RIGHT", "left") is True

    def test_special_moves_not_reversals(self):
        """Test that special moves are not considered reversals."""
        assert is_reverse("EMPTY", "UP") is False
        assert is_reverse("UP", "EMPTY") is False
        assert is_reverse("INVALID_REVERSAL", "DOWN") is False
        assert is_reverse("SOMETHING_IS_WRONG", "LEFT") is False


class TestCalculateMoveChain:
    """Test cases for calculate_move_differences."""

    def test_positive_horizontal_difference(self):
        """Test when apple is to the right of head."""
        head_pos = [3, 5]
        apple_pos = [7, 5]
        result = calculate_move_differences(head_pos, apple_pos)
        assert "#RIGHT - #LEFT = 4" in result
        assert "(= 7 - 3)" in result

    def test_negative_horizontal_difference(self):
        """Test when apple is to the left of head."""
        head_pos = [7, 5]
        apple_pos = [3, 5]
        result = calculate_move_differences(head_pos, apple_pos)
        assert "#LEFT - #RIGHT = 4" in result
        assert "(= 7 - 3)" in result

    def test_positive_vertical_difference(self):
        """Test when apple is below head."""
        head_pos = [5, 2]
        apple_pos = [5, 6]
        result = calculate_move_differences(head_pos, apple_pos)
        assert "#UP - #DOWN = 4" in result
        assert "(= 6 - 2)" in result

    def test_negative_vertical_difference(self):
        """Test when apple is above head."""
        head_pos = [5, 6]
        apple_pos = [5, 2]
        result = calculate_move_differences(head_pos, apple_pos)
        assert "#DOWN - #UP = 4" in result
        assert "(= 6 - 2)" in result

    def test_zero_horizontal_difference(self):
        """Test when head and apple have same x coordinate."""
        head_pos = [5, 3]
        apple_pos = [5, 7]
        result = calculate_move_differences(head_pos, apple_pos)
        assert "#RIGHT - #LEFT = 0" in result
        assert "(= 5 - 5)" in result

    def test_zero_vertical_difference(self):
        """Test when head and apple have same y coordinate."""
        head_pos = [3, 5]
        apple_pos = [7, 5]
        result = calculate_move_differences(head_pos, apple_pos)
        assert "#UP - #DOWN = 0" in result
        assert "(= 5 - 5)" in result

    def test_diagonal_difference(self):
        """Test with both horizontal and vertical differences."""
        head_pos = [2, 3]
        apple_pos = [6, 7]
        result = calculate_move_differences(head_pos, apple_pos)
        
        # Should contain both horizontal and vertical differences
        assert "#RIGHT - #LEFT = 4" in result
        assert "#UP - #DOWN = 4" in result
        assert "and" in result

    def test_same_position(self):
        """Test when head and apple are at same position."""
        head_pos = [5, 5]
        apple_pos = [5, 5]
        result = calculate_move_differences(head_pos, apple_pos)
        
        assert "#RIGHT - #LEFT = 0" in result
        assert "#UP - #DOWN = 0" in result
        assert "(= 5 - 5)" in result

    def test_boundary_positions(self):
        """Test with boundary positions."""
        # Top-left to bottom-right
        head_pos = [0, 0]
        apple_pos = [9, 9]
        result = calculate_move_differences(head_pos, apple_pos)
        
        assert "#RIGHT - #LEFT = 9" in result
        assert "#UP - #DOWN = 9" in result

    def test_return_format(self):
        """Test that return format is consistent."""
        head_pos = [1, 2]
        apple_pos = [4, 6]
        result = calculate_move_differences(head_pos, apple_pos)
        
        # Should contain both parts connected by "and"
        parts = result.split(", and ")
        assert len(parts) == 2
        
        # Each part should contain the pattern
        for part in parts:
            assert " = " in part
            assert "(" in part and ")" in part

    def test_sequence_input_types(self):
        """Test that function accepts different sequence types."""
        # Test with lists
        result1 = calculate_move_differences([1, 2], [3, 4])
        
        # Test with tuples
        result2 = calculate_move_differences((1, 2), (3, 4))
        
        # Results should be identical
        assert result1 == result2

    def test_large_differences(self):
        """Test with large coordinate differences."""
        head_pos = [0, 0]
        apple_pos = [100, 50]
        result = calculate_move_differences(head_pos, apple_pos)
        
        assert "#RIGHT - #LEFT = 100" in result
        assert "#UP - #DOWN = 50" in result


class TestIntegration:
    """Integration tests for move utilities."""

    def test_normalize_and_reverse_check(self):
        """Test integration of normalize and reverse checking."""
        # Test that normalized moves still work with reverse checking
        move1 = normalize_direction("up")
        move2 = normalize_direction("down")
        
        assert is_reverse(move1, move2) is True

    def test_full_move_processing_pipeline(self):
        """Test a full move processing pipeline."""
        raw_moves = [" up ", "DOWN", "left\n", "Right"]
        
        # Normalize the moves
        normalized = normalize_directions(raw_moves)
        
        # Check for reversals in sequence
        reversals = []
        for i in range(1, len(normalized)):
            if is_reverse(normalized[i], normalized[i-1]):
                reversals.append((i-1, i))
        
        assert normalized == ["UP", "DOWN", "LEFT", "RIGHT"]
        assert (0, 1) in reversals  # UP -> DOWN is a reversal

    def test_move_difference_with_normalization(self):
        """Test that move differences work with various input types."""
        positions = [
            ([0, 0], [1, 1]),
            ((2, 3), (4, 5)),
            ([10, 5], [5, 10]),
        ]
        
        for head, apple in positions:
            result = calculate_move_differences(head, apple)
            assert isinstance(result, str)
            assert "RIGHT" in result or "LEFT" in result
            assert "UP" in result or "DOWN" in result


class TestMovesUtils:
    """Test cases for move utility functions."""

    def test_normalize_direction_valid_cases(self) -> None:
        """Test normalizing valid direction strings."""
        test_cases: List[Tuple[str, str]] = [
            ("up", "UP"),
            ("UP", "UP"),
            ("Up", "UP"),
            ("  right  ", "RIGHT"),
            ("down", "DOWN"),
            ("LEFT", "LEFT"),
            ("left", "LEFT")
        ]
        
        for input_dir, expected in test_cases:
            result: str = normalize_direction(input_dir)
            assert result == expected

    def test_normalize_direction_invalid_cases(self) -> None:
        """Test normalizing invalid direction strings."""
        invalid_directions: List[str] = [
            "",
            "INVALID",
            "north",
            "123",
            "up down",
            "U P"
        ]
        
        for invalid_dir in invalid_directions:
            result: str = normalize_direction(invalid_dir)
            assert result == "NONE"

    def test_get_opposite_direction_valid(self) -> None:
        """Test getting opposite directions for valid inputs."""
        opposite_pairs: List[Tuple[str, str]] = [
            ("UP", "DOWN"),
            ("DOWN", "UP"),
            ("LEFT", "RIGHT"),
            ("RIGHT", "LEFT")
        ]
        
        for direction, expected_opposite in opposite_pairs:
            result: str = get_opposite_direction(direction)
            assert result == expected_opposite

    def test_get_opposite_direction_invalid(self) -> None:
        """Test getting opposite direction for invalid input."""
        invalid_directions: List[str] = ["NONE", "INVALID", "", "DIAGONAL"]
        
        for invalid_dir in invalid_directions:
            result: str = get_opposite_direction(invalid_dir)
            assert result == "NONE"

    def test_is_valid_direction_valid_cases(self) -> None:
        """Test validating correct direction strings."""
        valid_directions: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        for direction in valid_directions:
            assert is_valid_direction(direction)

    def test_is_valid_direction_invalid_cases(self) -> None:
        """Test validating incorrect direction strings."""
        invalid_directions: List[str] = [
            "NONE", "INVALID", "", "up", "down", "123", "DIAGONAL"
        ]
        
        for direction in invalid_directions:
            assert not is_valid_direction(direction)

    def test_calculate_next_position_all_directions(self) -> None:
        """Test calculating next position for all valid directions."""
        start_pos: List[int] = [5, 5]
        
        expected_results: List[Tuple[str, List[int]]] = [
            ("UP", [5, 4]),
            ("DOWN", [5, 6]),
            ("LEFT", [4, 5]),
            ("RIGHT", [6, 5])
        ]
        
        for direction, expected_pos in expected_results:
            result: List[int] = calculate_next_position(start_pos, direction)
            assert result == expected_pos

    def test_calculate_next_position_invalid_direction(self) -> None:
        """Test calculating next position with invalid direction."""
        start_pos: List[int] = [5, 5]
        
        # Should return original position for invalid direction
        result: List[int] = calculate_next_position(start_pos, "INVALID")
        assert result == start_pos

    def test_calculate_next_position_edge_coordinates(self) -> None:
        """Test calculating next position at grid edges."""
        test_cases: List[Tuple[List[int], str, List[int]]] = [
            ([0, 0], "UP", [0, -1]),    # Top edge, going up (negative y)
            ([0, 0], "LEFT", [-1, 0]),  # Left edge, going left (negative x)
            ([9, 9], "DOWN", [9, 10]),  # Bottom edge, going down
            ([9, 9], "RIGHT", [10, 9])  # Right edge, going right
        ]
        
        for start_pos, direction, expected_pos in test_cases:
            result: List[int] = calculate_next_position(start_pos, direction)
            assert result == expected_pos

    def test_detect_collision_wall_boundaries(self) -> None:
        """Test wall collision detection at grid boundaries."""
        grid_size: int = 10
        
        collision_cases: List[Tuple[List[int], bool]] = [
            ([-1, 5], True),    # Left wall
            ([10, 5], True),    # Right wall
            ([5, -1], True),    # Top wall
            ([5, 10], True),    # Bottom wall
            ([5, 5], False),    # Inside grid
            ([0, 0], False),    # Top-left corner (valid)
            ([9, 9], False)     # Bottom-right corner (valid)
        ]
        
        for position, should_collide in collision_cases:
            result: bool = detect_collision(position, grid_size)
            assert result == should_collide

    def test_detect_collision_snake_positions(self) -> None:
        """Test collision detection with snake positions."""
        grid_size: int = 10
        snake_positions: List[List[int]] = [[5, 5], [5, 4], [5, 3], [4, 3]]
        
        collision_cases: List[Tuple[List[int], bool]] = [
            ([5, 5], True),     # Head position
            ([5, 4], True),     # Body position
            ([4, 3], True),     # Tail position
            ([6, 6], False),    # Empty position
            ([0, 0], False)     # Different empty position
        ]
        
        for position, should_collide in collision_cases:
            result: bool = detect_collision(position, grid_size, snake_positions)
            assert result == should_collide

    def test_filter_invalid_moves_reversal_filtering(self) -> None:
        """Test filtering out reversal moves."""
        moves: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
        current_direction: str = "UP"
        
        filtered: List[str] = filter_invalid_moves(moves, current_direction)
        
        # DOWN should be filtered out as it's opposite to UP
        expected: List[str] = ["UP", "LEFT", "RIGHT"]
        assert filtered == expected

    def test_filter_invalid_moves_no_current_direction(self) -> None:
        """Test filtering when no current direction is set."""
        moves: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        filtered: List[str] = filter_invalid_moves(moves)
        
        # No filtering should occur
        assert filtered == moves

    def test_filter_invalid_moves_empty_input(self) -> None:
        """Test filtering with empty moves list."""
        moves: List[str] = []
        
        filtered: List[str] = filter_invalid_moves(moves, "UP")
        
        assert filtered == []

    def test_filter_invalid_moves_invalid_directions(self) -> None:
        """Test filtering out invalid direction strings."""
        moves: List[str] = ["UP", "INVALID", "LEFT", "DIAGONAL", "RIGHT"]
        
        filtered: List[str] = filter_invalid_moves(moves)
        
        # Only valid directions should remain
        expected: List[str] = ["UP", "LEFT", "RIGHT"]
        assert filtered == expected

    def test_get_direction_vector_all_directions(self) -> None:
        """Test getting direction vectors for all valid directions."""
        expected_vectors: List[Tuple[str, Tuple[int, int]]] = [
            ("UP", (0, -1)),
            ("DOWN", (0, 1)),
            ("LEFT", (-1, 0)),
            ("RIGHT", (1, 0))
        ]
        
        for direction, expected_vector in expected_vectors:
            result: Tuple[int, int] = get_direction_vector(direction)
            assert result == expected_vector

    def test_get_direction_vector_invalid_direction(self) -> None:
        """Test getting direction vector for invalid direction."""
        result: Tuple[int, int] = get_direction_vector("INVALID")
        assert result == (0, 0)

    def test_manhattan_distance_calculation(self) -> None:
        """Test Manhattan distance calculation."""
        distance_cases: List[Tuple[List[int], List[int], int]] = [
            ([0, 0], [0, 0], 0),        # Same position
            ([0, 0], [1, 0], 1),        # Adjacent horizontally
            ([0, 0], [0, 1], 1),        # Adjacent vertically
            ([0, 0], [1, 1], 2),        # Diagonal
            ([0, 0], [3, 4], 7),        # Distant positions
            ([5, 5], [2, 1], 7)         # Different starting point
        ]
        
        for pos1, pos2, expected_distance in distance_cases:
            result: int = manhattan_distance(pos1, pos2)
            assert result == expected_distance

    def test_manhattan_distance_negative_coordinates(self) -> None:
        """Test Manhattan distance with negative coordinates."""
        distance_cases: List[Tuple[List[int], List[int], int]] = [
            ([-1, -1], [1, 1], 4),      # Both negative to positive
            ([0, 0], [-2, -3], 5),      # Positive to negative
            ([-2, -3], [-1, -1], 3)     # Both negative
        ]
        
        for pos1, pos2, expected_distance in distance_cases:
            result: int = manhattan_distance(pos1, pos2)
            assert result == expected_distance

    def test_get_adjacent_positions_center(self) -> None:
        """Test getting adjacent positions from center of grid."""
        center_pos: List[int] = [5, 5]
        
        adjacent: List[List[int]] = get_adjacent_positions(center_pos)
        
        expected_positions: List[List[int]] = [
            [5, 4],  # UP
            [5, 6],  # DOWN
            [4, 5],  # LEFT
            [6, 5]   # RIGHT
        ]
        
        assert len(adjacent) == 4
        for expected_pos in expected_positions:
            assert expected_pos in adjacent

    def test_get_adjacent_positions_corner(self) -> None:
        """Test getting adjacent positions from corner."""
        corner_pos: List[int] = [0, 0]
        
        adjacent: List[List[int]] = get_adjacent_positions(corner_pos)
        
        expected_positions: List[List[int]] = [
            [0, -1],  # UP (outside grid)
            [0, 1],   # DOWN
            [-1, 0],  # LEFT (outside grid)
            [1, 0]    # RIGHT
        ]
        
        assert len(adjacent) == 4
        for expected_pos in expected_positions:
            assert expected_pos in adjacent

    def test_get_adjacent_positions_filtered(self) -> None:
        """Test getting adjacent positions with boundary filtering."""
        pos: List[int] = [0, 0]
        grid_size: int = 10
        
        # Get only valid adjacent positions within grid
        adjacent: List[List[int]] = [
            adj_pos for adj_pos in get_adjacent_positions(pos)
            if 0 <= adj_pos[0] < grid_size and 0 <= adj_pos[1] < grid_size
        ]
        
        expected_valid: List[List[int]] = [
            [0, 1],   # DOWN
            [1, 0]    # RIGHT
        ]
        
        assert len(adjacent) == 2
        for expected_pos in expected_valid:
            assert expected_pos in adjacent

    def test_sequence_of_moves(self) -> None:
        """Test applying a sequence of moves."""
        start_pos: List[int] = [5, 5]
        moves_sequence: List[str] = ["UP", "UP", "RIGHT", "DOWN"]
        
        current_pos: List[int] = start_pos[:]
        expected_path: List[List[int]] = [
            [5, 5],   # Start
            [5, 4],   # After UP
            [5, 3],   # After UP
            [6, 3],   # After RIGHT
            [6, 4]    # After DOWN
        ]
        
        path: List[List[int]] = [current_pos[:]]
        for move in moves_sequence:
            current_pos = calculate_next_position(current_pos, move)
            path.append(current_pos[:])
        
        assert path == expected_path

    def test_complex_collision_scenario(self) -> None:
        """Test complex collision detection scenario."""
        grid_size: int = 5
        # Snake forming an L-shape
        snake_positions: List[List[int]] = [
            [2, 2], [2, 1], [2, 0], [3, 0], [4, 0]
        ]
        
        collision_test_cases: List[Tuple[List[int], bool]] = [
            ([2, 2], True),   # Snake head
            ([3, 0], True),   # Snake body
            ([1, 1], False),  # Empty adjacent
            ([2, 3], False),  # Empty below
            ([-1, 0], True),  # Wall collision
            ([5, 2], True)    # Wall collision
        ]
        
        for position, should_collide in collision_test_cases:
            result: bool = detect_collision(position, grid_size, snake_positions)
            assert result == should_collide

    def test_direction_consistency(self) -> None:
        """Test that direction operations are consistent."""
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            # Test that opposite of opposite equals original
            opposite: str = get_opposite_direction(direction)
            double_opposite: str = get_opposite_direction(opposite)
            assert double_opposite == direction
            
            # Test that direction vector and next position are consistent
            vector: Tuple[int, int] = get_direction_vector(direction)
            next_pos: List[int] = calculate_next_position([0, 0], direction)
            assert next_pos == [vector[0], vector[1]]

    def test_edge_case_zero_distance(self) -> None:
        """Test edge cases with zero distances."""
        same_pos: List[int] = [3, 7]
        
        # Distance to self should be 0
        distance: int = manhattan_distance(same_pos, same_pos)
        assert distance == 0
        
        # Adjacent positions should have distance 1
        adjacent: List[List[int]] = get_adjacent_positions(same_pos)
        for adj_pos in adjacent:
            adj_distance: int = manhattan_distance(same_pos, adj_pos)
            assert adj_distance == 1

    def test_moves_with_numpy_arrays(self) -> None:
        """Test move functions work with numpy arrays."""
        start_pos_np: NDArray[np.int_] = np.array([5, 5])
        start_pos_list: List[int] = start_pos_np.tolist()
        
        # Functions should work with list conversion
        next_pos: List[int] = calculate_next_position(start_pos_list, "UP")
        distance: int = manhattan_distance(start_pos_list, [6, 6])
        
        assert next_pos == [5, 4]
        assert distance == 2 