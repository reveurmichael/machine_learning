#!/usr/bin/env python3
"""
Test script to verify the specific cases mentioned by the user.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import SSOT utilities directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extensions" / "heuristics-v0.04"))
from ssot_utils import ssot_bfs_pathfind, ssot_calculate_valid_moves

def test_case_1():
    """Test case 1: Head at [4, 0], Apple at [8, 0]"""
    print("=== Test Case 1 ===")
    head_pos = [4, 0]
    apple_pos = [8, 0]
    grid_size = 10
    
    # Snake body from the board visualization
    # Based on the board, the snake body forms a pattern
    snake_positions = [
        [2, 0],  # tail
        [3, 0],
        [4, 0],  # head
    ]
    
    print(f"Head: {head_pos}")
    print(f"Apple: {apple_pos}")
    print(f"Snake: {snake_positions}")
    
    # Test valid moves
    valid_moves = ssot_calculate_valid_moves(head_pos, snake_positions, grid_size)
    print(f"Valid moves: {valid_moves}")
    
    # Test BFS pathfinding
    obstacles = set(tuple(p) for p in snake_positions[:-1] if len(p) >= 2)
    path = ssot_bfs_pathfind(head_pos, apple_pos, obstacles, grid_size)
    path_length = len(path) - 1 if path else None
    print(f"BFS path: {path}")
    print(f"Path length: {path_length}")
    print(f"Expected: 4 (RIGHT×4)")
    print(f"Correct: {path_length == 4}")

def test_case_2():
    """Test case 2: Head at [5, 0], Apple at [8, 0]"""
    print("\n=== Test Case 2 ===")
    head_pos = [5, 0]
    apple_pos = [8, 0]
    grid_size = 10
    
    # Snake body from the board visualization
    snake_positions = [
        [2, 0],  # tail
        [3, 0],
        [4, 0],
        [5, 0],  # head
    ]
    
    print(f"Head: {head_pos}")
    print(f"Apple: {apple_pos}")
    print(f"Snake: {snake_positions}")
    
    # Test valid moves
    valid_moves = ssot_calculate_valid_moves(head_pos, snake_positions, grid_size)
    print(f"Valid moves: {valid_moves}")
    
    # Test BFS pathfinding
    obstacles = set(tuple(p) for p in snake_positions[:-1] if len(p) >= 2)
    path = ssot_bfs_pathfind(head_pos, apple_pos, obstacles, grid_size)
    path_length = len(path) - 1 if path else None
    print(f"BFS path: {path}")
    print(f"Path length: {path_length}")
    print(f"Expected: 3 (RIGHT×3)")
    print(f"Correct: {path_length == 3}")

def test_case_2_with_up_collision():
    """Test case 2 with UP collision scenario"""
    print("\n=== Test Case 2 (UP Collision) ===")
    head_pos = [5, 0]
    apple_pos = [8, 0]
    grid_size = 10
    
    # Snake body with collision above head
    snake_positions = [
        [2, 0],  # tail
        [3, 0],
        [4, 0],
        [5, 1],  # body above head (collision)
        [5, 0],  # head
    ]
    
    print(f"Head: {head_pos}")
    print(f"Apple: {apple_pos}")
    print(f"Snake: {snake_positions}")
    
    # Test valid moves
    valid_moves = ssot_calculate_valid_moves(head_pos, snake_positions, grid_size)
    print(f"Valid moves: {valid_moves}")
    print(f"Expected: ['RIGHT'] (UP should be blocked)")
    print(f"Correct: {valid_moves == ['RIGHT']}")

if __name__ == "__main__":
    test_case_1()
    test_case_2()
    test_case_2_with_up_collision() 