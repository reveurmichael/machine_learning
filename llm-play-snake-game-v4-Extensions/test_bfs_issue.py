#!/usr/bin/env python3
"""
Test script to debug the specific BFS pathfinding issue.
"""

from utils.ssot_utils import ssot_bfs_pathfind, ssot_calculate_valid_moves

def test_specific_case():
    """Test the specific case from the JSONL entry."""
    
    # Game state from the JSONL entry
    head_pos = [8, 7]
    apple_pos = [9, 5]
    grid_size = 10
    
    # Based on the board visualization and coordinate system:
    # Head is at [8, 7] (x=8, y=7)
    # Apple is at [9, 5] (x=9, y=5)
    # Snake body forms a path from bottom-left to the head
    
    # Let me reconstruct the snake body based on the board:
    # The snake appears to be in a pattern like this:
    snake_positions = [
        [2, 2],  # tail (bottom-left of snake body)
        [2, 3],
        [2, 4], 
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [3, 9],
        [4, 9],
        [5, 9],
        [6, 9],
        [7, 9],
        [8, 9],
        [8, 8],
        [8, 7],  # head
    ]
    
    print(f"Head position: {head_pos}")
    print(f"Apple position: {apple_pos}")
    print(f"Snake length: {len(snake_positions)}")
    
    # Calculate obstacles (all body except head)
    obstacles = set(tuple(p) for p in snake_positions[:-1] if len(p) >= 2)
    print(f"Obstacles count: {len(obstacles)}")
    
    # Test BFS pathfinding
    print(f"\nTesting BFS from {head_pos} to {apple_pos}...")
    path = ssot_bfs_pathfind(head_pos, apple_pos, obstacles, grid_size)
    print(f"Path found: {path}")
    print(f"Path length: {len(path) - 1 if path else None}")
    
    # Expected path: [8,7] → [9,7] → [9,6] → [9,5] (3 steps)
    expected_path = [[8, 7], [9, 7], [9, 6], [9, 5]]
    print(f"Expected path: {expected_path}")
    print(f"Expected length: 3")
    
    if path == expected_path:
        print("✅ BFS is working correctly!")
    else:
        print("❌ BFS is not finding the correct path!")
        
        # Debug: check if any positions in the expected path are blocked
        for i, pos in enumerate(expected_path):
            if tuple(pos) in obstacles:
                print(f"❌ Position {pos} (step {i}) is blocked by obstacles!")
            else:
                print(f"✅ Position {pos} (step {i}) is free")
        
        # Also check if the start or goal positions are blocked
        if tuple(head_pos) in obstacles:
            print(f"❌ Start position {head_pos} is blocked!")
        if tuple(apple_pos) in obstacles:
            print(f"❌ Goal position {apple_pos} is blocked!")

if __name__ == "__main__":
    test_specific_case() 