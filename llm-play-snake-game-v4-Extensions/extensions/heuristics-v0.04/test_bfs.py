#!/usr/bin/env python3
"""
Test script to debug BFS pathfinding issue.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Import directly from the local file
from ssot_utils import ssot_bfs_pathfind

def test_bfs_case():
    """Test the specific case mentioned by the user."""
    start = [7, 2]
    goal = [2, 1]
    obstacles = set()
    grid_size = 10
    
    print(f"Testing BFS from {start} to {goal}")
    print(f"Obstacles: {obstacles}")
    print(f"Grid size: {grid_size}")
    
    path = ssot_bfs_pathfind(start, goal, obstacles, grid_size)
    
    print(f"Path found: {path}")
    if path:
        print(f"Path length: {len(path) - 1}")
        print(f"Steps:")
        for i, pos in enumerate(path):
            print(f"  {i}: {pos}")
    else:
        print("No path found!")
    
    # Manual verification
    print(f"\nManual verification:")
    print(f"Manhattan distance: {abs(start[0] - goal[0]) + abs(start[1] - goal[1])}")
    print(f"Expected minimum path: LEFT×5 + DOWN = 6 steps")

if __name__ == "__main__":
    test_bfs_case() 