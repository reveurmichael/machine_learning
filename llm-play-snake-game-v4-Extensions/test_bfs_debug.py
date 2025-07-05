#!/usr/bin/env python3
"""
Debug script to test BFS pathfinding for the specific case.
"""

from collections import deque
from typing import List, Tuple

# Game constants
DIRECTIONS = {
    "UP": (0, 1),
    "DOWN": (0, -1), 
    "LEFT": (-1, 0),
    "RIGHT": (1, 0)
}

def bfs_pathfind(start: Tuple[int, int], goal: Tuple[int, int], 
                obstacles: set, grid_size: int) -> List[Tuple[int, int]]:
    """Find shortest path using Breadth-First Search."""
    print(f"[BFS DEBUG] Start: {start}, Goal: {goal}, Obstacles: {obstacles}")
    if start == goal:
        print(f"[BFS DEBUG] Start equals goal, returning [{start}]")
        return [start]

    # BFS initialization
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current_pos, path = queue.popleft()

        # Check all adjacent positions
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dx, dy = DIRECTIONS[direction]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)

            # Skip if out of bounds
            if not (0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size):
                continue

            # Skip if obstacle or already visited
            if next_pos in obstacles or next_pos in visited:
                continue

            # Create new path
            new_path = path + [next_pos]

            # Check if we reached the goal
            if next_pos == goal:
                print(f"[BFS DEBUG] Path found: {new_path}")
                return new_path

            # Add to queue for further exploration
            queue.append((next_pos, new_path))
            visited.add(next_pos)

    # No path found
    print(f"[BFS DEBUG] No path found from {start} to {goal}")
    return []

# Test the specific case
if __name__ == "__main__":
    # Case from the JSONL entry
    head = (3, 8)
    apple = (3, 9)
    snake = [(3, 8), (3, 7), (3, 6), (2, 6)]  # head to tail
    grid_size = 10
    
    # Obstacles should be body segments (excluding head)
    obstacles = set(snake[1:])  # [(3, 7), (3, 6), (2, 6)]
    
    print("=== BFS Debug Test ===")
    print(f"Head: {head}")
    print(f"Apple: {apple}")
    print(f"Snake: {snake}")
    print(f"Obstacles: {obstacles}")
    print(f"Manhattan distance: {abs(head[0] - apple[0]) + abs(head[1] - apple[1])}")
    
    # Test direct path
    direct_path = bfs_pathfind(head, apple, obstacles, grid_size)
    print(f"BFS path length: {len(direct_path) - 1 if direct_path else 'None'}")
    
    # Test what happens when we try to move UP
    up_pos = (head[0] + DIRECTIONS["UP"][0], head[1] + DIRECTIONS["UP"][1])
    print(f"UP position: {up_pos}")
    print(f"Is UP blocked by obstacles? {up_pos in obstacles}")
    print(f"Is UP in bounds? {0 <= up_pos[0] < grid_size and 0 <= up_pos[1] < grid_size}") 