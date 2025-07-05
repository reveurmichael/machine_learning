#!/usr/bin/env python3
"""
Detailed debug script to test BFS pathfinding for the specific problematic case.
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

def bfs_pathfind_detailed(start: Tuple[int, int], goal: Tuple[int, int], 
                         obstacles: set, grid_size: int) -> List[Tuple[int, int]]:
    """Find shortest path using Breadth-First Search with detailed logging."""
    print(f"[BFS DETAILED] Start: {start}, Goal: {goal}")
    print(f"[BFS DETAILED] Obstacles: {obstacles}")
    
    if start == goal:
        print(f"[BFS DETAILED] Start equals goal, returning [{start}]")
        return [start]

    # BFS initialization
    queue = deque([(start, [start])])
    visited = {start}
    
    print(f"[BFS DETAILED] Initial queue: {queue}")
    print(f"[BFS DETAILED] Initial visited: {visited}")

    while queue:
        current_pos, path = queue.popleft()
        print(f"[BFS DETAILED] Processing current_pos: {current_pos}, path: {path}")

        # Check all adjacent positions
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dx, dy = DIRECTIONS[direction]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            print(f"[BFS DETAILED] Checking {direction}: {next_pos}")

            # Skip if out of bounds
            if not (0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size):
                print(f"[BFS DETAILED] {next_pos} out of bounds")
                continue

            # Skip if obstacle or already visited
            if next_pos in obstacles:
                print(f"[BFS DETAILED] {next_pos} is obstacle")
                continue
            if next_pos in visited:
                print(f"[BFS DETAILED] {next_pos} already visited")
                continue

            # Create new path
            new_path = path + [next_pos]
            print(f"[BFS DETAILED] New path: {new_path}")

            # Check if we reached the goal
            if next_pos == goal:
                print(f"[BFS DETAILED] Goal reached! Path: {new_path}")
                return new_path

            # Add to queue for further exploration
            queue.append((next_pos, new_path))
            visited.add(next_pos)
            print(f"[BFS DETAILED] Added {next_pos} to queue, visited: {visited}")

    # No path found
    print(f"[BFS DETAILED] No path found from {start} to {goal}")
    return []

# Test the specific case from the JSONL entry
if __name__ == "__main__":
    # Case from the JSONL entry that shows wrong path length
    head = (3, 8)
    apple = (3, 9)
    snake = [(3, 8), (3, 7), (3, 6), (2, 6)]  # head to tail
    grid_size = 10
    
    # Obstacles should be body segments (excluding head)
    obstacles = set(snake[1:])  # [(3, 7), (3, 6), (2, 6)]
    
    print("=== Detailed BFS Debug Test ===")
    print(f"Head: {head}")
    print(f"Apple: {apple}")
    print(f"Snake: {snake}")
    print(f"Obstacles: {obstacles}")
    print(f"Manhattan distance: {abs(head[0] - apple[0]) + abs(head[1] - apple[1])}")
    
    # Test direct path
    direct_path = bfs_pathfind_detailed(head, apple, obstacles, grid_size)
    print(f"BFS path length: {len(direct_path) - 1 if direct_path else 'None'}")
    
    # Test what happens when we try to move UP
    up_pos = (head[0] + DIRECTIONS["UP"][0], head[1] + DIRECTIONS["UP"][1])
    print(f"UP position: {up_pos}")
    print(f"Is UP blocked by obstacles? {up_pos in obstacles}")
    print(f"Is UP in bounds? {0 <= up_pos[0] < grid_size and 0 <= up_pos[1] < grid_size}")
    
    # Test what happens when we try to move RIGHT
    right_pos = (head[0] + DIRECTIONS["RIGHT"][0], head[1] + DIRECTIONS["RIGHT"][1])
    print(f"RIGHT position: {right_pos}")
    print(f"Is RIGHT blocked by obstacles? {right_pos in obstacles}")
    
    # Test what happens when we try to move LEFT
    left_pos = (head[0] + DIRECTIONS["LEFT"][0], head[1] + DIRECTIONS["LEFT"][1])
    print(f"LEFT position: {left_pos}")
    print(f"Is LEFT blocked by obstacles? {left_pos in obstacles}")
    
    # Test what happens when we try to move DOWN
    down_pos = (head[0] + DIRECTIONS["DOWN"][0], head[1] + DIRECTIONS["DOWN"][1])
    print(f"DOWN position: {down_pos}")
    print(f"Is DOWN blocked by obstacles? {down_pos in obstacles}") 