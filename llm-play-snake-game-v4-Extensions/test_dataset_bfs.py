#!/usr/bin/env python3
"""
Test the dataset generator's BFS implementation for the specific case.
"""

from typing import List, Optional, Set, Tuple

def dataset_bfs_pathfind(start: List[int], goal: List[int], obstacles: Set[Tuple[int, int]], grid_size: int) -> Optional[List[List[int]]]:
    """
    BFS pathfinding from start to goal, avoiding obstacles.
    
    Args:
        start: Starting position [x, y]
        goal: Goal position [x, y]
        obstacles: Set of obstacle positions as tuples (x, y)
        grid_size: Size of the game grid
        
    Returns:
        List of positions forming the path from start to goal, or None if no path exists
    """
    if start == goal:
        return [start]
    
    # Convert to tuples for set operations
    start_tuple = tuple(start)
    goal_tuple = tuple(goal)
    
    if start_tuple in obstacles or goal_tuple in obstacles:
        return None
    
    # BFS queue: (position, path)
    queue = [(start_tuple, [start])]
    visited = {start_tuple}
    
    # Direction vectors for BFS
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
    
    while queue:
        current_pos, path = queue.pop(0)
        
        for dx, dy in directions:
            next_x = current_pos[0] + dx
            next_y = current_pos[1] + dy
            next_pos = (next_x, next_y)
            
            # Check bounds
            if not (0 <= next_x < grid_size and 0 <= next_y < grid_size):
                continue
            
            # Check if visited or obstacle
            if next_pos in visited or next_pos in obstacles:
                continue
            
            # Check if goal reached
            if next_pos == goal_tuple:
                return path + [[next_x, next_y]]
            
            # Add to queue
            visited.add(next_pos)
            queue.append((next_pos, path + [[next_x, next_y]]))
    
    return None

# Test the specific case from the JSONL entry
if __name__ == "__main__":
    # Case from the JSONL entry that shows wrong path length
    # Snake structure: tail at index 0, head at index -1
    head = [3, 8]
    apple = [3, 9]
    snake = [[2, 6], [3, 6], [3, 7], [3, 8]]  # tail to head
    grid_size = 10
    
    # Obstacles should be body segments (excluding head)
    obstacles = set(tuple(p) for p in snake[:-1])  # [(2, 6), (3, 6), (3, 7)]
    
    print("=== Dataset Generator BFS Test ===")
    print(f"Head: {head}")
    print(f"Apple: {apple}")
    print(f"Snake: {snake}")
    print(f"Obstacles: {obstacles}")
    print(f"Manhattan distance: {abs(head[0] - apple[0]) + abs(head[1] - apple[1])}")
    
    # Test direct path
    direct_path = dataset_bfs_pathfind(head, apple, obstacles, grid_size)
    print(f"Dataset BFS path: {direct_path}")
    print(f"Dataset BFS path length: {len(direct_path) - 1 if direct_path else 'None'}")
    
    # Test what happens when we try to move UP
    up_pos = (head[0] + 0, head[1] + 1)
    print(f"UP position: {up_pos}")
    print(f"Is UP blocked by obstacles? {up_pos in obstacles}")
    print(f"Is UP in bounds? {0 <= up_pos[0] < grid_size and 0 <= up_pos[1] < grid_size}")
    
    # Test what happens when we try to move RIGHT
    right_pos = (head[0] + 1, head[1] + 0)
    print(f"RIGHT position: {right_pos}")
    print(f"Is RIGHT blocked by obstacles? {right_pos in obstacles}")
    
    # Test what happens when we try to move LEFT
    left_pos = (head[0] - 1, head[1] + 0)
    print(f"LEFT position: {left_pos}")
    print(f"Is LEFT blocked by obstacles? {left_pos in obstacles}")
    
    # Test what happens when we try to move DOWN
    down_pos = (head[0] + 0, head[1] - 1)
    print(f"DOWN position: {down_pos}")
    print(f"Is DOWN blocked by obstacles? {down_pos in obstacles}") 