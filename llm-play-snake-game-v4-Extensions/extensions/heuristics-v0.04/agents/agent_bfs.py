"""
BFS Agent - Breadth-First Search pathfinding for Snake Game v0.04
--------------------

This module implements a BFS (Breadth-First Search) agent that finds
the shortest path to the apple while avoiding walls and its own body.

v0.04 Enhancement: Generates natural language explanations for each move
to create rich datasets for LLM fine-tuning.

The agent implements the SnakeAgent protocol, making it compatible with
the existing game engine infrastructure.

Design Patterns:
- Strategy Pattern: BFS algorithm encapsulated as a strategy
- Protocol Pattern: Implements SnakeAgent interface for compatibility
"""

from __future__ import annotations
from collections import deque
from typing import List, Tuple, TYPE_CHECKING

# Use standardized path setup
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from config import DIRECTIONS
from utils.moves_utils import position_to_direction

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class BFSAgent:
    """
    Breadth-First Search agent for Snake game with explanation generation.
    
    This agent uses BFS to find the shortest path from the snake's head
    to the apple, avoiding obstacles (walls and snake body).
    
    v0.04 Feature: Generates detailed natural language explanations
    for each move decision to support LLM fine-tuning datasets.
    
    Algorithm:
    1. Start from current head position
    2. Explore all valid adjacent positions using BFS
    3. Return the first move in the shortest path to apple
    4. Generate explanation describing the reasoning
    5. If no path exists, return "NO_PATH_FOUND" with explanation
    
    Design Patterns:
    - Strategy Pattern: BFS pathfinding strategy
    - Template Method: Generic pathfinding with BFS implementation
    """
    
    def __init__(self):
        """Initialize BFS agent."""
        self.algorithm_name = "BFS"
        
    def get_move(self, game: HeuristicGameLogic) -> str | None:
        """
        Get next move using BFS pathfinding (legacy method for v0.03 compatibility).
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        move, _ = self.get_move_with_explanation(game)
        return move
        
    def get_move_with_explanation(self, game: HeuristicGameLogic) -> Tuple[str, str]:
        """
        Get next move using BFS pathfinding with detailed explanation.
        
        v0.04 Enhancement: Returns both move and natural language explanation
        for LLM fine-tuning dataset generation.
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Tuple of (move, explanation) where:
            - move: Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
            - explanation: Natural language description of the reasoning
        """
        try:
            # Extract game state
            head_pos = tuple(game.head_position)
            apple_pos = tuple(game.apple_position)
            snake_positions = {tuple(pos) for pos in game.snake_positions}
            grid_size = game.grid_size
            
            # Find path using BFS
            path = self._bfs_pathfind(head_pos, apple_pos, snake_positions, grid_size)
            
            if not path or len(path) < 2:
                explanation = self._generate_no_path_explanation(
                    head_pos, apple_pos, snake_positions, grid_size
                )
                return "NO_PATH_FOUND", explanation
                
            # Get first move in path
            next_pos = path[1]  # path[0] is current head position
            direction = position_to_direction(head_pos, next_pos)
            
            # Generate explanation for this move
            explanation = self._generate_move_explanation(
                head_pos, apple_pos, snake_positions, path, direction
            )
            
            return direction, explanation
            
        except Exception as e:
            error_explanation = f"BFS agent encountered an error: {str(e)}. Unable to compute safe path."
            print(f"BFS Agent error: {e}")
            return "NO_PATH_FOUND", error_explanation
    
    def _generate_move_explanation(self, head_pos: Tuple[int, int], apple_pos: Tuple[int, int], 
                                 snake_positions: set, path: List[Tuple[int, int]], 
                                 direction: str) -> str:
        """
        Generate natural language explanation for the chosen move.
        
        Args:
            head_pos: Current head position
            apple_pos: Apple position
            snake_positions: Set of snake body positions
            path: BFS path to apple
            direction: Chosen direction
            
        Returns:
            Natural language explanation of the move reasoning
        """
        path_length = len(path) - 1  # Exclude starting position
        body_count = len(snake_positions)
        
        # Calculate distance to apple
        manhattan_distance = abs(apple_pos[0] - head_pos[0]) + abs(apple_pos[1] - head_pos[1])
        
        # Determine relative position of apple
        apple_direction = self._get_apple_direction(head_pos, apple_pos)
        
        # Check for obstacles in the chosen direction
        obstacles_avoided = self._count_obstacles_in_path(path, snake_positions)
        
        # Generate comprehensive explanation
        explanation_parts = [
            f"BFS found shortest path of length {path_length} from {head_pos} to apple at {apple_pos}."
        ]
        
        if body_count > 0:
            explanation_parts.append(f"Successfully avoiding {obstacles_avoided} snake body segments.")
        
        explanation_parts.append(f"The apple is {apple_direction} of current position.")
        explanation_parts.append(f"Moving {direction} (step 1 of {path_length}) follows optimal path.")
        
        if path_length == manhattan_distance:
            explanation_parts.append("This path is perfectly optimal with no detours needed.")
        else:
            detour_steps = path_length - manhattan_distance
            explanation_parts.append(f"Path includes {detour_steps} detour steps to avoid obstacles.")
        
        return " ".join(explanation_parts)
    
    def _generate_no_path_explanation(self, head_pos: Tuple[int, int], apple_pos: Tuple[int, int],
                                    snake_positions: set, grid_size: int) -> str:
        """
        Generate explanation when no path to apple is found.
        
        Args:
            head_pos: Current head position
            apple_pos: Apple position
            snake_positions: Set of snake body positions
            grid_size: Grid size
            
        Returns:
            Natural language explanation of why no path exists
        """
        body_count = len(snake_positions)
        manhattan_distance = abs(apple_pos[0] - head_pos[0]) + abs(apple_pos[1] - head_pos[1])
        
        # Check if apple is blocked by body
        apple_neighbors = self._get_neighbors(apple_pos, grid_size)
        blocked_neighbors = sum(1 for pos in apple_neighbors if pos in snake_positions)
        
        explanation_parts = [
            f"BFS could not find any path from {head_pos} to apple at {apple_pos}."
        ]
        
        if blocked_neighbors == len(apple_neighbors):
            explanation_parts.append("The apple is completely surrounded by snake body segments.")
        elif body_count > (grid_size * grid_size) // 2:
            explanation_parts.append(f"Snake body ({body_count} segments) has created too many obstacles.")
        else:
            explanation_parts.append("Available space is insufficient to reach the apple safely.")
        
        explanation_parts.append(f"Manhattan distance to apple is {manhattan_distance}, but path is blocked.")
        explanation_parts.append("Need to find alternative strategy or wait for tail to move.")
        
        return " ".join(explanation_parts)
    
    def _get_apple_direction(self, head_pos: Tuple[int, int], apple_pos: Tuple[int, int]) -> str:
        """Get relative direction description of apple from head."""
        dx = apple_pos[0] - head_pos[0]
        dy = apple_pos[1] - head_pos[1]
        
        if dx == 0 and dy == 0:
            return "at the same position as"
        
        directions = []
        if dy < 0:
            directions.append("above")
        elif dy > 0:
            directions.append("below")
            
        if dx > 0:
            directions.append("to the right")
        elif dx < 0:
            directions.append("to the left")
        
        if len(directions) == 2:
            return f"{directions[0]} and {directions[1]}"
        elif len(directions) == 1:
            return directions[0]
        else:
            return "at the same position as"
    
    def _count_obstacles_in_path(self, path: List[Tuple[int, int]], snake_positions: set) -> int:
        """Count how many snake body segments are near the path."""
        obstacles_near_path = 0
        for pos in path:
            # Check adjacent positions for snake body
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                adjacent_pos = (pos[0] + dx, pos[1] + dy)
                if adjacent_pos in snake_positions:
                    obstacles_near_path += 1
        return obstacles_near_path
    
    def _get_neighbors(self, pos: Tuple[int, int], grid_size: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (pos[0] + dx, pos[1] + dy)
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                neighbors.append(neighbor)
        return neighbors
    
    def _bfs_pathfind(self, start: Tuple[int, int], goal: Tuple[int, int], 
                     obstacles: set, grid_size: int) -> List[Tuple[int, int]]:
        """
        Find shortest path using Breadth-First Search.
        
        Args:
            start: Starting position (head)
            goal: Goal position (apple)
            obstacles: Set of obstacle positions (snake body)
            grid_size: Size of the game grid
            
        Returns:
            List of positions representing the path, or empty list if no path
        """
        if start == goal:
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
                    return new_path
                    
                # Add to queue for further exploration
                queue.append((next_pos, new_path))
                visited.add(next_pos)
        
        # No path found
        return []
    

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"BFSAgent(algorithm={self.algorithm_name})" 