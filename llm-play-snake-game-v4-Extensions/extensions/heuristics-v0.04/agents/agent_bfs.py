from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Agent - Breadth-First Search pathfinding for Snake Game v0.04
--------------------

This module implements a BFS (Breadth-First Search) agent that finds
the shortest path to the apple while avoiding walls and its own body.

v0.04 Enhancement: Generates natural language explanations for each move
to create rich datasets for LLM fine-tuning.

The agent implements the BaseAgent protocol, making it compatible with
the existing game engine infrastructure.

Design Patterns:
- Strategy Pattern: BFS algorithm encapsulated as a strategy
- Protocol Pattern: Implements BaseAgent interface for compatibility
"""

from collections import deque
from typing import List, Tuple, TYPE_CHECKING

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports
from config.game_constants import DIRECTIONS, VALID_MOVES
from utils.moves_utils import position_to_direction
from utils.print_utils import print_error

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
        Get next move using BFS pathfinding (simplified interface).
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        move, _ = self.get_move_with_explanation(game)
        return move
        
    def get_move_with_explanation(self, game: HeuristicGameLogic) -> Tuple[str, dict]:
        """
        Get next move using BFS pathfinding with detailed explanation.
        
        v0.04 Enhancement: Returns both move and natural language explanation
        for LLM fine-tuning dataset generation.
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Tuple of (move, explanation_dict) where:
            - move: Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
            - explanation_dict: Dictionary with explanation and metrics
        """
        try:
            # Extract game state
            head_pos = tuple(game.head_position)
            apple_pos = tuple(game.apple_position)
            snake_positions = {tuple(pos) for pos in game.snake_positions}
            grid_size = game.grid_size
            
            # ------------------------ Extra metrics for v0.04 JSONL ------------------------
            valid_moves = self._get_valid_moves(head_pos, snake_positions, grid_size)
            remaining_free_cells = self._count_remaining_free_cells(snake_positions, grid_size)
            manhattan_distance = abs(apple_pos[0] - head_pos[0]) + abs(apple_pos[1] - head_pos[1])
            # --------------------------------------------------------------------------------
            
            # Find path using BFS
            path = self._bfs_pathfind(head_pos, apple_pos, snake_positions, grid_size)
            
            if not path or len(path) < 2:
                natural_language_summary = self._generate_no_path_explanation(
                    head_pos, apple_pos, snake_positions, grid_size
                )
                metrics = {
                    "manhattan_distance": manhattan_distance,
                    "path_length": 0,
                    "obstacles_near_path": 0,
                    "remaining_free_cells": remaining_free_cells,
                    "valid_moves": valid_moves,
                    "final_chosen_direction": "NO_PATH_FOUND",
                }

                explanation_dict = {
                    "strategy_phase": "NO_PATH",
                    "metrics": metrics,
                    "explanation_steps": [natural_language_summary],
                    "natural_language_summary": natural_language_summary,
                }
                return "NO_PATH_FOUND", explanation_dict
                
            # Get first move in path
            next_pos = path[1]  # path[0] is current head position
            direction = position_to_direction(head_pos, next_pos)
            
            # Generate explanation for this move
            explanation_steps_text = self._generate_move_explanation(
                head_pos=head_pos,
                apple_pos=apple_pos,
                snake_positions=snake_positions,
                path=path,
                direction=direction,
                valid_moves=valid_moves,
                manhattan_distance=manhattan_distance,
                remaining_free_cells=remaining_free_cells
            )
            
            # ----------------------- Structured metrics -----------------------
            metrics = {
                "manhattan_distance": manhattan_distance,
                "path_length": len(path) - 1,
                "obstacles_near_path": self._count_obstacles_in_path(path, snake_positions),
                "remaining_free_cells": remaining_free_cells,
                "valid_moves": valid_moves,
                "final_chosen_direction": direction,
            }

            explanation_dict = {
                "strategy_phase": "APPLE_PATH",
                "metrics": metrics,
                "explanation_steps": explanation_steps_text.split("\n"),
                "natural_language_summary": f"BFS chooses {direction} as the first step in a shortest path of length {metrics['path_length']} to the apple.",
            }

            return direction, explanation_dict
            
        except Exception as e:
            error_explanation = f"BFS agent encountered an error: {str(e)}. Unable to compute safe path."
            print_error(f"BFS Agent error: {e}")
            explanation_dict = {
                "strategy_phase": "ERROR",
                "metrics": {},
                "explanation_steps": [error_explanation],
                "natural_language_summary": error_explanation,
            }
            return "NO_PATH_FOUND", explanation_dict
    
    def _generate_move_explanation(self, head_pos: Tuple[int, int], apple_pos: Tuple[int, int], 
                                 snake_positions: set, path: List[Tuple[int, int]], 
                                 direction: str,
                                 valid_moves: List[str],
                                 manhattan_distance: int,
                                 remaining_free_cells: int) -> str:
        """
        Generate rich, step-by-step natural language explanation for the chosen move.
        
        The explanation follows a **standardised template** so that downstream
        JSONL records are consistent across all heuristic agents.  It explicitly
        lists intermediate reasoning steps, quantitative metrics, and a final
        conclusion section.  This improves LLM-readability while remaining
        human-friendly.
        
        Args:
            head_pos: Current head position.
            apple_pos: Apple position.
            snake_positions: Set of snake body positions.
            path: Full BFS path to the apple (including head & goal).
            direction: Chosen move direction (UP/DOWN/LEFT/RIGHT).
            valid_moves: List of valid immediate moves.
            manhattan_distance: Manhattan distance from head to apple.
            remaining_free_cells: Count of free cells on the board.
        """
        path_length = len(path) - 1  # Exclude starting position
        obstacles_avoided = self._count_obstacles_in_path(path, snake_positions)

        # ------------------------------------------------------------------
        # TEMPLATE – Broken into logical sections (Instruction ↔ Analysis)
        # ------------------------------------------------------------------
        explanation_parts = [
            "Step-by-step BFS reasoning:",
            f"1️⃣ Identify valid immediate moves: {valid_moves}.",
            "2️⃣ Discard moves leading to collisions with walls or snake body segments.",
            f"3️⃣ For the remaining safe moves, run BFS to the apple at {apple_pos}.",
            f"4️⃣ BFS discovered a shortest path of length {path_length} from {head_pos} to the apple.",
            f"5️⃣ The chosen move '{direction}' is the first step on that path, bringing the snake closer to the target.",
            "",  # Spacer line
            "Additional analysis:",
            f"- Manhattan distance to apple: {manhattan_distance}.",
            f"- Obstacles encountered/avoided along the path: {obstacles_avoided}.",
            f"- Remaining free cells on the board: {remaining_free_cells}.",
            f"- Path optimality: {'Perfectly optimal – no detours.' if path_length == manhattan_distance else f'Includes {path_length - manhattan_distance} detour step(s) to avoid obstacles.'}",
            "",  # Spacer line
            "Conclusion:",
            f"By moving '{direction}', the snake safely advances toward the apple along the shortest known route, minimising risk and maximising future reward."
        ]

        return "\n".join(explanation_parts)
    
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
    
    # --------------------------------------------------------------------------
    # Helper utilities (v0.04 additions)
    # --------------------------------------------------------------------------
    def _get_valid_moves(self, head_pos: Tuple[int, int], snake_positions: set, grid_size: int) -> List[str]:
        """Return list of currently valid moves based on obstacles and walls."""
        valid_moves = []
        for move in VALID_MOVES:
            dx, dy = DIRECTIONS[move]
            next_pos = (head_pos[0] + dx, head_pos[1] + dy)
            if (
                0 <= next_pos[0] < grid_size and
                0 <= next_pos[1] < grid_size and
                next_pos not in snake_positions
            ):
                valid_moves.append(move)
        return valid_moves

    def _count_remaining_free_cells(self, snake_positions: set, grid_size: int) -> int:
        """Count how many empty cells are not occupied by the snake body."""
        total_cells = grid_size * grid_size
        return total_cells - len(snake_positions)

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"BFSAgent(algorithm={self.algorithm_name})" 
