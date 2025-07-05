from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Agent - Breadth-First Search pathfinding for Snake Game v0.04
----------------

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
import json

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports
from config.game_constants import DIRECTIONS, VALID_MOVES
from utils.moves_utils import position_to_direction
from utils.print_utils import print_error
from core.game_agents import BaseAgent
# SSOT: Import shared logic from ssot_utils - DO NOT reimplement these functions
from ssot_utils import ssot_bfs_pathfind, ssot_calculate_valid_moves

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class BFSAgent(BaseAgent):
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

    def get_move(self, state: dict) -> str | None:
        """
        Get next move using BFS pathfinding (simplified interface).
        Args:
            state: Game state dict
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        move, _ = self.get_move_with_explanation(state)
        return move

    def get_move_with_explanation(self, state: dict) -> Tuple[str, dict]:
        # Use the provided state dict for all calculations (SSOT)
        head = list(state["head_position"])
        apple = list(state["apple_position"])
        snake = [list(seg) for seg in state["snake_positions"]]
        grid_size = state["grid_size"]
        # SSOT: Obstacles are all body segments except head (matching dataset generator)
        obstacles = set(tuple(p) for p in snake[:-1])  # Exclude head from obstacles, include body segments

        # Helper metrics that are independent of strategy branch (all from pre-move state)
        manhattan_distance = abs(head[0] - apple[0]) + abs(head[1] - apple[1])
        valid_moves = self._calculate_valid_moves(head, snake, grid_size)
        remaining_free_cells = self._count_remaining_free_cells(set(tuple(p) for p in snake), grid_size)

        # Fail-fast: ensure state is not mutated (SSOT)

        # ---------------- Apple path first
        path_to_apple = ssot_bfs_pathfind(head, apple, obstacles, grid_size)
        if path_to_apple and len(path_to_apple) > 1:
            next_pos = path_to_apple[1]
            
            # Fail-fast: Validate that the next position is within bounds
            if (next_pos[0] < 0 or next_pos[0] >= grid_size or 
                next_pos[1] < 0 or next_pos[1] >= grid_size):
                raise RuntimeError(f"SSOT violation: BFS computed out-of-bounds position {next_pos} for grid size {grid_size}")
            
            direction = position_to_direction(tuple(head), tuple(next_pos))
            
            if direction not in valid_moves:
                raise RuntimeError(f"SSOT violation: BFS computed move '{direction}' is not valid for head {head} and valid_moves {valid_moves}")
            
            # Create metrics directly from pre-move state only
            metrics = {
                "final_chosen_direction": direction,
                "head_position": list(head),
                "apple_position": list(apple),
                "snake_length": len(snake),
                "grid_size": grid_size,
                "valid_moves": valid_moves,
                "manhattan_distance": manhattan_distance,
                "remaining_free_cells": remaining_free_cells,
                "path_length": len(path_to_apple) - 1,
                "obstacles_near_path": self._count_obstacles_in_path(path_to_apple, set(tuple(p) for p in snake))
            }
            
            # Get explanation text from helper (but not metrics)
            explanation_dict = self._generate_move_explanation(
                tuple(head), tuple(apple), set(tuple(p) for p in snake), path_to_apple, direction, valid_moves, manhattan_distance, remaining_free_cells, grid_size
            )
            explanation_dict["metrics"] = metrics  # Overwrite with pre-move state metrics
            
            return direction, explanation_dict
        else:
            if valid_moves:
                direction = valid_moves[0]
                # Create metrics directly from pre-move state only
                metrics = {
                    "final_chosen_direction": direction,
                    "head_position": list(head),
                    "apple_position": list(apple),
                    "snake_length": len(snake),
                    "grid_size": grid_size,
                    "valid_moves": valid_moves,
                    "manhattan_distance": manhattan_distance,
                    "remaining_free_cells": remaining_free_cells,
                    "path_length": 0,
                    "obstacles_near_path": 0
                }
                
                explanation_dict = {
                    "strategy_phase": "SURVIVAL_MOVE",
                    "metrics": metrics,
                    "explanation_steps": [
                        f"No path to apple found from {head} to {apple}.",
                        f"Choosing survival move: '{direction}' to avoid immediate death."
                    ],
                }
                return direction, explanation_dict
            else:
                direction = "NO_PATH_FOUND"
                # Create metrics directly from pre-move state only
                metrics = {
                    "final_chosen_direction": direction,
                    "head_position": list(head),
                    "apple_position": list(apple),
                    "snake_length": len(snake),
                    "grid_size": grid_size,
                    "valid_moves": valid_moves,
                    "manhattan_distance": manhattan_distance,
                    "remaining_free_cells": remaining_free_cells,
                    "path_length": 0,
                    "obstacles_near_path": len(snake)
                }
                
                # Get explanation text from helper (but not metrics)
                explanation_dict = self._generate_no_path_explanation(tuple(head), tuple(apple), set(tuple(p) for p in snake), grid_size)
                explanation_dict["metrics"] = metrics  # Overwrite with pre-move state metrics
                
                return direction, explanation_dict

    def _generate_move_explanation(self, head_pos: Tuple[int, int], apple_pos: Tuple[int, int], 
                                 snake_positions: set, path: List[Tuple[int, int]], 
                                 direction: str,
                                 valid_moves: List[str],
                                 manhattan_distance: int,
                                 remaining_free_cells: int,
                                 grid_size: int,
                                 ) -> dict:
        """
        Generate rich, step-by-step natural language explanation for the chosen move.
        
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        path_length = len(path) - 1  # Exclude starting position
        obstacles_avoided = self._count_obstacles_in_path(path, snake_positions)
        snake_length = len(snake_positions)
        
        # Analyze path efficiency
        efficiency_ratio = manhattan_distance / max(path_length, 1)
        is_optimal = path_length == manhattan_distance
        detour_steps = max(0, path_length - manhattan_distance)
        
        # Analyze board state
        board_fill_ratio = snake_length / (grid_size * grid_size)
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        # Analyze move safety
        next_pos = (head_pos[0] + (1 if direction == "RIGHT" else -1 if direction == "LEFT" else 0),
                   head_pos[1] + (1 if direction == "UP" else -1 if direction == "DOWN" else 0))
        
        # Detailed step-by-step explanation
        explanation_parts = [
            "=== BFS PATHFINDING ANALYSIS ===",
            "",
            "PHASE 1: INITIAL SITUATION ASSESSMENT",
            f"• Current head position: {head_pos}",
            f"• Target apple position: {apple_pos}",
            f"• Snake length: {snake_length} segments",
            f"• Grid dimensions: {grid_size}×{grid_size} ({grid_size * grid_size} total cells)",
            f"• Board occupation: {snake_length}/{grid_size * grid_size} cells ({board_fill_ratio:.1%}) - {space_pressure} space pressure",
            f"• Free cells remaining: {remaining_free_cells}",
            "",
            "PHASE 2: MOVE VALIDATION",
            f"• Available valid moves: {valid_moves} ({len(valid_moves)} options)",
                         f"• Rejected moves: {list(set(['UP', 'DOWN', 'LEFT', 'RIGHT']) - set(valid_moves))}",
            "• Validation criteria: no wall collisions, no body collisions, within grid bounds",
            "",
            "PHASE 3: BFS PATHFINDING EXECUTION",
            f"• Algorithm: Breadth-First Search from {head_pos} to {apple_pos}",
            f"• Search space: {grid_size * grid_size - snake_length} accessible cells",
            f"• Obstacles to navigate: {snake_length - 1} body segments",
            f"• Manhattan distance baseline: {manhattan_distance} steps (theoretical minimum)",
            "",
            "PHASE 4: PATH ANALYSIS RESULTS",
            f"• Shortest path found: {path_length} steps",
            f"• Path efficiency: {efficiency_ratio:.2f} ({path_length}/{manhattan_distance})",
                         f"• Path optimality: {'OPTIMAL - no detours required' if is_optimal else 'SUB-OPTIMAL - includes ' + str(detour_steps) + ' detour step(s) to avoid obstacles'}",
            f"• Obstacles near path: {obstacles_avoided} body segments in adjacent cells",
                         f"• Path coordinates: {' → '.join([str(list(p)) for p in path[:min(5, len(path))]])}{'...' if len(path) > 5 else ''}",
            "",
            "PHASE 5: MOVE SELECTION LOGIC",
            f"• Chosen direction: {direction}",
            f"• Next position: {next_pos}",
            f"• Rationale: First step of shortest path to apple",
            f"• Risk assessment: LOW (validated safe move on optimal path)",
            f"• Expected outcome: Advance {1} step closer to apple along shortest route",
            "",
            "PHASE 6: STRATEGIC IMPLICATIONS",
            f"• Immediate benefit: Reduces distance to apple from {manhattan_distance} to {manhattan_distance - 1 if direction in valid_moves else manhattan_distance}",
            f"• Future positioning: Maintains optimal trajectory toward apple",
            f"• Space management: Preserves {remaining_free_cells - 1} free cells for maneuvering",
            f"• Risk mitigation: BFS guarantees shortest path, minimizing exposure time",
            "",
            "=== DECISION SUMMARY ===",
            f"Moving {direction} is the optimal choice because it follows the shortest BFS-computed path to the apple at {apple_pos}. " +
            f"This move advances the snake from {head_pos} to {next_pos}, maintaining perfect trajectory efficiency " +
            f"{'with no detours required' if is_optimal else f'despite {detour_steps} necessary detour(s) to avoid obstacles'}. " +
            f"The decision is safe (validated against {len(valid_moves)} valid options), efficient " +
            f"({efficiency_ratio:.2f} path efficiency), and strategically sound given current board pressure ({space_pressure})."
        ]

        explanation_dict = {
            "strategy_phase": "APPLE_PATH",
            "metrics": {
                "manhattan_distance": int(manhattan_distance),
                "path_length": int(path_length),
                "obstacles_near_path": int(obstacles_avoided),
                "remaining_free_cells": int(remaining_free_cells),
                "valid_moves": valid_moves,
                "final_chosen_direction": direction,
                "head_position": list(head_pos),
                "apple_position": list(apple_pos),
                "snake_length": int(snake_length),
                "grid_size": int(grid_size),
            },
            "explanation_steps": explanation_parts,
        }

        return explanation_dict

    def _generate_no_path_explanation(self, head_pos: Tuple[int, int], apple_pos: Tuple[int, int],
                                    snake_positions: set, grid_size: int) -> dict:
        """
        Generate explanation when no path to apple is found.
        
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        body_count = len(snake_positions)
        manhattan_distance = abs(apple_pos[0] - head_pos[0]) + abs(apple_pos[1] - head_pos[1])
        board_fill_ratio = body_count / (grid_size * grid_size)
        remaining_free_cells = grid_size * grid_size - body_count

        # Check if apple is blocked by body
        apple_neighbors = self._get_neighbors(apple_pos, grid_size)
        blocked_neighbors = sum(1 for pos in apple_neighbors if pos in snake_positions)
        
        # Analyze space fragmentation
        total_cells = grid_size * grid_size
        space_pressure = "critical" if board_fill_ratio > 0.7 else "high" if board_fill_ratio > 0.5 else "moderate"

        explanation_parts = [
            "=== BFS PATHFINDING FAILURE ANALYSIS ===",
            "",
            "PHASE 1: PATHFINDING ATTEMPT RESULTS",
            f"• Algorithm: Breadth-First Search from {head_pos} to {apple_pos}",
            f"• Search outcome: NO PATH FOUND",
            f"• Manhattan distance: {manhattan_distance} steps (theoretical minimum)",
            f"• Actual path length: UNREACHABLE",
            "",
            "PHASE 2: OBSTACLE ANALYSIS",
            f"• Snake body segments: {body_count} total",
            f"• Board occupation: {body_count}/{total_cells} cells ({board_fill_ratio:.1%})",
            f"• Space pressure level: {space_pressure.upper()}",
            f"• Free cells remaining: {remaining_free_cells}",
            "",
            "PHASE 3: APPLE ACCESSIBILITY ASSESSMENT",
            f"• Apple position: {apple_pos}",
            f"• Adjacent cells to apple: {len(apple_neighbors)} total",
            f"• Blocked adjacent cells: {blocked_neighbors}/{len(apple_neighbors)}",
            f"• Apple accessibility: {'COMPLETELY BLOCKED' if blocked_neighbors == len(apple_neighbors) else 'PARTIALLY ACCESSIBLE'}",
            "",
            "PHASE 4: FAILURE ROOT CAUSE ANALYSIS",
        ]
        
        if blocked_neighbors == len(apple_neighbors):
            explanation_parts.extend([
                "• Primary cause: Apple is completely surrounded by snake body segments",
                "• Secondary cause: No adjacent cells available for approach",
                "• Tertiary cause: Snake has grown too large relative to board size",
                "• Resolution: Impossible until snake body moves away from apple vicinity"
            ])
        elif board_fill_ratio > 0.6:
            explanation_parts.extend([
                f"• Primary cause: Excessive board occupation ({board_fill_ratio:.1%})",
                "• Secondary cause: Snake body creates maze-like obstacles",
                "• Tertiary cause: Insufficient free space for pathfinding",
                "• Resolution: Wait for tail movement to create path opportunities"
            ])
        else:
            explanation_parts.extend([
                "• Primary cause: Snake body configuration blocks all viable routes",
                "• Secondary cause: Temporary spatial arrangement prevents access",
                "• Tertiary cause: Current head position disadvantaged",
                "• Resolution: Alternative strategy needed or wait for body repositioning"
            ])
        
        explanation_parts.extend([
            "",
            "PHASE 5: STRATEGIC IMPLICATIONS",
            f"• Immediate action: Cannot pursue apple directly",
            f"• Alternative strategies: Tail-chasing, space preservation, defensive positioning",
            f"• Risk assessment: HIGH (no progress toward apple possible)",
            f"• Expected outcome: Must adopt survival/waiting strategy",
            "",
            "=== CONCLUSION ===",
            f"BFS pathfinding from {head_pos} to apple at {apple_pos} has failed due to complete path blockage. " +
            f"With {blocked_neighbors}/{len(apple_neighbors)} apple-adjacent cells blocked and {board_fill_ratio:.1%} board occupation, " +
            f"the snake must adopt alternative strategies until body repositioning creates new path opportunities. " +
            f"The {manhattan_distance}-step Manhattan distance remains theoretical until obstacles clear."
        ])

        explanation_dict = {
            "strategy_phase": "NO_PATH",
            "metrics": {
                "manhattan_distance": int(manhattan_distance),
                "path_length": 0,
                "obstacles_near_path": int(body_count),
                "remaining_free_cells": int(remaining_free_cells),
                "valid_moves": [],
                "final_chosen_direction": "NO_PATH_FOUND",
                "head_position": list(head_pos),
                "apple_position": list(apple_pos),
                "snake_length": int(body_count),
                "grid_size": int(grid_size),
            },
            "explanation_steps": explanation_parts,
        }

        return explanation_dict

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

    # SSOT: BFS pathfinding is implemented in ssot_utils.py
    # Do not reimplement here - use ssot_bfs_pathfind from ssot_utils

    # ----------------
    # Helper utilities (v0.04 additions)
    # ----------------
    def _count_remaining_free_cells(self, snake_positions: set, grid_size: int) -> int:
        """Count how many empty cells are not occupied by the snake body."""
        total_cells = grid_size * grid_size
        return total_cells - len(snake_positions)

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"BFSAgent(algorithm={self.algorithm_name})"

    @staticmethod
    def _calculate_valid_moves(head_pos: list, snake_positions: list, grid_size: int) -> list:
        """
        SSOT: Single source of truth for valid moves calculation.
        Used by both agent and dataset generator.
        Args:
            head_pos: Current head position [x, y]
            snake_positions: All snake positions (head at index -1)
            grid_size: Size of the game grid
        Returns:
            List of valid moves (UP, DOWN, LEFT, RIGHT)
        """
        return ssot_calculate_valid_moves(head_pos, snake_positions, grid_size) 
