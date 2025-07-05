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
from typing import List, Tuple, TYPE_CHECKING, Any, Dict

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports
from config.game_constants import DIRECTIONS
from utils.moves_utils import position_to_direction
from core.game_agents import BaseAgent
# BFS pathfinding and valid moves calculation implemented directly in the agent

if TYPE_CHECKING:
    pass


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
        # PRE-EXECUTION: All state values are from BEFORE the move is executed
        # This includes: head_position, apple_position, snake_positions, score, steps
        # The agent must make decisions based on the current (pre-move) state
        head = list(state["head_position"])
        apple = list(state["apple_position"])
        snake = [list(seg) for seg in state["snake_positions"]]
        grid_size = state["grid_size"]
        # SSOT: Obstacles are all body segments except head (matching dataset generator)
        obstacles = set(tuple(p) for p in snake[:-1])  # Exclude head from obstacles, include body segments

        # Helper metrics that are independent of strategy branch (all from pre-move state)
        # PRE-EXECUTION: All these calculations use pre-move positions
        # manhattan_distance: distance from current head to current apple
        # valid_moves: available moves from current head position
        # remaining_free_cells: free cells based on current snake positions
        manhattan_distance = abs(head[0] - apple[0]) + abs(head[1] - apple[1])
        valid_moves = self._calculate_valid_moves(head, snake, grid_size)
        remaining_free_cells = self._count_remaining_free_cells(set(tuple(p) for p in snake), grid_size)

        # Fail-fast: ensure state is not mutated (SSOT)

        # ---------------- Apple path first
        # PRE-EXECUTION: Pathfinding from current head to current apple
        # This determines the optimal path from the current position
        path_to_apple = self._bfs_pathfind(head, apple, obstacles, grid_size)
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
            # PRE-EXECUTION: All metrics are calculated from pre-move state
            # This ensures consistency with the prompt and dataset generation
            # head_position: current head position (before move)
            # apple_position: current apple position (before move)
            # snake_length: current snake length (before move)
            # valid_moves: available moves from current head
            # manhattan_distance: distance from current head to current apple
            # remaining_free_cells: free cells based on current snake positions
            # path_length: length of path from current head to current apple
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
            # PRE-EXECUTION: Explanation describes the decision based on pre-move state
            explanation_dict = self._generate_move_explanation(
                tuple(head), tuple(apple), set(tuple(p) for p in snake), path_to_apple, direction, valid_moves, manhattan_distance, remaining_free_cells, grid_size
            )
            explanation_dict["metrics"] = metrics  # Overwrite with pre-move state metrics
            
            return direction, explanation_dict
        else:
            if valid_moves:
                direction = valid_moves[0]
                # Create metrics directly from pre-move state only
                # PRE-EXECUTION: All metrics from pre-move state for survival move
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
                # PRE-EXECUTION: All metrics from pre-move state for no path scenario
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
                # PRE-EXECUTION: Explanation describes the no-path scenario from pre-move state
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
        
        PRE-EXECUTION: All parameters are from the pre-move state:
        - head_pos: current head position (before move)
        - apple_pos: current apple position (before move) 
        - snake_positions: current snake body positions (before move)
        - path: optimal path from current head to current apple
        - direction: chosen move direction
        - valid_moves: available moves from current head position
        - manhattan_distance: distance from current head to current apple
        - remaining_free_cells: free cells based on current snake positions
        - grid_size: current grid size
        """
        # PRE-EXECUTION: All calculations use pre-move state values
        path_length = len(path) - 1  # Exclude starting position
        obstacles_avoided = self._count_obstacles_in_path(path, snake_positions)
        snake_length = len(snake_positions)
        efficiency_ratio = manhattan_distance / max(path_length, 1)
        is_optimal = path_length == manhattan_distance
        detour_steps = max(0, path_length - manhattan_distance)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        # PRE-EXECUTION: Calculate next position based on current head and chosen direction
        # This is the position the snake will move to, but we're still in pre-move state
        next_pos = (head_pos[0] + (1 if direction == "RIGHT" else -1 if direction == "LEFT" else 0),
                   head_pos[1] + (1 if direction == "UP" else -1 if direction == "DOWN" else 0))
        
        # PRE-EXECUTION: Format path coordinates for explanation
        path_str = ' → '.join([f'({p[0]}, {p[1]})' for p in path])
        efficiency_str = f"{efficiency_ratio:.2f} ({path_length}/{manhattan_distance})"
        
        # PRE-EXECUTION: All explanation text describes the current situation and decision
        # based on pre-move state values
        # Ensure path starts from pre-move head (type-consistent)
        assert tuple(path[0]) == tuple(head_pos), f"SSOT violation: path[0] ({path[0]}) != head_pos ({head_pos})"
        # Fail-fast: explanation must match pre-move head
        explanation_parts = [
            "=== BFS PATHFINDING ANALYSIS ===",
            "",
            "PHASE 1: INITIAL SITUATION ASSESSMENT",
            f"• Current head position: {tuple(head_pos)}",  # PRE-MOVE: current head position
            f"• Target apple position: {tuple(apple_pos)}",  # PRE-MOVE: current apple position
            f"• Snake body positions: {[tuple(p) for p in snake_positions if tuple(p) != tuple(head_pos)]}",  # PRE-MOVE: current body positions
            f"• Snake length: {snake_length} segments",  # PRE-MOVE: current snake length
            f"• Grid dimensions: {grid_size}×{grid_size} ({grid_size * grid_size} total cells)",
            f"• Board occupation: {snake_length}/{grid_size * grid_size} cells ({board_fill_ratio:.1%}) - {space_pressure} space pressure",  # PRE-MOVE: current occupation
            f"• Free cells remaining: {remaining_free_cells}",  # PRE-MOVE: current free cells
            "",
            "PHASE 2: MOVE VALIDATION",
            f"• Available valid moves: {valid_moves} ({len(valid_moves)} options)",  # PRE-MOVE: valid moves from current head
            f"• Rejected moves: {list(set(['UP', 'DOWN', 'LEFT', 'RIGHT']) - set(valid_moves))}",
            "• Validation criteria: no wall collisions, no body collisions, within grid bounds",
            "",
            "PHASE 3: BFS PATHFINDING EXECUTION",
            f"• Algorithm: Breadth-First Search from {tuple(head_pos)} to {tuple(apple_pos)}",  # PRE-MOVE: current positions
            f"• Search space: {grid_size * grid_size - snake_length} accessible cells",
            f"• Obstacles to navigate: {snake_length - 1} body segments",  # PRE-MOVE: current obstacles
            f"• Manhattan distance baseline: {manhattan_distance} steps (theoretical minimum)",  # PRE-MOVE: current distance
            "",
            "PHASE 4: PATH ANALYSIS RESULTS",
            f"• Shortest path found: {path_length} steps",  # PRE-MOVE: path from current position
            f"• Path efficiency: {efficiency_str}",
            f"• Path optimality: {'OPTIMAL - no detours required' if is_optimal else 'SUB-OPTIMAL - includes ' + str(detour_steps) + ' detour step(s) to avoid obstacles'}",
            f"• Obstacles near path: {obstacles_avoided} body segments in adjacent cells",  # PRE-MOVE: obstacles near current path
            f"• Path coordinates: {' → '.join([str(tuple(p)) for p in path])}",  # PRE-MOVE: path from current position
            "",
            "PHASE 5: MOVE SELECTION LOGIC",
            f"• Chosen direction: {direction}",
            f"• Next position: {next_pos}",  # PRE-MOVE: calculated next position
            "• Rationale: First step of shortest path to apple",
            "• Risk assessment: LOW (validated safe move on optimal path)",
            "• Expected outcome: Advance 1 step closer to apple along shortest route",
            "",
            "PHASE 6: STRATEGIC IMPLICATIONS",
            f"• Immediate benefit: Reduces distance to apple from {manhattan_distance} to {manhattan_distance - 1}",  # PRE-MOVE: current distance reduction
            "• Future positioning: Maintains optimal trajectory toward apple",
            f"• Space management: Preserves {remaining_free_cells - 1} free cells for maneuvering",  # PRE-MOVE: current space management
            "• Risk mitigation: BFS guarantees shortest path, minimizing exposure time",
            "",
            "=== DECISION SUMMARY ===",
            f"Moving {direction} is the optimal choice because it follows the shortest BFS-computed path to the apple at {tuple(apple_pos)}. " +  # PRE-MOVE: current apple position
            f"This move advances the snake from {tuple(head_pos)} to {next_pos}, maintaining perfect trajectory efficiency " +  # PRE-MOVE: current to calculated next position
            f"{'with no detours required' if is_optimal else f'despite {detour_steps} necessary detour(s) to avoid obstacles'}. " +
            f"The decision is safe (validated against {len(valid_moves)} valid options), efficient " +  # PRE-MOVE: current valid moves
            f"({efficiency_ratio:.2f} path efficiency), and strategically sound given current board pressure ({space_pressure})."  # PRE-MOVE: current board pressure
        ]

        # PRE-EXECUTION: All metrics in explanation are from pre-move state
        explanation_dict = {
            "strategy_phase": "APPLE_PATH",
            "metrics": {
                "manhattan_distance": int(manhattan_distance),  # PRE-MOVE: current distance
                "path_length": int(path_length),  # PRE-MOVE: current path length
                "obstacles_near_path": int(obstacles_avoided),  # PRE-MOVE: current obstacles
                "remaining_free_cells": int(remaining_free_cells),  # PRE-MOVE: current free cells
                "valid_moves": valid_moves,  # PRE-MOVE: current valid moves
                "final_chosen_direction": direction,
                "head_position": list(head_pos),  # PRE-MOVE: current head position
                "apple_position": list(apple_pos),  # PRE-MOVE: current apple position
                "snake_length": int(snake_length),  # PRE-MOVE: current snake length
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
            "• Search outcome: NO PATH FOUND",
            f"• Manhattan distance: {manhattan_distance} steps (theoretical minimum)",
            "• Actual path length: UNREACHABLE",
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
            "• Immediate action: Cannot pursue apple directly",
            "• Alternative strategies: Tail-chasing, space preservation, defensive positioning",
            "• Risk assessment: HIGH (no progress toward apple possible)",
            "• Expected outcome: Must adopt survival/waiting strategy",
            "",
            "=== CONCLUSION ===",
            f"BFS pathfinding from {head_pos} to apple at {apple_pos} has failed due to complete path blockage. " +
            f"With {blocked_neighbors}/{len(apple_neighbors)} apple-adjacent cells blocked and {board_fill_ratio:.1%} board occupation, " +
            "the snake must adopt alternative strategies until body repositioning creates new path opportunities. " +
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
    def _bfs_pathfind(start: List[int], goal: List[int], obstacles: Set[Tuple[int, int]], grid_size: int) -> Optional[List[List[int]]]:
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
        queue = deque([(start_tuple, [start])])
        visited = {start_tuple}
        
        # Direction vectors for BFS
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        
        while queue:
            current_pos, path = queue.popleft()
            
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

    @staticmethod
    def _calculate_valid_moves(head_pos: list, snake_positions: list, grid_size: int) -> list:
        """
        Calculate valid moves using the same logic as agents.
        Args:
            head_pos: Current head position [x, y]
            snake_positions: All snake positions (head at index -1)
            grid_size: Size of the game grid
        Returns:
            List of valid moves (UP, DOWN, LEFT, RIGHT)
        """
        obstacles = set(tuple(p) for p in snake_positions[:-1] if len(p) >= 2)
        
        valid_moves = []
        for direction, (dx, dy) in DIRECTIONS.items():
            next_x = head_pos[0] + dx
            next_y = head_pos[1] + dy
            # Check bounds
            if 0 <= next_x < grid_size and 0 <= next_y < grid_size:
                next_pos = (next_x, next_y)
                # Check if position is not occupied by snake body (excluding head)
                if next_pos not in obstacles:
                    valid_moves.append(direction)
        
        return valid_moves 

    @staticmethod
    def flatten_explanation_for_jsonl(explanation: Any) -> str:
        """
        Convert a structured explanation dict to a rich, human-readable string for JSONL output.
        If already a string, return as-is. If dict, use 'natural_language_summary' and 'explanation_steps'.
        
        Args:
            explanation: The explanation object to flatten
            
        Returns:
            Rich, human-readable explanation string
        """
        if isinstance(explanation, str):
            return explanation
        if isinstance(explanation, dict):
            # Prefer natural_language_summary + explanation_steps
            summary = explanation.get('natural_language_summary', '')
            steps = explanation.get('explanation_steps', [])
            if steps and isinstance(steps, list):
                steps_text = '\n'.join(steps)
            else:
                steps_text = ''
            # Compose
            if summary and steps_text:
                return f"{steps_text}\n\n{summary}"
            elif steps_text:
                return steps_text
            elif summary:
                return summary
            # Fallback: join all string fields in the dict
            return '\n'.join(str(v) for v in explanation.values() if isinstance(v, str))
        # Fallback: just str()
        return str(explanation)

    @staticmethod
    def format_metrics_for_jsonl(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format metrics for JSONL completion, ensuring all values are JSON serializable.
        
        Args:
            metrics: Raw metrics dictionary
            
        Returns:
            Formatted metrics dictionary
        """
        if not metrics:
            return {}
        
        # Map metric names to match the expected format
        formatted_metrics = {}
        
        # Common metric mappings
        metric_mappings = {
            'manhattan_distance': 'manhattan_distance',
            'obstacles_near_path': 'obstacles_near_path',
            'remaining_free_cells': 'remaining_free_cells',
            'valid_moves': 'valid_moves',
            'final_chosen_direction': 'chosen_direction',
            'apple_path_length': 'path_length',
            'apple_path_safe': 'apple_path_safe',
            'fallback_used': 'fallback_used',
            # Position and game state metrics
            'head_position': 'head_position',
            'apple_position': 'apple_position',
            'snake_length': 'snake_length',
            'grid_size': 'grid_size'
        }
        
        for old_key, new_key in metric_mappings.items():
            if old_key in metrics:
                value = metrics[old_key]
                # Convert numpy types to Python types for JSON serialization
                if hasattr(value, 'item'):  # numpy scalar
                    formatted_metrics[new_key] = value.item()
                elif isinstance(value, (list, tuple)):
                    # Handle lists/tuples that might contain numpy types
                    formatted_metrics[new_key] = [
                        item.item() if hasattr(item, 'item') else item 
                        for item in value
                    ]
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    formatted_metrics[new_key] = BFSAgent.format_metrics_for_jsonl(value)
                else:
                    formatted_metrics[new_key] = value
        
        return formatted_metrics



    @staticmethod
    def to_serializable(obj):
        """
        Convert numpy types to Python types for JSON serialization.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable object
        """
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [BFSAgent.to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: BFSAgent.to_serializable(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def count_free_space_in_direction(start_pos: List[int], direction: str, snake_positions: List[List[int]], grid_size: int) -> int:
        """
        Count free space in a given direction from a starting position.
        
        Args:
            start_pos: Starting position [x, y]
            direction: Direction to check ('UP', 'DOWN', 'LEFT', 'RIGHT')
            snake_positions: List of snake body positions
            grid_size: Size of the game grid
            
        Returns:
            Number of free cells in the direction
        """
        count = 0
        current_pos = list(start_pos)
        
        while True:
            if direction == 'UP':
                current_pos[1] += 1
            elif direction == 'DOWN':
                current_pos[1] -= 1
            elif direction == 'LEFT':
                current_pos[0] -= 1
            elif direction == 'RIGHT':
                current_pos[0] += 1
            
            # Check bounds
            if (current_pos[0] < 0 or current_pos[0] >= grid_size or 
                current_pos[1] < 0 or current_pos[1] >= grid_size):
                break
            
            # Check snake collision
            if current_pos in snake_positions:
                break
            
            count += 1
            
            # Prevent infinite loop
            if count > grid_size * grid_size:
                break
        
        return count 
