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
from typing import List, Tuple, TYPE_CHECKING, Any, Dict, Set, Optional

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root

ensure_project_root()

# Import from project root using absolute imports
from config.game_constants import DIRECTIONS
from utils.moves_utils import position_to_direction
from core.game_agents import BaseAgent

# Import utility functions from the new utility modules
from extensions.common.utils.game_state_utils import (
    extract_head_position,
    extract_body_positions,
    extract_apple_position,
    extract_grid_size,
    to_serializable,
    format_metrics_for_jsonl,
    flatten_explanation_for_jsonl,
)
from heuristics_utils import (
    count_obstacles_in_path,
    get_neighbors,
    count_remaining_free_cells,
    calculate_valid_moves,
    count_free_space_in_direction,
    calculate_manhattan_distance,
    calculate_valid_moves_ssot,
    bfs_pathfind,
)

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

        # SSOT: Use centralized utilities for all position and calculation extractions
        head = extract_head_position(state)
        apple = list(state["apple_position"])
        snake = [list(seg) for seg in state["snake_positions"]]
        grid_size = state["grid_size"]

        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(state)
        obstacles = set(tuple(p) for p in body_positions)  # Use body_positions directly

        # SSOT: Use centralized calculations for all metrics
        manhattan_distance = calculate_manhattan_distance(state)
        valid_moves = calculate_valid_moves_ssot(state)
        remaining_free_cells = count_remaining_free_cells(
            set(tuple(p) for p in snake), grid_size
        )

        # Fail-fast: ensure state is not mutated (SSOT)

        # ---------------- Apple path first
        # PRE-EXECUTION: Pathfinding from current head to current apple
        # This determines the optimal path from the current position
        path_to_apple = bfs_pathfind(head, apple, obstacles, grid_size)
        if path_to_apple and len(path_to_apple) > 1:
            next_pos = path_to_apple[1]

            # Fail-fast: Validate that the next position is within bounds
            if (
                next_pos[0] < 0
                or next_pos[0] >= grid_size
                or next_pos[1] < 0
                or next_pos[1] >= grid_size
            ):
                raise RuntimeError(
                    f"SSOT violation: BFS computed out-of-bounds position {next_pos} for grid size {grid_size}"
                )

            direction = position_to_direction(tuple(head), tuple(next_pos))

            if direction not in valid_moves:
                raise RuntimeError(
                    f"SSOT violation: BFS computed move '{direction}' is not valid for head {head} and valid_moves {valid_moves}"
                )

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
            # SSOT: Use centralized utilities for all position extractions
            ssot_head = extract_head_position(state)
            ssot_apple = extract_apple_position(state)

            metrics = {
                "final_chosen_direction": direction,
                "head_position": list(
                    ssot_head
                ),  # PRE-MOVE: current head position (SSOT)
                "apple_position": list(
                    ssot_apple
                ),  # PRE-MOVE: current apple position (SSOT)
                "snake_length": len(snake),  # PRE-MOVE: current snake length
                "grid_size": grid_size,
                "valid_moves": valid_moves,  # PRE-MOVE: current valid moves
                "manhattan_distance": manhattan_distance,  # PRE-MOVE: current distance
                "remaining_free_cells": remaining_free_cells,  # PRE-MOVE: current free cells
                "path_length": len(path_to_apple) - 1,
                "obstacles_near_path": count_obstacles_in_path(
                    path_to_apple, set(tuple(p) for p in snake)
                ),
                "apple_path_safe": True,
            }

            # Get explanation text from helper (but not metrics)
            # PRE-EXECUTION: Explanation describes the decision based on pre-move state
            explanation_dict = self._generate_move_explanation(
                state,
                path_to_apple,
                direction,
                valid_moves,
                manhattan_distance,
                remaining_free_cells,
            )
            explanation_dict["metrics"] = (
                metrics  # Overwrite with pre-move state metrics (SSOT)
            )

            # FAIL-FAST: Validate explanation head position matches input state
            explanation_head = explanation_dict.get("metrics", {}).get("head_position")
            if explanation_head != head:
                raise RuntimeError(
                    f"[SSOT] Agent explanation head {explanation_head} != input state head {head}. "
                    f"This indicates the agent is using a different state than provided."
                )

            return direction, explanation_dict
        else:
            # Original BFS Safe Greedy survival strategy: try tail-chasing first
            if valid_moves:
                # Try to find path to tail (always safe)
                tail = snake[-1]
                path_to_tail = bfs_pathfind(head, tail, obstacles, grid_size)

                if path_to_tail and len(path_to_tail) > 1:
                    direction = position_to_direction(head, path_to_tail[1])
                    # SSOT: Use centralized utilities for all position extractions
                    ssot_head = extract_head_position(state)
                    ssot_apple = extract_apple_position(state)
                    metrics = {
                        "final_chosen_direction": direction,
                        "head_position": list(
                            ssot_head
                        ),  # PRE-MOVE: current head position (SSOT)
                        "apple_position": list(
                            ssot_apple
                        ),  # PRE-MOVE: current apple position (SSOT)
                        "snake_length": len(snake),
                        "grid_size": grid_size,
                        "valid_moves": valid_moves,
                        "manhattan_distance": manhattan_distance,
                        "remaining_free_cells": remaining_free_cells,
                        "path_length": len(path_to_tail) - 1,
                        "obstacles_near_path": 0,
                        "apple_path_safe": False,
                        "survival_strategy": "tail_chasing",
                    }

                    explanation_dict = {
                        "strategy_phase": "SURVIVAL_MOVE",
                        "metrics": metrics,
                        "explanation_steps": [
                            f"No path to apple found from {head} to {apple}.",
                            f"Using survival strategy: chasing tail with path length {len(path_to_tail) - 1}.",
                            f"Moving {direction} to follow tail at {tail}.",
                        ],
                    }

                    # FAIL-FAST: Validate explanation head position matches input state
                    explanation_head = explanation_dict.get("metrics", {}).get(
                        "head_position"
                    )
                    if explanation_head != head:
                        raise RuntimeError(
                            f"[SSOT] Agent survival explanation head {explanation_head} != input state head {head}. "
                            f"This indicates the agent is using a different state than provided."
                        )

                    return direction, explanation_dict
                else:
                    # No path to tail, use any valid move
                    direction = valid_moves[0]
                    # SSOT: Use centralized utilities for all position extractions
                    ssot_head = extract_head_position(state)
                    ssot_apple = extract_apple_position(state)
                    metrics = {
                        "final_chosen_direction": direction,
                        "head_position": list(
                            ssot_head
                        ),  # PRE-MOVE: current head position (SSOT)
                        "apple_position": list(
                            ssot_apple
                        ),  # PRE-MOVE: current apple position (SSOT)
                        "snake_length": len(snake),
                        "grid_size": grid_size,
                        "valid_moves": valid_moves,
                        "manhattan_distance": manhattan_distance,
                        "remaining_free_cells": remaining_free_cells,
                        "path_length": 0,
                        "obstacles_near_path": 0,
                        "apple_path_safe": False,
                        "survival_strategy": "random_valid_move",
                    }

                    explanation_dict = {
                        "strategy_phase": "SURVIVAL_MOVE",
                        "metrics": metrics,
                        "explanation_steps": [
                            f"No path to apple or tail found from {head}.",
                            f"Using survival strategy: choosing first valid move '{direction}'.",
                        ],
                    }

                    # FAIL-FAST: Validate explanation head position matches input state
                    explanation_head = explanation_dict.get("metrics", {}).get(
                        "head_position"
                    )
                    if explanation_head != head:
                        raise RuntimeError(
                            f"[SSOT] Agent survival explanation head {explanation_head} != input state head {head}. "
                            f"This indicates the agent is using a different state than provided."
                        )

                    return direction, explanation_dict
            else:
                # No valid moves - game over
                return "NO_PATH_FOUND", {"strategy_phase": "GAME_OVER", "metrics": {}}

    def _generate_move_explanation(
        self,
        game_state: dict,
        path: List[Tuple[int, int]],
        direction: str,
        valid_moves: List[str],
        manhattan_distance: int,
        remaining_free_cells: int,
    ) -> dict:
        """
        Generate rich, step-by-step natural language explanation for the chosen move.
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.

        PRE-EXECUTION: All parameters are from the pre-move state:
        - game_state: complete game state dict (before move)
        - path: optimal path from current head to current apple
        - direction: chosen move direction
        - valid_moves: available moves from current head position
        - manhattan_distance: distance from current head to current apple
        - remaining_free_cells: free cells based on current snake positions
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = extract_head_position(game_state)
        apple_pos = list(game_state.get("apple_position", [0, 0]))
        grid_size = game_state.get("grid_size", 10)

        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(game_state)

        # PRE-EXECUTION: All calculations use pre-move state values
        path_length = len(path) - 1  # Exclude starting position
        obstacles_avoided = count_obstacles_in_path(
            path, set(tuple(p) for p in body_positions)
        )
        snake_length = len(game_state.get("snake_positions", []))
        efficiency_ratio = manhattan_distance / max(path_length, 1)
        is_optimal = path_length == manhattan_distance
        detour_steps = max(0, path_length - manhattan_distance)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        space_pressure = (
            "low"
            if board_fill_ratio < 0.3
            else "medium" if board_fill_ratio < 0.6 else "high"
        )

        # PRE-EXECUTION: Calculate next position based on current head and chosen direction
        # This is the position the snake will move to, but we're still in pre-move state
        next_pos = (
            head_pos[0]
            + (1 if direction == "RIGHT" else -1 if direction == "LEFT" else 0),
            head_pos[1]
            + (1 if direction == "UP" else -1 if direction == "DOWN" else 0),
        )

        # PRE-EXECUTION: Format path coordinates for explanation
        path_str = " → ".join([f"({p[0]}, {p[1]})" for p in path])
        efficiency_str = f"{efficiency_ratio:.2f} ({path_length}/{manhattan_distance})"

        # PRE-EXECUTION: All explanation text describes the current situation and decision
        # based on pre-move state values
        # Ensure path starts from pre-move head (type-consistent)
        assert tuple(path[0]) == tuple(
            head_pos
        ), f"SSOT violation: path[0] ({path[0]}) != head_pos ({head_pos})"
        # Fail-fast: explanation must match pre-move head
        explanation_parts = [
            "=== BFS PATHFINDING ANALYSIS ===",
            "",
            "PHASE 1: INITIAL SITUATION ASSESSMENT",
            f"• Current head position: {tuple(head_pos)}",  # PRE-MOVE: current head position
            f"• Target apple position: {tuple(apple_pos)}",  # PRE-MOVE: current apple position
            f"• Snake body positions: {[tuple(p) for p in body_positions]}",  # PRE-MOVE: current body positions (SSOT: same as dataset generator)
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
            f"Moving {direction} is the optimal choice because it follows the shortest BFS-computed path to the apple at {tuple(apple_pos)}. "  # PRE-MOVE: current apple position
            + f"This move advances the snake from {tuple(head_pos)} to {next_pos}, maintaining perfect trajectory efficiency "  # PRE-MOVE: current to calculated next position
            + f"{'with no detours required' if is_optimal else f'despite {detour_steps} necessary detour(s) to avoid obstacles'}. "
            + f"The decision is safe (validated against {len(valid_moves)} valid options), efficient "  # PRE-MOVE: current valid moves
            + f"({efficiency_ratio:.2f} path efficiency), and strategically sound given current board pressure ({space_pressure}).",  # PRE-MOVE: current board pressure
        ]

        # PRE-EXECUTION: All metrics in explanation are from pre-move state
        explanation_dict = {
            "strategy_phase": "APPLE_PATH",
            "metrics": {
                "manhattan_distance": int(
                    manhattan_distance
                ),  # PRE-MOVE: current distance
                "path_length": int(path_length),  # PRE-MOVE: current path length
                "obstacles_near_path": int(
                    obstacles_avoided
                ),  # PRE-MOVE: current obstacles
                "remaining_free_cells": int(
                    remaining_free_cells
                ),  # PRE-MOVE: current free cells
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

    def _generate_no_path_explanation(self, game_state: dict) -> dict:
        """
        Generate explanation when no path to apple is found.

        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        # SSOT: Extract positions using exact same logic as dataset_generator.py
        head_pos = extract_head_position(game_state)
        apple_pos = extract_apple_position(game_state)
        grid_size = extract_grid_size(game_state)

        # SSOT: Use exact same body_positions logic as dataset_generator.py
        body_positions = extract_body_positions(game_state)

        snake_positions = game_state.get("snake_positions", [])
        body_count = len(snake_positions)
        manhattan_distance = calculate_manhattan_distance(game_state)
        board_fill_ratio = body_count / (grid_size * grid_size)
        remaining_free_cells = grid_size * grid_size - body_count

        # Check if apple is blocked by body
        apple_neighbors = get_neighbors(tuple(apple_pos), grid_size)
        blocked_neighbors = sum(
            1
            for pos in apple_neighbors
            if pos in set(tuple(p) for p in snake_positions)
        )

        # Analyze space fragmentation
        total_cells = grid_size * grid_size
        space_pressure = (
            "critical"
            if board_fill_ratio > 0.7
            else "high" if board_fill_ratio > 0.5 else "moderate"
        )

        explanation_parts = [
            "=== BFS PATHFINDING FAILURE ANALYSIS ===",
            "",
            "PHASE 1: PATHFINDING ATTEMPT RESULTS",
            f"• Algorithm: Breadth-First Search from {tuple(head_pos)} to {tuple(apple_pos)}",
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
            f"• Apple position: {tuple(apple_pos)}",
            f"• Adjacent cells to apple: {len(apple_neighbors)} total",
            f"• Blocked adjacent cells: {blocked_neighbors}/{len(apple_neighbors)}",
            f"• Apple accessibility: {'COMPLETELY BLOCKED' if blocked_neighbors == len(apple_neighbors) else 'PARTIALLY ACCESSIBLE'}",
            "",
            "PHASE 4: FAILURE ROOT CAUSE ANALYSIS",
        ]

        if blocked_neighbors == len(apple_neighbors):
            explanation_parts.extend(
                [
                    "• Primary cause: Apple is completely surrounded by snake body segments",
                    "• Secondary cause: No adjacent cells available for approach",
                    "• Tertiary cause: Snake has grown too large relative to board size",
                    "• Resolution: Impossible until snake body moves away from apple vicinity",
                ]
            )
        elif board_fill_ratio > 0.6:
            explanation_parts.extend(
                [
                    f"• Primary cause: Excessive board occupation ({board_fill_ratio:.1%})",
                    "• Secondary cause: Snake body creates maze-like obstacles",
                    "• Tertiary cause: Insufficient free space for pathfinding",
                    "• Resolution: Wait for tail movement to create path opportunities",
                ]
            )
        else:
            explanation_parts.extend(
                [
                    "• Primary cause: Snake body configuration blocks all viable routes",
                    "• Secondary cause: Temporary spatial arrangement prevents access",
                    "• Tertiary cause: Current head position disadvantaged",
                    "• Resolution: Alternative strategy needed or wait for body repositioning",
                ]
            )

        explanation_parts.extend(
            [
                "",
                "PHASE 5: STRATEGIC IMPLICATIONS",
                "• Immediate action: Cannot pursue apple directly",
                "• Alternative strategies: Tail-chasing, space preservation, defensive positioning",
                "• Risk assessment: HIGH (no progress toward apple possible)",
                "• Expected outcome: Must adopt survival/waiting strategy",
                "",
                "=== CONCLUSION ===",
                f"BFS pathfinding from {tuple(head_pos)} to apple at {tuple(apple_pos)} has failed due to complete path blockage. "
                + f"With {blocked_neighbors}/{len(apple_neighbors)} apple-adjacent cells blocked and {board_fill_ratio:.1%} board occupation, "
                + "the snake must adopt alternative strategies until body repositioning creates new path opportunities. "
                + f"The {manhattan_distance}-step Manhattan distance remains theoretical until obstacles clear.",
            ]
        )

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

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"BFSAgent(algorithm={self.algorithm_name})"

    def _choose_survival_direction(
        self,
        head: List[int],
        valid_moves: List[str],
        snake: List[List[int]],
        grid_size: int,
    ) -> str:
        """
        Choose the best survival direction by evaluating free space in each direction.

        Args:
            head: Current head position [x, y]
            valid_moves: List of valid move directions
            snake: List of snake body positions
            grid_size: Size of the game grid

        Returns:
            Best direction to move for survival
        """
        snake_set = set(tuple(p) for p in snake)
        best_direction = valid_moves[0]  # Default to first valid move
        max_free_space = -1

        for direction in valid_moves:
            # Calculate next position
            dx, dy = DIRECTIONS[direction]
            next_x = head[0] + dx
            next_y = head[1] + dy
            next_pos = (next_x, next_y)

            # Count free space in this direction using BFS
            free_space = count_free_space_in_direction(next_pos, snake_set, grid_size)

            if free_space > max_free_space:
                max_free_space = free_space
                best_direction = direction

        return best_direction

    def _count_free_space_in_direction_bfs(
        self, start_pos: List[int], snake_set: set, grid_size: int
    ) -> int:
        """
        Count free space using BFS from a starting position.

        Args:
            start_pos: Starting position [x, y]
            snake_set: Set of snake body positions as tuples
            grid_size: Size of the game grid

        Returns:
            Number of free cells reachable from start_pos
        """
        if tuple(start_pos) in snake_set:
            return 0

        visited = set()
        queue = deque([tuple(start_pos)])
        visited.add(tuple(start_pos))

        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT

        while queue:
            current_pos = queue.popleft()

            for dx, dy in directions:
                next_x = current_pos[0] + dx
                next_y = current_pos[1] + dy
                next_pos = (next_x, next_y)

                # Check bounds
                if not (0 <= next_x < grid_size and 0 <= next_y < grid_size):
                    continue

                # Check if visited or occupied by snake
                if next_pos in visited or next_pos in snake_set:
                    continue

                # Add to queue
                visited.add(next_pos)
                queue.append(next_pos)

        return len(visited)
