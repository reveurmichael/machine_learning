from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Agent - Blueprint Template for BFS Token Variants
----------------

This module implements a BFS (Breadth-First Search) agent that serves as a
blueprint template for token-limited variants. It contains only core BFS logic
and basic explanation structure with minimal JSONL output (~100 tokens).

This is a BLUEPRINT TEMPLATE - not used for actual dataset generation.
Token variants (BFS-512, BFS-1024, etc.) inherit from this class and add
their own detailed JSONL generation logic.

Design Patterns:
- Template Method: Base class provides algorithm structure
- Inheritance: Token variants extend this blueprint
- Single Responsibility: Core BFS logic only
"""

from collections import deque
from typing import List, Tuple, TYPE_CHECKING, Dict, Any

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
)
from heuristics_utils import (
    count_obstacles_in_path,
    count_remaining_free_cells,
    count_free_space_in_direction,
    calculate_manhattan_distance,
    calculate_valid_moves_ssot,
    bfs_pathfind,
)

if TYPE_CHECKING:
    pass


class BFSAgent(BaseAgent):
    """
    Breadth-First Search agent blueprint template.

    This is a BLUEPRINT TEMPLATE for token-limited variants.
    Contains only core BFS logic and basic explanation structure.
    Token variants inherit from this class and add their own JSONL generation.

    Algorithm:
    1. Start from current head position
    2. Explore all valid adjacent positions using BFS
    3. Return the first move in the shortest path to apple
    4. Generate basic explanation structure (~100 tokens)
    5. If no path exists, return "NO_PATH_FOUND" with basic explanation

    Design Patterns:
    - Template Method: Provides algorithm structure for inheritance
    - Single Responsibility: Core BFS logic only
    """

    def __init__(self):
        """Initialize BFS agent blueprint."""
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
        """
        Core BFS pathfinding with basic explanation structure.
        
        This is a BLUEPRINT TEMPLATE - token variants override this method
        to add their own detailed explanations and JSONL generation.
        """
        # SSOT: Use centralized utilities for all position and calculation extractions
        head = extract_head_position(state)
        apple = list(state["apple_position"])
        snake = [list(seg) for seg in state["snake_positions"]]
        grid_size = state["grid_size"]

        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(state)
        obstacles = set(tuple(p) for p in body_positions)

        # SSOT: Use centralized calculations for all metrics
        manhattan_distance = calculate_manhattan_distance(state)
        valid_moves = calculate_valid_moves_ssot(state)
        remaining_free_cells = count_remaining_free_cells(
            set(tuple(p) for p in snake), grid_size
        )

        # ---------------- Apple path first
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

            # Basic metrics structure (token variants will enhance this)
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
                "obstacles_near_path": count_obstacles_in_path(
                    path_to_apple, set(tuple(p) for p in snake)
                ),
                "apple_path_safe": True,
            }

            # Basic explanation structure (token variants will enhance this)
            explanation_dict = self._generate_basic_explanation(
                state,
                path_to_apple,
                direction,
                valid_moves,
                manhattan_distance,
                remaining_free_cells,
            )
            explanation_dict["metrics"] = metrics

            return direction, explanation_dict
        else:
            # Basic survival strategy
            if valid_moves:
                direction = valid_moves[0]
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
                    "obstacles_near_path": 0,
                    "apple_path_safe": False,
                    "survival_strategy": "random_valid_move",
                }

                explanation_dict = {
                    "strategy_phase": "SURVIVAL_MOVE",
                    "metrics": metrics,
                    "explanation_steps": [
                        f"No path to apple found from {head} to {apple}.",
                        f"Using survival strategy: choosing first valid move '{direction}'.",
                    ],
                }

                return direction, explanation_dict
            else:
                return "NO_PATH_FOUND", {"strategy_phase": "GAME_OVER", "metrics": {}}

    def _generate_basic_explanation(
        self,
        game_state: dict,
        path: List[Tuple[int, int]],
        direction: str,
        valid_moves: List[str],
        manhattan_distance: int,
        remaining_free_cells: int,
    ) -> dict:
        """
        Generate basic explanation structure for blueprint template (~100 tokens).
        
        Token variants will override this method to provide detailed explanations.
        This method provides only the basic structure and core metrics.
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = extract_head_position(game_state)
        apple_pos = list(game_state.get("apple_position", [0, 0]))
        grid_size = game_state.get("grid_size", 10)

        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(game_state)

        # Basic calculations
        path_length = len(path) - 1
        snake_length = len(game_state.get("snake_positions", []))
        obstacles_avoided = count_obstacles_in_path(
            path, set(tuple(p) for p in body_positions)
        )

        # Basic explanation structure (~100 tokens)
        explanation_parts = [
            f"BFS: Found path to apple ({path_length} steps)",
            f"Moving {direction} from {head_pos}",
            f"Distance: {manhattan_distance}, Free cells: {remaining_free_cells}",
            f"Rationale: Shortest path to apple",
        ]

        return {
            "strategy_phase": "APPLE_PATH",
            "explanation_steps": explanation_parts,
        }

    def __str__(self) -> str:
        """String representation of BFS agent."""
        return f"BFSAgent({self.algorithm_name})"

    def _choose_survival_direction(
        self,
        head: List[int],
        valid_moves: List[str],
        snake: List[List[int]],
        grid_size: int,
    ) -> str:
        """
        Choose survival direction when no path to apple exists.
        
        This method implements a simple survival strategy by choosing
        the direction with the most free space.
        """
        if not valid_moves:
            return "NO_PATH_FOUND"

        # Simple survival: choose direction with most free space
        best_direction = valid_moves[0]
        max_free_space = 0

        for direction in valid_moves:
            free_space = count_free_space_in_direction(
                {
                    "snake_positions": snake,
                    "grid_size": grid_size,
                    "head_position": head,
                },
                direction,
            )
            if free_space > max_free_space:
                max_free_space = free_space
                best_direction = direction

        return best_direction

    def _count_free_space_in_direction_bfs(
        self, start_pos: List[int], snake_set: set, grid_size: int
    ) -> int:
        """
        Count free space in a direction using BFS.
        
        This is a helper method for survival strategies.
        """
        if not snake_set:
            return grid_size * grid_size

        visited = set()
        queue = deque([(start_pos[0], start_pos[1])])
        visited.add((start_pos[0], start_pos[1]))
        count = 0

        while queue:
            x, y = queue.popleft()
            count += 1

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (
                    0 <= new_x < grid_size
                    and 0 <= new_y < grid_size
                    and (new_x, new_y) not in visited
                    and (new_x, new_y) not in snake_set
                ):
                    visited.add((new_x, new_y))
                    queue.append((new_x, new_y))

        return count

    def generate_jsonl_record(self, game_state: dict, move: str, explanation: dict, 
                            game_id: int = 1, round_num: int = 1) -> Dict[str, Any]:
        """
        SSOT: Single method to generate complete JSONL record for base BFS.
        
        This centralizes all JSONL generation logic in the agent, making the pipeline
        short and flexible while maintaining SSOT compliance.
        
        Args:
            game_state: Pre-move game state
            move: The chosen move direction
            explanation: Agent's move explanation with metrics
            game_id: Game identifier
            round_num: Round number
            
        Returns:
            Complete JSONL record ready for writing
        """
        # SSOT: Extract all data from provided game_state (pre-move state)
        head_pos = extract_head_position(game_state)
        body_positions = extract_body_positions(game_state)
        apple_position = game_state.get("apple_position", [0, 0])
        grid_size = game_state.get("grid_size", 10)
        
        # SSOT: Validate move against game state
        valid_moves = calculate_valid_moves_ssot(game_state)
        if move not in valid_moves:
            raise RuntimeError(
                f"SSOT violation: move '{move}' not in valid moves {valid_moves} "
                f"for head {head_pos} in game {game_id} round {round_num}"
            )
        
        # SSOT: Validate explanation head consistency
        if isinstance(explanation, dict) and "metrics" in explanation:
            explanation_head = explanation["metrics"].get("head_position")
            if explanation_head and tuple(explanation_head) != tuple(head_pos):
                raise RuntimeError(
                    f"SSOT violation: explanation head {explanation_head} != "
                    f"game state head {head_pos} for game {game_id} round {round_num}"
                )
        
        # Extract explanation text
        if isinstance(explanation, dict) and "explanation_steps" in explanation:
            explanation_text = "\n".join(explanation["explanation_steps"])
        else:
            raise RuntimeError(
                f"SSOT violation: explanation missing 'explanation_steps' for game {game_id}"
            )
        
        # Build metrics for completion
        base_metrics = {
            "valid_moves": valid_moves,
            "manhattan_distance": explanation.get("metrics", {}).get("manhattan_distance", 0),
        }
        
        # Generate prompt and completion using centralized methods
        prompt = self.format_prompt(game_state)
        completion = self.format_completion(move, explanation_text, base_metrics)
        
        return {
            "prompt": prompt,
            "completion": completion,
        }

    def format_prompt(self, game_state: dict) -> str:
        """Return a simple prompt string built from *game_state*."""
        grid_size = game_state.get("grid_size", 10)
        head_pos = extract_head_position(game_state)
        apple_pos = game_state.get("apple_position", [0, 0])
        snake_len = len(game_state.get("snake_positions", []))

        prompt_parts = [
            f"Snake on {grid_size}x{grid_size} grid.",
            f"Head: {head_pos}, Apple: {apple_pos}, Length: {snake_len}.",
            "Choose next move (UP, DOWN, LEFT, RIGHT):",
        ]

        return "\n".join(prompt_parts)

    def format_completion(self, move: str, explanation_text: str, metrics: dict) -> str:
        """Return a simple completion string for the JSONL entry."""
        parts = [explanation_text.strip()]
        parts.append(f"\nConclusion: {move.upper()}")
        return "\n".join(parts)
