from __future__ import annotations
from typing import List, Tuple, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Safe Greedy Agent - Blueprint Template for BFS-SAFE-GREEDY Token Variants
----------------

This module implements a SAFE-GREEDY agent that serves as a blueprint template
for token-limited variants. It contains only core BFS-SAFE-GREEDY logic and
basic explanation structure with minimal JSONL output (~100 tokens).

This is a BLUEPRINT TEMPLATE - not used for actual dataset generation.
Token variants (BFS-SAFE-GREEDY-4096, etc.) inherit from this class and add
their own detailed JSONL generation logic.

Algorithm:
1. Find shortest path to apple using BFS
2. Validate path safety (can snake reach tail after move?)
3. If safe, follow apple path
4. If unsafe, follow tail (always safe)
5. If no paths exist, use any valid move

Design Patterns:
- Inheritance: Extends BFSAgent with safety validation
- Template Method: Base class provides algorithm structure
- Single Responsibility: Core BFS-SAFE-GREEDY logic only
"""

from typing import TYPE_CHECKING

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root

ensure_project_root()

# Import from project root using absolute imports
from utils.moves_utils import position_to_direction

# Import extension-specific components using relative imports
from .agent_bfs import BFSAgent
from extensions.common.utils.game_state_utils import (
    extract_head_position,
    extract_body_positions,
)
from heuristics_utils import (
    calculate_manhattan_distance,
    calculate_valid_moves_ssot,
    count_remaining_free_cells,
    bfs_pathfind,
)

if TYPE_CHECKING:
    pass


class BFSSafeGreedyAgent(BFSAgent):
    """
    BFS Safe Greedy Agent: Blueprint template for token variants.

    This is a BLUEPRINT TEMPLATE for token-limited variants.
    Contains only core BFS-SAFE-GREEDY logic and basic explanation structure.
    Token variants inherit from this class and add their own JSONL generation.

    Inheritance Pattern:
    - Inherits from BFSAgent (reuses helper methods and patterns)
    - Overrides get_move_with_explanation() to add safety validation
    - Maintains consistent naming and structure with BFS agent

    Algorithm Enhancement:
    1. Find shortest path to apple using BFS
    2. Validate path safety (can snake reach tail afterward?)
    3. If safe, follow apple path
    4. If unsafe, chase tail (always safe)
    5. Last resort: any valid move

    KISS: No unnecessary fallbacks, fail-fast on SSOT violations
    """

    def __init__(self) -> None:
        """Initialize BFS Safe Greedy agent blueprint."""
        super().__init__()  # Initialize parent BFS agent
        self.algorithm_name = "BFS-SAFE-GREEDY"
        self.name = "BFS Safe Greedy"
        self.description = (
            "Enhanced BFS with safety validation. Inherits core BFS pathfinding "
            "from BFSAgent and adds safety checks to avoid getting trapped. "
            "Falls back to tail-chasing when apple path is unsafe."
        )

    def get_move(self, state: dict) -> str | None:
        """
        Get next move using safe BFS pathfinding (simplified interface).
        Args:
            state: Game state dict
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        move, _ = self.get_move_with_explanation(state)
        return move

    def get_move_with_explanation(self, state: dict) -> Tuple[str, dict]:
        """
        SAFE-GREEDY agent: Prioritizes safety over greed.
        KISS: Fail fast on any SSOT violations.

        This is a BLUEPRINT TEMPLATE - token variants override this method
        to add their own detailed explanations and JSONL generation.
        """
        # SSOT: Use centralized utilities from parent BFSAgent for all position extractions
        snake_positions = state.get("snake_positions", [])
        head_pos = extract_head_position(state)
        apple_pos = state.get("apple_position", [0, 0])
        grid_size = state.get("grid_size", 10)

        # SSOT: Use centralized body positions calculation from parent
        body_positions = extract_body_positions(state)
        obstacles = set(tuple(p) for p in body_positions)

        # SSOT: Use centralized calculations for all metrics from parent
        manhattan_distance = calculate_manhattan_distance(state)
        valid_moves = calculate_valid_moves_ssot(state)
        remaining_free_cells = count_remaining_free_cells(
            set(tuple(p) for p in snake_positions), grid_size
        )

        # ---------------- 1. Try safe apple path first
        path_to_apple = bfs_pathfind(head_pos, apple_pos, obstacles, grid_size)
        if path_to_apple and len(path_to_apple) > 1:
            next_pos = path_to_apple[1]
            direction = position_to_direction(tuple(head_pos), tuple(next_pos))

            # Fail-fast: validate bounds and valid moves
            if (
                next_pos[0] < 0
                or next_pos[0] >= grid_size
                or next_pos[1] < 0
                or next_pos[1] >= grid_size
            ):
                raise RuntimeError(
                    f"SSOT violation: BFS-SAFE-GREEDY computed out-of-bounds position {next_pos} for grid size {grid_size}"
                )

            if direction not in valid_moves:
                raise RuntimeError(
                    f"SSOT violation: BFS-SAFE-GREEDY computed move '{direction}' is not valid for head {head_pos} and valid_moves {valid_moves}"
                )

            # Safety validation: can snake reach tail after this move?
            if self._is_move_safe(state, next_pos):
                # Safe path found
                metrics = {
                    "final_chosen_direction": direction,
                    "head_position": list(head_pos),
                    "apple_position": list(apple_pos),
                    "snake_length": len(snake_positions),
                    "grid_size": grid_size,
                    "valid_moves": valid_moves,
                    "manhattan_distance": manhattan_distance,
                    "remaining_free_cells": remaining_free_cells,
                    "path_length": len(path_to_apple) - 1,
                    "apple_path_safe": True,
                }
                explanation_dict = self._generate_basic_safe_apple_explanation(
                    state,
                    path_to_apple,
                    direction,
                    valid_moves,
                    manhattan_distance,
                    remaining_free_cells,
                    metrics,
                )
                return direction, explanation_dict

        # ---------------- 2. Apple path unsafe or not found, try tail-chasing
        tail = snake_positions[-1]
        path_to_tail = bfs_pathfind(head_pos, tail, obstacles, grid_size)
        if path_to_tail and len(path_to_tail) > 1:
            next_pos = path_to_tail[1]
            direction = position_to_direction(tuple(head_pos), tuple(next_pos))

            # Fail-fast: validate bounds and valid moves
            if (
                next_pos[0] < 0
                or next_pos[0] >= grid_size
                or next_pos[1] < 0
                or next_pos[1] >= grid_size
            ):
                raise RuntimeError(
                    f"SSOT violation: BFS-SAFE-GREEDY tail-chase computed out-of-bounds position {next_pos} for grid size {grid_size}"
                )

            if direction not in valid_moves:
                raise RuntimeError(
                    f"SSOT violation: BFS-SAFE-GREEDY tail-chase move '{direction}' is not valid for head {head_pos} and valid_moves {valid_moves}"
                )

            metrics = {
                "final_chosen_direction": direction,
                "head_position": list(head_pos),
                "apple_position": list(apple_pos),
                "snake_length": len(snake_positions),
                "grid_size": grid_size,
                "valid_moves": valid_moves,
                "manhattan_distance": manhattan_distance,
                "remaining_free_cells": remaining_free_cells,
                "path_length": len(path_to_tail) - 1,
                "apple_path_safe": False,
            }

            explanation_dict = self._generate_basic_tail_chase_explanation(
                state,
                tail,
                path_to_tail,
                direction,
                valid_moves,
                manhattan_distance,
                remaining_free_cells,
                metrics,
            )

            return direction, explanation_dict

        # ---------------- 3. Last resort: any valid move
        if valid_moves:
            direction = valid_moves[0]
            metrics = {
                "final_chosen_direction": direction,
                "head_position": list(head_pos),
                "apple_position": list(apple_pos),
                "snake_length": len(snake_positions),
                "grid_size": grid_size,
                "valid_moves": valid_moves,
                "manhattan_distance": manhattan_distance,
                "remaining_free_cells": remaining_free_cells,
                "path_length": 0,
                "apple_path_safe": False,
            }

            explanation_dict = self._generate_basic_survival_explanation(
                state,
                direction,
                valid_moves,
                manhattan_distance,
                remaining_free_cells,
                metrics,
            )

            return direction, explanation_dict
        else:
            direction = "NO_PATH_FOUND"
            metrics = {
                "final_chosen_direction": direction,
                "head_position": list(head_pos),
                "apple_position": list(apple_pos),
                "snake_length": len(snake_positions),
                "grid_size": grid_size,
                "valid_moves": valid_moves,
                "manhattan_distance": manhattan_distance,
                "remaining_free_cells": remaining_free_cells,
                "path_length": 0,
                "apple_path_safe": False,
            }
            explanation_dict = self._generate_basic_no_moves_explanation(
                state, valid_moves, manhattan_distance, remaining_free_cells, metrics
            )
            return direction, explanation_dict

    def _is_move_safe(self, game_state: dict, next_pos: List[int]) -> bool:
        """
        Validate if a move is safe (snake can reach tail afterward).

        SSOT: Extract positions using exact same logic as dataset_generator.py
        to guarantee true single source of truth.

        PRE-EXECUTION: This method validates safety based on the current game state
        and a predicted next position. All parameters are from pre-move state:
        - game_state: complete game state dict (before move)
        - next_pos: where the head will be after the move (predicted)

        The method simulates what would happen after the move and checks if
        the snake could still reach its tail from the new position.

        Args:
            game_state: Complete game state dict containing all positions
            next_pos: Predicted head position after the move (PRE-MOVE prediction)

        Returns:
            True if the move is safe (snake can reach tail afterward), False otherwise
        """
        # SSOT: Extract positions using exact same logic as dataset_generator.py
        snake_positions = game_state.get("snake_positions", [])
        head_pos = extract_head_position(game_state)
        grid_size = game_state.get("grid_size", 10)

        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(game_state)
        obstacles = set(tuple(p) for p in body_positions)

        # Simulate the move: head moves to next_pos, tail disappears
        new_snake = [list(next_pos)] + snake_positions[
            :-1
        ]  # Head moves, tail disappears
        new_obstacles = set(
            tuple(p) for p in new_snake[1:]
        )  # All body segments except new head

        # Check if snake can reach its tail from new position
        tail = snake_positions[-1]  # Original tail position
        path_to_tail = bfs_pathfind(next_pos, tail, new_obstacles, grid_size)

        return path_to_tail is not None and len(path_to_tail) > 1

    def _generate_basic_safe_apple_explanation(
        self,
        game_state: dict,
        path: List[List[int]],
        direction: str,
        valid_moves: List[str],
        manhattan_distance: int,
        remaining_free_cells: int,
        metrics: dict,
    ) -> dict:
        """
        Generate basic safe apple explanation (~100 tokens).

        Token variants will override this method to provide detailed explanations.
        """
        head_pos = extract_head_position(game_state)
        apple_pos = list(game_state.get("apple_position", [0, 0]))
        path_length = len(path) - 1

        explanation_parts = [
            f"SAFE-GREEDY: Found safe path to apple ({path_length} steps)",
            f"Moving {direction} from {head_pos}",
            f"Distance: {manhattan_distance}, Free cells: {remaining_free_cells}",
            f"Rationale: Safe shortest path to apple",
        ]

        return {
            "strategy_phase": "SAFE_APPLE_PATH",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def _generate_basic_tail_chase_explanation(
        self,
        game_state: dict,
        tail: List[int],
        path: List[List[int]],
        direction: str,
        valid_moves: List[str],
        manhattan_distance: int,
        remaining_free_cells: int,
        metrics: dict,
    ) -> dict:
        """
        Generate basic tail chase explanation (~100 tokens).

        Token variants will override this method to provide detailed explanations.
        """
        head_pos = extract_head_position(game_state)
        path_length = len(path) - 1

        explanation_parts = [
            f"SAFE-GREEDY: Apple path unsafe, chasing tail ({path_length} steps)",
            f"Moving {direction} from {head_pos} to {tail}",
            f"Distance: {manhattan_distance}, Free cells: {remaining_free_cells}",
            f"Rationale: Safe tail-chasing strategy",
        ]

        return {
            "strategy_phase": "TAIL_CHASE",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def _generate_basic_survival_explanation(
        self,
        game_state: dict,
        direction: str,
        valid_moves: List[str],
        manhattan_distance: int,
        remaining_free_cells: int,
        metrics: dict,
    ) -> dict:
        """
        Generate basic survival explanation (~100 tokens).

        Token variants will override this method to provide detailed explanations.
        """
        head_pos = extract_head_position(game_state)

        explanation_parts = [
            f"SAFE-GREEDY: No safe paths found, using survival move",
            f"Moving {direction} from {head_pos}",
            f"Distance: {manhattan_distance}, Free cells: {remaining_free_cells}",
            f"Rationale: Last resort survival strategy",
        ]

        return {
            "strategy_phase": "SURVIVAL_MOVE",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def _generate_basic_no_moves_explanation(
        self,
        game_state: dict,
        valid_moves: List[str],
        manhattan_distance: int,
        remaining_free_cells: int,
        metrics: dict,
    ) -> dict:
        """
        Generate basic no moves explanation (~100 tokens).

        Token variants will override this method to provide detailed explanations.
        """
        head_pos = extract_head_position(game_state)

        explanation_parts = [
            f"SAFE-GREEDY: No valid moves available",
            f"Head at {head_pos}, no escape possible",
            f"Distance: {manhattan_distance}, Free cells: {remaining_free_cells}",
            f"Rationale: Game over - snake trapped",
        ]

        return {
            "strategy_phase": "GAME_OVER",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def __str__(self) -> str:
        """String representation of BFS Safe Greedy agent."""
        return f"BFSSafeGreedyAgent({self.algorithm_name})"

    def generate_jsonl_record(
        self,
        game_state: dict,
        move: str,
        explanation: dict,
        game_id: int = 1,
        round_num: int = 1,
    ) -> Dict[str, Any]:
        """
        SSOT: Single method to generate complete JSONL record for base BFS-SAFE-GREEDY.

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
            "manhattan_distance": explanation.get("metrics", {}).get(
                "manhattan_distance", 0
            ),
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
