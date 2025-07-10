from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS 512 Token Agent - Concise BFS pathfinding for Snake Game v0.04
----------------

This module implements a token-limited BFS agent (512 tokens) that inherits
from the standard BFS agent but generates very concise explanations.

Design Patterns:
- Inheritance: Extends BFSAgent with token-limited explanations
- Strategy Pattern: Same BFS pathfinding, different explanation generation
- SSOT: Uses all parent methods, only overrides explanation generation
"""

from typing import List, Tuple

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import extension-specific components using relative imports
from .agent_bfs import BFSAgent
from extensions.common.utils.game_state_utils import (
    extract_head_position, extract_body_positions
)
from heuristics_utils import count_obstacles_in_path


class BFS512TokenAgent(BFSAgent):
    """
    BFS Agent with 512-token limited explanations.
    
    Inheritance Pattern:
    - Inherits from BFSAgent (reuses all pathfinding logic)
    - Overrides _generate_move_explanation() for concise output
    - Maintains identical algorithm behavior with shorter explanations
    
    Token Limit: ~512 tokens (very concise explanations)
    """

    def __init__(self):
        """Initialize BFS 512-token agent, extending base BFS."""
        super().__init__()  # Initialize parent BFS agent
        self.algorithm_name = "BFS-512"
        # Control whether to include ASCII board representation in prompts (saves tokens)
        self.include_board_representation = False
        
        # ----- Dataset-generation customisation switches -----
        self.include_danger_assessment = False
        self.include_apple_direction = False
        self.include_free_space = False

    def _generate_move_explanation(self, game_state: dict, path: List[Tuple[int, int]], 
                                 direction: str, valid_moves: List[str],
                                 manhattan_distance: int, remaining_free_cells: int) -> dict:
        """
        Generate concise explanation for the chosen move (512 tokens max).
        
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = extract_head_position(game_state)
        apple_pos = list(game_state.get('apple_position', [0, 0]))
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(game_state)
        
        # Calculate basic metrics
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        next_pos = (head_pos[0] + (1 if direction == "RIGHT" else -1 if direction == "LEFT" else 0),
                   head_pos[1] + (1 if direction == "UP" else -1 if direction == "DOWN" else 0))
        
        # Concise explanation
        explanation_parts = [
            f"Path found: {path_length} steps",
            f"Moving {direction} to {next_pos}",
            "",
            f"Rationale: Move {direction} advances optimally toward apple.",
        ]

        # Metrics matching parent format
        explanation_dict = {
            "strategy_phase": "APPLE_PATH",
            "metrics": {
                "manhattan_distance": int(manhattan_distance),
                "path_length": int(path_length),
                "obstacles_near_path": count_obstacles_in_path(path, set(tuple(p) for p in body_positions)),
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

    def format_metrics_for_completion(self, metrics: dict, additional_metrics: dict = None) -> str:
        """
        Format metrics for completion text. Users can override this method to customize
        which metrics to include and how to format them.
        
        Args:
            metrics: Agent's own metrics dictionary
            additional_metrics: Additional metrics from dataset generator (optional)
        
        Returns:
            Formatted metrics string for completion
        """
        # Default implementation - users can override for custom formatting
        formatted_metrics = []
        
        # Include basic metrics
        if 'valid_moves' in metrics:
            formatted_metrics.append(f"- Valid moves: {metrics['valid_moves']}")
        
        if 'manhattan_distance' in metrics:
            formatted_metrics.append(f"- Manhattan distance to apple: {metrics['manhattan_distance']}")
        
        # Include additional metrics if provided
        if additional_metrics:
            if 'apple_direction' in additional_metrics:
                formatted_metrics.append(f"- Apple direction: {additional_metrics['apple_direction']}")
            
            if 'free_space' in additional_metrics:
                formatted_metrics.append(f"- Free space: {additional_metrics['free_space']}")
        
        return "\n".join(formatted_metrics) if formatted_metrics else ""

    # ------------------------------------------------------------------
    # DatasetGenerator will call the following two hooks (if present) to
    # let the agent fully control JSONL prompt / completion generation.
    # ------------------------------------------------------------------

    def format_prompt(self, game_state: dict) -> str:  # noqa: D401 – simple description is OK
        """Return a concise prompt string built from *game_state*.

        This implementation intentionally keeps the prompt extremely short
        to stay well within a 512-token budget.  Additional details (board
        ASCII art, danger assessment, etc.) can be enabled by toggling the
        corresponding *include_* attributes.
        """
        grid_size = game_state.get("grid_size", 10)
        head_pos = extract_head_position(game_state)
        apple_pos = game_state.get("apple_position", [0, 0])
        snake_len = len(game_state.get("snake_positions", []))

        prompt_parts = [
            f"Snake on {grid_size}x{grid_size} grid.",
            f"Head: {head_pos}, Apple: {apple_pos}, Length: {snake_len}.",
        ]

        # Optional board representation (rarely used for 512-token agent)
        if self.include_board_representation:
            from utils.board_utils import create_text_board  # local import – avoids heavy global import cost

            board_text = create_text_board(
                grid_size,
                head_pos,
                game_state.get("snake_positions", []),
                apple_pos,
            )
            prompt_parts.append("Board:\n" + board_text)

        prompt_parts.append("Choose next move (UP, DOWN, LEFT, RIGHT):")

        return "\n".join(prompt_parts)

    def format_completion(self, move: str, explanation_text: str, metrics: dict) -> str:  # noqa: D401
        """Return a concise completion string for the JSONL entry."""
        parts = [explanation_text.strip()]

        # Optionally append a tiny metrics section
        if (
            self.include_danger_assessment
            or self.include_apple_direction
            or self.include_free_space
        ):
            metrics_summary = self.format_metrics_for_completion(metrics, metrics)
            if metrics_summary:
                parts.append("\nMetrics:\n" + metrics_summary)

        parts.append(f"\nConclusion: {move.upper()}")
        return "\n".join(parts)


