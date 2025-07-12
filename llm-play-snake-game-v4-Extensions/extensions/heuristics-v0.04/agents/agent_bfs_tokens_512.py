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

from typing import List, Tuple, Dict, Any

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import extension-specific components using relative imports
from .agent_bfs import BFSAgent
from extensions.common.utils.game_state_utils import (
    extract_head_position, extract_body_positions, extract_grid_size
)
from heuristics_utils import count_obstacles_in_path, calculate_valid_moves_ssot, count_free_space_in_direction
from extensions.common.utils.game_analysis_utils import calculate_apple_direction, calculate_danger_assessment


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
        
        # ----- JSONL Generation Control Switches -----
        # Control whether to include ASCII board representation in prompts (saves tokens)
        self.include_board_representation = False
        self.include_danger_assessment = False
        self.include_apple_direction = False
        self.include_free_space = False
        self.include_metrics_in_completion = False

    def generate_jsonl_record(self, game_state: dict, move: str, explanation: dict, 
                            game_id: int = 1, round_num: int = 1) -> Dict[str, Any]:
        """
        SSOT: Single method to generate complete JSONL record.
        
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
        grid_size = extract_grid_size(game_state)
        
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
        
        # Calculate comprehensive metrics for detailed JSONL output
        additional_metrics = {}
        
        # Always include apple direction for detailed format
        additional_metrics["apple_direction"] = calculate_apple_direction(head_pos, apple_position)
        
        # Always include danger assessment for detailed format
        additional_metrics["danger_assessment"] = calculate_danger_assessment(
            head_pos, body_positions, grid_size, move
        )
        
        # Always include free space analysis for detailed format
        additional_metrics["free_space"] = {
            "up": count_free_space_in_direction(game_state, "UP"),
            "down": count_free_space_in_direction(game_state, "DOWN"),
            "left": count_free_space_in_direction(game_state, "LEFT"),
            "right": count_free_space_in_direction(game_state, "RIGHT"),
        }
        
        # Build comprehensive metrics for completion
        base_metrics = {
            "valid_moves": valid_moves,
            "manhattan_distance": explanation.get("metrics", {}).get("manhattan_distance", 0),
        }
        base_metrics.update(additional_metrics)
        
        # Generate prompt and completion using centralized methods
        prompt = self.format_prompt(game_state)
        completion = self.format_completion(move, explanation_text, base_metrics)
        
        return {
            "prompt": prompt,
            "completion": completion,
        }

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

    def format_prompt(self, game_state: dict) -> str:  # noqa: D401 – simple description is OK
        """Return a detailed prompt string built from *game_state*.

        This implementation provides comprehensive game state information
        to enable detailed analysis and decision-making.
        """
        grid_size = game_state.get("grid_size", 10)
        head_pos = extract_head_position(game_state)
        apple_pos = game_state.get("apple_position", [0, 0])
        body_positions = extract_body_positions(game_state)
        snake_len = len(game_state.get("snake_positions", []))

        prompt_parts = [
            f"You are playing Snake on a {grid_size}x{grid_size} grid. The coordinate system is (0,0) at bottom-left to ({grid_size-1},{grid_size-1}) at top-right. Movement: UP=y+1, DOWN=y-1, RIGHT=x+1, LEFT=x-1.",
            "",
            "Current game state:",
            f"- Snake head position: ({head_pos[0]}, {head_pos[1]})",
            f"- Apple position: ({apple_pos[0]}, {apple_pos[1]})",
            f"- Snake body positions: {body_positions}",
            f"- Snake length: {snake_len}",
            "",
            "What is the best move to make? Consider:",
            "1. Path to the apple",
            "2. Avoiding collisions with walls and snake body", 
            "3. Maximizing score and survival",
            "",
            "Choose from: UP, DOWN, LEFT, RIGHT"
        ]

        return "\n".join(prompt_parts)

    def format_completion(self, move: str, explanation_text: str, metrics: dict) -> str:  # noqa: D401
        """Return a detailed completion string for the JSONL entry."""
        # Extract path information from explanation
        path_info = ""
        if "Path found:" in explanation_text:
            path_line = [line for line in explanation_text.split('\n') if line.startswith("Path found:")]
            if path_line:
                path_info = path_line[0]
        
        # Extract move information
        move_info = ""
        if f"Moving {move}" in explanation_text:
            move_lines = [line for line in explanation_text.split('\n') if f"Moving {move}" in line]
            if move_lines:
                move_info = move_lines[0]
        
        parts = []
        
        # Add path and move information
        if path_info:
            parts.append(path_info)
        if move_info:
            parts.append(move_info)
        
        # Add comprehensive metrics section
        parts.append("")
        parts.append("Metrics:")
        
        # Format metrics in the requested style
        if 'valid_moves' in metrics:
            parts.append(f"- Valid moves: {metrics['valid_moves']}")
        
        if 'manhattan_distance' in metrics:
            parts.append(f"- Manhattan distance to apple: {metrics['manhattan_distance']}")
        
        if 'apple_direction' in metrics:
            parts.append(f"- Apple direction: {metrics['apple_direction']}")
        
        if 'danger_assessment' in metrics:
            parts.append(f"- Danger assessment: {metrics['danger_assessment']}")
        
        if 'free_space' in metrics:
            parts.append(f"- Free space: {metrics['free_space']}")
        
        parts.append("")
        parts.append(f"Conclusion: The move is: {move.upper()}")
        
        return "\n".join(parts)


