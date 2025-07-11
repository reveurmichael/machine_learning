from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS 1024 Token Agent - Moderately detailed BFS pathfinding for Snake Game v0.04
----------------

This module implements a token-limited BFS agent (1024 tokens) that inherits
from the standard BFS agent but generates moderately detailed explanations.

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
from heuristics_utils import count_obstacles_in_path, calculate_manhattan_distance, calculate_valid_moves_ssot, count_free_space_in_direction
from extensions.common.utils.game_analysis_utils import calculate_apple_direction, calculate_danger_assessment


class BFS1024TokenAgent(BFSAgent):
    """
    BFS Agent with 1024-token limited explanations.
    
    Inheritance Pattern:
    - Inherits from BFSAgent (reuses all pathfinding logic)
    - Overrides _generate_move_explanation() for moderate detail
    - Maintains identical algorithm behavior with medium explanations
    
    Token Limit: ~1024 tokens (moderately detailed explanations)
    """

    def __init__(self):
        """Initialize BFS 1024-token agent, extending base BFS."""
        super().__init__()  # Initialize parent BFS agent
        self.algorithm_name = "BFS-1024"
        # Control whether to include ASCII board representation in prompts (saves tokens)
        self.include_board_representation = True
        
        # ----- JSONL Generation Control Switches -----
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
        
        # Calculate optional metrics based on switches
        additional_metrics = {}
        if self.include_apple_direction:
            additional_metrics["apple_direction"] = calculate_apple_direction(head_pos, apple_position)
        
        if self.include_danger_assessment:
            additional_metrics["danger_assessment"] = calculate_danger_assessment(
                head_pos, body_positions, grid_size, move
            )
        
        if self.include_free_space:
            additional_metrics["free_space"] = {
                "up": count_free_space_in_direction(game_state, "UP"),
                "down": count_free_space_in_direction(game_state, "DOWN"),
                "left": count_free_space_in_direction(game_state, "LEFT"),
                "right": count_free_space_in_direction(game_state, "RIGHT"),
            }
        
        # Build metrics for completion
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
        Generate moderately detailed explanation for the chosen move (1024 tokens max).
        
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = extract_head_position(game_state)
        apple_pos = list(game_state.get('apple_position', [0, 0]))
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(game_state)
        
        # Calculate metrics
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        efficiency_ratio = manhattan_distance / max(path_length, 1)
        is_optimal = path_length == manhattan_distance
        board_fill_ratio = snake_length / (grid_size * grid_size)
        obstacles_avoided = count_obstacles_in_path(path, set(tuple(p) for p in body_positions))
        
        next_pos = (head_pos[0] + (1 if direction == "RIGHT" else -1 if direction == "LEFT" else 0),
                   head_pos[1] + (1 if direction == "UP" else -1 if direction == "DOWN" else 0))
        
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        # Moderate detail explanation
        explanation_parts = [
            "=== BFS PATHFINDING ANALYSIS ===",
            "",
            "SITUATION ASSESSMENT:",
            f"• Head position: {tuple(head_pos)}",
            f"• Apple position: {tuple(apple_pos)}",
            f"• Snake length: {snake_length} segments",
            f"• Grid: {grid_size}×{grid_size}, board fill: {board_fill_ratio:.1%} ({space_pressure} pressure)",
            f"• Free cells: {remaining_free_cells}",
            "",
            "PATHFINDING RESULTS:",
            f"• Manhattan distance: {manhattan_distance} steps",
            f"• BFS path length: {path_length} steps",
            f"• Path efficiency: {efficiency_ratio:.2f} ({'optimal' if is_optimal else 'suboptimal'})",
            f"• Valid moves available: {valid_moves} ({len(valid_moves)} options)",
            f"• Obstacles near path: {obstacles_avoided}",
            "",
            "MOVE DECISION:",
            f"• Chosen direction: {direction}",
            f"• Next position: {next_pos}",
            f"• Rationale: {'Optimal BFS path' if is_optimal else 'Best available BFS path'} to apple",
            "",
            "STRATEGIC ANALYSIS:",
            f"Moving {direction} follows the BFS-computed shortest path from {tuple(head_pos)} to {tuple(apple_pos)}. " +
            f"This advances optimally toward the apple while maintaining {remaining_free_cells - 1} free cells " +
            f"for future maneuvering. Path validated as safe with {space_pressure} board pressure."
        ]

        # Metrics matching parent format
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

    def format_prompt(self, game_state: dict) -> str:
        """Return a prompt string built from *game_state*."""
        grid_size = game_state.get("grid_size", 10)
        head_pos = extract_head_position(game_state)
        apple_pos = game_state.get("apple_position", [0, 0])
        snake_len = len(game_state.get("snake_positions", []))

        prompt_parts = [
            f"Snake on {grid_size}x{grid_size} grid.",
            f"Head: {head_pos}, Apple: {apple_pos}, Length: {snake_len}.",
        ]

        # Optional board representation
        if self.include_board_representation:
            from utils.board_utils import create_text_board

            board_text = create_text_board(
                grid_size,
                head_pos,
                game_state.get("snake_positions", []),
                apple_pos,
            )
            prompt_parts.append("Board:\n" + board_text)

        prompt_parts.append("Choose next move (UP, DOWN, LEFT, RIGHT):")

        return "\n".join(prompt_parts)

    def format_completion(self, move: str, explanation_text: str, metrics: dict) -> str:
        """Return a completion string for the JSONL entry."""
        parts = [explanation_text.strip()]

        # Optionally append metrics section based on switches
        if self.include_metrics_in_completion and any([
            self.include_danger_assessment,
            self.include_apple_direction, 
            self.include_free_space
        ]):
            metrics_summary = self._format_metrics_summary(metrics)
            if metrics_summary:
                parts.append("\nMetrics:\n" + metrics_summary)

        parts.append(f"\nConclusion: {move.upper()}")
        return "\n".join(parts)

    def _format_metrics_summary(self, metrics: dict) -> str:
        """Format metrics for completion text based on enabled switches."""
        formatted_metrics = []
        
        # Include basic metrics
        if 'valid_moves' in metrics:
            formatted_metrics.append(f"- Valid moves: {metrics['valid_moves']}")
        
        if 'manhattan_distance' in metrics:
            formatted_metrics.append(f"- Manhattan distance to apple: {metrics['manhattan_distance']}")
        
        # Include optional metrics based on switches
        if self.include_apple_direction and 'apple_direction' in metrics:
            formatted_metrics.append(f"- Apple direction: {metrics['apple_direction']}")
        
        if self.include_free_space and 'free_space' in metrics:
            formatted_metrics.append(f"- Free space: {metrics['free_space']}")
            
        if self.include_danger_assessment and 'danger_assessment' in metrics:
            formatted_metrics.append(f"- Danger assessment: {metrics['danger_assessment']}")
        
        return "\n".join(formatted_metrics)
