from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS 4096 Token Agent - Full detailed BFS pathfinding for Snake Game v0.04
----------------

This module implements a full inheritance BFS agent (4096 tokens) that inherits
completely from the standard BFS agent with no modifications.

Design Patterns:
- Full Inheritance: Complete inheritance from BFSAgent with no overrides
- Strategy Pattern: Identical BFS pathfinding and explanation generation
- SSOT: Uses all parent methods without any modifications
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

class BFS4096TokenAgent(BFSAgent):
    """
    BFS Agent with full 4096-token explanations (identical to original BFS).
    
    Full Inheritance Pattern:
    - Complete inheritance from BFSAgent with no overrides
    - Maintains identical algorithm behavior and explanation generation
    - Only changes algorithm_name for identification purposes
    
    Token Limit: ~4096 tokens (full detailed explanations, identical to BFS)
    """

    def __init__(self):
        """Initialize BFS 4096-token agent, exactly like base BFS."""
        super().__init__()  # Initialize parent BFS agent
        self.algorithm_name = "BFS-4096"
        # Control whether to include ASCII board representation in prompts (saves tokens)
        self.include_board_representation = True
        
        # ----- JSONL Generation Control Switches -----
        self.include_danger_assessment = False
        self.include_apple_direction = False
        self.include_free_space = False
        self.include_metrics_in_completion = False
        
    # No method overrides - this agent is exactly identical to BFSAgent
    # except for the algorithm_name for identification purposes

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


