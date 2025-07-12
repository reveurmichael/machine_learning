from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Safe Greedy 4096 Token Agent - Full detailed BFS with Safety Validation for Snake Game v0.04
----------------

This module implements a token-limited BFS-SAFE-GREEDY agent (4096 tokens) that inherits
from the standard BFS-SAFE-GREEDY agent but generates full detailed explanations.

Design Patterns:
- Inheritance: Extends BFSSafeGreedyAgent with token-limited explanations
- Strategy Pattern: Same BFS-SAFE-GREEDY pathfinding, different explanation generation
- SSOT: Uses all parent methods, only overrides explanation generation
"""

from typing import List, Dict, Any

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import extension-specific components using relative imports
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent
from extensions.common.utils.game_state_utils import (
    extract_head_position, extract_body_positions, extract_grid_size
)
from heuristics_utils import calculate_valid_moves_ssot, count_free_space_in_direction
from extensions.common.utils.game_analysis_utils import calculate_apple_direction, calculate_danger_assessment

class BFSSafeGreedy4096TokenAgent(BFSSafeGreedyAgent):
    """
    BFS Safe Greedy Agent with full 4096-token explanations.
    
    Inheritance Pattern:
    - Inherits from BFSSafeGreedyAgent (reuses all pathfinding logic)
    - Overrides _generate_basic_*_explanation() methods for detailed output
    - Maintains identical algorithm behavior with full explanations
    
    Token Limit: ~4096 tokens (full detailed explanations)
    """

    def __init__(self) -> None:
        """Initialize BFS Safe Greedy 4096-token agent, extending base BFS-SAFE-GREEDY."""
        super().__init__()  # Initialize parent BFS Safe Greedy agent
        self.algorithm_name = "BFS-SAFE-GREEDY-4096"
        
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

    def _generate_basic_safe_apple_explanation(self, game_state: dict, path: List[List[int]], 
                                        direction: str, valid_moves: List[str],
                                        manhattan_distance: int, remaining_free_cells: int,
                                        metrics: dict) -> dict:
        """
        Generate full detailed explanation for safe apple path move (4096 tokens max).
        
        SSOT Compliance: All coordinates and positions are taken directly from
        the recorded game state, ensuring perfect consistency with dataset generation.
        """
        # SSOT: Use centralized utilities for all position extractions
        head_pos = extract_head_position(game_state)
        apple_pos = list(game_state.get('apple_position', [0, 0]))
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use centralized body positions calculation
        body_positions = extract_body_positions(game_state)
        
        # Calculate detailed metrics
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        efficiency_ratio = manhattan_distance / max(path_length, 1)
        is_optimal = path_length == manhattan_distance
        detour_steps = max(0, path_length - manhattan_distance)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        next_pos = (head_pos[0] + (1 if direction == "RIGHT" else -1 if direction == "LEFT" else 0),
                   head_pos[1] + (1 if direction == "UP" else -1 if direction == "DOWN" else 0))
        
        efficiency_str = f"{efficiency_ratio:.2f} ({path_length}/{manhattan_distance})"
        
        # Full detailed explanation (identical to original BFS-SAFE-GREEDY)
        explanation_parts = [
            "=== BFS-SAFE-GREEDY PATHFINDING ANALYSIS ===",
            "",
            "PHASE 1: INITIAL SITUATION ASSESSMENT",
            f"• Current head position: {tuple(head_pos)}",
            f"• Target apple position: {tuple(apple_pos)}",
            f"• Snake body positions: {[tuple(p) for p in body_positions]}",
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
            f"• Algorithm: Breadth-First Search from {tuple(head_pos)} to {tuple(apple_pos)}",
            f"• Search space: {grid_size * grid_size - snake_length} accessible cells",
            f"• Obstacles to navigate: {snake_length - 1} body segments",
            f"• Manhattan distance baseline: {manhattan_distance} steps (theoretical minimum)",
            "",
            "PHASE 4: PATH ANALYSIS RESULTS",
            f"• Shortest path found: {path_length} steps",
            f"• Path efficiency: {efficiency_str}",
            f"• Path optimality: {'OPTIMAL - no detours required' if is_optimal else 'SUB-OPTIMAL - includes ' + str(detour_steps) + ' detour step(s)'}",
            f"• Path coordinates: {' → '.join([str(tuple(p)) for p in path])}",
            "",
            "PHASE 5: SAFETY VALIDATION",
            "• Safety check: Validating that snake can reach tail after move",
            f"• Next position: {next_pos}",
            f"• Tail reachability: CONFIRMED (path exists from {next_pos} to tail)",
            "• Safety status: SAFE (move will not trap the snake)",
            "",
            "PHASE 6: MOVE SELECTION LOGIC",
            f"• Chosen direction: {direction}",
            f"• Next position: {next_pos}",
            "• Rationale: First step of shortest path to apple (SAFE)",
            "• Risk assessment: LOW (validated safe move on optimal path)",
            "• Expected outcome: Advance 1 step closer to apple along shortest route",
            "",
            "PHASE 7: STRATEGIC IMPLICATIONS",
            f"• Immediate benefit: Reduces distance to apple from {manhattan_distance} to {manhattan_distance - 1}",
            "• Future positioning: Maintains optimal trajectory toward apple",
            f"• Space management: Preserves {remaining_free_cells - 1} free cells for maneuvering",
            "• Risk mitigation: BFS guarantees shortest path, safety validation prevents trapping",
            "",
            "=== DECISION SUMMARY ===",
            f"Moving {direction} is the optimal SAFE choice because it follows the shortest BFS-computed path to the apple at {tuple(apple_pos)}. " +
            f"This move advances the snake from {tuple(head_pos)} to {next_pos}, maintaining perfect trajectory efficiency " +
            f"{'with no detours required' if is_optimal else f'despite {detour_steps} necessary detour(s) to avoid obstacles'}. " +
            "The decision is both safe (validated tail reachability) and efficient " +
            f"({efficiency_ratio:.2f} path efficiency), making it strategically sound given current board pressure ({space_pressure})."
        ]

        return {
            "strategy_phase": "SAFE_APPLE_PATH",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def _generate_basic_tail_chase_explanation(self, game_state: dict, tail: List[int], path: List[List[int]], 
                                       direction: str, valid_moves: List[str], manhattan_distance: int, 
                                       remaining_free_cells: int, metrics: dict) -> dict:
        """Generate full detailed explanation for tail chase strategy."""
        # SSOT: Extract positions using exact same logic as dataset_generator.py
        snake_positions = game_state.get('snake_positions', [])
        head_pos = extract_head_position(game_state)
        apple_pos = game_state.get('apple_position', [0, 0])
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use exact same body_positions logic as dataset_generator.py
        body_positions = extract_body_positions(game_state)
        
        path_length = len(path) - 1
        snake_length = len(snake_positions)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        explanation_parts = [
            "=== BFS-SAFE-GREEDY ANALYSIS: TAIL CHASE STRATEGY ===",
            "",
            "PHASE 1: PRIMARY STRATEGY FAILURE ANALYSIS",
            f"• Primary strategy attempted: Apple pathfinding to {tuple(apple_pos)}",
            "• Primary strategy result: FAILED (unsafe or no path found)",
            "• Failure reason: Safety validation rejected apple path",
            "• Risk detected: Potential self-trapping if pursuing apple",
            "• Algorithm response: Activate secondary strategy",
            "",
            "PHASE 2: SECONDARY STRATEGY - TAIL CHASING",
            "• Strategy priority: SECONDARY (defensive positioning)",
            f"• Target: Snake tail at {tuple(tail)}",
            "• Rationale: Tail chasing is always safe (tail moves away)",
            f"• Available valid moves: {valid_moves} ({len(valid_moves)} options)",
            f"• BFS pathfinding from {tuple(head_pos)} to {tuple(tail)}",
            f"• Tail chase path found: {path_length} steps",
            f"• Path coordinates: {' → '.join([str(tuple(p)) for p in path[:min(4, len(path))]])}{'...' if len(path) > 4 else ''}",
            "",
            "PHASE 3: TAIL CHASE SAFETY ANALYSIS",
            "• Safety guarantee: ABSOLUTE (tail moves as snake advances)",
            "• Self-collision risk: ZERO (impossible to catch moving tail)",
            "• Space preservation: Maintains current board position",
            "• Future opportunities: Keeps options open for apple pursuit",
            f"• Board pressure: {space_pressure} ({board_fill_ratio:.1%} occupation)",
            "",
            "PHASE 4: STRATEGIC POSITIONING",
            f"• Current head position: {tuple(head_pos)}",
            f"• Chosen direction: {direction}",
            f"• Next position: Following tail at distance {path_length}",
            f"• Apple distance: {manhattan_distance} steps (for future reference)",
            f"• Space management: Preserving {remaining_free_cells} free cells",
            "• Positioning benefit: Maintains mobility while avoiding risks",
            "",
            "PHASE 5: SAFE-GREEDY DEFENSIVE LOGIC",
            "• Algorithm strength: Never pursues risky apple paths",
            "• Fallback reliability: Tail chasing provides guaranteed safe moves",
            "• vs Standard BFS: Would attempt unsafe apple path",
            "• vs Pure Conservative: Would avoid apple even when safe",
            "• Adaptive behavior: Switches strategies based on safety assessment",
            "",
            "=== CONCLUSION ===",
            "BFS-Safe-Greedy activated tail chase strategy after determining apple pursuit was unsafe. " +
            f"Moving {direction} toward tail at {tuple(tail)} provides guaranteed safety while maintaining " +
            "board position. This defensive strategy preserves the snake's survival until safer " +
            "apple pursuit opportunities emerge, demonstrating the algorithm's adaptive safety-first approach."
        ]

        return {
            "strategy_phase": "TAIL_CHASE",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def _generate_basic_survival_explanation(self, game_state: dict, direction: str, valid_moves: List[str], 
                                     manhattan_distance: int, remaining_free_cells: int, metrics: dict) -> dict:
        """Generate full detailed explanation for survival move strategy."""
        # SSOT: Extract positions using exact same logic as dataset_generator.py
        snake_positions = game_state.get('snake_positions', [])
        head_pos = extract_head_position(game_state)
        apple_pos = game_state.get('apple_position', [0, 0])
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use exact same body_positions logic as dataset_generator.py
        body_positions = extract_body_positions(game_state)
        
        snake_length = len(snake_positions)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        
        explanation_parts = [
            "=== BFS-SAFE-GREEDY ANALYSIS: SURVIVAL MODE ===",
            "",
            "PHASE 1: CRITICAL SITUATION ASSESSMENT",
            "• Algorithm: BFS-Safe-Greedy in emergency survival mode",
            f"• Current head position: {tuple(head_pos)}",
            f"• Snake length: {snake_length} segments",
            f"• Board occupation: {board_fill_ratio:.1%} (CRITICAL density)",
            f"• Free cells remaining: {remaining_free_cells}",
            f"• Available moves: {valid_moves} ({len(valid_moves)} emergency options)",
            "",
            "PHASE 2: STRATEGY CASCADE FAILURE",
            "• PRIMARY strategy (safe apple path): FAILED",
            "• SECONDARY strategy (tail chase): FAILED", 
            "• TERTIARY strategy (survival move): ACTIVATED",
            "• Situation severity: CRITICAL (limited options remaining)",
            "• Risk level: MAXIMUM (immediate survival at stake)",
            "",
            "PHASE 3: EMERGENCY MOVE SELECTION",
            f"• Chosen direction: {direction}",
            "• Rationale: First available valid move to avoid immediate death",
            "• Safety assessment: UNKNOWN (no path validation possible)",
            "• Risk acceptance: MAXIMUM (survival over safety)",
            "• Expected outcome: Avoid immediate collision, hope for better positioning",
            "",
            "PHASE 4: CRITICAL SPACE ANALYSIS",
            f"• Board density: {board_fill_ratio:.1%} (CRITICAL threshold exceeded)",
            f"• Snake body positions: {[tuple(p) for p in body_positions]}",
            f"• Apple position: {tuple(apple_pos)} (unreachable)",
            f"• Manhattan distance to apple: {manhattan_distance} (irrelevant in survival mode)",
            "• Space fragmentation: SEVERE (limited maneuvering room)",
            "",
            "PHASE 5: SURVIVAL STRATEGY IMPLICATIONS",
            "• Immediate priority: Avoid death in next move",
            "• Secondary priority: Create space for future moves",
            "• Long-term strategy: Wait for tail movement to create opportunities",
            "• Recovery potential: LOW (depends on tail movement pattern)",
            "• Algorithm adaptation: Switched from greedy to pure survival",
            "",
            "=== CONCLUSION ===",
            "BFS-Safe-Greedy has entered emergency survival mode due to complete strategy failure. " +
            f"Moving {direction} is a last-resort action to avoid immediate death, with no guarantee of " +
            "long-term survival. The algorithm has exhausted all safe strategies and now operates in " +
            f"pure survival mode with {len(valid_moves)} emergency options remaining."
        ]

        return {
            "strategy_phase": "SURVIVAL_MODE",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def _generate_basic_no_moves_explanation(self, game_state: dict, valid_moves: List[str], 
                                     manhattan_distance: int, remaining_free_cells: int, metrics: dict) -> dict:
        """Generate full detailed explanation for no valid moves scenario."""
        # SSOT: Extract positions using exact same logic as dataset_generator.py
        snake_positions = game_state.get('snake_positions', [])
        head_pos = extract_head_position(game_state)
        apple_pos = game_state.get('apple_position', [0, 0])
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use exact same body_positions logic as dataset_generator.py
        body_positions = extract_body_positions(game_state)
        
        snake_length = len(snake_positions)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        
        explanation_parts = [
            "=== BFS-SAFE-GREEDY ANALYSIS: GAME OVER ===",
            "",
            "PHASE 1: CRITICAL SITUATION ASSESSMENT",
            "• Algorithm: BFS-Safe-Greedy facing game termination",
            f"• Current head position: {tuple(head_pos)}",
            f"• Snake length: {snake_length} segments",
            f"• Board occupation: {board_fill_ratio:.1%} (CRITICAL density)",
            f"• Free cells remaining: {remaining_free_cells}",
            f"• Available moves: {valid_moves} ({len(valid_moves)} options - NONE VALID)",
            "",
            "PHASE 2: STRATEGY CASCADE COMPLETE FAILURE",
            "• PRIMARY strategy (safe apple path): FAILED",
            "• SECONDARY strategy (tail chase): FAILED", 
            "• TERTIARY strategy (survival move): FAILED",
            "• QUATERNARY strategy (any valid move): FAILED",
            "• Situation severity: TERMINAL (no options remaining)",
            "• Risk level: MAXIMUM (game over imminent)",
            "",
            "PHASE 3: GAME OVER ANALYSIS",
            f"• Snake body positions: {[tuple(p) for p in body_positions]}",
            f"• Apple position: {tuple(apple_pos)} (unreachable)",
            f"• Manhattan distance to apple: {manhattan_distance} (irrelevant)",
            "• Space fragmentation: COMPLETE (no free cells accessible)",
            "• Movement impossibility: ALL DIRECTIONS BLOCKED",
            "",
            "PHASE 4: TERMINATION CAUSE ANALYSIS",
            "• Primary cause: Snake has completely surrounded itself",
            "• Secondary cause: No valid moves available from current position",
            "• Tertiary cause: Board density reached critical threshold",
            "• Quaternary cause: No escape path exists",
            "• Resolution: IMPOSSIBLE (game over)",
            "",
            "PHASE 5: ALGORITHM PERFORMANCE ASSESSMENT",
            "• BFS-Safe-Greedy performance: MAXIMUM (survived until no options)",
            "• Strategy effectiveness: OPTIMAL (exhausted all possibilities)",
            "• Safety validation: SUCCESSFUL (prevented premature termination)",
            "• vs Standard BFS: Would have terminated earlier",
            "• vs Pure Conservative: Would have terminated earlier",
            "• Algorithm achievement: Reached natural game end",
            "",
            "=== CONCLUSION ===",
            "BFS-Safe-Greedy has reached the natural end of the game with no valid moves remaining. " +
            f"The snake at position {tuple(head_pos)} is completely surrounded and cannot move in any direction. " +
            "This represents the algorithm's maximum possible performance - it survived until the game became " +
            "mathematically impossible to continue, demonstrating optimal safety-first strategy execution."
        ]

        return {
            "strategy_phase": "GAME_OVER",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
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
        if hasattr(self, 'include_board_representation') and self.include_board_representation:
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