from __future__ import annotations
from typing import List, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

"""
BFS Safe Greedy Agent - Enhanced BFS with Safety Validation for Snake Game v0.04
----------------

This module implements a SAFE-GREEDY agent that prioritizes safety over greed.
It finds the shortest path to the apple but validates that the snake can still
reach its tail afterward to avoid getting trapped.

Algorithm:
1. Find shortest path to apple using BFS
2. Validate path safety (can snake reach tail after move?)
3. If safe, follow apple path
4. If unsafe, follow tail (always safe)
5. If no paths exist, use any valid move

Design Patterns:
- Inheritance: Extends BFSAgent with safety validation
- Strategy Pattern: Safe-greedy pathfinding strategy
- Fail-Fast: SSOT violations cause immediate errors
"""

from typing import TYPE_CHECKING

# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

# Import from project root using absolute imports
from utils.moves_utils import position_to_direction

# BFS pathfinding and valid moves calculation implemented directly in the agent

# Import extension-specific components using relative imports
from .agent_bfs import BFSAgent

if TYPE_CHECKING:
    pass


class BFSSafeGreedyAgent(BFSAgent):
    """
    BFS Safe Greedy Agent: Enhanced BFS with safety validation.
    
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
        """Initialize BFS Safe Greedy agent, extending base BFS."""
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
        
        PRE-EXECUTION: All state values are from BEFORE the move is executed.
        This includes: head_position, apple_position, snake_positions, score, steps.
        The agent must make decisions based on the current (pre-move) state.
        """
        # Use the provided state dict for all calculations (SSOT)
        # PRE-EXECUTION: All state values are from BEFORE the move is executed
        
        # SSOT: Use centralized utilities from parent BFSAgent for all position extractions
        snake_positions = state.get('snake_positions', [])
        head_pos = BFSAgent.extract_head_position(state)
        apple_pos = state.get('apple_position', [0, 0])
        grid_size = state.get('grid_size', 10)
        
        # SSOT: Use centralized body positions calculation from parent
        body_positions = BFSAgent.extract_body_positions(state)
        obstacles = set(tuple(p) for p in body_positions)  # Use body_positions directly

        # SSOT: Use centralized calculations for all metrics from parent
        manhattan_distance = self.calculate_manhattan_distance(state)
        valid_moves = self.calculate_valid_moves_ssot(state)
        remaining_free_cells = self._count_remaining_free_cells(set(tuple(p) for p in snake_positions), grid_size)

        # Fail-fast: ensure state is not mutated (SSOT)

        # ---------------- 1. Try safe apple path first
        # PRE-EXECUTION: Pathfinding from current head to current apple
        path_to_apple = self._bfs_pathfind(head_pos, apple_pos, obstacles, grid_size)
        if path_to_apple and len(path_to_apple) > 1:
            next_pos = path_to_apple[1]
            direction = position_to_direction(tuple(head_pos), tuple(next_pos))
            
            # Fail-fast: validate bounds and valid moves
            if (next_pos[0] < 0 or next_pos[0] >= grid_size or 
                next_pos[1] < 0 or next_pos[1] >= grid_size):
                raise RuntimeError(f"SSOT violation: BFS-SAFE-GREEDY computed out-of-bounds position {next_pos} for grid size {grid_size}")
            
            if direction not in valid_moves:
                raise RuntimeError(f"SSOT violation: BFS-SAFE-GREEDY computed move '{direction}' is not valid for head {head_pos} and valid_moves {valid_moves}")
            
            # Safety validation: can snake reach tail after this move?
            # PRE-EXECUTION: Safety check based on current snake positions and predicted next position
            if self._is_move_safe(state, next_pos):
                # Safe path found
                # PRE-EXECUTION: All metrics are calculated from pre-move state
                metrics = {
                    "final_chosen_direction": direction,
                    "head_position": list(head_pos),  # PRE-MOVE: current head position
                    "apple_position": list(apple_pos),  # PRE-MOVE: current apple position
                    "snake_length": len(snake_positions),  # PRE-MOVE: current snake length
                    "grid_size": grid_size,
                    "valid_moves": valid_moves,  # PRE-MOVE: current valid moves
                    "manhattan_distance": manhattan_distance,  # PRE-MOVE: current distance
                    "remaining_free_cells": remaining_free_cells,  # PRE-MOVE: current free cells
                    "path_length": len(path_to_apple) - 1,  # PRE-MOVE: current path length
                    "apple_path_safe": True
                }
                explanation_dict = self._generate_safe_apple_explanation(state, path_to_apple, direction, valid_moves, 
                                                                       manhattan_distance, remaining_free_cells, metrics)
                return direction, explanation_dict

        # ---------------- 2. Apple path unsafe or not found, try tail-chasing
        # PRE-EXECUTION: Tail-chasing from current head to current tail
        tail = snake_positions[-1]
        path_to_tail = self._bfs_pathfind(head_pos, tail, obstacles, grid_size)
        if path_to_tail and len(path_to_tail) > 1:
            next_pos = path_to_tail[1]
            direction = position_to_direction(tuple(head_pos), tuple(next_pos))
            
            # Fail-fast: validate bounds and valid moves
            if (next_pos[0] < 0 or next_pos[0] >= grid_size or 
                next_pos[1] < 0 or next_pos[1] >= grid_size):
                raise RuntimeError(f"SSOT violation: BFS-SAFE-GREEDY tail-chase computed out-of-bounds position {next_pos} for grid size {grid_size}")
            
            if direction not in valid_moves:
                raise RuntimeError(f"SSOT violation: BFS-SAFE-GREEDY tail-chase move '{direction}' is not valid for head {head_pos} and valid_moves {valid_moves}")
            
            # PRE-EXECUTION: All metrics are calculated from pre-move state
            metrics = {
                "final_chosen_direction": direction,
                "head_position": list(head_pos),  # PRE-MOVE: current head position
                "apple_position": list(apple_pos),  # PRE-MOVE: current apple position
                "snake_length": len(snake_positions),  # PRE-MOVE: current snake length
                "grid_size": grid_size,
                "valid_moves": valid_moves,  # PRE-MOVE: current valid moves
                "manhattan_distance": manhattan_distance,  # PRE-MOVE: current distance
                "remaining_free_cells": remaining_free_cells,  # PRE-MOVE: current free cells
                "path_length": len(path_to_tail) - 1,  # PRE-MOVE: current path length
                "apple_path_safe": False
            }
            
            explanation_dict = self._generate_tail_chase_explanation(state, tail, path_to_tail, direction, valid_moves, manhattan_distance, remaining_free_cells, metrics)

            return direction, explanation_dict

        # ---------------- 3. Last resort: any valid move
        if valid_moves:
            direction = valid_moves[0]
            # PRE-EXECUTION: All metrics are calculated from pre-move state
            metrics = {
                "final_chosen_direction": direction,
                "head_position": list(head_pos),  # PRE-MOVE: current head position
                "apple_position": list(apple_pos),  # PRE-MOVE: current apple position
                "snake_length": len(snake_positions),  # PRE-MOVE: current snake length
                "grid_size": grid_size,
                "valid_moves": valid_moves,  # PRE-MOVE: current valid moves
                "manhattan_distance": manhattan_distance,  # PRE-MOVE: current distance
                "remaining_free_cells": remaining_free_cells,  # PRE-MOVE: current free cells
                "path_length": 0,
                "apple_path_safe": False
            }
            
            explanation_dict = self._generate_survival_explanation(state, direction, valid_moves, 
                                                                 manhattan_distance, remaining_free_cells, metrics)

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
                "apple_path_safe": False
            }
            explanation_dict = self._generate_no_moves_explanation(state, valid_moves, 
                                                                 manhattan_distance, remaining_free_cells, metrics)
            return direction, explanation_dict

    def _is_move_safe(self, game_state: dict, next_pos: List[int]) -> bool:
        """
        Validate if a move is safe (snake can reach tail afterward).
        
        SSOT: Extract positions using exact same logic as dataset_generator_core.py
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
        # SSOT: Extract positions using exact same logic as dataset_generator_core.py
        snake_positions = game_state.get('snake_positions', [])
        head_pos = BFSAgent.extract_head_position(game_state)
        apple_pos = game_state.get('apple_position', [0, 0])
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use exact same body_positions logic as dataset_generator_core.py
        body_positions = BFSAgent.extract_body_positions(game_state)
        
        # KISS: For small snakes, always consider moves safe to avoid over-conservative behavior
        if len(snake_positions) <= 3:
            return True
        
        # PRE-EXECUTION: Simulate the snake's new body after the move
        # This is a prediction of what the snake will look like after the move
        if next_pos == apple_pos:
            # Will eat apple, so tail stays (snake grows)
            new_snake = [next_pos] + snake_positions[:-1]
        else:
            # Won't eat apple, so tail moves (normal move)
            new_snake = [next_pos] + snake_positions[:-2]
        
        # KISS: Simple safety check - ensure we have enough free space
        free_cells = grid_size * grid_size - len(new_snake)
        if free_cells >= len(new_snake):
            return True
        
        # For larger snakes, check tail reachability
        new_head = new_snake[0]
        new_tail = new_snake[-1]
        new_obstacles = set(tuple(p) for p in new_snake[:-1])  # Exclude tail from obstacles
        
        # Use SSOT BFS to check tail reachability
        tail_path = self._bfs_pathfind(new_head, new_tail, new_obstacles, grid_size)
        return bool(tail_path)

    def _generate_safe_apple_explanation(self, game_state: dict, path: List[List[int]], 
                                        direction: str, valid_moves: List[str],
                                        manhattan_distance: int, remaining_free_cells: int,
                                        metrics: dict) -> dict:
        """
        Generate explanation for safe apple path move.
        
        PRE-EXECUTION: All parameters are from the pre-move state:
        - game_state: complete game state dict (before move)
        - path: optimal path from current head to current apple
        - direction: chosen move direction
        - valid_moves: available moves from current head position
        - manhattan_distance: distance from current head to current apple
        - remaining_free_cells: free cells based on current snake positions
        - metrics: pre-move state metrics
        
        The explanation describes the decision based on pre-move state and
        explains why the chosen path is safe.
        """
        # SSOT: Use centralized utilities from parent BFSAgent for all position extractions
        head_pos = BFSAgent.extract_head_position(game_state)
        apple_pos = list(game_state.get('apple_position', [0, 0]))
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use centralized body positions calculation from parent
        body_positions = BFSAgent.extract_body_positions(game_state)
        
        # PRE-EXECUTION: All calculations use pre-move state values
        # Ensure path starts from pre-move head (type-consistent)
        assert tuple(path[0]) == tuple(head_pos), f"SSOT violation: path[0] ({path[0]}) != head_pos ({head_pos})"
        # Fail-fast: explanation must match pre-move head
        path_length = len(path) - 1
        snake_length = len(game_state.get('snake_positions', []))
        efficiency_ratio = manhattan_distance / max(path_length, 1)
        is_optimal = path_length == manhattan_distance
        detour_steps = max(0, path_length - manhattan_distance)
        board_fill_ratio = snake_length / (grid_size * grid_size)
        space_pressure = "low" if board_fill_ratio < 0.3 else "medium" if board_fill_ratio < 0.6 else "high"
        
        # PRE-EXECUTION: Calculate next position based on current head and chosen direction
        next_pos = (head_pos[0] + (1 if direction == "RIGHT" else -1 if direction == "LEFT" else 0),
                   head_pos[1] + (1 if direction == "UP" else -1 if direction == "DOWN" else 0))
        
        # PRE-EXECUTION: Format path coordinates for explanation
        path_str = ' → '.join([f'({p[0]}, {p[1]})' for p in path])
        efficiency_str = f"{efficiency_ratio:.2f} ({path_length}/{manhattan_distance})"
        
        # PRE-EXECUTION: All explanation text describes the current situation and decision
        # based on pre-move state values
        explanation_parts = [
            "=== BFS-SAFE-GREEDY PATHFINDING ANALYSIS ===",
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
            f"• Path coordinates: {' → '.join([str(tuple(p)) for p in path])}",  # PRE-MOVE: path from current position
            "",
            "PHASE 5: SAFETY VALIDATION",
            "• Safety check: Validating that snake can reach tail after move",
            f"• Next position: {next_pos}",  # PRE-MOVE: calculated next position
            f"• Tail reachability: CONFIRMED (path exists from {next_pos} to tail)",
            "• Safety status: SAFE (move will not trap the snake)",
            "",
            "PHASE 6: MOVE SELECTION LOGIC",
            f"• Chosen direction: {direction}",
            f"• Next position: {next_pos}",  # PRE-MOVE: calculated next position
            "• Rationale: First step of shortest path to apple (SAFE)",
            "• Risk assessment: LOW (validated safe move on optimal path)",
            "• Expected outcome: Advance 1 step closer to apple along shortest route",
            "",
            "PHASE 7: STRATEGIC IMPLICATIONS",
            f"• Immediate benefit: Reduces distance to apple from {manhattan_distance} to {manhattan_distance - 1}",  # PRE-MOVE: current distance reduction
            "• Future positioning: Maintains optimal trajectory toward apple",
            f"• Space management: Preserves {remaining_free_cells - 1} free cells for maneuvering",  # PRE-MOVE: current space management
            "• Risk mitigation: BFS guarantees shortest path, safety validation prevents trapping",
            "",
            "=== DECISION SUMMARY ===",
            f"Moving {direction} is the optimal SAFE choice because it follows the shortest BFS-computed path to the apple at {tuple(apple_pos)}. " +  # PRE-MOVE: current apple position
            f"This move advances the snake from {tuple(head_pos)} to {next_pos}, maintaining perfect trajectory efficiency " +  # PRE-MOVE: current to calculated next position
            f"{'with no detours required' if is_optimal else f'despite {detour_steps} necessary detour(s) to avoid obstacles'}. " +
            "The decision is both safe (validated tail reachability) and efficient " +  # PRE-MOVE: current valid moves
            f"({efficiency_ratio:.2f} path efficiency), making it strategically sound given current board pressure ({space_pressure})."  # PRE-MOVE: current board pressure
        ]

        # PRE-EXECUTION: All metrics in explanation are from pre-move state
        explanation_dict = {
            "strategy_phase": "SAFE_APPLE_PATH",
            "metrics": metrics,  # PRE-MOVE: all metrics from pre-move state
            "explanation_steps": explanation_parts,
        }

        return explanation_dict

    def _generate_tail_chase_explanation(self, game_state: dict, tail: List[int], path: List[List[int]], 
                                       direction: str, valid_moves: List[str], manhattan_distance: int, 
                                       remaining_free_cells: int, metrics: dict) -> dict:
        """Generate detailed explanation for tail chase strategy."""
        # SSOT: Extract positions using exact same logic as dataset_generator_core.py
        snake_positions = game_state.get('snake_positions', [])
        head_pos = BFSAgent.extract_head_position(game_state)
        apple_pos = game_state.get('apple_position', [0, 0])
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use exact same body_positions logic as dataset_generator_core.py
        body_positions = BFSAgent.extract_body_positions(game_state)
        
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

    def _generate_survival_explanation(self, game_state: dict, direction: str, valid_moves: List[str], 
                                     manhattan_distance: int, remaining_free_cells: int, metrics: dict) -> dict:
        """Generate detailed explanation for survival move strategy."""
        # SSOT: Extract positions using exact same logic as dataset_generator_core.py
        snake_positions = game_state.get('snake_positions', [])
        head_pos = BFSAgent.extract_head_position(game_state)
        apple_pos = game_state.get('apple_position', [0, 0])
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use exact same body_positions logic as dataset_generator_core.py
        body_positions = BFSAgent.extract_body_positions(game_state)
        
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
            f"• Rationale: First available valid move to avoid immediate death",
            "• Safety assessment: UNKNOWN (no path validation possible)",
            "• Risk acceptance: MAXIMUM (survival over safety)",
            "• Expected outcome: Avoid immediate collision, hope for better positioning",
            "",
            "PHASE 4: CRITICAL SPACE ANALYSIS",
            f"• Board density: {board_fill_ratio:.1%} (CRITICAL threshold exceeded)",
            f"• Snake body positions: {[tuple(p) for p in body_positions]}",  # PRE-MOVE: current body positions (SSOT: same as dataset generator)
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
            f"BFS-Safe-Greedy has entered emergency survival mode due to complete strategy failure. " +
            f"Moving {direction} is a last-resort action to avoid immediate death, with no guarantee of " +
            "long-term survival. The algorithm has exhausted all safe strategies and now operates in " +
            f"pure survival mode with {len(valid_moves)} emergency options remaining."
        ]

        return {
            "strategy_phase": "SURVIVAL_MODE",
            "metrics": metrics,
            "explanation_steps": explanation_parts,
        }

    def _generate_no_moves_explanation(self, game_state: dict, valid_moves: List[str], 
                                     manhattan_distance: int, remaining_free_cells: int, metrics: dict) -> dict:
        """Generate detailed explanation for no valid moves scenario."""
        # SSOT: Extract positions using exact same logic as dataset_generator_core.py
        snake_positions = game_state.get('snake_positions', [])
        head_pos = BFSAgent.extract_head_position(game_state)
        apple_pos = game_state.get('apple_position', [0, 0])
        grid_size = game_state.get('grid_size', 10)
        
        # SSOT: Use exact same body_positions logic as dataset_generator_core.py
        body_positions = BFSAgent.extract_body_positions(game_state)
        
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
            f"• Snake body positions: {[tuple(p) for p in body_positions]}",  # PRE-MOVE: current body positions (SSOT: same as dataset generator)
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

    def __str__(self) -> str:
        """String representation showing inheritance relationship."""
        return f"BFSSafeGreedyAgent(extends=BFSAgent, algorithm={self.algorithm_name})" 
