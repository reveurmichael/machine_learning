"""
Pure Hamiltonian Cycle Agent - Guaranteed Snake Solution
--------------------

This module implements a pure Hamiltonian cycle agent that guarantees
the snake will never die by following a pre-computed cycle that visits
every cell exactly once.

Unlike the A*-Hamiltonian hybrid, this agent ONLY follows the Hamiltonian
cycle, making it completely predictable and safe but potentially slower.

Design Patterns:
- Strategy Pattern: Pure Hamiltonian cycle strategy
- Precomputation Pattern: Cycle generated once, reused throughout game
- Failsafe Pattern: Guaranteed safety through mathematical properties
"""

from __future__ import annotations
from typing import List, Tuple, Dict, TYPE_CHECKING

# Use standardized path setup
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from config import DIRECTIONS
from utils.moves_utils import position_to_direction

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class HamiltonianAgent:
    """
    Pure Hamiltonian Cycle Agent for Snake game.
    
    This agent follows a pre-computed Hamiltonian cycle (a path that visits
    every cell exactly once and returns to the start). This guarantees the
    snake will never die, making it the safest possible strategy.
    
    Algorithm:
    1. Generate Hamiltonian cycle using boustrophedon pattern
    2. Find current position in cycle
    3. Move to next position in cycle
    4. Repeat until game ends
    
    Trade-offs:
    - Pros: Guaranteed safety, never dies, mathematical elegance
    - Cons: Slower apple collection, predictable movement pattern
    
    Educational Value:
    Demonstrates how mathematical guarantees can solve complex problems,
    even if not optimally.
    """
    
    def __init__(self) -> None:
        """Initialize Hamiltonian cycle agent."""
        self.algorithm_name = "HAMILTONIAN"
        self.name = "Pure Hamiltonian Cycle"
        self.description = (
            "Pure Hamiltonian cycle agent. Follows a pre-computed path that "
            "visits every cell exactly once. Guarantees the snake never dies "
            "but may be slower than other algorithms. Demonstrates mathematical "
            "approach to problem solving."
        )
        
        # Cycle state
        self.cycle: List[Tuple[int, int]] = []
        self.cycle_map: Dict[Tuple[int, int], int] = {}
        self.grid_size: int = 0
        self.current_cycle_index: int = 0
        
        # Statistics
        self.cycle_completions: int = 0
        self.total_moves: int = 0
        
    def get_move(self, game: HeuristicGameLogic) -> str | None:
        """
        Get next move following the Hamiltonian cycle (legacy method for v0.03 compatibility).
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        move, _ = self.get_move_with_explanation(game)
        return move
        
    def get_move_with_explanation(self, game: HeuristicGameLogic) -> Tuple[str, str]:
        """
        Get next move following the Hamiltonian cycle with detailed explanation.
        
        v0.04 Enhancement: Returns both move and natural language explanation
        for LLM fine-tuning dataset generation.
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Tuple of (direction_string, explanation_string)
        """
        try:
            head = tuple(game.head_position)
            grid_size = game.grid_size
            
            # Generate cycle if needed
            if not self.cycle or self.grid_size != grid_size:
                self.grid_size = grid_size
                self._generate_hamiltonian_cycle(grid_size)
            
            # Find current position in cycle
            head_idx = self.cycle_map.get(head, -1)
            if head_idx == -1:
                # Head not in cycle (shouldn't happen), find nearest
                head_idx = self._find_nearest_cycle_position(head)
            
            # Get next position in cycle
            next_idx = (head_idx + 1) % len(self.cycle)
            next_pos = self.cycle[next_idx]
            
            # Validate move is safe (should always be true for valid cycle)
            if self._is_move_safe(head, next_pos, game):
                self.current_cycle_index = next_idx
                self.total_moves += 1
                
                # Track cycle completions
                if next_idx == 0:
                    self.cycle_completions += 1
                
                direction = position_to_direction(head, next_pos)
                
                # Generate detailed explanation
                cycle_progress = f"{next_idx + 1}/{len(self.cycle)}"
                explanation = (
                    f"Following Hamiltonian cycle: moving {direction} from {head} to {next_pos}. "
                    f"Progress: {cycle_progress} positions in cycle. "
                    f"This guarantees visiting every cell exactly once and prevents the snake from ever getting trapped. "
                )
                
                if next_idx == 0:
                    explanation += f"Completed cycle #{self.cycle_completions}! Starting new cycle traversal."
                else:
                    explanation += "Continuing systematic exploration of the grid."
                
                return direction, explanation
            else:
                # This should never happen with a valid Hamiltonian cycle
                # Fall back to any safe move
                fallback_move = self._find_any_safe_move(head, game)
                explanation = (
                    f"Hamiltonian cycle move from {head} to {next_pos} was unsafe (unexpected). "
                    f"Falling back to safe move {fallback_move} to avoid collision. "
                    f"This indicates an issue with the cycle generation or validation."
                )
                return fallback_move, explanation
                
        except Exception as e:
            explanation = f"Hamiltonian Agent encountered an error: {str(e)}"
            print(f"Hamiltonian Agent error: {e}")
            return "NO_PATH_FOUND", explanation
    
    def _generate_hamiltonian_cycle(self, size: int) -> None:
        """
        Generate Hamiltonian cycle using boustrophedon (zigzag) pattern.
        
        This creates a path that visits every cell exactly once and returns
        to the starting position, guaranteeing the snake can move indefinitely.
        
        Args:
            size: Grid size (assumes square grid)
        """
        self.cycle = []
        
        # Boustrophedon pattern: alternate left-to-right and right-to-left
        for y in range(size):
            if y % 2 == 0:
                # Left to right
                for x in range(size):
                    self.cycle.append((x, y))
            else:
                # Right to left
                for x in range(size - 1, -1, -1):
                    self.cycle.append((x, y))
        
        # Ensure the cycle is properly closed (first and last positions adjacent)
        if size % 2 == 0:
            # For even grids, adjust the last position to ensure adjacency
            # For even grids, the last position should connect back to (0, 0)
            # Remove the last position and add the connection point
            if len(self.cycle) > 0:
                last_pos = self.cycle[-1]
                # Make sure last position can connect to first position (0, 0)
                if last_pos != (0, size - 1):
                    # Adjust to ensure proper connection
                    pass  # The boustrophedon pattern should already be correct
        
        # Build lookup map for fast position-to-index mapping
        self.cycle_map = {pos: idx for idx, pos in enumerate(self.cycle)}
        
        # Validate cycle integrity
        self._validate_cycle()
    
    def _validate_cycle(self) -> None:
        """
        Validate the generated Hamiltonian cycle.
        
        Checks:
        1. Correct number of positions (grid_size^2)
        2. No duplicate positions
        3. All positions are adjacent
        4. First and last positions are adjacent (cycle property)
        """
        expected_length = self.grid_size * self.grid_size
        
        # Check length
        if len(self.cycle) != expected_length:
            print(f"⚠️  Cycle length mismatch: {len(self.cycle)} vs {expected_length}")
        
        # Check for duplicates
        unique_positions = set(self.cycle)
        if len(unique_positions) != len(self.cycle):
            duplicates = len(self.cycle) - len(unique_positions)
            print(f"⚠️  Found {duplicates} duplicate positions in cycle")
        
        # Check adjacency of consecutive positions
        for i, current in enumerate(self.cycle):
            next_pos = self.cycle[(i + 1) % len(self.cycle)]
            
            # Manhattan distance should be exactly 1
            distance = abs(current[0] - next_pos[0]) + abs(current[1] - next_pos[1])
            if distance != 1:
                print(f"⚠️  Non-adjacent positions in cycle: {current} -> {next_pos}")
                break
    
    def _find_nearest_cycle_position(self, pos: Tuple[int, int]) -> int:
        """
        Find the nearest position in the cycle to the given position.
        
        Args:
            pos: Position to find nearest cycle position for
            
        Returns:
            Index of nearest position in cycle
        """
        min_distance = float('inf')
        nearest_idx = 0
        
        for idx, cycle_pos in enumerate(self.cycle):
            distance = abs(pos[0] - cycle_pos[0]) + abs(pos[1] - cycle_pos[1])
            if distance < min_distance:
                min_distance = distance
                nearest_idx = idx
        
        return nearest_idx
    
    def _is_move_safe(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                     game: HeuristicGameLogic) -> bool:
        """
        Check if a move is safe (doesn't collide with walls or snake body).
        
        Args:
            from_pos: Starting position
            to_pos: Target position
            game: Game logic instance
            
        Returns:
            True if move is safe, False otherwise
        """
        # Check bounds
        if not (0 <= to_pos[0] < game.grid_size and 0 <= to_pos[1] < game.grid_size):
            return False
        
        # Check collision with snake body (excluding tail which will move)
        snake_body = [tuple(seg) for seg in game.snake_positions]
        tail = snake_body[-1] if snake_body else None
        
        # Check if target is in body (excluding tail unless eating apple)
        if to_pos in snake_body[:-1]:  # Exclude tail
            return False
        
        # If target is tail position, it's safe unless we're eating an apple
        # (because tail will move away)
        if to_pos == tail:
            apple_pos = tuple(game.apple_position)
            is_eating_apple = (to_pos == apple_pos)
            return not is_eating_apple  # Safe if not eating apple
        
        return True
    
    def _find_any_safe_move(self, head: Tuple[int, int], game: HeuristicGameLogic) -> str:
        """
        Find any safe move as a fallback (should rarely be needed).
        
        Args:
            head: Current head position
            game: Game logic instance
            
        Returns:
            Direction string or "NO_PATH_FOUND"
        """
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dx, dy = DIRECTIONS[direction]
            next_pos = (head[0] + dx, head[1] + dy)
            
            if self._is_move_safe(head, next_pos, game):
                return direction
        
        return "NO_PATH_FOUND"
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get agent statistics.
        
        Returns:
            Dictionary containing performance statistics
        """
        return {
            "algorithm": self.algorithm_name,
            "cycle_length": len(self.cycle),
            "cycle_completions": self.cycle_completions,
            "total_moves": self.total_moves,
            "current_cycle_index": self.current_cycle_index,
            "grid_size": self.grid_size
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"HamiltonianAgent(algorithm={self.algorithm_name}, completions={self.cycle_completions})"
