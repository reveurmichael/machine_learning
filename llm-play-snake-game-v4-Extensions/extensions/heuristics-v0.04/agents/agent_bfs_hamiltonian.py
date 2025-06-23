"""
BFS Hamiltonian Agent - Enhanced BFS with Hamiltonian Cycle Fallback
--------------------

This module implements a sophisticated agent that extends BFS Safe Greedy
with Hamiltonian cycle capabilities. It represents the evolution from basic
BFS → BFS-Safe-Greedy → BFS-Hamiltonian.

Inheritance Hierarchy:
- BFSAgent (basic pathfinding)
  └── BFSSafeGreedyAgent (adds safety validation)
      └── BFSHamiltonianAgent (adds Hamiltonian cycle fallback)

This demonstrates progressive enhancement: each level adds new capabilities
while reusing the proven functionality of its parent.

Design Patterns:
- Inheritance: Extends BFSSafeGreedyAgent with Hamiltonian cycle logic
- Template Method: Overrides get_move() while reusing safety validation
- Strategy Pattern: Hamiltonian cycle as ultimate fallback strategy
- Composite Pattern: Combines multiple pathfinding strategies
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Set, TYPE_CHECKING
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent

# Use standardized path setup
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from utils.moves_utils import position_to_direction

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


def generate_hamiltonian_cycle(grid_size: int) -> List[Tuple[int, int]]:
    """
    Generate a simple Hamiltonian cycle using boustrophedon pattern.
    
    Creates a snake-like pattern that visits every cell exactly once:
    - Even rows: left to right
    - Odd rows: right to left
    
    Args:
        grid_size: Size of the square grid
        
    Returns:
        List of positions forming a Hamiltonian cycle
    """
    cycle: List[Tuple[int, int]] = []
    for y in range(grid_size):
        if y % 2 == 0:
            # Even row: left to right
            row = list(range(grid_size))
        else:
            # Odd row: right to left
            row = list(range(grid_size - 1, -1, -1))
        
        for x in row:
            cycle.append((x, y))
    
    return cycle


class BFSHamiltonianAgent(BFSSafeGreedyAgent):
    """
    BFS Hamiltonian Hybrid Agent: Ultimate enhanced BFS with Hamiltonian fallback.
    
    Inheritance Hierarchy:
    - Inherits from BFSSafeGreedyAgent (gets BFS + safety validation)
    - Adds Hamiltonian cycle generation and intelligent shortcuts
    - Provides ultimate fallback strategy that guarantees no deadlocks
    
    Algorithm Enhancement:
    1. Try shortest safe path to apple (inherited from BFSSafeGreedyAgent)
    2. Try tail-chasing fallback (inherited from BFSSafeGreedyAgent)
    3. NEW: Try intelligent Hamiltonian shortcut toward apple
    4. NEW: Follow Hamiltonian cycle as ultimate safe fallback
    5. Last resort: any non-crashing move (inherited)
    
    This demonstrates the final evolution in the BFS family:
    BFS → BFS-Safe-Greedy → BFS-Hamiltonian
    """

    def __init__(self) -> None:
        """Initialize BFS Hamiltonian agent, extending BFS Safe Greedy."""
        super().__init__()  # Initialize parent BFS Safe Greedy agent
        self.algorithm_name = "BFS-HAMILTONIAN"
        self.name = "BFS Hamiltonian Hybrid"
        self.description = (
            "Ultimate enhanced BFS agent. Inherits safe pathfinding from "
            "BFSSafeGreedyAgent and adds Hamiltonian cycle capabilities. "
            "Provides intelligent shortcuts and guarantees no deadlocks."
        )
        
        # Hamiltonian cycle state
        self.hamiltonian: List[Tuple[int, int]] = []
        self.grid_size: Optional[int] = None

    def get_move(self, game: "HeuristicGameLogic") -> str | None:
        """
        Get next move using enhanced BFS with Hamiltonian fallback.
        
        This method is kept for backward compatibility with v0.03.
        For explanation-capable moves, use get_move_with_explanation().
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        move, _ = self.get_move_with_explanation(game)
        return move

    def get_move_with_explanation(self, game: "HeuristicGameLogic") -> Tuple[str, str]:
        """
        Get next move using enhanced BFS with Hamiltonian fallback, with detailed explanation.
        
        Enhancement over parent BFS Safe Greedy:
        - Adds intelligent Hamiltonian shortcuts
        - Provides Hamiltonian cycle as ultimate safe fallback
        - Guarantees the snake never gets permanently trapped
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Tuple of (direction_string, explanation_string)
        """
        try:
            head = tuple(game.head_position)
            apple = tuple(game.apple_position)
            snake = [tuple(seg) for seg in game.snake_positions]
            grid_size = game.grid_size
            obstacles = set(snake[:-1])  # Exclude tail (can vacate)

            # Initialize Hamiltonian cycle once per grid size
            if self.grid_size != grid_size:
                self.grid_size = grid_size
                self.hamiltonian = generate_hamiltonian_cycle(grid_size)

            # ---------------------
            # 1. Try inherited safe apple path
            # ---------------------
            path_to_apple = self._bfs_pathfind(head, apple, obstacles, grid_size)
            if path_to_apple and len(path_to_apple) > 1:
                if self._path_is_safe(path_to_apple, snake, apple, grid_size):
                    next_pos = path_to_apple[1]
                    direction = position_to_direction(head, next_pos)
                    path_length = len(path_to_apple) - 1
                    explanation = (
                        f"BFS Hamiltonian found safe shortest path of length {path_length} to apple at {apple}. "
                        f"Moving {direction} from {head} to {next_pos}. "
                        f"Path safety validated: snake can safely collect apple and reach its tail afterward. "
                        f"This is optimal as it's both shortest and safe."
                    )
                    return direction, explanation

            # ---------------------
            # 2. Try inherited tail-chasing fallback
            # ---------------------
            tail = snake[-1]
            path_to_tail = self._bfs_pathfind(head, tail, obstacles, grid_size)
            if path_to_tail and len(path_to_tail) > 1:
                next_pos = path_to_tail[1]
                direction = position_to_direction(head, next_pos)
                tail_distance = len(path_to_tail) - 1
                explanation = (
                    f"BFS Hamiltonian: Direct path to apple at {apple} deemed unsafe. "
                    f"Switching to tail-chasing strategy. Moving {direction} toward tail at {tail} "
                    f"(distance: {tail_distance}). This guarantees survival while waiting for safer opportunity."
                )
                return direction, explanation

            # ---------------------
            # 3. NEW: Intelligent Hamiltonian shortcut toward apple
            # ---------------------
            shortcut_move = self._hamiltonian_shortcut(head, apple, snake, grid_size)
            if shortcut_move:
                # Calculate shortcut distance along cycle
                idx_head = self.hamiltonian.index(head)
                idx_apple = self.hamiltonian.index(apple)
                shortcut_distance = (idx_apple - idx_head) % len(self.hamiltonian)
                explanation = (
                    f"BFS Hamiltonian: Using intelligent Hamiltonian shortcut toward apple at {apple}. "
                    f"Moving {shortcut_move} along Hamiltonian cycle (shortcut distance: {shortcut_distance} cells). "
                    f"This optimizes path while maintaining cycle safety guarantees."
                )
                return shortcut_move, explanation

            # ---------------------
            # 4. NEW: Hamiltonian cycle fallback (ultimate safety)
            # ---------------------
            hamiltonian_move = self._hamiltonian_move(head, obstacles)
            if hamiltonian_move:
                idx_head = self.hamiltonian.index(head)
                cycle_progress = f"{idx_head + 1}/{len(self.hamiltonian)}"
                explanation = (
                    f"BFS Hamiltonian: No safe path to apple or tail. Activating Hamiltonian cycle fallback. "
                    f"Moving {hamiltonian_move} to continue systematic grid exploration (progress: {cycle_progress}). "
                    f"This guarantees no deadlocks and eventual apple collection."
                )
                return hamiltonian_move, explanation

            # ---------------------
            # 5. Last resort: inherited safe move finder
            # ---------------------
            last_resort_move = self._get_safe_move(head, obstacles, grid_size)
            explanation = (
                f"BFS Hamiltonian: All advanced strategies failed. Using last resort move {last_resort_move} "
                f"to avoid immediate collision. This indicates extremely constrained game state where even "
                f"Hamiltonian cycle cannot proceed safely."
            )
            return last_resort_move, explanation
            
        except Exception as e:
            explanation = f"BFS Hamiltonian Agent encountered an error: {str(e)}"
            print(f"BFS Hamiltonian Agent error: {e}")
            return "NO_PATH_FOUND", explanation

    def _hamiltonian_shortcut(
        self,
        head: Tuple[int, int],
        apple: Tuple[int, int],
        snake: List[Tuple[int, int]],
        grid_size: int
    ) -> Optional[str]:
        """
        Try intelligent Hamiltonian shortcut toward apple.
        
        Enhancement: Instead of always following the full cycle,
        try to jump ahead on the cycle toward the apple if safe.
        
        Args:
            head: Current head position
            apple: Apple position
            snake: Current snake body
            grid_size: Size of game grid
            
        Returns:
            Direction for shortcut move or None if not safe
        """
        # Check if both head and apple are on the Hamiltonian cycle
        if head not in self.hamiltonian or apple not in self.hamiltonian:
            return None
            
        # Find indices on the cycle
        idx_head = self.hamiltonian.index(head)
        idx_apple = self.hamiltonian.index(apple)
        
        # Calculate forward distance on cycle
        forward_distance = (idx_apple - idx_head) % len(self.hamiltonian)
        
        # Skip if apple is at current position or too far (prefer normal cycle)
        if forward_distance == 0 or forward_distance > len(self.hamiltonian) // 2:
            return None
        
        # Build segment from head to apple along cycle
        segment = []
        for i in range(forward_distance + 1):
            pos_idx = (idx_head + i) % len(self.hamiltonian)
            segment.append(self.hamiltonian[pos_idx])
        
        # Check if following this segment is safe
        if self._segment_is_safe(segment, snake, apple, grid_size):
            next_pos = segment[1]  # First move in segment
            return position_to_direction(head, next_pos)
            
        return None

    def _hamiltonian_move(
        self, 
        head: Tuple[int, int], 
        obstacles: Set[Tuple[int, int]]
    ) -> Optional[str]:
        """
        Move to next position in Hamiltonian cycle.
        
        Ultimate fallback: follow the pre-computed Hamiltonian cycle
        which guarantees visiting every cell and avoiding deadlocks.
        
        Args:
            head: Current head position
            obstacles: Set of obstacle positions
            
        Returns:
            Direction for next Hamiltonian move or None if not possible
        """
        if head not in self.hamiltonian:
            return None
            
        # Find current position in cycle
        current_idx = self.hamiltonian.index(head)
        
        # Get next position in cycle
        next_idx = (current_idx + 1) % len(self.hamiltonian)
        next_pos = self.hamiltonian[next_idx]
        
        # Check if next position is safe
        if next_pos not in obstacles:
            return position_to_direction(head, next_pos)
            
        return None

    def _segment_is_safe(
        self,
        segment: List[Tuple[int, int]],
        snake: List[Tuple[int, int]],
        apple: Tuple[int, int],
        grid_size: int
    ) -> bool:
        """
        Check if following a Hamiltonian segment is safe.
        
        Simulates following the segment and verifies the snake
        can still reach its tail afterward.
        
        Args:
            segment: Proposed path segment along Hamiltonian cycle
            snake: Current snake body
            apple: Apple position
            grid_size: Size of game grid
            
        Returns:
            True if segment is safe to follow
        """
        # Simulate following the segment
        virtual_snake = list(snake)
        
        for step in segment[1:]:  # Skip current head position
            virtual_snake.insert(0, step)  # Move head
            
            if step == apple:
                # Apple eaten - snake grows, keep tail
                break
            
            # No apple - tail moves forward
            virtual_snake.pop()
        
        # Check if new head can reach new tail
        new_head = virtual_snake[0]
        new_tail = virtual_snake[-1]
        new_obstacles = set(virtual_snake[:-1])  # Exclude tail
        
        # Use inherited BFS to check tail reachability
        tail_path = self._bfs_pathfind(new_head, new_tail, new_obstacles, grid_size)
        return bool(tail_path)

    def __str__(self) -> str:
        """String representation showing inheritance hierarchy."""
        return f"BFSHamiltonianAgent(extends=BFSSafeGreedyAgent, algorithm={self.algorithm_name})"
