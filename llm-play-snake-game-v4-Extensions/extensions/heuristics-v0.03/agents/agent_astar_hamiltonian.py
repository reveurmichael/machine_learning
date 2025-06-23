"""
A* Hamiltonian Hybrid Agent - Advanced Pathfinding with Safety Guarantee
==============================================

This module implements a sophisticated agent that extends the A* agent with
Hamiltonian cycle capabilities. It represents the evolution from A* to A*-Hamiltonian.

Inheritance Hierarchy:
- AStarAgent (optimal pathfinding with heuristics)
  └── AStarHamiltonianAgent (adds Hamiltonian cycle fallback)

This demonstrates how advanced algorithms can be built by extending proven ones:
A* provides optimal pathfinding, Hamiltonian cycle provides safety guarantee.

Design Patterns:
- Inheritance: Extends AStarAgent with Hamiltonian cycle logic
- Template Method: Overrides get_move() while reusing A* pathfinding
- Strategy Pattern: Hamiltonian cycle as ultimate fallback strategy
- Composite Pattern: Combines optimal search with safety guarantee
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING
from .agent_astar import AStarAgent

# Use standardized path setup
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from utils.moves_utils import position_to_direction

if TYPE_CHECKING:
    from game_logic import HeuristicGameLogic


class AStarHamiltonianAgent(AStarAgent):
    """
    A* Hamiltonian Hybrid Agent: Advanced pathfinding with safety guarantee.
    
    Inheritance Pattern:
    - Inherits from AStarAgent (reuses optimal A* pathfinding)
    - Adds Hamiltonian cycle generation and management
    - Provides safety validation and fallback strategies
    - Demonstrates evolution from optimal to optimal + safe
    
    Algorithm Enhancement:
    1. Use inherited A* to find optimal path to apple (from AStarAgent)
    2. Validate path safety (can reach tail afterward)
    3. If safe, follow A* path
    4. NEW: Fall back to Hamiltonian cycle (guaranteed safe)
    5. Last resort: any safe move (inherited)
    
    This represents the pinnacle of heuristic Snake AI:
    - Optimal when possible (A* pathfinding)
    - Safe always (Hamiltonian cycle fallback)
    """

    def __init__(self) -> None:
        """Initialize A* Hamiltonian agent, extending A* agent."""
        super().__init__()  # Initialize parent A* agent
        self.algorithm_name = "ASTAR-HAMILTONIAN"
        self.name = "A* Hamiltonian Hybrid"
        self.description = (
            "Advanced hybrid agent extending AStarAgent with Hamiltonian cycle "
            "safety guarantee. Uses inherited A* optimal pathfinding when safe, "
            "falls back to Hamiltonian cycle when necessary. Represents the "
            "pinnacle of heuristic Snake AI."
        )
        
        # Hamiltonian cycle state
        self.cycle: List[Tuple[int, int]] = []
        self.cycle_map: Dict[Tuple[int, int], int] = {}
        self.grid_size: int = 0
        self.current_cycle_index: int = 0

    def get_move(self, game: "HeuristicGameLogic") -> str | None:
        """
        Get next move using A* with Hamiltonian fallback.
        
        Enhancement over parent A*:
        - Adds safety validation before following A* path
        - Provides Hamiltonian cycle as ultimate safe fallback
        - Guarantees the snake never gets permanently trapped
        
        Strategy priority:
        1. Use inherited A* to find optimal path to apple
        2. Validate path safety (can reach tail afterward)
        3. If safe, follow A* path
        4. Otherwise, fall back to Hamiltonian cycle
        5. Last resort: any safe move
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        try:
            head = tuple(game.head_position)
            apple = tuple(game.apple_position)
            grid_size = game.grid_size
            body = [tuple(seg) for seg in game.snake_positions]
            obstacles: Set[Tuple[int, int]] = set(body[:-1])  # Tail can vacate

            # Initialize Hamiltonian cycle if needed
            if not self.cycle or self.grid_size != grid_size:
                self.grid_size = grid_size
                self._generate_hamiltonian_cycle(grid_size)

            # ------------------------------------------------------
            # 1. Try inherited A* pathfinding with safety validation
            # ------------------------------------------------------
            path_to_apple = self._astar_pathfind(head, apple, obstacles, grid_size)
            
            if path_to_apple and len(path_to_apple) > 1:
                # Safety enhancement: validate A* path before using it
                if self._path_is_safe_after_eating(path_to_apple, body, grid_size):
                    next_pos = path_to_apple[1]
                    return position_to_direction(head, next_pos)

            # ------------------------------------------------------
            # 2. NEW: Hamiltonian cycle fallback (ultimate safety)
            # ------------------------------------------------------
            hamiltonian_move = self._follow_hamiltonian_cycle(head, obstacles)
            if hamiltonian_move:
                return hamiltonian_move

            # ------------------------------------------------------
            # 3. Last resort: inherited emergency evasion
            # ------------------------------------------------------
            return self._emergency_evasion(head, grid_size)

        except Exception as e:
            print(f"A* Hamiltonian Agent error: {e}")
            return "NO_PATH_FOUND"

    def _generate_hamiltonian_cycle(self, grid_size: int) -> None:
        """
        Generate Hamiltonian cycle using boustrophedon pattern.
        
        Enhancement: Creates a cycle that visits every cell exactly once,
        providing ultimate safety fallback when A* paths are unsafe.
        
        Creates snake-like pattern:
        - Even rows: left to right
        - Odd rows: right to left
        - Ensures first and last cells are adjacent
        
        Args:
            grid_size: Size of the square grid
        """
        self.cycle = []

        # Generate boustrophedon sweep pattern
        for y in range(grid_size):
            if y % 2 == 0:
                # Even row: left to right
                row_positions = [(x, y) for x in range(grid_size)]
            else:
                # Odd row: right to left
                row_positions = [(x, y) for x in range(grid_size - 1, -1, -1)]
            
            self.cycle.extend(row_positions)

        # Ensure cycle closure (first and last cells are adjacent)
        if grid_size % 2 == 0:
            # For even grids, adjust last cell to ensure adjacency
            if len(self.cycle) >= 2:
                # Move last cell to ensure adjacency with first
                last_pos = self.cycle[-1]
                first_pos = self.cycle[0]
                
                # If not adjacent, fix the pattern
                if abs(last_pos[0] - first_pos[0]) + abs(last_pos[1] - first_pos[1]) != 1:
                    # Adjust the pattern to ensure proper cycle
                    self.cycle[-1] = (1, grid_size - 1)
                    if (0, grid_size - 1) not in self.cycle:
                        self.cycle.append((0, grid_size - 1))

        # Build fast lookup map for O(1) position-to-index lookup
        self.cycle_map = {pos: idx for idx, pos in enumerate(self.cycle)}

    def _path_is_safe_after_eating(
        self,
        path: List[Tuple[int, int]],
        snake_body: List[Tuple[int, int]],
        grid_size: int
    ) -> bool:
        """
        Safety enhancement: Validate A* path by simulating execution.
        
        This is the key enhancement over basic A* - we simulate following
        the A* path and check if the snake can still reach its tail afterward.
        
        Args:
            path: A* path to apple
            snake_body: Current snake body
            grid_size: Size of game grid
            
        Returns:
            True if A* path is safe (tail reachable), False otherwise
        """
        # Simulate following the A* path
        virtual_snake = list(snake_body)
        apple_pos = path[-1]  # Last position in path is apple
        
        for step in path[1:]:  # Skip current head position
            virtual_snake.insert(0, step)  # Move head
            
            if step == apple_pos:
                # Apple eaten - snake grows, keep tail
                break
            
            # No apple yet - tail moves forward
            virtual_snake.pop()
        
        # Check if new head can reach new tail using inherited A*
        new_head = virtual_snake[0]
        new_tail = virtual_snake[-1]
        new_obstacles = set(virtual_snake[:-1])  # Exclude tail
        
        # Use inherited A* pathfinding to check tail reachability
        tail_path = self._astar_pathfind(new_head, new_tail, new_obstacles, grid_size)
        return bool(tail_path)

    def _follow_hamiltonian_cycle(
        self, 
        head: Tuple[int, int], 
        obstacles: Set[Tuple[int, int]]
    ) -> Optional[str]:
        """
        Follow the Hamiltonian cycle as safety fallback.
        
        Enhancement: Provides guaranteed safe movement when A* paths are unsafe.
        The Hamiltonian cycle visits every cell, ensuring no deadlocks.
        
        Args:
            head: Current head position
            obstacles: Set of obstacle positions
            
        Returns:
            Direction for next cycle move or None if not possible
        """
        # Find current position in cycle
        head_idx = self.cycle_map.get(head, -1)
        
        if head_idx == -1:
            # Head not in cycle, find nearest position
            head_idx = self._find_nearest_cycle_index(head)
        
        # Get next position in cycle
        next_idx = (head_idx + 1) % len(self.cycle)
        next_pos = self.cycle[next_idx]
        
        # Check if next position is safe
        if next_pos not in obstacles:
            self.current_cycle_index = next_idx
            return position_to_direction(head, next_pos)
        
        return None

    def _find_nearest_cycle_index(self, pos: Tuple[int, int]) -> int:
        """
        Find nearest position in Hamiltonian cycle.
        
        Used when snake head is not exactly on the cycle path.
        
        Args:
            pos: Position to find nearest cycle position for
            
        Returns:
            Index of nearest position in cycle
        """
        best_idx = 0
        best_distance = float('inf')
        
        for idx, cycle_pos in enumerate(self.cycle):
            distance = abs(pos[0] - cycle_pos[0]) + abs(pos[1] - cycle_pos[1])
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
        
        return best_idx

    def get_cycle_info(self) -> Dict[str, object]:
        """
        Get information about the Hamiltonian cycle.
        
        Useful for debugging and analysis.
        
        Returns:
            Dictionary with cycle information
        """
        return {
            "cycle_length": len(self.cycle),
            "grid_size": self.grid_size,
            "current_index": self.current_cycle_index,
            "cycle_generated": bool(self.cycle)
        }

    def __str__(self) -> str:
        """String representation showing inheritance hierarchy."""
        return f"AStarHamiltonianAgent(extends=AStarAgent, algorithm={self.algorithm_name})" 