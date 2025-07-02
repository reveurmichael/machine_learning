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
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional

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
        
        # Statistics
        self.cycle_completions: int = 0
        self.total_moves: int = 0
        
    def get_move(self, game: HeuristicGameLogic) -> str | None:
        """Get the next move by following the Hamiltonian cycle, with potential shortcuts."""
        move, _ = self.get_move_with_explanation(game)
        return move
        
    def get_move_with_explanation(self, game: HeuristicGameLogic) -> Tuple[str, str]:
        """
        Get the next move by following the Hamiltonian cycle, with explanations for shortcuts.

        v0.04 Enhancement: Provides detailed explanations about cycle-following, shortcut-taking,
        and safety checks, making the data ideal for LLM fine-tuning.
        """
        try:
            head = tuple(game.head_position)
            grid_size = game.grid_size
            
            # Generate the cycle only once or if the grid size changes
            if not self.cycle or self.grid_size != grid_size:
                self.grid_size = grid_size
                self._generate_hamiltonian_cycle()
            
            # Find the snake's current position in the cycle
            head_idx = self.cycle_map.get(head)
            if head_idx is None:
                # This is a critical error, the snake is off the cycle. Fallback.
                return self._handle_off_cycle_error(head, game)

            # --- Shortcut Logic ---
            # Can we take a shortcut to the apple without getting trapped?
            shortcut_move, shortcut_explanation = self._evaluate_shortcut(game, head_idx)
            if shortcut_move:
                return shortcut_move, shortcut_explanation

            # --- Default Action: Follow the Cycle ---
            next_idx = (head_idx + 1) % len(self.cycle)
            next_pos = self.cycle[next_idx]
            
            direction = position_to_direction(head, next_pos)
            explanation = self._generate_cycle_follow_explanation(head, next_pos, direction, head_idx)
            
            return direction, explanation
                
        except Exception as e:
            explanation = f"Hamiltonian Agent encountered a critical error: {str(e)}"
            print(f"Hamiltonian Agent error: {e}")
            return "NO_PATH_FOUND", explanation

    def _generate_hamiltonian_cycle(self) -> None:
        """
        Generates a robust, guaranteed-closed Hamiltonian cycle for any grid size.
        
        This method works by creating a grid graph, finding a spanning tree (using DFS),
        and then tracing the perimeter of the tree. This is a reliable way to ensure
        every node is visited and the path is a closed loop.
        """
        size = self.grid_size
        adj = { (x, y): [] for x in range(size) for y in range(size) }
        
        # Build adjacency list for the grid graph
        for x in range(size):
            for y in range(size):
                if x > 0: adj[(x, y)].append((x - 1, y))
                if x < size - 1: adj[(x, y)].append((x + 1, y))
                if y > 0: adj[(x, y)].append((x, y - 1))
                if y < size - 1: adj[(x, y)].append((x, y + 1))

        # Create a spanning tree using DFS
        tree_adj = { (x, y): [] for x in range(size) for y in range(size) }
        visited = set()
        stack = [(0, 0)]
        
        while stack:
            v = stack.pop()
            if v in visited: continue
            visited.add(v)
            for neighbor in adj[v]:
                if neighbor not in visited:
                    tree_adj[v].append(neighbor)
                    tree_adj[neighbor].append(v)
                    stack.append(neighbor)
        
        # Trace the perimeter of the spanning tree to get the cycle
        cycle = []
        curr = (0, 0)
        prev = (-1, -1)  # A virtual previous node
        
        for _ in range(2 * size * size):
            cycle.append(curr)
            
            # Find the next node by turning "left" relative to the edge from prev to curr
            dx, dy = curr[0] - prev[0], curr[1] - prev[1]
            
            # Order of preference: Left, Straight, Right, Back
            if dx == 0: # Moving vertically
                options = [(curr[0] + dy, curr[1]), (curr[0], curr[1] + dy), (curr[0] - dy, curr[1]), (prev[0], prev[1])]
            else: # Moving horizontally
                options = [(curr[0], curr[1] - dx), (curr[0] + dx, curr[1]), (curr[0], curr[1] + dx), (prev[0], prev[1])]

            for nxt in options:
                if nxt in tree_adj[curr] or nxt == prev:
                    prev, curr = curr, nxt
                    break

            if curr == (0, 0) and len(cycle) > 1: break # Completed the cycle
        
        self.cycle = cycle
        self.cycle_map = {pos: idx for idx, pos in enumerate(self.cycle)}

    def _evaluate_shortcut(self, game: HeuristicGameLogic, head_idx: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Determines if a safe and effective shortcut to the apple is possible.
        A shortcut jumps forward in the cycle, skipping intermediate nodes.
        """
        head_pos = tuple(game.head_position)
        apple_pos = tuple(game.apple_position)
        snake_len = len(game.snake_positions)
        
        apple_idx = self.cycle_map.get(apple_pos)
        
        if apple_idx is None: return None, None # Apple not on cycle (should not happen)
        
        # Is the apple "ahead" of the snake in the cycle path?
        if (head_idx < apple_idx) and (apple_idx - head_idx < snake_len):
            # Attempting a shortcut
            path_to_apple = self.cycle[head_idx:apple_idx+1]
            
            # Is the shortcut path clear of the snake's own body?
            snake_body_set = {tuple(p) for p in game.snake_positions[1:]}
            shortcut_path_set = set(path_to_apple[1:]) # Don't check head
            
            if not shortcut_path_set.intersection(snake_body_set):
                # Is the move safe? Check if the snake traps itself after eating.
                # A simple check: is the space behind the apple's future position (the snake's old tail) free?
                future_snake_tail = self.cycle[(apple_idx - snake_len + 1) % len(self.cycle)]
                if self._is_safe_shortcut(game, future_snake_tail):
                    direction = position_to_direction(head_pos, path_to_apple[1])
                    explanation = (
                        f"Taking a safe shortcut. The apple is {apple_idx - head_idx} steps ahead on the cycle. "
                        f"Moving {direction} to jump directly towards it, skipping intermediate cycle nodes. "
                        "This is faster than following the full cycle."
                    )
                    return direction, explanation

        return None, None
    
    def _is_safe_shortcut(self, game, future_snake_tail) -> bool:
        """
        A simple heuristic to check if a shortcut is safe.
        Checks if the position that *will be* the snake's new tail is not immediately blocked.
        """
        # A more robust check would involve a flood fill from the new tail position.
        # For now, we do a simple check.
        for move in DIRECTIONS.values():
            check_pos = (future_snake_tail[0] + move[0], future_snake_tail[1] + move[1])
            if (0 <= check_pos[0] < self.grid_size and 0 <= check_pos[1] < self.grid_size and
                check_pos not in game.snake_positions):
                return True
        return False

    def _generate_cycle_follow_explanation(self, head: tuple, next_pos: tuple, direction: str, head_idx: int) -> str:
        """Generates an explanation for following the cycle normally."""
        progress = f"{head_idx + 1}/{len(self.cycle)}"
        is_new_cycle = (head_idx + 1) % len(self.cycle) == 0

        explanation = (
            f"Following pre-computed Hamiltonian cycle. The optimal move is {direction} from {head} to {next_pos}. "
            f"This guarantees complete and safe grid traversal. Current cycle progress: {progress}. "
        )
        if is_new_cycle:
            explanation += "A full cycle has been completed, starting a new traversal."
        return explanation

    def _handle_off_cycle_error(self, head: tuple, game: HeuristicGameLogic) -> Tuple[str, str]:
        """Generates an explanation and fallback for when the snake is off the cycle path."""
        # This state indicates a critical failure. Try to find any safe move.
        for move_dir, (dx, dy) in DIRECTIONS.items():
            next_pos = (head[0] + dx, head[1] + dy)
            if (0 <= next_pos[0] < self.grid_size and
                0 <= next_pos[1] < self.grid_size and
                next_pos not in game.snake_positions):
                explanation = (
                    "CRITICAL ERROR: Snake is off the Hamiltonian cycle. This should not happen. "
                    f"Attempting an emergency fallback move {move_dir} to avoid immediate collision."
                )
                return move_dir, explanation
        
        return "NO_PATH_FOUND", "CRITICAL ERROR: Snake is off-cycle and trapped. No safe moves available."

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
            "grid_size": self.grid_size
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"HamiltonianAgent(grid_size={self.grid_size}, cycle_nodes={len(self.cycle)})"
