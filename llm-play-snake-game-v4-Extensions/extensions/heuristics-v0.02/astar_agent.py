"""
A* Pathfinding Agent for Snake Game
===================================

This module implements the A* (A-star) pathfinding algorithm for Snake gameplay.
A* is an informed search algorithm that uses a heuristic function to guide the
search towards the goal, making it more efficient than uninformed search like BFS.

The algorithm uses f(n) = g(n) + h(n) where:
- g(n) = actual cost from start to node n
- h(n) = heuristic cost from node n to goal (Manhattan distance)
- f(n) = estimated total cost of path through node n

Key Features:
- Manhattan distance heuristic (admissible for grid-based movement)
- Priority queue for optimal node exploration
- Collision detection integration
- Same interface as BFS agent for easy substitution

Design Patterns:
- Strategy Pattern: Interchangeable pathfinding algorithm
- Template Method: Consistent agent interface
"""

import heapq
from typing import List, Tuple, Optional, Set, Dict
from config.game_constants import DIRECTIONS


class AStarAgent:
    """
    A* pathfinding agent for Snake game.
    
    Implements the A* algorithm with Manhattan distance heuristic to find
    optimal paths from the snake head to the apple position while avoiding
    collisions with walls and the snake's own body.
    
    The agent uses a priority queue to explore the most promising nodes first,
    making it more efficient than BFS while still guaranteeing optimal paths.
    
    Design Patterns:
    - Strategy Pattern: Pluggable pathfinding algorithm
    - Command Pattern: get_move() returns direction commands
    """
    
    def __init__(self):
        """Initialize the A* pathfinding agent."""
        self.name = "A*"
        self.description = "A* pathfinding with Manhattan distance heuristic"
    
    def get_move(self, game: "HeuristicGameLogic") -> str | None:
        """
        Get next move using A* pathfinding.
        
        Args:
            game: Game logic instance containing current game state
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT) or "NO_PATH_FOUND"
        """
        try:
            # Extract game state
            head_pos = tuple(game.head_position)
            apple_pos = tuple(game.apple_position)
            snake_positions = {tuple(pos) for pos in game.snake_positions}
            grid_size = game.grid_size
            
            # Find path using A*
            path = self._astar_pathfind(head_pos, apple_pos, snake_positions, grid_size)
            
            if not path or len(path) < 2:
                return "NO_PATH_FOUND"
                
            # Get first move in path
            next_pos = path[1]  # path[0] is current head position
            direction = self._get_direction(head_pos, next_pos)
            
            return direction
            
        except Exception as e:
            print(f"A* Agent error: {e}")
            return "NO_PATH_FOUND"
    
    def _astar_pathfind(self, start: Tuple[int, int], goal: Tuple[int, int], 
                       obstacles: Set[Tuple[int, int]], grid_size: int) -> List[Tuple[int, int]]:
        """
        A* pathfinding algorithm implementation.
        
        Uses Manhattan distance as heuristic function and priority queue
        for optimal node exploration.
        
        Args:
            start: Starting position (snake head)
            goal: Goal position (apple)
            obstacles: Set of obstacle positions (snake body)
            grid_size: Size of the game grid
            
        Returns:
            List of positions forming the optimal path, or empty list if no path
        """
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        
        # Track visited nodes and their g_scores
        g_scores: Dict[Tuple[int, int], int] = {start: 0}
        
        # Track path reconstruction
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Set of visited nodes
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            # Get node with lowest f_score
            f_score, g_score, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
                
            # Mark as processed
            closed_set.add(current)
            
            # Check if we reached the goal
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            # Explore neighbors
            for dx, dy in DIRECTIONS.values():
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if out of bounds
                if not self._is_valid_position(neighbor, grid_size):
                    continue
                    
                # Skip if collision with snake body (but allow goal position)
                if neighbor in obstacles and neighbor != goal:
                    continue
                
                # Skip if already processed
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score + 1
                
                # Check if this path is better
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    # Record best path to this neighbor
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    
                    # Calculate f_score using Manhattan distance heuristic
                    h_score = self._manhattan_distance(neighbor, goal)
                    f_score_neighbor = tentative_g + h_score
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score_neighbor, tentative_g, neighbor))
        
        # No path found
        return []
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two positions.
        
        Manhattan distance is admissible for grid-based movement, guaranteeing
        optimal paths when used as a heuristic in A*.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Manhattan distance between the positions
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _is_valid_position(self, pos: Tuple[int, int], grid_size: int) -> bool:
        """
        Check if position is within grid boundaries.
        
        Args:
            pos: Position to check
            grid_size: Size of the game grid
            
        Returns:
            True if position is valid, False otherwise
        """
        x, y = pos
        return 0 <= x < grid_size and 0 <= y < grid_size
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the optimal path from came_from mapping.
        
        Args:
            came_from: Dictionary mapping each node to its predecessor
            current: Goal node to trace back from
            
        Returns:
            List of positions forming the complete path from start to goal
        """
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        # Reverse to get path from start to goal
        return path[::-1]
    
    def _get_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """
        Convert position difference to direction string.
        
        Uses the same coordinate system as the base game for consistency.
        
        Args:
            from_pos: Starting position
            to_pos: Target position
            
        Returns:
            Direction string (UP, DOWN, LEFT, RIGHT)
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Map coordinate differences to directions
        # Based on DIRECTIONS = {"UP": (0, 1), "RIGHT": (1, 0), "DOWN": (0, -1), "LEFT": (-1, 0)}
        if dx == 0 and dy == 1:
            return "UP"
        elif dx == 0 and dy == -1:
            return "DOWN"
        elif dx == 1 and dy == 0:
            return "RIGHT"
        elif dx == -1 and dy == 0:
            return "LEFT"
        else:
            return "NO_PATH_FOUND"  # Invalid move 