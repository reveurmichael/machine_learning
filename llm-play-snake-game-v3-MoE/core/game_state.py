"""
Game state management for the Snake game.
Separates state management from game logic.
"""

import numpy as np
from config import GRID_SIZE

class GameState:
    """Manages the state of the Snake game independently of game logic."""
    
    def __init__(self, grid_size=GRID_SIZE):
        """Initialize the game state.
        
        Args:
            grid_size: Size of the game grid
        """
        if not isinstance(grid_size, int) or grid_size <= 0:
            raise ValueError(f"grid_size must be a positive integer, got {grid_size}")
            
        # Grid configuration
        self.grid_size = grid_size
        self.board = np.zeros((grid_size, grid_size))
        
        # Game state
        self.snake_positions = np.array([[grid_size//2, grid_size//2]])  # Start in middle
        self.head_position = self.snake_positions[-1]
        self.apple_position = None  # Will be set by game logic
        self.current_direction = None
        self.steps = 0
        self.score = 0
        self.last_collision_type = None
        
        # Board entity codes
        self.board_info = {
            "empty": 0,
            "snake": 1,
            "apple": 2
        }
        
        # Apple history for replay
        self.apple_positions_history = []
        self.replay_mode = False
    
    def reset(self):
        """Reset the game state to initial values."""
        self.snake_positions = np.array([[self.grid_size//2, self.grid_size//2]])
        self.head_position = self.snake_positions[-1]
        self.apple_position = None
        self.current_direction = None
        self.steps = 0
        self.score = 0
        self.last_collision_type = None
        self.apple_positions_history = []
        self.board.fill(self.board_info["empty"])
    
    def update_board(self):
        """Update the game board with current snake and apple positions."""
        # Clear the board
        self.board.fill(self.board_info["empty"])
        
        # Place the snake
        for x, y in self.snake_positions:
            self.board[y, x] = self.board_info["snake"]
        
        # Place the apple if it exists
        if self.apple_position is not None:
            x, y = self.apple_position
            self.board[y, x] = self.board_info["apple"]
    
    def add_apple_to_history(self, position):
        """Add an apple position to the history.
        
        Args:
            position: Position to add as [x, y]
        """
        self.apple_positions_history.append(position.copy())
    
    def get_apple_from_history(self, index):
        """Get an apple position from history.
        
        Args:
            index: Index in the history to retrieve
            
        Returns:
            Apple position as [x, y] or None if index is invalid
        """
        if 0 <= index < len(self.apple_positions_history):
            return self.apple_positions_history[index].copy()
        return None
    
    def set_apple_position(self, position):
        """Set the apple position.
        
        Args:
            position: Position to set as [x, y]
            
        Returns:
            Boolean indicating if the position was valid and set successfully
        """
        try:
            x, y = position
            
            # Validate position
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                print(f"Invalid apple position: {position}")
                return False
                
            # Check if position is empty
            if any(np.array_equal([x, y], pos) for pos in self.snake_positions):
                print(f"Cannot place apple on snake: {position}")
                return False
                
            # Set the position
            self.apple_position = np.array([x, y])
            return True
            
        except Exception as e:
            print(f"Error setting apple position: {e}")
            return False
    
    def move_snake(self, new_head):
        """Move the snake by adding a new head position.
        
        Args:
            new_head: New head position as [x, y]
        """
        self.snake_positions = np.vstack([self.snake_positions, new_head])
        self.head_position = new_head
    
    def remove_tail(self):
        """Remove the tail of the snake."""
        self.snake_positions = self.snake_positions[1:]
    
    def increment_score(self):
        """Increment the game score."""
        self.score += 1
    
    def increment_steps(self):
        """Increment the step counter."""
        self.steps += 1
    
    def set_collision_type(self, collision_type):
        """Set the type of collision that ended the game.
        
        Args:
            collision_type: String indicating collision type ('wall', 'self', etc.)
        """
        self.last_collision_type = collision_type
    
    def set_direction(self, direction):
        """Set the current direction of movement.
        
        Args:
            direction: Direction vector as [dx, dy]
        """
        self.current_direction = direction 