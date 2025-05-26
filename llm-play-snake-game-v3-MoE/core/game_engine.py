"""
Game engine for the Snake game.
Provides core game logic that can run with or without a GUI.
"""

import numpy as np
from config import GRID_SIZE, DIRECTIONS

class GameEngine:
    """Base class for the Snake game engine."""
    
    def __init__(self, grid_size=GRID_SIZE, use_gui=True):
        """Initialize the game engine.
        
        Args:
            grid_size: Size of the game grid
            use_gui: Whether to use GUI for display
        """
        if not isinstance(grid_size, int) or grid_size <= 0:
            raise ValueError(f"grid_size must be a positive integer, got {grid_size}")
            
        # Game state variables
        self.grid_size = grid_size
        self.board = np.zeros((grid_size, grid_size))
        self.snake_positions = np.array([[grid_size//2, grid_size//2]])  # Start in middle
        self.head_position = self.snake_positions[-1]
        self.apple_position = self._generate_apple()
        self.current_direction = None
        self.steps = 0
        self.score = 0
        self.last_collision_type = None  # Will be 'wall', 'self', 'max_steps', or 'empty_moves'
        
        # Board entity codes
        self.board_info = {
            "empty": 0,
            "snake": 1,
            "apple": 2
        }
        
        # Track apple positions history for replay
        self.apple_positions_history = []
        self.apple_positions_history.append(self.apple_position.copy())
        self.replay_mode = False
        
        # GUI settings
        self.use_gui = use_gui
        self.gui = None
        
        # Initialize the board
        self._update_board()
    
    def set_gui(self, gui_instance):
        """Set the GUI instance to use for display.
        
        Args:
            gui_instance: Instance of a GUI class
        """
        self.gui = gui_instance
        self.use_gui = (gui_instance is not None)
    
    def reset(self):
        """Reset the game to the initial state."""
        # Reset game state
        self.snake_positions = np.array([[self.grid_size//2, self.grid_size//2]])
        self.head_position = self.snake_positions[-1]
        self.apple_position = self._generate_apple()
        self.current_direction = None
        self.steps = 0
        self.score = 0
        self.last_collision_type = None
        
        # Reset apple history
        self.apple_positions_history = []
        self.apple_positions_history.append(self.apple_position.copy())
        
        # Update the board
        self._update_board()
        
        # Draw if GUI is available
        if self.use_gui and self.gui:
            self.draw()
    
    def draw(self):
        """Draw the current game state if GUI is available."""
        if self.use_gui and self.gui:
            # Specific drawing handled by the GUI implementation
            pass
    
    def _update_board(self):
        """Update the game board with current snake and apple positions."""
        # Clear the board
        self.board.fill(self.board_info["empty"])
        
        # Place the snake (board is indexed as [y][x] since it's a 2D array)
        for x, y in self.snake_positions:
            self.board[y, x] = self.board_info["snake"]
        
        # Place the apple
        x, y = self.apple_position
        self.board[y, x] = self.board_info["apple"]
    
    def _generate_apple(self):
        """Generate a new apple at a random empty position.
        
        Returns:
            Array of [x, y] coordinates for the new apple
        """
        while True:
            # Generate random position
            x, y = np.random.randint(0, self.grid_size, 2)
            
            # Check if position is empty (not occupied by snake)
            if not any(np.array_equal([x, y], pos) for pos in self.snake_positions):
                return np.array([x, y])
    
    def set_apple_position(self, position):
        """Set the apple position manually (for replay purposes).
        
        Args:
            position: Position to place the apple as [x, y]
            
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
            
            # Update the board
            self._update_board()
            
            # Update display if GUI is available
            if self.use_gui and self.gui:
                self.draw()
                
            return True
        except Exception as e:
            print(f"Error setting apple position: {e}")
            return False
    
    def make_move(self, direction_key):
        """Execute a move in the specified direction.
        
        Args:
            direction_key: String key of the direction to move in ("UP", "DOWN", etc.)
            
        Returns:
            Tuple of (game_active, apple_eaten) where:
                game_active: Boolean indicating if the game is still active
                apple_eaten: Boolean indicating if an apple was eaten on this move
        """
        # Get direction vector
        if direction_key not in DIRECTIONS:
            print(f"Invalid direction: {direction_key}, defaulting to RIGHT")
            direction_key = "RIGHT"
        
        direction = DIRECTIONS[direction_key]
        
        # Don't allow reversing direction directly
        if (self.current_direction is not None and 
            np.array_equal(np.array(direction), -np.array(self.current_direction))):
            print(f"Tried to reverse direction: {direction_key}. Using current direction instead.")
            direction = self.current_direction
            direction_key = self._get_current_direction_key()
        
        # Update current direction
        self.current_direction = direction
        
        # Calculate new head position according to our coordinate system
        head_x, head_y = self.head_position
        
        # Apply direction vector to head position
        new_head = np.array([
            head_x + direction[0],  # Apply dx to x-coordinate
            head_y + direction[1]   # Apply dy to y-coordinate
        ])
        
        # Debug log
        print(f"Moving {direction_key}: Head from ({head_x}, {head_y}) to ({new_head[0]}, {new_head[1]})")
        
        # Check for collisions
        wall_collision, body_collision = self._check_collision(new_head)
        
        if wall_collision:
            print(f"Game over! Snake hit wall moving {direction_key}")
            self.last_collision_type = 'wall'
            return False, False  # Game over, no apple eaten
            
        if body_collision:
            print(f"Game over! Snake hit itself moving {direction_key}")
            self.last_collision_type = 'self'
            return False, False  # Game over, no apple eaten
        
        # Move the snake: add new head
        self.snake_positions = np.vstack([self.snake_positions, new_head])
        self.head_position = new_head
        
        # Check if apple is eaten
        apple_eaten = False
        if np.array_equal(new_head, self.apple_position):
            self.score += 1
            print(f"Apple eaten! Score: {self.score}")
            
            # Generate a new apple
            if self.replay_mode and len(self.apple_positions_history) > self.score:
                # In replay mode, use the next apple position from history
                next_apple = self.apple_positions_history[self.score]
                self.apple_position = next_apple.copy()
                print(f"Replay: using predefined apple position: ({next_apple[0]}, {next_apple[1]})")
            else:
                # Generate a new random apple position
                self.apple_position = self._generate_apple()
                # In normal mode, record this position for future replay
                if not self.replay_mode:
                    self.apple_positions_history.append(self.apple_position.copy())
                    print(f"Recorded new apple position: ({self.apple_position[0]}, {self.apple_position[1]})")
            
            apple_eaten = True
        else:
            # Remove the tail if no apple is eaten
            self.snake_positions = self.snake_positions[1:]
            
        # Update the board
        self._update_board()
        
        # Increment steps
        self.steps += 1
        
        # Update display if GUI is available
        if self.use_gui and self.gui:
            self.draw()
        
        return True, apple_eaten  # Game continues, indicates if apple was eaten
    
    def _check_collision(self, position):
        """Check if a position collides with wall or snake body.
        
        Args:
            position: Position to check as [x, y]
            
        Returns:
            Tuple of (wall_collision, body_collision) booleans
        """
        x, y = position
        
        # Check wall collision
        wall_collision = (x < 0 or x >= self.grid_size or 
                         y < 0 or y >= self.grid_size)
        
        # Check body collision (skip head position which is at index 0)
        body_collision = False
        if len(self.snake_positions) > 1:
            body_collision = any(np.array_equal(position, pos) 
                               for pos in self.snake_positions[1:])
        
        return wall_collision, body_collision
    
    def _get_current_direction_key(self):
        """Get the string key for the current direction.
        
        Returns:
            String key ("UP", "DOWN", "LEFT", "RIGHT") for current direction
        """
        if self.current_direction is None:
            return "RIGHT"  # Default direction
            
        for key, value in DIRECTIONS.items():
            if np.array_equal(self.current_direction, value):
                return key
                
        return "RIGHT"  # Default if no match found 