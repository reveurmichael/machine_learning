import numpy as np
from .config import SNAKE_SIZE, DIRECTIONS

class SnakeGameEnv:
    """Core Snake game environment without GUI dependencies."""
    
    def __init__(self, grid_size=SNAKE_SIZE):
        self.grid_size = grid_size
        self.board = np.zeros((grid_size, grid_size))
        self.snake_positions = np.array([[grid_size//2, grid_size//2]])
        self.head_position = self.snake_positions[-1]
        self.apple_position = self._generate_apple()
        self.current_direction = None
        self.steps = 0
        self.score = 0
        self.generation = 1
        
        # Board entity codes
        self.board_info = {
            "empty": 0,
            "snake": 1,
            "apple": 2
        }
        
        # Initialize the board
        self._update_board()
    
    def reset(self):
        """Reset the game to initial state."""
        self.snake_positions = np.array([[self.grid_size//2, self.grid_size//2]])
        self.head_position = self.snake_positions[-1]
        self.apple_position = self._generate_apple()
        self.current_direction = None
        self.steps = 0
        self.score = 0
        self.generation += 1
        self._update_board()
        return self.get_state()
    
    def _update_board(self):
        """Update the game board with current snake and apple positions."""
        self.board.fill(self.board_info["empty"])
        for x, y in self.snake_positions:
            self.board[y, x] = self.board_info["snake"]
        x, y = self.apple_position
        self.board[y, x] = self.board_info["apple"]
    
    def _generate_apple(self):
        """Generate a new apple at a random empty position."""
        while True:
            x, y = np.random.randint(0, self.grid_size, 2)
            if not any(np.array_equal([x, y], pos) for pos in self.snake_positions):
                return np.array([x, y])
    
    def make_move(self, direction_key):
        """Execute a move in the specified direction."""
        if direction_key not in DIRECTIONS:
            direction_key = "RIGHT"
        
        direction = DIRECTIONS[direction_key]
        
        if (self.current_direction is not None and 
            np.array_equal(np.array(direction), -np.array(self.current_direction))):
            direction = self.current_direction
            direction_key = self._get_current_direction_key()
        
        self.current_direction = direction
        head_x, head_y = self.head_position
        new_head = np.array([
            head_x + direction[0],
            head_y + direction[1]
        ])
        
        wall_collision, body_collision = self._check_collision(new_head)
        
        if wall_collision or body_collision:
            return False, False
        
        self.snake_positions = np.vstack([self.snake_positions, new_head])
        self.head_position = new_head
        
        apple_eaten = False
        if np.array_equal(new_head, self.apple_position):
            self.score += 1
            self.apple_position = self._generate_apple()
            apple_eaten = True
        else:
            self.snake_positions = self.snake_positions[1:]
        
        self._update_board()
        self.steps += 1
        
        return True, apple_eaten
    
    def _check_collision(self, position):
        """Check if a position collides with wall or snake body."""
        x, y = position
        wall_collision = (x < 0 or x >= self.grid_size or 
                        y < 0 or y >= self.grid_size)
        
        body_collision = False
        for pos in self.snake_positions[:-1]:
            if np.array_equal(position, pos):
                body_collision = True
                break
                
        return wall_collision, body_collision
    
    def _get_current_direction_key(self):
        """Get the string key for the current direction."""
        if self.current_direction is None:
            return None
        for key, value in DIRECTIONS.items():
            if np.array_equal(self.current_direction, value):
                return key
        return None
    
    def get_state(self):
        """Get the current game state."""
        return {
            'board': self.board.copy(),
            'head_position': self.head_position.copy(),
            'apple_position': self.apple_position.copy(),
            'score': self.score,
            'steps': self.steps
        }