"""
Main Snake game module.
Handles game logic, state management, and interaction with the LLM agent.
"""

import numpy as np
import re
import pygame
from gui import DrawWindow
from config import ROW, DIRECTIONS

class SnakeGame:
    """Main class for the Snake game logic and rendering."""
    
    def __init__(self, row=ROW):
        """Initialize the Snake game.
        
        Args:
            row: Number of rows/columns in the game grid
        """
        # Game state variables
        self.row = row
        self.board = np.zeros((row, row))
        self.snake_positions = np.array([[row//2, row//2]])  # Start in middle
        self.head_position = self.snake_positions[-1]
        self.apple_position = self._generate_apple()
        self.current_direction = None
        self.steps = 0
        self.score = 0
        self.generation = 1
        
        # Game board entity codes
        self.board_info = {
            "empty": 0,
            "snake": 1,
            "apple": 2
        }
        
        # Store the planned sequence of moves from LLM
        self.planned_moves = []
        self.last_llm_response = ""
        self.processed_response = ""  # For display purposes
        
        # Initialize the UI
        pygame.display.set_caption("LLM Snake Agent")
        self.window = DrawWindow()
        
        # Place the snake and apple on the board
        self._update_board()
    
    def _generate_apple(self):
        """Generate a new apple at a random empty position.
        
        Returns:
            Tuple of (y, x) coordinates for the new apple
        """
        while True:
            # Generate random position
            y, x = np.random.randint(0, self.row, 2)
            
            # Check if position is empty (not occupied by snake)
            if not any(np.array_equal([y, x], pos) for pos in self.snake_positions):
                return np.array([y, x])
    
    def _update_board(self):
        """Update the game board with current snake and apple positions."""
        # Clear the board
        self.board.fill(self.board_info["empty"])
        
        # Place the snake
        for y, x in self.snake_positions:
            self.board[y, x] = self.board_info["snake"]
        
        # Place the apple
        self.board[self.apple_position[0], self.apple_position[1]] = self.board_info["apple"]
    
    def _check_collision(self, position, is_eating_apple_flag):
        """Check if a position collides with the walls or snake body.
        
        Args:
            position: [y, x] position to check
            is_eating_apple_flag: Boolean indicating if this move would eat an apple
            
        Returns:
            Tuple of (wall_collision, body_collision) booleans
        """
        y, x = position
        
        # Check wall collision
        wall_collision = (y < 0 or y >= self.row or x < 0 or x >= self.row)
        
        # Default to no collision
        body_collision = False
        
        # Handle empty snake case (shouldn't happen normally)
        if len(self.snake_positions) == 0:
            return wall_collision, False
        
        # Get current snake structure for clarity
        current_tail = self.snake_positions[0]  # First position is tail
        current_head = self.snake_positions[-1]  # Last position is head
        
        if is_eating_apple_flag:
            # CASE: Eating an apple - tail will NOT move
            # Check collision with all segments EXCEPT the current head
            # (since the head will move to the new position)
            
            # Check all segments except the head
            body_segments = self.snake_positions[:-1]  # [tail, body1, body2, ..., bodyN]
            
            # Check if new head position collides with any body segment (including tail)
            body_collision = any(np.array_equal(position, pos) for pos in body_segments)
            
        else:
            # CASE: Normal move (not eating apple) - tail WILL move
            # Only need to check for collision with body segments, excluding both
            # the current tail (which will move) and the current head (which will be replaced)
            
            if len(self.snake_positions) > 2:
                # If snake has body segments between tail and head
                # Check segments excluding tail and head: [body1, body2, ..., bodyN]
                body_segments = self.snake_positions[1:-1]
                body_collision = any(np.array_equal(position, pos) for pos in body_segments)
            else:
                # Snake has only head and tail (or just head), no body segments to collide with
                body_collision = False
        
        return wall_collision, body_collision
    
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
            print(f"Invalid direction from LLM: {direction_key}, defaulting to RIGHT")
            # Use default of RIGHT if the LLM returns an invalid direction
            direction_key = "RIGHT"
        
        direction = DIRECTIONS[direction_key]
        
        # Don't allow reversing direction directly
        if (self.current_direction is not None and 
            np.array_equal(np.array(direction), -np.array(self.current_direction))):
            print(f"LLM tried to reverse direction: {direction_key}. Using current direction instead.")
            # Trying to reverse direction, use current direction instead
            direction = self.current_direction
            direction_key = self._get_current_direction_key()
        
        # Update current direction
        self.current_direction = direction
        
        # Calculate new head position
        head_y, head_x = self.head_position
        new_head = np.array([head_y + direction[1], head_x + direction[0]])
        
        # Check if the new head position is where the apple is
        is_eating_apple = np.array_equal(new_head, self.apple_position)
        
        # Check for collisions - pass the apple flag to handle collisions correctly
        wall_collision, body_collision = self._check_collision(new_head, is_eating_apple_flag=is_eating_apple)
        
        # Check if game is over due to collision
        if wall_collision:
            print(f"Game over! Snake hit wall moving {direction_key}")
            return False, False  # Game over, no apple eaten
        
        if body_collision:
            print(f"Game over! Snake hit itself moving {direction_key}")
            return False, False  # Game over, no apple eaten
        
        # Move the snake: add new head
        self.snake_positions = np.vstack([self.snake_positions, new_head])
        self.head_position = new_head
        
        # Check if apple is eaten
        apple_eaten = False
        if is_eating_apple:
            self.score += 1
            print(f"Apple eaten! Score: {self.score}")
            # Generate a new apple
            self.apple_position = self._generate_apple()
            apple_eaten = True
        else:
            # Remove the tail if no apple is eaten
            self.snake_positions = self.snake_positions[1:]
        
        # Update the board
        self._update_board()
        
        # Increment steps
        self.steps += 1
        
        return True, apple_eaten  # Game continues, indicates if apple was eaten
    
    def get_state_representation(self):
        """Generate a text representation of the game state.
        
        Returns:
            String representation of the game board
        """
        # Create coordinate header
        board_str = "  "
        # Add top wall with column numbers
        board_str += "+".join(["-" for _ in range(self.row+1)]) + "\n"
        board_str += "  " + " ".join([f"{x}" for x in range(self.row)]) + "\n"
        
        # Create the board with row numbers and side walls
        for y in range(self.row):
            board_str += f"{y}|"  # Left wall with row number
            for x in range(self.row):
                if self.board[y, x] == self.board_info["empty"]:
                    board_str += ". "  # Empty space
                elif self.board[y, x] == self.board_info["snake"]:
                    # Check if this is the head
                    if np.array_equal([y, x], self.head_position):
                        # Show direction of the head
                        if self.current_direction is None:
                            board_str += "H "  # Head with no direction
                        elif np.array_equal(self.current_direction, DIRECTIONS["UP"]):
                            board_str += "H↑"  # Head facing up
                        elif np.array_equal(self.current_direction, DIRECTIONS["RIGHT"]):
                            board_str += "H→"  # Head facing right
                        elif np.array_equal(self.current_direction, DIRECTIONS["DOWN"]):
                            board_str += "H↓"  # Head facing down
                        elif np.array_equal(self.current_direction, DIRECTIONS["LEFT"]):
                            board_str += "H←"  # Head facing left
                    else:
                        board_str += "S "  # Snake body (changed from # to S for clarity)
                elif self.board[y, x] == self.board_info["apple"]:
                    board_str += "A "  # Apple (changed from @ to A for clarity)
            board_str += "|" + "\n"  # Right wall
        
        # Add bottom wall
        board_str += "  " + "+".join(["-" for _ in range(self.row+1)]) + "\n"
        
        # Add a legend
        board_str += "\nLegend:\n"
        board_str += ". = Empty space\n"
        board_str += "S = Snake body\n"
        board_str += "A = Apple\n"
        board_str += "H = Snake head (no direction, hence can go any direction)\n"
        board_str += "H↑ = Snake head (facing UP)\n"
        board_str += "H→ = Snake head (facing RIGHT)\n"
        board_str += "H↓ = Snake head (facing DOWN)\n"
        board_str += "H← = Snake head (facing LEFT)\n"
        board_str += "| and - = Walls (game boundary)\n"
        
        # Add coordinate explanation
        board_str += "\nCoordinates are (row, column) with (0,0) at the top-left corner\n"
        
        # Add head and apple positions for clarity
        head_y, head_x = self.head_position
        apple_y, apple_x = self.apple_position
        board_str += f"\nSnake head position: ({head_y}, {head_x})\n"
        board_str += f"Apple position: ({apple_y}, {apple_x})\n"
        
        # Add directional guidance
        board_str += "\nDirectional guide:\n"
        board_str += "- UP: Decreases row coordinate (moves toward row 0)\n"
        board_str += f"- DOWN: Increases row coordinate (moves toward row {self.row - 1})\n"
        board_str += "- LEFT: Decreases column coordinate (moves toward column 0)\n"
        board_str += f"- RIGHT: Increases column coordinate (moves toward column {self.row - 1})\n"
        
        return board_str
    
    def parse_llm_response(self, response):
        """Parse the LLM's response to extract multiple sequential moves.
        
        Args:
            response: Text response from the LLM
            
        Returns:
            The next move to make as a direction key string ("UP", "DOWN", "LEFT", "RIGHT")
            or None if no valid moves were found
        """
        # Store the raw response for display
        self.last_llm_response = response
        
        # Process the response for display
        self._process_response_for_display(response)
        
        # Clear previous planned moves
        self.planned_moves = []
        
        # Print raw response for debugging
        print(f"Parsing LLM response: '{response[:50]}...'")
        
        # Look for numbered lists of directions using different patterns
        # Pattern 1: numbered list (1. UP\n2. RIGHT)
        numbered_list = re.findall(r'(\d+)\.?\s+(UP|DOWN|LEFT|RIGHT)', response, re.IGNORECASE)
        
        # Pattern 2: steps labeled with "Step X: DIRECTION"
        step_pattern = re.findall(r'Step\s+(\d+):\s+(UP|DOWN|LEFT|RIGHT)', response, re.IGNORECASE)
        
        # Pattern 3: simple directions separated by commas or newlines
        simple_list = re.findall(r'\b(UP|DOWN|LEFT|RIGHT)\b', response, re.IGNORECASE)
        
        # Combine and sort the numbered lists
        combined_list = []
        
        # Add numbered list entries
        for num_str, direction in numbered_list:
            try:
                num = int(num_str)
                combined_list.append((num, direction.upper()))
            except ValueError:
                pass
                
        # Add step pattern entries
        for num_str, direction in step_pattern:
            try:
                num = int(num_str)
                combined_list.append((num, direction.upper()))
            except ValueError:
                pass
        
        # Sort by number
        combined_list.sort(key=lambda x: x[0])
        
        # Extract directions from sorted list
        if combined_list:
            self.planned_moves = [direction for _, direction in combined_list]
            print(f"Found {len(self.planned_moves)} moves in the numbered list: {self.planned_moves}")
        # If no numbered list, use the simple list
        elif simple_list:
            self.planned_moves = [direction.upper() for direction in simple_list]
            print(f"Found {len(self.planned_moves)} moves in simple list: {self.planned_moves}")
        else:
            # No valid moves found
            print("No valid directions found. Not moving.")
        
        # Get the next move from the sequence (or None if empty)
        if self.planned_moves:
            return self.planned_moves.pop(0)
        return None
            
    def _process_response_for_display(self, response):
        """Process the LLM response for display purposes.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Processed response text ready for display
        """
        processed = response
            
        # Limit to a reasonable length for display
        self.processed_response = processed[:1000]
        return self.processed_response
        
    def get_display_response(self):
        """Get the processed LLM response for display.
        
        Returns:
            Processed LLM response text
        """
        return self.processed_response
        
    def get_next_planned_move(self):
        """Get the next move from the planned sequence.
        
        Returns:
            Next direction or None if no more planned moves
        """
        if self.planned_moves:
            return self.planned_moves.pop(0)
        return None
        
    def has_planned_moves(self):
        """Check if there are still planned moves available.
        
        Returns:
            Boolean indicating if there are more planned moves
        """
        return len(self.planned_moves) > 0
    
    def _get_current_direction_key(self):
        """Get the string key for the current direction.
        
        Returns:
            Direction key ("UP", "DOWN", "LEFT", "RIGHT")
        """
        for key, value in DIRECTIONS.items():
            if np.array_equal(self.current_direction, value):
                return key
        return "RIGHT"  # Default
    
    def update(self):
        """Update the game state."""
        # State is fully updated after each move; no additional logic required at present.
    
    def draw(self):
        """Draw the current game state."""
        # Draw the board
        self.window.draw_board(self.board, self.board_info, self.head_position)
        
        # Draw game info and LLM response
        self.window.draw_game_info(
            score=self.score, 
            steps=self.steps, 
            planned_moves=self.planned_moves,
            llm_response=self.processed_response
        )
    
    def reset(self):
        """Reset the game to the initial state."""
        # Reset game state
        self.snake_positions = np.array([[self.row//2, self.row//2]])  # Start in middle
        self.head_position = self.snake_positions[-1]
        self.apple_position = self._generate_apple()
        self.current_direction = None
        self.steps = 0
        self.score = 0
        self.generation += 1
        self.planned_moves = []
        self.processed_response = ""
        
        # Clear any key states that might be stuck
        pygame.event.clear()
        
        # Update the board
        self._update_board()
        
        # Redraw the screen
        self.draw()
        
        # Return the current state representation
        return self.get_state_representation() 