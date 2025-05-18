"""
Main Snake game module.
Handles game logic, state management, and interaction with the LLM agent.
"""

import numpy as np
import re
import pygame
from gui import DrawWindow
from config import SNAKE_SIZE, DIRECTIONS, PROMPT_TEMPLATE_TEXT

class SnakeGame:
    """Main class for the Snake game logic and rendering."""
    
    def __init__(self, row=SNAKE_SIZE):
        """Initialize the Snake game.
        
        Args:
            row: Number of rows/columns in the game grid (default is SNAKE_SIZE from config)
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
        
        # Check for collision with wall
        if (new_head[0] < 0 or new_head[0] >= self.row or 
            new_head[1] < 0 or new_head[1] >= self.row):
            print(f"Game over! Snake hit wall moving {direction_key}")
            return False, False  # Game over, no apple eaten
        
        # Check for collision with self (except tail which will move)
        for pos in self.snake_positions[:-1]:  # Skip the tail
            if np.array_equal(new_head, pos):
                print(f"Game over! Snake hit itself moving {direction_key}")
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
        """Generate a variable-based representation of the game state.
        
        Returns:
            A string representing the game state for the LLM prompt
        """
        # Get head position - (row, col) format
        head_y, head_x = self.head_position
        head_pos = f"({head_y}, {head_x})"
        
        # Get current direction
        if self.current_direction is None:
            current_direction = "NONE"
        else:
            current_direction = self._get_current_direction_key()
        
        # Get body cells (excluding head)
        body_cells = []
        for y, x in self.snake_positions[:-1]:  # Exclude the head
            body_cells.append(f"({y}, {x})")
        body_cells_str = "[" + ", ".join(body_cells) + "]"
        
        # Get apple position
        apple_y, apple_x = self.apple_position
        apple_pos = f"({apple_y}, {apple_x})"
        
        # Create a prompt from the template text using string replacements
        prompt = PROMPT_TEMPLATE_TEXT
        prompt = prompt.replace("HEAD_POS", head_pos)
        prompt = prompt.replace("DIRECTION", current_direction)
        prompt = prompt.replace("BODY_CELLS", body_cells_str)
        prompt = prompt.replace("APPLE_POS", apple_pos)
        prompt = prompt.replace("SIZE", str(SNAKE_SIZE))
        
        # Add current score and steps
        prompt += f"\nCurrent Score: {self.score}\nSteps Taken: {self.steps}\n"
        
        return prompt
    
    def parse_llm_response(self, response):
        """Parse the LLM's response to extract multiple sequential moves.
        
        Args:
            response: Text response from the LLM in JSON format
            
        Returns:
            The next move to make as a direction key string ("UP", "DOWN", "LEFT", "RIGHT")
        """
        # Store the raw response for display
        self.last_llm_response = response
        
        # Process the response for display
        self._process_response_for_display(response)
        
        # Default direction if parsing fails
        default_direction = "RIGHT" if self.current_direction is None else self._get_current_direction_key()
        
        # Clear previous planned moves
        self.planned_moves = []
        
        # Print raw response for debugging
        print(f"Parsing LLM response: '{response[:50]}...'")
        
        # Try to parse JSON response
        try:
            # Look for JSON structure in the response
            import json
            import re
            
            # First try to extract JSON using a more flexible approach
            json_match = re.search(r'\{.*?"moves"\s*:\s*\[.*?\].*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Try to parse the JSON
                try:
                    json_data = json.loads(json_str)
                    if "moves" in json_data and isinstance(json_data["moves"], list):
                        moves = json_data["moves"]
                        # Validate and convert moves to uppercase
                        valid_moves = [move.upper() for move in moves 
                                     if isinstance(move, str) and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
                        if valid_moves:
                            self.planned_moves = valid_moves
                            print(f"Found {len(self.planned_moves)} moves in JSON: {self.planned_moves}")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {e}, falling back to regex patterns")
            
            # If we still don't have moves, try another JSON pattern
            if not self.planned_moves:
                # Look for arrays of directions in quotes
                move_arrays = re.findall(r'\[\s*"([^"]+)"\s*(?:,\s*"([^"]+)"\s*)*\]', response)
                if move_arrays:
                    for move_group in move_arrays:
                        for move in move_group:
                            if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                                self.planned_moves.append(move.upper())
                    if self.planned_moves:
                        print(f"Found {len(self.planned_moves)} moves in array format: {self.planned_moves}")
                
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            
        # If JSON parsing failed, fall back to regex patterns
        if not self.planned_moves:
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
        
        # If we still have no moves, use default
        if not self.planned_moves:
            # Fallback to default
            self.planned_moves = [default_direction]
            print(f"No valid directions found, using default: {default_direction}")
        
        # Filter out invalid reversal moves before returning
        if len(self.planned_moves) > 1 and self.current_direction is not None:
            filtered_moves = []
            last_direction = self._get_current_direction_key()
            
            for move in self.planned_moves:
                # Skip if this move would be a reversal of the last direction
                if (last_direction == "UP" and move == "DOWN") or \
                   (last_direction == "DOWN" and move == "UP") or \
                   (last_direction == "LEFT" and move == "RIGHT") or \
                   (last_direction == "RIGHT" and move == "LEFT"):
                    print(f"Filtering out invalid reversal move: {move} after {last_direction}")
                else:
                    filtered_moves.append(move)
                    last_direction = move
            
            # Update planned moves with filtered list
            if filtered_moves:
                self.planned_moves = filtered_moves
                print(f"After filtering reversals: {len(self.planned_moves)} moves: {self.planned_moves}")
            else:
                # If all moves were filtered out, use default
                self.planned_moves = [default_direction]
                print(f"All moves were invalid reversals, using default: {default_direction}")
        
        # Get the next move from the sequence (or default if empty)
        if self.planned_moves:
            next_move = self.planned_moves.pop(0)
            return next_move
        else:
            return default_direction
            
    def _process_response_for_display(self, response):
        """Process the LLM response for display purposes.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Processed response text ready for display
        """
        # Extract just the generated code part if possible
        if "GENERATED_CODE:" in response:
            processed = response.split("GENERATED_CODE:", 1)[1].strip()
        else:
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
        # Nothing to do for now, state is fully updated after each move
        pass
    
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