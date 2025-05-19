"""
Main Snake game module.
Handles game logic, state management, and interaction with the LLM agent.
"""

import numpy as np
import re
import json
import pygame
from gui import DrawWindow
from config import SNAKE_SIZE, DIRECTIONS, PROMPT_TEMPLATE_TEXT

class SnakeGame:
    """Main class for the Snake game logic and rendering."""
    
    def __init__(self, grid_size=SNAKE_SIZE):
        """Initialize the Snake game.
        
        Args:
            grid_size: Number of cells in each dimension of the game grid (default is SNAKE_SIZE from config)
        """
        # Game state variables
        self.grid_size = grid_size
        self.board = np.zeros((grid_size, grid_size))
        self.snake_positions = np.array([[grid_size//2, grid_size//2]])  # Start in middle, [x, y]
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
        
        # LLM interaction state
        self.planned_moves = []
        self.last_llm_response = ""
        self.processed_response = ""
        
        # Initialize the UI
        pygame.display.set_caption("LLM Snake Agent")
        self.window = DrawWindow()
        
        # Generate the first apple and initialize the board
        self.apple_position = self._generate_apple()
        self._update_board()
        
        # Verify coordinate system
        self._verify_coordinate_system()
    
    #-----------------------
    # Game State Management
    #-----------------------
    
    def reset(self):
        """Reset the game to the initial state.
        
        Returns:
            String representation of the game state
        """
        # Reset game state
        self.snake_positions = np.array([[self.grid_size//2, self.grid_size//2]])  # Start in middle, [x, y]
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
        
        # Update the board and redraw
        self._update_board()
        self.draw()
        
        # Return the current state representation for LLM
        return self.get_state_representation()
    
    def update(self):
        """Update the game state.
        Note: Currently, the state is fully updated after each move.
        """
        pass
    
    def draw(self):
        """Draw the current game state."""
        # Draw the board with snake and apple
        self.window.draw_board(self.board, self.board_info, self.head_position)
        
        # Draw game info and LLM response
        self.window.draw_game_info(
            score=self.score, 
            steps=self.steps, 
            planned_moves=self.planned_moves,
            llm_response=self.processed_response
        )
    
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
    
    #-----------------------
    # Movement & Collision
    #-----------------------
    
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
        
        # Calculate new head position according to our coordinate system:
        # In config.py and prompt, we define:
        # UP = (0, 1) → increases y
        # DOWN = (0, -1) → decreases y
        # RIGHT = (1, 0) → increases x
        # LEFT = (-1, 0) → decreases x
        head_x, head_y = self.head_position
        
        # Apply direction vector to head position
        # direction[0] affects x-coordinate
        # direction[1] affects y-coordinate
        new_head = np.array([
            head_x + direction[0],  # Apply dx to x-coordinate
            head_y + direction[1]   # Apply dy to y-coordinate
        ])
        
        # Debug log
        print(f"Moving {direction_key}: Head from ({head_x}, {head_y}) to ({new_head[0]}, {new_head[1]})")
        
        # Validate move follows coordinate system
        self._validate_move(self.head_position, new_head, direction_key)
        
        # Check for collisions
        wall_collision, body_collision = self._check_collision(new_head)
        
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
    
    def _check_collision(self, position):
        """Check if a position collides with wall or snake body.
        
        Args:
            position: Position to check as [x, y]
            
        Returns:
            Tuple of (collides_with_wall, collides_with_body)
        """
        x, y = position
        
        # Check for collision with wall
        wall_collision = (x < 0 or x >= self.grid_size or 
                        y < 0 or y >= self.grid_size)
        
        # Check for collision with self (except tail which will move)
        body_collision = False
        for pos in self.snake_positions[:-1]:  # Skip the tail
            if np.array_equal(position, pos):
                body_collision = True
                break
                
        return wall_collision, body_collision
    
    def _get_current_direction_key(self):
        """Get the string key for the current direction.
        
        Returns:
            Direction key ("UP", "DOWN", "LEFT", "RIGHT") or "NONE" if no direction is set
        """
        if self.current_direction is None:
            return "NONE"
            
        for key, value in DIRECTIONS.items():
            if np.array_equal(self.current_direction, value):
                return key
        
        # If direction doesn't match any known direction (shouldn't happen),
        # return a safe default
        print("Warning: Unknown direction vector, defaulting to RIGHT")
        return "RIGHT"
    
    #-----------------------
    # Coordinate System
    #-----------------------
    
    def _verify_coordinate_system(self):
        """Verify the coordinate system is consistent.
        
        Logs details of the coordinate system to ensure the game is set up correctly
        according to our configuration.
        """
        print("\n==== COORDINATE SYSTEM VERIFICATION ====")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        
        print("\nDirection vectors (dx, dy):")
        for dir_name, dir_vector in DIRECTIONS.items():
            dx, dy = dir_vector
            print(f"  {dir_name}: ({dx}, {dy}) - x changes by {dx}, y changes by {dy}")
        
        print("\nCoordinate system rules:")
        print("  • Game uses [x, y] format for all coordinates")
        print(f"  • Origin (0,0) is at the BOTTOM-LEFT of the grid")
        print(f"  • First element affects x-axis (LEFT/RIGHT)")
        print(f"  • Second element affects y-axis (UP/DOWN)")
        print(f"  • UP: y increases (moves toward y={self.grid_size-1})")
        print(f"  • DOWN: y decreases (moves toward y=0)")
        print(f"  • RIGHT: x increases (moves toward x={self.grid_size-1})")
        print(f"  • LEFT: x decreases (moves toward x=0)")
        
        # Verify with test moves
        print("\nTest moves from current position:")
        x, y = self.head_position
        print(f"  Current head position [x,y]: [{x},{y}]")
        
        for dir_name, dir_vector in DIRECTIONS.items():
            dx, dy = dir_vector
            new_x = x + dx
            new_y = y + dy
            print(f"  {dir_name}: [{x},{y}] → [{new_x},{new_y}] (applying vector {dir_vector})")
        
        print("=========================================\n")
    
    def _validate_move(self, current_pos, new_pos, direction_key):
        """Validate that a move follows the coordinate system rules.
        
        Args:
            current_pos: Current position as [x, y]
            new_pos: New position after move as [x, y]
            direction_key: Direction key string (UP, DOWN, LEFT, RIGHT)
            
        Returns:
            Boolean indicating if the move is valid according to coordinate system
        """
        valid = True
        head_x, head_y = current_pos
        new_x, new_y = new_pos
        
        # Validate the move follows our coordinate system rules
        if direction_key == "UP" and new_y <= head_y:
            print(f"Warning: UP move should increase y-coordinate but didn't: {head_y} → {new_y}")
            valid = False
        elif direction_key == "DOWN" and new_y >= head_y:
            print(f"Warning: DOWN move should decrease y-coordinate but didn't: {head_y} → {new_y}")
            valid = False
        elif direction_key == "RIGHT" and new_x <= head_x:
            print(f"Warning: RIGHT move should increase x-coordinate but didn't: {head_x} → {new_x}")
            valid = False
        elif direction_key == "LEFT" and new_x >= head_x:
            print(f"Warning: LEFT move should decrease x-coordinate but didn't: {head_x} → {new_x}")
            valid = False
            
        return valid
    
    #-----------------------
    # LLM Interaction
    #-----------------------
    
    def get_state_representation(self):
        """Generate a variable-based representation of the game state.
        
        Returns:
            A string representing the game state for the LLM prompt that follows the template
            defined in config.py, with variables replaced with actual game state values.
        """
        # Get head position in (x, y) format for prompt
        head_x, head_y = self.head_position
        head_pos = f"({head_x}, {head_y})"
        
        # Get current direction
        if self.current_direction is None:
            current_direction = "NONE"
        else:
            current_direction = self._get_current_direction_key()
        
        # Get body cells (excluding head)
        body_cells = []
        for x, y in self.snake_positions[:-1]:  # Exclude the head
            body_cells.append(f"({x}, {y})")
        body_cells_str = "[" + ", ".join(body_cells) + "]"
        
        # Get apple position
        apple_x, apple_y = self.apple_position
        apple_pos = f"({apple_x}, {apple_y})"
        
        # Create a prompt from the template text using string replacements
        prompt = PROMPT_TEMPLATE_TEXT
        prompt = prompt.replace("HEAD_POS", head_pos)
        prompt = prompt.replace("CURRENT_DIRECTION", current_direction)
        prompt = prompt.replace("BODY_CELLS", body_cells_str)
        prompt = prompt.replace("APPLE_POS", apple_pos)
        
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
        
        # Print raw response snippet for debugging
        print(f"Parsing LLM response: '{response[:50]}...'")
        
        # Method 1: Try to extract from JSON code block
        json_data = self._extract_json_from_code_block(response)
        
        # Method 2: Try to extract JSON from regular text if code block fails
        if not json_data or "moves" not in json_data or not json_data["moves"]:
            json_data = self._extract_json_from_text(response)
            
        # Extract moves from JSON if found
        if json_data and "moves" in json_data and isinstance(json_data["moves"], list):
            valid_moves = [move.upper() for move in json_data["moves"] 
                         if isinstance(move, str) and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
            if valid_moves:
                self.planned_moves = valid_moves
                print(f"Found {len(self.planned_moves)} moves in JSON: {self.planned_moves}")
        
        # Method 3: Try regex patterns if JSON parsing failed
        if not self.planned_moves:
            text_moves = self._extract_moves_from_text(response)
            if text_moves:
                self.planned_moves = text_moves
        
        # Method 4: Try finding arrays if other methods failed
        if not self.planned_moves:
            array_moves = self._extract_moves_from_arrays(response)
            if array_moves:
                self.planned_moves = array_moves
                print(f"Found {len(self.planned_moves)} moves in array format: {self.planned_moves}")
        
        # If we still have no moves, use default
        if not self.planned_moves:
            # Fallback to default
            self.planned_moves = [default_direction]
            print(f"No valid directions found, using default: {default_direction}")
        
        # Filter out invalid reversal moves
        self.planned_moves = self._filter_invalid_reversals(self.planned_moves)
        
        # Get the next move from the sequence (or default if empty)
        if self.planned_moves:
            next_move = self.planned_moves.pop(0)
            return next_move
        else:
            return default_direction
    
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
    
    def get_display_response(self):
        """Get the processed LLM response for display.
        
        Returns:
            Processed LLM response text
        """
        return self.processed_response
    
    #-----------------------
    # LLM Response Parsing Utilities
    #-----------------------
    
    def _extract_json_from_code_block(self, response):
        """Extract JSON data from a code block in the response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted JSON data as a dictionary, or None if not found/invalid
        """
        try:
            # Look for JSON code block which is common in LLM responses
            json_block_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
            if not json_block_match:
                return None
                
            json_str = json_block_match.group(1)
            # Clean the JSON string
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            # Remove any trailing commas before closing brackets or braces
            json_str = re.sub(r',\s*(\]|\})', r'\1', json_str)
            
            data = json.loads(json_str)
            return data
        except Exception as e:
            print(f"Failed to parse JSON code block: {e}")
            return None
    
    def _extract_json_from_text(self, response):
        """Extract JSON data directly from text without code block.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted JSON data as a dictionary, or None if not found/invalid
        """
        try:
            # Try extracting JSON object outside of code blocks
            json_match = re.search(r'\{.*?"moves"\s*:\s*\[.*?\].*?\}', response, re.DOTALL)
            if not json_match:
                return None
                
            json_str = json_match.group(0)
            # Clean the JSON string
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            # Remove any trailing commas before closing brackets or braces
            json_str = re.sub(r',\s*(\]|\})', r'\1', json_str)
            
            # Now try to parse the cleaned JSON
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}, trying simplified parsing")
            
            # Try to extract just the moves array
            try:
                moves_array_match = re.search(r'"moves"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
                if not moves_array_match:
                    return None
                    
                moves_array = moves_array_match.group(1)
                # Extract quoted strings
                move_matches = re.findall(r'"([^"]+)"', moves_array)
                valid_moves = [move.upper() for move in move_matches 
                              if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
                if valid_moves:
                    return {"moves": valid_moves}
            except Exception:
                pass
                
            return None
        except Exception as e:
            print(f"Error in JSON parsing: {e}")
            return None
    
    def _extract_moves_from_text(self, response):
        """Extract moves from plain text using regex patterns.
        
        Args:
            response: LLM response text
            
        Returns:
            List of extracted moves, or empty list if none found
        """
        moves = []
        
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
            moves = [direction for _, direction in combined_list]
            print(f"Found {len(moves)} moves in the numbered list: {moves}")
        # If no numbered list, use the simple list
        elif simple_list:
            moves = [direction.upper() for direction in simple_list]
            print(f"Found {len(moves)} moves in simple list: {moves}")
            
        return moves
    
    def _extract_moves_from_arrays(self, response):
        """Extract moves from array notation in text.
        
        Args:
            response: LLM response text
            
        Returns:
            List of extracted moves, or empty list if none found
        """
        moves = []
        
        # Look for arrays of directions in quotes
        move_arrays = re.findall(r'\[\s*"([^"]+)"\s*(?:,\s*"([^"]+)"\s*)*\]', response)
        if move_arrays:
            for move_group in move_arrays:
                for move in move_group:
                    if move and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                        moves.append(move.upper())
                        
        return moves
    
    def _filter_invalid_reversals(self, moves):
        """Filter out invalid reversal moves from a sequence.
        
        Args:
            moves: List of move directions
            
        Returns:
            Filtered list of moves with invalid reversals removed
        """
        if not moves or len(moves) <= 1:
            return moves
            
        filtered_moves = []
        last_direction = self._get_current_direction_key()
        
        for move in moves:
            # Skip if this move would be a reversal of the last direction
            if (last_direction == "UP" and move == "DOWN") or \
               (last_direction == "DOWN" and move == "UP") or \
               (last_direction == "LEFT" and move == "RIGHT") or \
               (last_direction == "RIGHT" and move == "LEFT"):
                print(f"Filtering out invalid reversal move: {move} after {last_direction}")
            else:
                filtered_moves.append(move)
                last_direction = move
        
        # If all moves were filtered out, use default
        if not filtered_moves and self.current_direction is not None:
            filtered_moves = [self._get_current_direction_key()]
            print(f"All moves were invalid reversals, using default: {filtered_moves[0]}")
            
        return filtered_moves
    
    def _process_response_for_display(self, response):
        """Process the LLM response for display purposes.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Processed response text ready for display
        """
        try:
            # Extract just the generated code part if possible
            if "GENERATED_CODE:" in response:
                processed = response.split("GENERATED_CODE:", 1)[1].strip()
            else:
                processed = response
                
            # Limit to a reasonable length for display
            if len(processed) > 1500:
                processed = processed[:1500] + "...\n(response truncated)"
                
            self.processed_response = processed
            return self.processed_response
        except Exception as e:
            print(f"Error processing response for display: {e}")
            self.processed_response = "Error processing response"
            return self.processed_response 