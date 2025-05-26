"""
Snake game implementation with LLM integration.
Extends the base game engine with LLM-specific functionality.
"""

import re
import traceback
from core.game_engine import GameEngine
from utils.snake_utils import calculate_move_differences
from utils.llm_utils import parse_llm_response
from config import PROMPT_TEMPLATE_TEXT

class SnakeGame(GameEngine):
    """Snake game with LLM agent integration."""
    
    def __init__(self, grid_size=None, use_gui=True):
        """Initialize the snake game.
        
        Args:
            grid_size: Size of the game grid (default from config)
            use_gui: Whether to use GUI for display
        """
        super().__init__(grid_size, use_gui)
        
        # LLM interaction state
        self.planned_moves = []
        self.last_llm_response = ""
        self.processed_response = ""
        
        # Performance tracking
        self.response_times = []  # List of primary LLM response times in seconds
        self.secondary_response_times = []  # List of secondary LLM response times in seconds
    
    def set_gui(self, gui_instance):
        """Set the GUI instance to use for display.
        
        Args:
            gui_instance: Instance of a GUI class for the snake game
        """
        super().set_gui(gui_instance)
    
    def draw(self):
        """Draw the current game state if GUI is available."""
        if self.use_gui and self.gui:
            self.gui.draw_board(self.board, self.board_info, self.head_position)
            self.gui.draw_game_info(
                score=self.score,
                steps=self.steps,
                planned_moves=self.planned_moves,
                llm_response=self.processed_response
            )
    
    def reset(self):
        """Reset the game to the initial state."""
        super().reset()
        self.planned_moves = []
        self.last_llm_response = ""
        self.processed_response = ""
        
        # Return the current state representation for LLM
        return self.get_state_representation()
    
    def format_body_cells_str(self, snake_positions, exclude_head=True):
        """Format the snake body cells as a string representation.
        
        Args:
            snake_positions: List of [x, y] coordinates of the snake segments
            exclude_head: Whether to exclude the head from the output (default: True)
            
        Returns:
            String representation of body cells in format: "[(x1,y1), (x2,y2), ...]"
        """
        body_cells = []
        positions = snake_positions[:-1] if exclude_head else snake_positions
        
        # Optionally reverse the positions to start from the segment adjacent to head
        for x, y in reversed(positions):
            body_cells.append(f"({x},{y})")
            
        return "[" + ", ".join(body_cells) + "]"
    
    def get_state_representation(self):
        """Generate a variable-based representation of the game state.
        
        Returns:
            A string representing the game state for the LLM prompt that follows the template
            defined in config.py, with variables replaced with actual game state values.
        """
        # Get head position in (x, y) format for prompt
        head_x, head_y = self.head_position
        head_pos = f"({head_x},{head_y})"
        
        # Get current direction
        if self.current_direction is None:
            current_direction = "NONE"
        else:
            current_direction = self._get_current_direction_key()
        
        # Get body cells (excluding head)
        body_cells_str = self.format_body_cells_str(self.snake_positions)
        
        # Get apple position
        apple_x, apple_y = self.apple_position
        apple_pos = f"({apple_x},{apple_y})"
        
        # Calculate the expected move differences
        move_differences = calculate_move_differences(self.head_position, self.apple_position)
        
        # Create a prompt from the template text using string replacements
        prompt = PROMPT_TEMPLATE_TEXT
        prompt = prompt.replace("TEXT_TO_BE_REPLACED_HEAD_POS", head_pos)
        prompt = prompt.replace("TEXT_TO_BE_REPLACED_CURRENT_DIRECTION", current_direction)
        prompt = prompt.replace("TEXT_TO_BE_REPLACED_BODY_CELLS", body_cells_str)
        prompt = prompt.replace("TEXT_TO_BE_REPLACED_APPLE_POS", apple_pos)
        prompt = prompt.replace("TEXT_TO_BE_REPLACED_ON_THE_TOPIC_OF_MOVES_DIFFERENCE", move_differences)
        
        return prompt
    
    def parse_llm_response(self, response):
        """Parse the LLM's response to extract multiple sequential moves.
        
        Args:
            response: Text response from the LLM in JSON format
            
        Returns:
            The next move to make as a direction key string ("UP", "DOWN", "LEFT", "RIGHT")
            or None if no valid moves were found
        """
        try:
            from utils.text_utils import process_response_for_display
            return parse_llm_response(response, process_response_for_display, self)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            traceback.print_exc()
            
            # Store the raw response for display
            self.last_llm_response = response
            
            # Process a simplified version for display
            self.processed_response = f"ERROR: Failed to parse LLM response\n\n{response[:200]}..."
            
            # Clear previous planned moves
            self.planned_moves = []
            
            return None
    
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
    
    def add_response_time(self, duration):
        """Add a primary LLM response time to the tracking list.
        
        Args:
            duration: Response time duration in seconds
        """
        self.response_times.append(duration)
    
    def add_secondary_response_time(self, duration):
        """Add a secondary LLM response time to the tracking list.
        
        Args:
            duration: Response time duration in seconds
        """
        self.secondary_response_times.append(duration)
    
    def get_average_response_time(self):
        """Get the average primary LLM response time.
        
        Returns:
            Average response time in seconds, or 0 if no responses
        """
        if not self.response_times:
            return 0
        return sum(self.response_times) / len(self.response_times)
    
    def get_average_secondary_response_time(self):
        """Get the average secondary LLM response time.
        
        Returns:
            Average secondary response time in seconds, or 0 if no responses
        """
        if not self.secondary_response_times:
            return 0
        return sum(self.secondary_response_times) / len(self.secondary_response_times)
    
    def get_steps_per_apple(self):
        """Get the average number of steps taken to eat one apple.
        
        Returns:
            Average steps per apple, or 0 if no apples eaten
        """
        if self.score == 0:
            return 0
        return self.steps / self.score 