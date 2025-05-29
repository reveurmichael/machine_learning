"""
Snake game implementation with LLM integration.
Extends the base game controller with LLM-specific functionality.
"""

import traceback
from core.game_controller import GameController
from llm.prompt_utils import prepare_snake_prompt
from llm.parsing_utils import parse_llm_response
from utils.text_utils import process_response_for_display
from config import GRID_SIZE

class GameLogic(GameController):
    """Snake game with LLM agent integration."""
    
    def __init__(self, grid_size=None, use_gui=True):
        """Initialize the snake game.
        
        Args:
            grid_size: Size of the game grid (default from config)
            use_gui: Whether to use GUI for display
        """
        # Use default grid size from config if none provided
        if grid_size is None:
            grid_size = GRID_SIZE
            
        super().__init__(grid_size, use_gui)
        
        # LLM interaction state
        self.planned_moves = []
        self.processed_response = ""
    
    @property
    def head(self):
        """Get the current head position.
        
        Returns:
            Tuple of (x, y) coordinates of the snake's head
        """
        return tuple(self.head_position)
    
    @property
    def apple(self):
        """Get the current apple position.
        
        Returns:
            Tuple of (x, y) coordinates of the apple
        """
        return tuple(self.apple_position)
    
    @property
    def body(self):
        """Get the snake body positions (excluding head).
        
        Returns:
            List of (x, y) tuples for body segments
        """
        # Convert each position to a tuple and exclude the head (last element)
        return [tuple(pos) for pos in self.snake_positions[:-1]]
    
    def draw(self):
        """Draw the current game state if GUI is available."""
        if self.use_gui and self.gui:
            self.gui.draw_board(self.board, self.board_info, self.head_position)
            
            # Create game info dictionary
            game_info = {
                'score': self.score,
                'steps': self.steps,
                'planned_moves': self.planned_moves,
                'llm_response': self.processed_response
            }
            
            self.gui.draw_game_info(game_info)
    
    def reset(self):
        """Reset the game to the initial state."""
        super().reset()
        self.planned_moves = []
        self.processed_response = ""
        
        # Return the current state representation for LLM
        return self.get_state_representation()
    
    def get_state_representation(self):
        """Generate a representation of the game state for the LLM prompt.
        
        Returns:
            String representation of the game state using the template from config.py
        """
        # Get current direction as a string
        current_direction = self._get_current_direction_key() if self.current_direction is not None else "NONE"
        
        # Use the utility function from llm_utils
        return prepare_snake_prompt(
            head_position=self.head_position,
            body_positions=self.body,
            apple_position=self.apple_position,
            current_direction=current_direction
        )
    
    def parse_llm_response(self, response):
        """Parse the LLM's response to extract multiple sequential moves.
        
        Args:
            response: Text response from the LLM in JSON format
            
        Returns:
            The next move to make as a direction key string ("UP", "DOWN", "LEFT", "RIGHT")
            or None if no valid moves were found
        """
        try:
            return parse_llm_response(response, process_response_for_display, self)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            traceback.print_exc()
            
            # Store the raw response for display
            self.processed_response = f"ERROR: Failed to parse LLM response\n\n{response[:200]}..."
            
            # Clear previous planned moves
            self.planned_moves = []
            
            # Update game state to record error
            self.game_state.record_error_move()
            
            return None
    
    def get_next_planned_move(self):
        """Get the next move from the planned sequence.
        
        Returns:
            Next direction or None if no more planned moves
        """
        if self.planned_moves:
            next_move = self.planned_moves.pop(0)
            return next_move
        return None 