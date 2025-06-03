"""
Base GUI components for the Snake game.
Provides common setup and drawing functionality.
"""

import pygame
from config import (
    COLORS, WINDOW_WIDTH, WINDOW_HEIGHT, GRID_SIZE
)

class BaseGUI:
    """Base class for UI setup."""
    
    def __init__(self):
        """Initialize the UI dimensions."""
        self.width = WINDOW_WIDTH
        self.width_plus = 200
        self.height = WINDOW_HEIGHT
        self.info_panel = self.width - self.height  # Used for layout calculations in derived classes
        self.grid_size = GRID_SIZE
        self.pixel = self.height // self.grid_size
        
        # These will be initialized in derived classes
        self.screen = None
        self.font = None
        self.small_font = None
    
    def init_display(self, caption="Snake Game"):
        """Initialize the pygame display.
        
        Args:
            caption: Window caption text
        """
        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()
            
        if not pygame.font.get_init():
            pygame.font.init()
            
        # Set up the screen
        self.screen = pygame.display.set_mode((self.width + self.width_plus, self.height))
        pygame.display.set_caption(caption)
        
        # Set up fonts
        self.font = pygame.font.SysFont("arial", 18)
        self.small_font = pygame.font.SysFont("arial", 12)
        
        # Calculate the width of the text panel
        self.text_panel_width = self.width + self.width_plus - self.height - 40
    
    def draw_apple(self, apple_position, flip_y=False):
        """Draw the apple at the given position.
        
        Args:
            apple_position: [x,y] position of the apple
            flip_y: Whether y-coordinate is already flipped (for GameGUI compatibility)
        """
        x, y = apple_position
        
        # Calculate display position
        # If flip_y is True, y is already flipped in GameGUI, so use it directly
        # Otherwise transform from Cartesian coordinates
        y_display = y if flip_y else (self.grid_size - 1 - y)
            
        # Draw rectangle for apple
        rect = pygame.Rect(
            x * self.pixel,
            y_display * self.pixel,
            self.pixel - 5,
            self.pixel - 5
        )
        
        pygame.draw.rect(self.screen, COLORS['APPLE'], rect)
    
    def clear_game_area(self):
        """Clear the game board area."""
        # Draw background rectangle for game section without any padding
        pygame.draw.rect(
            self.screen, 
            COLORS['BACKGROUND'], 
            (0, 0, self.height, self.height)
        )
    
    def clear_info_panel(self):
        """Clear the information panel area."""
        # Draw background rectangle for info panel without separation line
        pygame.draw.rect(
            self.screen,
            COLORS['APP_BG'],
            (self.height, 0, self.width+self.width_plus-self.height, self.height)
        )
    
    def render_text_area(self, text, x, y, width, height, max_lines=20):
        """Render text inside a scrollable text area.
        
        Args:
            text: Text to render
            x: X position of text area
            y: Y position of text area
            width: Width of text area
            height: Height of text area
            max_lines: Maximum number of lines to render
        """
        # Setup text area with border
        pygame.draw.rect(
            self.screen,
            COLORS['GREY'],  # Light gray background for text area
            (x, y, width, height),
            0,  # Filled rectangle
            3   # Rounded corners
        )
        
        pygame.draw.rect(
            self.screen,
            COLORS['GREY'],  # Darker border
            (x, y, width, height),
            2,  # Border only (not filled)
            3   # Rounded corners
        )
        
        # Calculate text parameters
        avg_char_width = self.small_font.size("m")[0]
        max_chars_per_line = int(width / avg_char_width * 1.7)
        line_height = 20
        padding_left = 10
        
        # Split into lines and render
        y_offset = y + 10  # Starting y position inside the text area
        lines = text.split('\n')
        
        for line in lines[:max_lines]:
            # Skip empty lines
            if not line.strip():
                y_offset += line_height
                continue
                
            # Further split lines that are too long
            for i in range(0, len(line), max_chars_per_line):
                segment = line[i:i+max_chars_per_line]
                if segment:
                    text_surface = self.small_font.render(segment, True, COLORS['BLACK'])
                    self.screen.blit(text_surface, (x + padding_left, y_offset))
                    y_offset += line_height
                    
                    # Stop if we've reached the bottom of the text area
                    if y_offset > y + height - line_height:
                        return 
    
    def draw_game_info(self, game_info):
        """Draw game information overlay.
        
        Args:
            game_info: Dictionary containing game information:
                - score: Current score
                - steps: Current step count
                - planned_moves: List of planned moves
                - llm_response: LLM response text
                - secondary_llm: Name of secondary LLM (optional)
                - paused: Whether the game is paused (optional)
        """
        if not self.screen:
            return
        
        # Get values from the dictionary
        score = game_info.get('score', 0)
        steps = game_info.get('steps', 0)
        planned_moves = game_info.get('planned_moves', [])
        llm_response = game_info.get('llm_response', '')
        secondary_llm = game_info.get('secondary_llm', None)
        paused = game_info.get('paused', False) 