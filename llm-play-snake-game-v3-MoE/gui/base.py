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
        self.info_panel = self.width - self.height
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
        self.font = pygame.font.SysFont("freesansbold.ttf", 24)
        self.small_font = pygame.font.SysFont("freesansbold.ttf", 16)
        
        # Calculate the width of the text panel
        self.text_panel_width = self.width + self.width_plus - self.height - 40
    
    def draw_snake(self, snake_positions, flip_y=False):
        """Draw the snake at the given positions.
        
        Args:
            snake_positions: List of [x,y] positions for snake segments
            flip_y: Whether to flip the y-coordinate (for different coordinate systems)
        """
        for i, position in enumerate(snake_positions):
            x, y = position
            
            # Handle flipping y coordinate if needed
            if flip_y:
                display_y = self.grid_size - 1 - y
            else:
                display_y = y
                
            # Draw rectangle for snake segment
            rect = pygame.Rect(
                x * self.pixel,
                display_y * self.pixel,
                self.pixel - 5,
                self.pixel - 5
            )
            
            # Draw head in different color
            if i == 0:
                pygame.draw.rect(self.screen, COLORS['SNAKE_HEAD'], rect)
            else:
                pygame.draw.rect(self.screen, COLORS['SNAKE_BODY'], rect)
    
    def draw_apple(self, apple_position, flip_y=False):
        """Draw the apple at the given position.
        
        Args:
            apple_position: [x,y] position of the apple
            flip_y: Whether to flip the y-coordinate
        """
        x, y = apple_position
        
        # Handle flipping y coordinate if needed
        if flip_y:
            display_y = self.grid_size - 1 - y
        else:
            display_y = y
            
        # Draw rectangle for apple
        rect = pygame.Rect(
            x * self.pixel,
            display_y * self.pixel,
            self.pixel - 5,
            self.pixel - 5
        )
        
        pygame.draw.rect(self.screen, COLORS['APPLE'], rect)
    
    def draw_walls(self):
        """Draw the walls/borders of the game board."""
        wall_thickness = 2
        
        # Draw four walls
        pygame.draw.rect(
            self.screen,
            COLORS['WHITE'],  # Wall color
            (0, 0, self.height, wall_thickness)  # Top wall
        )
        pygame.draw.rect(
            self.screen,
            COLORS['WHITE'],  # Wall color
            (0, 0, wall_thickness, self.height)  # Left wall
        )
        pygame.draw.rect(
            self.screen,
            COLORS['WHITE'],  # Wall color
            (0, self.height - wall_thickness, self.height, wall_thickness)  # Bottom wall
        )
        pygame.draw.rect(
            self.screen,
            COLORS['WHITE'],  # Wall color
            (self.height - wall_thickness, 0, wall_thickness, self.height)  # Right wall
        )
    
    def clear_game_area(self):
        """Clear the game board area."""
        # Draw background rectangle for game section
        pygame.draw.rect(
            self.screen, 
            COLORS['BACKGROUND'], 
            (0, 0, self.height+1, self.height+1)
        )
    
    def clear_info_panel(self):
        """Clear the information panel area."""
        # Draw background rectangle for info panel
        pygame.draw.rect(
            self.screen,
            COLORS['APP_BG'],
            (self.height+1, 0, self.width+self.width_plus, self.height)
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