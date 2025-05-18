"""
GUI module for the Snake game.
Handles drawing the game board and text information.
"""

import pygame
import sys
import time
import numpy as np
from config import (
    SNAKE_C, APPLE_C, BG, APP_BG, GRID_BG, BLACK, WHITE, GREY, GREY2, GREY3,
    APP_WIDTH, APP_HEIGHT, ROW, TIME_DELAY, TIME_TICK, DIRECTIONS
)

# Define a new color for the snake head
SNAKE_HEAD_C = (255, 140, 0)  # Bright orange for the head

# Import after config to avoid circular imports
# Only import when running as main
import importlib.util

class SetUp:
    """Base class for UI setup."""
    
    def __init__(self):
        """Initialize the UI dimensions."""
        self.width = APP_WIDTH
        self.width_plus = 200
        self.height = APP_HEIGHT
        self.info_panel = self.width - self.height
        self.n_row = ROW
        self.pixel = self.height // self.n_row

class DrawWindow(SetUp):
    """Main UI class for drawing the game window."""
    
    def __init__(self):
        """Initialize the drawing window."""
        super().__init__()
        self.screen = pygame.display.set_mode((APP_WIDTH+self.width_plus, APP_HEIGHT))
        self.font = pygame.font.SysFont("freesansbold.ttf", 24)
        self.small_font = pygame.font.SysFont("freesansbold.ttf", 16)
        # Calculate the width of the text panel
        self.text_panel_width = self.width + self.width_plus - self.height - 40  # Account for margins
        
    def draw_board(self, board, board_info, head_position=None):
        """Draw the game board with snake and apple.
        
        Args:
            board: 2D array representing the game board
            board_info: Dictionary with board entity information
            head_position: Position of the snake's head
        """
        rects = []
        # Draw background for game section
        rects.append(pygame.draw.rect(
            self.screen, 
            BG, 
            (0, 0, self.height+1, self.height+1)
        ))

        # Draw board entities
        for y, row in enumerate(board):
            for x, value in enumerate(row):
                if value == board_info["snake"]:  # snake
                    # Use a different color for the head
                    if head_position is not None and y == head_position[0] and x == head_position[1]:
                        rects.append(pygame.draw.rect(
                            self.screen, 
                            SNAKE_HEAD_C,  # Head color
                            (x*self.pixel, y*self.pixel, self.pixel-5, self.pixel-5)
                        ))
                    else:
                        rects.append(pygame.draw.rect(
                            self.screen, 
                            SNAKE_C,  # Body color
                            (x*self.pixel, y*self.pixel, self.pixel-5, self.pixel-5)
                        ))
                elif value == board_info["apple"]:  # apple
                    rects.append(pygame.draw.rect(
                        self.screen, 
                        APPLE_C, 
                        (x*self.pixel, y*self.pixel, self.pixel-5, self.pixel-5)
                    ))
                    
        # Draw walls/borders
        wall_thickness = 2
        rects.append(pygame.draw.rect(
            self.screen,
            WHITE,  # Wall color
            (0, 0, self.height, wall_thickness)  # Top wall
        ))
        rects.append(pygame.draw.rect(
            self.screen,
            WHITE,  # Wall color
            (0, 0, wall_thickness, self.height)  # Left wall
        ))
        rects.append(pygame.draw.rect(
            self.screen,
            WHITE,  # Wall color
            (0, self.height - wall_thickness, self.height, wall_thickness)  # Bottom wall
        ))
        rects.append(pygame.draw.rect(
            self.screen,
            WHITE,  # Wall color
            (self.height - wall_thickness, 0, wall_thickness, self.height)  # Right wall
        ))
                
        pygame.display.update(rects)

    def draw_game_info(self, score, steps, planned_moves=None, llm_response=None):
        """Draw game information and LLM response.
        
        Args:
            score: Current game score
            steps: Current step count
            planned_moves: List of planned moves from LLM
            llm_response: Processed LLM response text ready for display
        """
        # Clear info panel
        info_panel_rect = pygame.draw.rect(
            self.screen,
            APP_BG,
            (self.height+1, 0, self.width+self.width_plus, self.height)
        )
        
        # Draw score and steps
        score_text = self.font.render(f"Score: {score}", True, BLACK)
        steps_text = self.font.render(f"Steps: {steps}", True, BLACK)
        
        self.screen.blit(score_text, (self.height + 20, 20))
        self.screen.blit(steps_text, (self.height + 20, 60))
        
        # Draw planned moves if available
        if planned_moves:
            moves_text = self.font.render("Planned moves:", True, BLACK)
            self.screen.blit(moves_text, (self.height + 20, 100))
            
            # Display each planned move
            moves_str = ", ".join(planned_moves)
            moves_display = self.font.render(moves_str, True, GREY3)
            self.screen.blit(moves_display, (self.height + 20, 130))
        
        # Draw LLM response if available
        if llm_response:
            response_title = self.font.render("LLM Response:", True, BLACK)
            self.screen.blit(response_title, (self.height + 20, 170))
            
            # Calculate how many characters fit in the text panel
            # This is an approximation based on monospace font
            avg_char_width = self.small_font.size("m")[0]
            max_chars_per_line = int(self.text_panel_width / avg_char_width * 1.7)
            
            # Setup text area with border
            response_area_rect = pygame.draw.rect(
                self.screen,
                GREY2,  # Light gray background for text area
                (self.height + 20, 200, self.text_panel_width, 350),
                0,  # Filled rectangle
                3   # Rounded corners
            )
            
            pygame.draw.rect(
                self.screen,
                GREY3,  # Darker border
                (self.height + 20, 200, self.text_panel_width, 350),
                2,  # Border only (not filled)
                3   # Rounded corners
            )
            
            # The response is already processed by SnakeGame._process_response_for_display()
            # No need to process it again here
            response_text = llm_response
            
            # Text rendering variables
            y_offset = 210  # Starting y position inside the text area
            line_height = 20
            padding_left = 10  # Left padding inside text area
            
            # Split response into lines and wrap
            lines = response_text.split('\n')
            for line in lines[:20]:  
                # Skip empty lines
                if not line.strip():
                    y_offset += line_height
                    continue
                    
                # Further split lines that are too long
                for i in range(0, len(line), max_chars_per_line):
                    segment = line[i:i+max_chars_per_line]
                    if segment:
                        text_surface = self.small_font.render(segment, True, BLACK)
                        self.screen.blit(text_surface, (self.height + 20 + padding_left, y_offset))
                        y_offset += line_height
                        
                        # Stop if we've reached the bottom of the text area
                        if y_offset > 530:  # 200 (start) + 350 (height) - 20 (padding)
                            break
                            
                # Stop if we've reached the bottom of the text area
                if y_offset > 530:
                    break
        
        pygame.display.flip()

# Add the main function at the end of the file
if __name__ == "__main__":
    # Import the necessary modules
    if importlib.util.find_spec("snake_game"):
        from snake_game import SnakeGame
    else:
        print("Error: snake_game module not found. Make sure it's in the same directory.")
        sys.exit(1)
    
    # Initialize pygame
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Snake Game - Human Control")
    
    # Initialize game
    game = SnakeGame()
    
    # Set up game variables
    clock = pygame.time.Clock()
    running = True
    game_active = True
    game_count = 0
    steps = 0
    score = 0
    time_delay = TIME_DELAY
    time_tick = TIME_TICK
    
    # Direction mapping for WASD keys
    key_direction_map = {
        pygame.K_w: "UP",
        pygame.K_s: "DOWN",
        pygame.K_a: "LEFT",
        pygame.K_d: "RIGHT",
        pygame.K_UP: "UP",
        pygame.K_DOWN: "DOWN",
        pygame.K_LEFT: "LEFT",
        pygame.K_RIGHT: "RIGHT"
    }
    
    # Display instructions
    print("\nSnake Game - Human Control")
    print("Controls:")
    print("  W or Up Arrow = UP")
    print("  S or Down Arrow = DOWN")
    print("  A or Left Arrow = LEFT")
    print("  D or Right Arrow = RIGHT")
    print("  R = Reset Game")
    print("  ESC = Quit Game")
    print("  SPACE = Toggle Speed\n")
    print("Game started! Use WASD keys to control the snake.")
    
    # Main game loop
    while running:
        # Handle events
        next_direction = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in key_direction_map and game_active:
                    next_direction = key_direction_map[event.key]
                elif event.key == pygame.K_SPACE:
                    # Toggle speed
                    if time_delay == TIME_DELAY:
                        print("⚡ Speed mode enabled")
                        time_delay, time_tick = 0, 0
                    else:
                        print("🐢 Normal speed mode")
                        time_delay, time_tick = TIME_DELAY, TIME_TICK
                elif event.key == pygame.K_r:
                    # Reset game
                    game.reset()
                    game_active = True
                    print("🔄 Game reset")
        
        # Update game state if a key was pressed
        if game_active and next_direction:
            # Execute the move and check if game continues
            game_active, _ = game.make_move(next_direction)
            print(f"Move: {next_direction}, Score: {game.score}")
            
            # Check if game is over
            if not game_active:
                game_count += 1
                print(f"❌ Game over! Score: {game.score}, Steps: {game.steps}")
                
                # Wait a moment before allowing restart
                pygame.time.delay(1000)  # Wait 1 second
        
        # Update and draw the game
        if game_active:
            game.update()
        game.draw()
        
        # Control game speed
        pygame.time.delay(time_delay)
        clock.tick(time_tick)
    
    # Clean up
    pygame.quit()
    sys.exit() 