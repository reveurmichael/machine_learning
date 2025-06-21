"""
Simple snake game.

NOTE: This represents our "vague idea" implementation - v0!
This shows how you can quickly prototype a working solution with basic Python.
We start simple with functions and global variables to get something working first.
In v1, we'll see how a little bit of OOP can help us organize and improve this foundation.
"""

import pygame
import random
import re

# Game constants - hardcoded for quick prototyping
# NOTE: Starting simple with hardcoded values - easy to change later with proper config
GRID_SIZE = 10
CELL_SIZE = 60
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
INFO_PANEL_WIDTH = 400

# Colors - keeping them organized and readable
SNAKE_C = (209, 204, 192)           # Snake body color
APPLE_C = (192, 57, 43)             # Apple color  
BG = (44, 44, 84)                   # Game background
APP_BG = (240, 240, 240)            # Application background
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (113, 128, 147)
GREY2 = (189, 195, 199)
GREY3 = (87, 96, 111)
SNAKE_HEAD_C = (255, 140, 0)        # Bright orange for the head

# Global game state - simple and direct approach for v0
snake_positions = []        # List of [y, x] positions of snake segments
snake_direction = None      # Current movement direction as (dx, dy) tuple
apple_position = [0, 0]     # Position of the apple as [y, x]
score = 0                   # Current game score
steps = 0                   # Total steps taken
game_over = False           # Game over flag
screen = None               # Pygame screen surface
font = None                 # Large font for UI
small_font = None           # Small font for detailed text
planned_moves = []          # Queue of moves planned by LLM
last_llm_response = ""      # Last response from LLM for display

def init_pygame():
    """
    Initialize pygame and create the game window.
    
    Sets up the pygame display, fonts, and window caption.
    This function gets our graphics system ready to go.
    """
    global screen, font, small_font
    
    pygame.init()
    # Create window with game area + info panel
    screen = pygame.display.set_mode((WINDOW_WIDTH + INFO_PANEL_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("LLM Snake Agent (Non-OOP)")
    
    # Initialize fonts for UI text
    font = pygame.font.SysFont("freesansbold.ttf", 24)
    small_font = pygame.font.SysFont("freesansbold.ttf", 16)

def init_game():
    """
    Initialize the game state to starting conditions.
    
    Resets all game variables to start a fresh game.
    Notice how we need to reset many variables - 
    in v1, a Game class will encapsulate this state better.
    """
    global snake_positions, snake_direction, apple_position, score, steps
    global game_over, planned_moves, last_llm_response
    
    # Start snake in the middle of the grid
    start_x = GRID_SIZE // 2
    start_y = GRID_SIZE // 2
    snake_positions = [[start_y, start_x]]  # [y, x] format like original
    
    # Reset all game state
    snake_direction = None      # No initial direction
    score = 0
    steps = 0
    game_over = False
    planned_moves = []          # Clear any planned moves
    last_llm_response = ""      # Clear LLM display
    
    # Place the first apple randomly
    place_new_apple()

def place_new_apple():
    """
    Place an apple at a random empty position on the grid.
    
    Keeps trying random positions until it finds one not occupied by the snake.
    Simple and effective approach for our v0 implementation.
    """
    global apple_position
    
    while True:
        # Generate random coordinates
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        
        # Make sure apple isn't placed on any part of the snake
        if [y, x] not in snake_positions:
            apple_position = [y, x]
            break

def move_snake(direction_key):
    """
    Move the snake in the specified direction.
    
    Args:
        direction_key (str): Direction to move ("UP", "DOWN", "LEFT", "RIGHT")
        
    Returns:
        tuple: (game_active, apple_eaten) 
               - game_active: True if game continues, False if game over
               - apple_eaten: True if an apple was consumed this move
               
    This function handles the core game mechanics. 
    """
    global snake_positions, score, steps, game_over, snake_direction, planned_moves
    
    # Can't move if game is already over
    if game_over:
        return False, False
    
    # Direction mappings - convert string to coordinate deltas
    directions = {
        "UP": (0, -1),      # Move up (decrease y)
        "RIGHT": (1, 0),    # Move right (increase x)
        "DOWN": (0, 1),     # Move down (increase y)
        "LEFT": (-1, 0)     # Move left (decrease x)
    }
    
    # Validate direction, default to RIGHT if invalid
    if direction_key not in directions:
        direction_key = "RIGHT"
    
    direction = directions[direction_key]
    
    # Prevent snake from reversing directly into itself
    if (snake_direction is not None and 
        direction == (-snake_direction[0], -snake_direction[1])):
        # Keep current direction instead of reversing
        direction = snake_direction
        direction_key = get_direction_key(direction)
    
    # Update current direction for next move
    snake_direction = direction
    
    # Calculate new head position
    head_y, head_x = snake_positions[0]  # Current head is first element
    new_head = [head_y + direction[1], head_x + direction[0]]
    
    # Check if this move would eat the apple
    is_eating_apple = (new_head == apple_position)
    
    # Collision detection
    # Check if new head position hits walls
    wall_collision = (new_head[0] < 0 or new_head[0] >= GRID_SIZE or 
                     new_head[1] < 0 or new_head[1] >= GRID_SIZE)
    
    # Check if new head position hits snake body
    body_collision = False
    if is_eating_apple:
        # When eating apple, tail doesn't move, so check collision with body except head
        body_segments = snake_positions[:-1]
        body_collision = new_head in body_segments
    else:
        # Normal move: tail will move, so check collision with middle segments only
        if len(snake_positions) > 2:
            body_segments = snake_positions[1:-1]  # Exclude head and tail
            body_collision = new_head in body_segments
    
    # Handle game over conditions
    if wall_collision:
        print(f"💀 Game over! Snake hit wall moving {direction_key}")
        game_over = True
        return False, False
    
    if body_collision:
        print(f"💀 Game over! Snake hit itself moving {direction_key}")
        game_over = True
        return False, False
    
    # Valid move - update snake position
    snake_positions.insert(0, new_head)  # Add new head at front
    
    # Handle apple consumption
    apple_eaten = False
    if is_eating_apple:
        score += 1  # Increase score
        print(f"🍎 Apple eaten! Score: {score}")
        place_new_apple()  # Generate new apple
        apple_eaten = True
        # Note: We don't remove the tail when eating an apple (snake grows)
    else:
        # Normal move - remove tail to maintain snake length
        snake_positions.pop()
    
    # Increment step counter
    steps += 1
    return True, apple_eaten

def get_direction_key(direction):
    """
    Convert direction tuple back to string key.
    
    Args:
        direction (tuple): Direction as (dx, dy) coordinates
        
    Returns:
        str: Direction key ("UP", "DOWN", "LEFT", "RIGHT")
        
    Helper function to reverse the direction mapping.
    """
    directions = {
        (0, -1): "UP",
        (1, 0): "RIGHT", 
        (0, 1): "DOWN",
        (-1, 0): "LEFT"
    }
    return directions.get(direction, "RIGHT")

def get_game_state():
    """
    Generate a text representation of the current game state for the LLM.
    
    Returns:
        str: Multi-line string showing the game board with coordinates
        
    Creates a visual representation that the LLM can understand:
    - H = Snake head (with direction arrow)
    - S = Snake body segments  
    - A = Apple
    - . = Empty space
    - Includes coordinate grid and walls for clarity
    """
    # Create coordinate header with grid lines
    board_str = "  "
    board_str += "+".join(["-" for _ in range(GRID_SIZE + 1)]) + "\n"
    board_str += "  " + " ".join([f"{x}" for x in range(GRID_SIZE)]) + "\n"
    
    # Build the game board row by row
    for y in range(GRID_SIZE):
        board_str += f"{y}|"  # Row number and left wall
        
        for x in range(GRID_SIZE):
            if [y, x] == apple_position:
                board_str += "A "  # Apple position
            elif [y, x] in snake_positions:
                if [y, x] == snake_positions[0]:  # Head position
                    # Show head with direction indicator
                    if snake_direction is None:
                        board_str += "H "
                    elif snake_direction == (0, -1):
                        board_str += "H↑"  # Facing up
                    elif snake_direction == (1, 0):
                        board_str += "H→"  # Facing right
                    elif snake_direction == (0, 1):
                        board_str += "H↓"  # Facing down
                    elif snake_direction == (-1, 0):
                        board_str += "H←"  # Facing left
                    else:
                        board_str += "H "
                else:
                    board_str += "S "  # Snake body segment
            else:
                board_str += ". "  # Empty space
        
        board_str += "|\n"  # Right wall and newline
    
    # Add bottom wall
    board_str += "  " + "+".join(["-" for _ in range(GRID_SIZE + 1)]) + "\n"
    
    return board_str

def parse_llm_response(response):
    """
    Parse the LLM's response to extract planned moves.
    
    Args:
        response (str): Raw text response from the LLM
        
    Returns:
        str or None: First move to execute, or None if no valid moves found
        
    This function extracts multiple moves from the LLM response.
    In v1, we might create a dedicated ResponseParser class for more sophisticated parsing.
    """
    global planned_moves, last_llm_response
    
    # Store response for display in UI
    last_llm_response = response
    planned_moves = []
    
    # Use regex to find direction words in the response
    directions_found = re.findall(r'\b(UP|DOWN|LEFT|RIGHT)\b', response, re.IGNORECASE)
    
    if directions_found:
        # Convert to uppercase and store as planned moves
        planned_moves = [d.upper() for d in directions_found]
        print(f"Found {len(planned_moves)} moves: {planned_moves}")
        return planned_moves[0] if planned_moves else None
    
    return None

def get_next_planned_move():
    """
    Get and remove the next move from the planned moves queue.
    
    Returns:
        str or None: Next planned move, or None if no moves left
        
    Simple queue implementation using list.pop(0).
    """
    if planned_moves:
        return planned_moves.pop(0)
    return None

def draw_game():
    """
    Draw the complete game interface using pygame.
    
    Renders the game board, snake, apple, walls, and information panel.
    This function handles all our rendering needs for v0.
    """
    if not screen:
        return
    
    # Clear the entire screen
    screen.fill(APP_BG)
    
    # Draw game board background
    game_area = pygame.Rect(0, 0, WINDOW_HEIGHT, WINDOW_HEIGHT)
    pygame.draw.rect(screen, BG, game_area)
    
    # Calculate cell size for the game grid
    cell_size = WINDOW_HEIGHT // GRID_SIZE
    
    # Render snake segments
    for i, (y, x) in enumerate(snake_positions):
        # Head gets special color, body gets normal color
        color = SNAKE_HEAD_C if i == 0 else SNAKE_C
        rect = pygame.Rect(x * cell_size, y * cell_size, cell_size - 5, cell_size - 5)
        pygame.draw.rect(screen, color, rect)
    
    # Render apple
    ay, ax = apple_position
    apple_rect = pygame.Rect(ax * cell_size, ay * cell_size, cell_size - 5, cell_size - 5)
    pygame.draw.rect(screen, APPLE_C, apple_rect)
    
    # Draw boundary walls
    wall_thickness = 2
    pygame.draw.rect(screen, WHITE, (0, 0, WINDOW_HEIGHT, wall_thickness))  # Top
    pygame.draw.rect(screen, WHITE, (0, 0, wall_thickness, WINDOW_HEIGHT))  # Left  
    pygame.draw.rect(screen, WHITE, (0, WINDOW_HEIGHT - wall_thickness, WINDOW_HEIGHT, wall_thickness))  # Bottom
    pygame.draw.rect(screen, WHITE, (WINDOW_HEIGHT - wall_thickness, 0, wall_thickness, WINDOW_HEIGHT))  # Right
    
    # Draw the information panel
    draw_info_panel()
    
    # Update the display
    pygame.display.flip()

def draw_info_panel():
    """
    Draw the information panel showing game stats and LLM response.
    
    Displays:
    - Current score and steps
    - Planned moves from LLM
    - LLM response text in a scrollable area
    
    This creates our user interface for monitoring the LLM's thinking.
    """
    info_x = WINDOW_HEIGHT + 20  # Start position for info panel
    
    # Render score and steps
    score_text = font.render(f"Score: {score}", True, BLACK)
    steps_text = font.render(f"Steps: {steps}", True, BLACK)
    
    screen.blit(score_text, (info_x, 20))
    screen.blit(steps_text, (info_x, 60))
    
    # Show planned moves if any exist
    if planned_moves:
        moves_text = font.render("Planned moves:", True, BLACK)
        screen.blit(moves_text, (info_x, 100))
        
        # Show first 10 planned moves
        moves_str = ", ".join(planned_moves[:10])
        moves_display = font.render(moves_str, True, GREY3)
        screen.blit(moves_display, (info_x, 130))
    
    # Display LLM response in a text area
    if last_llm_response:
        response_title = font.render("LLM Response:", True, BLACK)
        screen.blit(response_title, (info_x, 170))
        
        # Create text area with border
        text_area_width = INFO_PANEL_WIDTH - 40
        text_area_height = 350
        text_area_rect = pygame.Rect(info_x, 200, text_area_width, text_area_height)
        
        # Draw text area background and border
        pygame.draw.rect(screen, GREY2, text_area_rect, 0, 3)  # Background
        pygame.draw.rect(screen, GREY3, text_area_rect, 2, 3)  # Border
        
        # Render text line by line with word wrapping
        y_offset = 210
        line_height = 18
        max_chars_per_line = 35
        
        # Split response into lines and render each
        lines = last_llm_response.split('\n')
        for line in lines[:15]:  # Limit to 15 lines to fit in area
            if not line.strip():
                y_offset += line_height
                continue
            
            # Handle long lines by wrapping them
            for i in range(0, len(line), max_chars_per_line):
                segment = line[i:i+max_chars_per_line]
                if segment:
                    text_surface = small_font.render(segment, True, BLACK)
                    screen.blit(text_surface, (info_x + 10, y_offset))
                    y_offset += line_height
                    
                    # Stop if we've reached the bottom of the text area
                    if y_offset > 530:
                        break
            
            if y_offset > 530:
                break

# Simple accessor functions for getting game state
def is_game_over():
    """Check if the game is in a game over state."""
    return game_over

def get_score():
    """Get the current game score."""
    return score

def get_steps():
    """Get the current step count."""
    return steps

def reset_game():
    """
    Reset the game to initial state.
    
    Notice we need to handle the game_over flag separately.
    In v1, better encapsulation will make state management cleaner.
    """
    global game_over
    init_game()
    game_over = False
    print("🔄 Game reset!") 