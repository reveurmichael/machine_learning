"""
Snake Game Replay Module.
Allows replaying of previously recorded games based on logged moves.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
import pygame
from pygame.locals import *

from config import (
    SNAKE_C, APPLE_C, BG, APP_BG, GRID_BG, BLACK, WHITE, GREY, GRID_SIZE, DIRECTIONS,
    APP_WIDTH, APP_HEIGHT, SNAKE_HEAD_C
)

# Define the pause time between games
PAUSE_BETWEEN_GAMES_SECONDS = 1

# Import after config to avoid circular imports
from gui import SetUp
from utils.snake_utils import filter_invalid_reversals, extract_apple_positions

# Define a simple draw_grid function
def draw_grid(screen, pixel_size, grid_size, color):
    """Draw grid lines on the screen.
    
    Args:
        screen: Pygame screen surface
        pixel_size: Size of each grid cell in pixels
        grid_size: Number of cells in the grid
        color: Color of the grid lines
    """
    # Draw horizontal lines
    for y in range(grid_size + 1):
        pygame.draw.line(screen, color, (0, y * pixel_size), (grid_size * pixel_size, y * pixel_size), 1)
    
    # Draw vertical lines
    for x in range(grid_size + 1):
        pygame.draw.line(screen, color, (x * pixel_size, 0), (x * pixel_size, grid_size * pixel_size), 1)

class ReplaySnakeGame(SetUp):
    """
    A class to replay Snake games based on saved game data.
    """
    
    def __init__(self, log_dir, move_pause=1.0, auto_advance=False):
        """
        Initialize the replay environment.
        
        Args:
            log_dir: Directory containing game logs
            move_pause: Time in seconds to pause between moves
            auto_advance: Whether to automatically advance through games
        """
        super().__init__()
        
        # Initialize pygame and screen
        pygame.init()
        self.screen = pygame.display.set_mode((self.width + self.width_plus, self.height))
        pygame.display.set_caption('Snake Game Replay')
        
        # Initialize replay parameters
        self.log_dir = log_dir
        self.move_pause = move_pause
        self.auto_advance = auto_advance
        
        # Game state
        self.game_number = 1
        self.snake_positions = []
        self.apple_positions = []
        self.apple_index = 0
        self.moves = []
        self.move_index = 0
        self.moves_made = []
        self.game_score = 0
        self.game_steps = 0
        self.current_direction = None
        self.game_stats = {}
        self.last_move_time = time.time()
        self.running = True
    
    def load_game_data(self, game_number):
        """Load data for a specific game number.
        
        Args:
            game_number: The game number to load
            
        Returns:
            Boolean indicating if game data was loaded successfully
        """
        # Reset game state
        self.snake_positions = [[GRID_SIZE // 2, GRID_SIZE // 2]]
        self.apple_positions = []
        self.apple_index = 0
        self.moves = []
        self.move_index = 0
        self.moves_made = []
        self.game_score = 0
        self.game_steps = 0
        self.current_direction = None
        
        # Load game data from summary file
        summary_file = os.path.join(self.log_dir, f"game{game_number}_summary.json")
        
        if not os.path.exists(summary_file):
            print(f"Game {game_number} summary file not found.")
            return False
        
        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            # Extract game statistics
            self.game_stats = {
                'score': summary_data.get('score', 0),
                'steps': summary_data.get('steps', 0),
                'game_end_reason': summary_data.get('game_end_reason', 'Unknown'),
                'snake_length': summary_data.get('snake_length', 1)
            }
            
            # Load apple positions from JSON summary
            self.apple_positions = extract_apple_positions(self.log_dir, game_number)
            
            # Load moves from JSON summary
            if 'moves' in summary_data and summary_data['moves']:
                raw_moves = summary_data['moves']
                
                # Ensure moves are valid and no invalid reversals
                self.moves = filter_invalid_reversals(raw_moves)
                
                print(f"Loaded {len(self.moves)} moves for game {game_number}")
                
                if len(self.moves) != len(raw_moves):
                    print(f"Warning: Filtered out {len(raw_moves) - len(self.moves)} invalid moves")
                
                return len(self.moves) > 0
            else:
                print(f"No moves found for game {game_number}")
                return False
            
        except Exception as e:
            print(f"Error loading game data: {e}")
            return False
    
    def make_move(self, direction):
        """Make a move in the specified direction.
        
        Args:
            direction: Direction to move (UP, DOWN, LEFT, RIGHT)
            
        Returns:
            Boolean indicating if game continues
        """
        # Skip invalid direction
        if not direction or direction not in DIRECTIONS:
            print(f"Invalid direction: {direction}")
            return True
        
        # Get current head position
        head_x, head_y = self.snake_positions[0]
        
        # Calculate new head position based on direction
        if direction == "UP":
            head_y -= 1
        elif direction == "DOWN":
            head_y += 1
        elif direction == "LEFT":
            head_x -= 1
        elif direction == "RIGHT":
            head_x += 1
        
        # Check for collision with walls
        if head_x < 0 or head_x >= GRID_SIZE or head_y < 0 or head_y >= GRID_SIZE:
            print(f"Hit wall at ({head_x}, {head_y})")
            return False
        
        # Check for collision with self
        if [head_x, head_y] in self.snake_positions:
            print(f"Hit self at ({head_x}, {head_y})")
            return False
        
        # Update head position
        self.snake_positions.insert(0, [head_x, head_y])
        
        # Check for apple
        apple_eaten = False
        if self.apple_index < len(self.apple_positions):
            apple_pos = self.apple_positions[self.apple_index]
            if head_x == apple_pos[0] and head_y == apple_pos[1]:
                apple_eaten = True
                self.apple_index += 1
                self.game_score += 1
        
        # Remove tail if no apple eaten
        if not apple_eaten:
            self.snake_positions.pop()
        
        # Update game state
        self.current_direction = direction
        self.moves_made.append(direction)
        self.game_steps += 1
        
        return True
    
    def draw(self):
        """Draw the game state."""
        # Fill background
        self.screen.fill(APP_BG)
        
        # Draw grid
        draw_grid(self.screen, self.pixel, self.grid_size, GRID_BG)
        
        # Draw snake
        for i, position in enumerate(self.snake_positions):
            x, y = position
            
            # Convert grid position to pixel position
            rect = pygame.Rect(
                x * self.pixel,
                y * self.pixel,
                self.pixel,
                self.pixel
            )
            
            # Draw head in different color
            if i == 0:
                pygame.draw.rect(self.screen, SNAKE_HEAD_C, rect)
            else:
                pygame.draw.rect(self.screen, SNAKE_C, rect)
        
        # Draw apple if available
        if self.apple_index < len(self.apple_positions):
            apple_pos = self.apple_positions[self.apple_index]
            x, y = apple_pos
            
            # Convert grid position to pixel position
            rect = pygame.Rect(
                x * self.pixel,
                y * self.pixel,
                self.pixel,
                self.pixel
            )
            
            pygame.draw.rect(self.screen, APPLE_C, rect)
        
        # Draw game info
        font = pygame.font.SysFont('arial', 20)
        
        # Right panel info
        info_text = [
            f"Game: {self.game_number}",
            f"Score: {self.game_score}",
            f"Steps: {self.game_steps}",
            f"Moves: {self.move_index}/{len(self.moves)}",
            f"Current Direction: {self.current_direction or 'None'}",
            f"Press Space to pause/resume",
            f"Press N for next game",
            f"Press R to restart game",
            f"Press Esc to quit"
        ]
        
        y_offset = 20
        for text in info_text:
            text_surface = font.render(text, True, WHITE)
            self.screen.blit(text_surface, (self.height + 20, y_offset))
            y_offset += 30
        
        # Update display
        pygame.display.flip()
    
    def update(self):
        """Update game state for each frame."""
        current_time = time.time()
        
        # Check if it's time for the next move
        if current_time - self.last_move_time >= self.move_pause and self.move_index < len(self.moves):
            # Make the next move
            next_move = self.moves[self.move_index]
            self.move_index += 1
            
            # Update game state
            game_continues = self.make_move(next_move)
            
            # Update last move time
            self.last_move_time = current_time
            
            # Check if game is over
            if not game_continues:
                print(f"Game {self.game_number} over. Score: {self.game_score}, Steps: {self.game_steps}")
                
                # Check if we should advance to next game
                if self.auto_advance:
                    pygame.time.delay(PAUSE_BETWEEN_GAMES_SECONDS * 1000)
                    self.game_number += 1
                    self.load_game_data(self.game_number)
            
            # Check if we're out of moves
            elif self.move_index >= len(self.moves):
                print(f"Replay complete for game {self.game_number}. Score: {self.game_score}, Steps: {self.game_steps}")
                
                # Check if we should advance to next game
                if self.auto_advance:
                    pygame.time.delay(PAUSE_BETWEEN_GAMES_SECONDS * 1000)
                    self.game_number += 1
                    self.load_game_data(self.game_number)
    
    def run(self):
        """Run the replay loop."""
        clock = pygame.time.Clock()
        paused = False
        
        # Load first game
        if not self.load_game_data(self.game_number):
            print(f"Could not load game {self.game_number}. Trying next game.")
            self.game_number += 1
            if not self.load_game_data(self.game_number):
                print("No valid games found in log directory.")
                return
        
        # Main game loop
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                    elif event.key == K_SPACE:
                        paused = not paused
                        print("Replay " + ("paused" if paused else "resumed"))
                    elif event.key == K_n:
                        # Next game
                        self.game_number += 1
                        if not self.load_game_data(self.game_number):
                            print(f"Could not load game {self.game_number}. Staying with current game.")
                            self.game_number -= 1
                    elif event.key == K_r:
                        # Restart current game
                        self.load_game_data(self.game_number)
            
            # Update game state if not paused
            if not paused:
                self.update()
            
            # Draw game state
            self.draw()
            
            # Control game speed
            clock.tick(60)
        
        # Clean up
        pygame.quit()

def main():
    """Main function to run the replay."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Replay a Snake game.')
    parser.add_argument('--log-dir', type=str, required=True, help='Directory containing game logs')
    parser.add_argument('--game', type=int, help='Specific game number to replay')
    parser.add_argument('--move-pause', type=float, default=1.0, help='Pause between moves in seconds')
    parser.add_argument('--auto-advance', action='store_true', help='Automatically advance to next game')
    args = parser.parse_args()
    
    # Check if log directory exists
    if not os.path.isdir(args.log_dir):
        print(f"Log directory does not exist: {args.log_dir}")
        sys.exit(1)
    
    # Initialize and run replay
    replay = ReplaySnakeGame(args.log_dir, args.move_pause, args.auto_advance)
    
    # Set specific game if provided
    if args.game:
        replay.game_number = args.game
    
    # Run replay
    replay.run()

if __name__ == "__main__":
    main()
