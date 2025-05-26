"""
Human-controlled Snake Game.
Allows playing the Snake game with keyboard controls for fun.
"""

import sys
import pygame
import importlib.util
from colorama import Fore, init as colorama_init

# Import the game components from the project
from core.snake_game import SnakeGame
from gui.game_gui import GameGUI

# Game settings
TIME_DELAY = 100  # Milliseconds between moves
TIME_TICK = 60    # FPS target

def main():
    """Run the human-controlled Snake game."""
    # Initialize colorama for cross-platform color output
    colorama_init()
    
    # Initialize pygame
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Snake Game - Human Control")
    
    # Initialize game
    game = SnakeGame()
    gui = GameGUI()
    game.set_gui(gui)
    
    # Set up game variables
    clock = pygame.time.Clock()
    running = True
    game_active = True
    game_count = 0
    direction = "RIGHT"  # Initial direction
    
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
    print(f"{Fore.CYAN}Snake Game - Human Control{Fore.RESET}")
    print(f"{Fore.YELLOW}Controls:{Fore.RESET}")
    print("  W or Up Arrow = UP")
    print("  S or Down Arrow = DOWN")
    print("  A or Left Arrow = LEFT")
    print("  D or Right Arrow = RIGHT")
    print("  R = Reset Game")
    print("  ESC = Quit Game")
    print(f"{Fore.GREEN}Game started! Use WASD or arrow keys to control the snake.{Fore.RESET}")
    
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
                elif event.key in key_direction_map and not game.game_over:
                    next_direction = key_direction_map[event.key]
                elif event.key == pygame.K_r:
                    # Reset game
                    game.reset()
                    print(f"{Fore.CYAN}🔄 Game reset{Fore.RESET}")
        
        # Update game state if a key was pressed
        if not game.game_over and next_direction:
            # Store the previous direction for validation
            prev_direction = direction
            
            # Prevent 180-degree turns (snake can't turn back on itself)
            if (next_direction == "UP" and prev_direction == "DOWN") or \
               (next_direction == "DOWN" and prev_direction == "UP") or \
               (next_direction == "LEFT" and prev_direction == "RIGHT") or \
               (next_direction == "RIGHT" and prev_direction == "LEFT"):
                # Invalid move, continue in the same direction
                next_direction = prev_direction
            
            # Execute the move
            direction = next_direction
            game.move(direction)
            print(f"Move: {direction}, Score: {game.score}, Steps: {game.steps}")
            
            # Check if game is over
            if game.game_over:
                game_count += 1
                print(f"{Fore.RED}❌ Game over! Score: {game.score}, Steps: {game.steps}{Fore.RESET}")
                print(f"{Fore.YELLOW}Press R to start a new game or ESC to quit{Fore.RESET}")
        
        # Draw the game
        gui.draw(game)
        pygame.display.flip()
        
        # Control game speed
        pygame.time.delay(TIME_DELAY)
        clock.tick(TIME_TICK)
    
    # Clean up
    pygame.quit()
    print(f"{Fore.CYAN}Thanks for playing!{Fore.RESET}")
    sys.exit()

if __name__ == "__main__":
    main() 