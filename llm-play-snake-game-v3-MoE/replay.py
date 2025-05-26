"""
Replay module for the Snake game.
Handles replaying recorded games.
"""

import os
import json
import pygame
import argparse
import datetime
from colorama import Fore
from core.snake_game import SnakeGame
from gui.replay_gui import ReplayGUI


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Replay recorded Snake games")
    parser.add_argument("--log-dir", type=str, required=True,
                      help="Directory containing game logs")
    parser.add_argument("--game", type=int, default=None,
                      help="Specific game number to replay")
    parser.add_argument("--speed", type=float, default=1.0,
                      help="Replay speed multiplier")
    return parser.parse_args()


def main():
    """Main replay function."""
    # Parse arguments
    args = parse_args()
    
    # Initialize pygame
    pygame.init()
    
    # Create game instance
    game = SnakeGame()
    
    # Create GUI
    gui = ReplayGUI()
    game.set_gui(gui)
    
    # Load game data
    game_data = load_game_data(args.log_dir, args.game)
    if not game_data:
        print(f"{Fore.RED}No game data found{Fore.RESET}")
        return
        
    # Replay the game
    try:
        replay_game(game, gui, game_data, args.speed)
    except Exception as e:
        print(f"{Fore.RED}Error during replay: {str(e)}{Fore.RESET}")
    finally:
        pygame.quit()


def load_game_data(log_dir, game_number=None):
    """Load game data from logs.
    
    Args:
        log_dir: Directory containing game logs
        game_number: Specific game number to load
        
    Returns:
        List of game moves
    """
    # Find available game logs
    game_logs = []
    for filename in os.listdir(log_dir):
        if filename.startswith("game_") and filename.endswith(".json"):
            game_logs.append(os.path.join(log_dir, filename))
    
    if not game_logs:
        return None
        
    # Sort logs by timestamp
    game_logs.sort()
    
    # Load specific game if requested
    if game_number is not None:
        if game_number < 0 or game_number >= len(game_logs):
            print(f"{Fore.RED}Invalid game number: {game_number}{Fore.RESET}")
            return None
        game_logs = [game_logs[game_number]]
    
    # Load game data
    game_data = []
    for log_file in game_logs:
        with open(log_file, "r") as f:
            data = json.load(f)
            game_data.extend(data["moves"])
    
    return game_data


def replay_game(game, gui, game_data, speed=1.0):
    """Replay a recorded game.
    
    Args:
        game: SnakeGame instance
        gui: ReplayGUI instance
        game_data: List of game moves
        speed: Replay speed multiplier
    """
    # Game state
    current_move = 0
    paused = False
    
    # Main replay loop
    while True:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and current_move < len(game_data):
                    if not paused:
                        move = game_data[current_move]
                        game.move(move)
                        current_move += 1
                elif event.key == pygame.K_LEFT and current_move > 0:
                    if not paused:
                        game.reset()
                        current_move = 0
                        for i in range(current_move):
                            game.move(game_data[i])
                elif event.key == pygame.K_r:
                    game.reset()
                    current_move = 0
                elif event.key == pygame.K_q:
                    return
        
        # Draw the current state
        gui.draw(game, 1, 1, current_move)
        pygame.display.flip()
        
        # Control replay speed
        pygame.time.delay(int(1000 / speed))


if __name__ == "__main__":
    main()
