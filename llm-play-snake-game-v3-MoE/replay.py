"""
Replay module for the Snake game.
Handles replaying recorded games.
"""

import os
import json
import pygame
import argparse
from colorama import Fore, init as colorama_init

from core.snake_game import SnakeGame
from gui.replay_gui import ReplayGUI
from replay.replay_engine import ReplayEngine
from utils.log_utils import load_game_from_file


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Replay recorded Snake games")
    
    # Create a mutually exclusive group for log directory vs specific game file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--log-dir", type=str,
                      help="Directory containing game logs (e.g., logs/game_session_YYYYMMDD_HHMMSS)")
    group.add_argument("--game-file", type=str,
                      help="Path to a specific game summary file (e.g., logs/game_session_YYYYMMDD_HHMMSS/game1_summary.json)")
    
    # Other arguments
    parser.add_argument("--game", type=int, default=None,
                      help="Specific game number (0-indexed) within the session to replay. Only used with --log-dir.")
    parser.add_argument("--move-pause", type=float, default=1.0,
                      help="Pause time between moves in seconds (e.g., 0.5 for half speed, 2.0 for double speed)")
    
    return parser.parse_args()


def main():
    """Main replay function."""
    colorama_init() # Initialize colorama for cross-platform color output
    args = parse_args()
    
    pygame.init()
    pygame.font.init() # Ensure font system is initialized for GUIs
    
    # Create game instance (this is the model)
    game = SnakeGame() # You might pass grid_size etc. from config if replay needs it
    
    # Create GUI instance (this is the view)
    # GUI needs to be initialized before being passed to ReplayEngine if it does its own pygame setup
    gui = ReplayGUI() # Assuming ReplayGUI also calls pygame.init() or similar, or relies on it being called
    
    try:
        if args.game_file:
            # Load game directly from file
            moves_data, game_info = load_game_from_file(args.game_file)
            if not moves_data:
                print(f"{Fore.RED}No moves found in {args.game_file}{Fore.RESET}")
                return
                
            print(f"{Fore.GREEN}Loaded game from {args.game_file}{Fore.RESET}")
            print(f"{Fore.GREEN}Score: {game_info.get('score', 0)}, Steps: {game_info.get('steps', 0)}{Fore.RESET}")
            print(f"{Fore.GREEN}Starting replay with {len(moves_data)} moves. Speed: {args.move_pause}x{Fore.RESET}")
            
            # Create a custom replay engine just for this file
            custom_replay = ReplayEngine(
                game=game,
                gui=gui,
                log_dir=os.path.dirname(args.game_file),
                game_number=None,
                speed=args.move_pause
            )
            
            # Set the moves directly
            custom_replay.all_moves_for_game = moves_data
            custom_replay.game_number_display = 1
            
            # Run the replay engine
            custom_replay.run()
        else:
            # The ReplayEngine is the controller for the replay
            replay_engine_instance = ReplayEngine(
                game=game,
                gui=gui,
                log_dir=args.log_dir,
                game_number=args.game, # Pass the specific game number
                speed=args.move_pause
            )
            
            # Run the replay engine
            replay_engine_instance.run()
    except Exception as e:
        print(f"{Fore.RED}Error during replay: {str(e)}{Fore.RESET}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
    finally:
        pygame.quit()
        print(f"{Fore.CYAN}Pygame quit. Replay finished or exited.{Fore.RESET}")


if __name__ == "__main__":
    main()
