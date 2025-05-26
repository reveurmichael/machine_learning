"""
Snake Game Replay Module.
Allows replaying of previously recorded games based on logged moves.
"""

import os
import sys
import argparse
import pygame
from pygame.locals import *

from config import PAUSE_BETWEEN_GAMES_SECONDS
from replay.replay_engine import ReplayEngine
from gui.replay_gui import ReplayGUI

def main():
    """Main function to run the replay."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Replay a Snake game.')
    parser.add_argument('--log-dir', type=str, required=True, help='Directory containing game logs')
    parser.add_argument('--game', type=int, help='Specific game number to replay')
    parser.add_argument('--move-pause', type=float, default=1.0, help='Pause between moves in seconds')
    parser.add_argument('--auto-advance', action='store_true', help='Automatically advance to next game')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI (for validation only)')
    args = parser.parse_args()
    
    # Check if log directory exists
    if not os.path.isdir(args.log_dir):
        print(f"Log directory does not exist: {args.log_dir}")
        sys.exit(1)
    
    # Initialize replay engine
    use_gui = not args.no_gui
    replay_engine = ReplayEngine(
        log_dir=args.log_dir, 
        move_pause=args.move_pause, 
        auto_advance=args.auto_advance,
        use_gui=use_gui
    )
    
    # Set specific game if provided
    if args.game:
        replay_engine.game_number = args.game
    
    # Set up the GUI if needed
    if use_gui:
        gui = ReplayGUI()
        replay_engine.set_gui(gui)
        
        # Run the replay with GUI
        replay_engine.run()
    else:
        # Run the replay without GUI - just validate logs
        print("Running in no-GUI mode - validating game logs only")
        
        # Load first game
        if not replay_engine.load_game_data(replay_engine.game_number):
            print(f"Could not load game {replay_engine.game_number}. Trying next game.")
            replay_engine.game_number += 1
            if not replay_engine.load_game_data(replay_engine.game_number):
                print("No valid games found in log directory.")
                sys.exit(1)
        
        # Validate all games
        game_number = replay_engine.game_number
        total_games = 0
        valid_games = 0
        
        while True:
            if not replay_engine.load_game_data(game_number):
                print(f"Game {game_number} not found or invalid")
                break
                
            total_games += 1
            valid_games += 1
            
            print(f"Game {game_number}: {len(replay_engine.moves)} moves, {len(replay_engine.apple_positions)} apples")
            game_number += 1
        
        print(f"Validated {valid_games} out of {total_games} games")

if __name__ == "__main__":
    main()
