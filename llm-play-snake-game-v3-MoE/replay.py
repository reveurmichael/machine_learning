"""
Snake Game Replay Module.
Allows replaying of previously recorded games based on logged moves.
"""

import os
import sys
import argparse
import pygame
from pygame.locals import *

from config import PAUSE_BETWEEN_MOVES_SECONDS, TIME_DELAY, TIME_TICK
from replay.replay_engine import ReplayEngine
from gui.replay_gui import ReplayGUI

def main():
    """Main function to run the replay."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Replay a Snake game session with detailed visualization.')
    parser.add_argument('--log-dir', type=str, required=True, help='Directory containing game logs')
    parser.add_argument('--game', type=int, default=None, 
                      help='Specific game number (1-indexed) within the session to replay. If not specified, all games will be replayed in sequence.')
    parser.add_argument(
        "--move-pause",
        type=float,
        default=PAUSE_BETWEEN_MOVES_SECONDS,
        help=f"Pause between moves in seconds (default: {PAUSE_BETWEEN_MOVES_SECONDS})",
    )
    parser.add_argument('--auto-advance', action='store_true', help='Automatically advance to next game')
    parser.add_argument('--start-paused', action='store_true', help='Start replay in paused state')
    args = parser.parse_args()

    # Check if log directory exists
    if not os.path.isdir(args.log_dir):
        print(f"Log directory does not exist: {args.log_dir}")
        sys.exit(1)

    # Initialize replay engine
    replay_engine = ReplayEngine(
        log_dir=args.log_dir, 
        move_pause=args.move_pause, 
        auto_advance=args.auto_advance
    )

    # Set initial paused state if requested
    replay_engine.paused = args.start_paused

    # Set specific game if provided
    if args.game is not None:
        replay_engine.game_number = args.game
        print(f"Starting replay with game {args.game}")
    else:
        print("Starting replay with game 1. All games will be replayed in sequence.")

    # Set up the GUI
    gui = ReplayGUI()
    replay_engine.set_gui(gui)

    # Set window title
    pygame.display.set_caption(f"Snake Game Replay - {os.path.basename(args.log_dir)}")

    # Print keyboard controls for user reference
    print("\nKeyboard Controls:")
    print("  SPACE: Pause/Resume replay")
    print("  UP/DOWN: Increase/decrease replay speed")
    print("  LEFT/RIGHT: Navigate to previous/next game")
    print("  R: Restart current game")
    print("  N: Jump to next game")
    print("  S: Speed up (same as UP)")
    print("  D: Slow down (same as DOWN)")
    print("  ESC: Quit replay\n")

    # Run the replay
    try:
        # Initialize pygame timing
        pygame.time.delay(TIME_DELAY)
        pygame.time.Clock().tick(TIME_TICK)
        
        # Create enhanced event handler
        def enhanced_handle_events():
            """Event handler with keyboard controls for replay navigation."""
            for event in pygame.event.get():
                if event.type == QUIT:
                    replay_engine.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        replay_engine.running = False
                    elif event.key == K_SPACE:
                        replay_engine.paused = not replay_engine.paused
                        # Update GUI paused state
                        if replay_engine.use_gui and replay_engine.gui and hasattr(replay_engine.gui, 'set_paused'):
                            replay_engine.gui.set_paused(replay_engine.paused)
                        print("Replay " + ("paused" if replay_engine.paused else "resumed"))
                    elif event.key == K_UP or event.key == K_s:
                        # Speed up replay (decrease pause time)
                        replay_engine.pause_between_moves = max(0.1, replay_engine.pause_between_moves - 0.1)
                        print(f"Replay speed increased. Pause between moves: {replay_engine.pause_between_moves:.1f}s")
                    elif event.key == K_DOWN or event.key == K_d:
                        # Slow down replay (increase pause time)
                        replay_engine.pause_between_moves += 0.1
                        print(f"Replay speed decreased. Pause between moves: {replay_engine.pause_between_moves:.1f}s")
                    elif event.key == K_LEFT:
                        # Previous game
                        if replay_engine.game_number > 1:
                            replay_engine.game_number -= 1
                            if replay_engine.load_game_data(replay_engine.game_number):
                                print(f"Loaded previous game: {replay_engine.game_number}")
                            else:
                                replay_engine.game_number += 1
                                print(f"Could not load previous game. Staying with game {replay_engine.game_number}")
                        else:
                            print("Already at first game")
                    elif event.key == K_RIGHT or event.key == K_n:
                        # Next game
                        replay_engine.game_number += 1
                        if not replay_engine.load_game_data(replay_engine.game_number):
                            print(f"Could not load game {replay_engine.game_number}. Staying with current game.")
                            replay_engine.game_number -= 1
                    elif event.key == K_r:
                        # Restart current game
                        replay_engine.load_game_data(replay_engine.game_number)
        
        # Set the event handler
        replay_engine.handle_events = enhanced_handle_events
        
        # Run the replay engine
        replay_engine.run()
    except Exception as e:
        print(f"Error during replay: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up pygame
        if pygame.get_init():
            pygame.quit()

if __name__ == "__main__":
    main()
