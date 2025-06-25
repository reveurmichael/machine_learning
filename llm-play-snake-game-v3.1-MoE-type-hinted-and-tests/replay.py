"""
Snake Game Replay Module.
Allows replaying of previously recorded games based on logged moves.
"""

import os
import sys
import argparse
import pygame
from pygame.locals import (  # pylint: disable=no-name-in-module,unused-import
    QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_r
)

from config.ui_constants import TIME_DELAY, TIME_TICK
from config.game_constants import PAUSE_BETWEEN_MOVES_SECONDS
from replay.replay_engine import ReplayEngine
from gui.replay_gui import ReplayGUI


def parse_arguments():
    """Parse command line arguments for the replay CLI.

    Exposes the same interface that was previously embedded inside ``main`` so
    other modules (e.g. ``replay_web.py``) can import and reuse the exact same
    parser without having to duplicate option definitions.  The behaviour and
    defaults are **unchanged**.
    """

    parser = argparse.ArgumentParser(
        description="Replay a Snake game session with detailed visualization."
    )

    # Log directory (positional or --log-dir)
    parser.add_argument(
        "log_dir",
        type=str,
        nargs="?",
        help="Directory containing game logs",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        dest="log_dir_opt",
        help="Directory containing game logs (alternative to positional argument)",
    )

    # Optional fine-grained controls – identical to the original implementation
    parser.add_argument(
        "--game",
        type=int,
        default=None,
        help=(
            "Specific game number (1-indexed) within the session to replay. "
            "If not specified, all games will be replayed in sequence."
        ),
    )
    parser.add_argument(
        "--move-pause",
        type=float,
        default=PAUSE_BETWEEN_MOVES_SECONDS,
        help=(
            f"Pause between moves in seconds (default: {PAUSE_BETWEEN_MOVES_SECONDS})"
        ),
    )
    parser.add_argument(
        "--auto-advance",
        action="store_true",
        help="Automatically advance to next game",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Start replay in paused state",
    )

    return parser.parse_args()


def main():
    """Main function to run the replay."""

    # Reuse the shared CLI parser so behaviour remains identical.
    args = parse_arguments()

    # Use either positional argument or --log-dir option
    log_dir = args.log_dir_opt if args.log_dir_opt else args.log_dir

    if not log_dir:
        # reconstruct the parser to show help (avoid holding a ref)
        if "-h" not in sys.argv and "--help" not in sys.argv:
            print(
                "Error: Log directory must be specified either as a positional argument or using --log-dir"
            )
        # Show full help text to guide the user
        argparse.ArgumentParser(prog="replay.py").print_help()
        sys.exit(1)

    # Check if log directory exists
    if not os.path.isdir(log_dir):
        print(f"Log directory does not exist: {log_dir}")
        sys.exit(1)

    # Initialize replay engine
    replay_engine = ReplayEngine(
        log_dir=log_dir,
        move_pause=args.move_pause,
        auto_advance=args.auto_advance,
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
    pygame.display.set_caption(f"Snake Game Replay - {os.path.basename(log_dir)}")

    # Print keyboard controls for user reference
    print("\nKeyboard Controls:")
    print("  SPACE: Pause/Resume replay")
    print("  UP/DOWN: Increase/decrease replay speed")
    print("  LEFT/RIGHT: Navigate to previous/next game")
    print("  R: Restart current game")
    print("  ESC: Quit replay\n")

    # Run the replay
    try:
        # Initialize pygame timing
        pygame.time.delay(TIME_DELAY)
        pygame.time.Clock().tick(TIME_TICK)
        replay_engine.run()
    except Exception as exception:
        print(f"Error during replay: {exception}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up pygame
        if pygame.get_init():  # pylint: disable=no-member
            pygame.quit()  # pylint: disable=no-member


if __name__ == "__main__":
    main()
