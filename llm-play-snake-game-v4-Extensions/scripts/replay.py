"""
Snake Game Replay Module.
Allows replaying of previously recorded games based on logged moves.

This whole module is Task0 specific.
"""

# ---------------------
# Ensure execution directory & import paths are correct irrespective
# of where the user launches the script from (matches main.py wrapper).
# ---------------------

import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------
# Guarantee that 'utils' package is importable even when the user launches
# the script from inside the scripts/ directory.
# ---------------------


# Standard lib additional
import os

import argparse
import pygame
from pygame.locals import *

from config.ui_constants import TIME_DELAY, TIME_TICK
from config.game_constants import PAUSE_BETWEEN_MOVES_SECONDS
from replay import ReplayEngine
from gui.replay_gui import ReplayGUI


def get_parser() -> argparse.ArgumentParser:
    """Creates and returns the argument parser for the replay CLI.

    This function is separate from `parse_arguments` to allow other scripts
    (e.g., replay_web.py) to reuse the parser configuration without
    immediately parsing arguments.

    Returns:
        An argparse.ArgumentParser instance.
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
        "--pause-between-moves",
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

    return parser


def parse_arguments():
    """Parse command line arguments for the replay CLI.

    Exposes the same interface that was previously embedded inside ``main`` so
    other modules (e.g. ``replay_web.py``) can import and reuse the exact same
    parser without having to duplicate option definitions.  The behaviour and
    defaults are **unchanged**.
    """
    parser = get_parser()
    return parser.parse_args()


def main():
    """Main function to run the replay."""

    # Reuse the shared CLI parser so behaviour remains identical.
    args = parse_arguments()

    # Use either positional argument or --log-dir option
    log_dir = args.log_dir_opt if args.log_dir_opt else args.log_dir

    if not log_dir:
        # Show full help text to guide the user
        parser = get_parser()
        parser.print_help()
        sys.exit(1)

    # Check if log directory exists
    if not os.path.isdir(log_dir):
        print(f"Log directory does not exist: {log_dir}")
        sys.exit(1)

    # Initialize replay engine
    replay_engine = ReplayEngine(
        log_dir=log_dir,
        pause_between_moves=args.pause_between_moves,
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
