"""
Snake Game Web Replay Module.
Provides a web-based interface for replaying previously recorded games.
Reuses existing replay engine, constants, and game logic from the pygame implementation.

This whole module is Task0 specific.
"""

# pylint: disable=global-statement,too-many-return-statements

# sys.path bootstrap so 'utils' is importable when launched from /scripts
import sys
import pathlib
_repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from utils.path_utils import ensure_repo_root

# Ensure the script runs from the repository root for consistent paths.
ensure_repo_root()

import argparse
import threading
import time
from flask import Flask, render_template, request, jsonify
import logging
import os

from replay.replay_engine import ReplayEngine
from utils.network_utils import find_free_port
from scripts.replay import parse_arguments  # Re-use CLI parser from replay.py
from utils.web_utils import build_color_map, translate_end_reason, to_list

# Build absolute paths for flask folders
_static_folder = str(_repo_root / "web" / "static")
_template_folder = str(_repo_root / "web" / "templates")

# Initialize Flask app
app = Flask(__name__, static_folder=_static_folder, template_folder=_template_folder)

# Global replay engine instance
replay_engine = None
replay_thread = None
running = True

logging.getLogger("werkzeug").setLevel(logging.WARNING)


class WebReplayEngine(ReplayEngine):
    """Extended replay engine for web-based replay.
    Reuses the core functionality from ReplayEngine but adapts it for web display.
    """

    def __init__(self, log_dir, pause_between_moves=1.0, auto_advance=False):
        """Initialize the web replay engine.

        Args:
            log_dir: Directory containing game logs
            pause_between_moves: Time in seconds to pause between moves
            auto_advance: Whether to automatically advance through games
        """
        # Initialize without GUI since we're using web interface
        super().__init__(
            log_dir=log_dir,
            pause_between_moves=pause_between_moves,
            auto_advance=auto_advance,
            use_gui=False,
        )
        self.paused = True  # Start paused until client connects

    def get_current_state(self):
        """Get the current state for the web interface.

        Returns:
            Dictionary with current game state
        """
        # Start with the shared base state (numpy arrays, etc.)
        state = self._build_state_base()

        # Convert numpy arrays to lists for JSON serialisation
        state["snake_positions"] = to_list(state["snake_positions"])
        state["apple_position"] = to_list(state["apple_position"])

        # Web-specific enrichments
        state.update(
            {
                "pause_between_moves": self.pause_between_moves,
                "game_end_reason": translate_end_reason(self.game_end_reason),
                "grid_size": self.grid_size,
                "colors": build_color_map(),
            }
        )

        return state

    def run_web(self):
        """Run the replay engine for web interface.
        Adaptation of the run() method from ReplayEngine but without GUI updates.
        """
        # Load initial game data
        self.load_game_data(self.game_number)

        # Main loop - similar to run() but without GUI updates
        self.running = True
        while self.running:
            # Process game updates if not paused
            if not self.paused:
                self.update()

            # Sleep to control update rate
            time.sleep(0.1)


def replay_thread_function(log_dir, pause_between_moves, auto_advance):
    """Function to run the replay engine in a separate thread.

    Args:
        log_dir: Directory containing game logs
        pause_between_moves: Time in seconds to pause between moves
        auto_advance: Whether to automatically advance through games
    """
    global replay_engine

    # Initialize replay engine
    replay_engine = WebReplayEngine(
        log_dir=log_dir, pause_between_moves=pause_between_moves, auto_advance=auto_advance
    )

    # Run the replay engine
    replay_engine.run_web()


# Define routes
@app.route("/")
def index():
    """Render the main page."""
    return render_template("replay.html")


@app.route("/api/state")
def get_state():  # noqa: F401 – used via Flask routing
    """API endpoint to get the current game state."""
    if replay_engine is None:
        return jsonify({"error": "Replay engine not initialized"})

    return jsonify(replay_engine.get_current_state())


@app.route("/api/control", methods=["POST"])
def control():  # noqa: F401 – used via Flask routing
    """API endpoint to control the replay.
    Implements the same control functions as the keyboard handlers in the pygame version.
    """
    if replay_engine is None:
        return jsonify({"error": "Replay engine not initialized"})

    data = request.json
    command = data.get("command")

    if command == "pause":
        replay_engine.paused = True
        return jsonify({"status": "paused"})

    if command == "play":
        replay_engine.paused = False
        return jsonify({"status": "playing"})

    if command == "next_game":
        # Try to load next game - same logic as in replay.py
        replay_engine.game_number += 1
        if not replay_engine.load_game_data(replay_engine.game_number):
            replay_engine.game_number -= 1
            return jsonify({"status": "error", "message": "No next game"})
        return jsonify({"status": "ok"})

    if command == "prev_game":
        # Try to load previous game - same logic as in replay.py
        if replay_engine.game_number > 1:
            replay_engine.game_number -= 1
            replay_engine.load_game_data(replay_engine.game_number)
            return jsonify({"status": "ok"})
        return jsonify({"status": "error", "message": "Already at first game"})

    if command == "restart_game":
        # Restart current game - same logic as in replay.py
        replay_engine.load_game_data(replay_engine.game_number)
        return jsonify({"status": "ok"})

    if command == "speed_up":  # Note: 'speed_up' decreases move pause time
        # Decrease move pause time by 0.1s (minimum 0.1s)
        replay_engine.pause_between_moves = max(
            0.1, replay_engine.pause_between_moves - 0.1
        )
        # Return both the move pause time and its multiplier
        return jsonify(
            {
                "status": "ok",
                "pause_between_moves": replay_engine.pause_between_moves,
                "speed": 1.0 / replay_engine.pause_between_moves,
            }
        )

    if command == "speed_down":  # Note: 'speed_down' increases move pause time
        # Increase move pause time by 0.1s
        replay_engine.pause_between_moves += 0.1
        # Return both the move pause time and its multiplier
        return jsonify(
            {
                "status": "ok",
                "pause_between_moves": replay_engine.pause_between_moves,
                "speed": 1.0 / replay_engine.pause_between_moves,
            }
        )

    return jsonify({"status": "error", "message": "Unknown command"})


def main():
    """Main function to run the web replay.

    Follows the same two-step CLI parsing pattern used by ``main_web.py`` to
    keep ``replay_web.py`` free from argument-list duplication while still
    supporting additional web-specific flags (``--host`` / ``--port``).
    """

    global replay_thread

    # ---------------------
    # Step 1 – extract host / port first so we can pass the remaining CLI
    #          arguments to the shared replay parser without causing unknown
    #          option errors.
    # ---------------------
    host_port_parser = argparse.ArgumentParser(add_help=False)
    host_port_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host IP"
    )
    host_port_parser.add_argument(
        "--port", type=int, default=find_free_port(8000), help="Port number"
    )

    host_port_args, remaining_argv = host_port_parser.parse_known_args()

    # ---------------------
    # Step 2 – delegate the remaining arguments (log-dir, etc.) to the common
    #          replay CLI defined in ``replay.py``.
    # ---------------------
    argv_backup = sys.argv.copy()
    sys.argv = [sys.argv[0]] + remaining_argv
    try:
        args = parse_arguments()
    finally:
        sys.argv = argv_backup

    # Use either positional argument or --log-dir option
    log_dir = args.log_dir_opt if args.log_dir_opt else args.log_dir

    if not log_dir:
        print(
            "Error: Log directory must be specified either as a positional argument or using --log-dir"
        )
        host_port_parser.print_help()
        sys.exit(1)

    # Check if log directory exists
    if not os.path.isdir(log_dir):
        print(f"Log directory does not exist: {log_dir}")
        sys.exit(1)

    # Start replay engine in a separate thread
    replay_thread = threading.Thread(
        target=replay_thread_function,
        args=(log_dir, args.pause_between_moves, args.auto_advance),
    )
    replay_thread.daemon = True
    replay_thread.start()

    # If specific game is provided, set it after engine starts
    if args.game is not None:
        # Wait a bit for engine to initialize
        time.sleep(1)
        if replay_engine:
            replay_engine.game_number = args.game
            replay_engine.load_game_data(args.game)

    # Set initial paused state if requested (matching pygame replay.py)
    if replay_engine and not args.start_paused:
        replay_engine.paused = False

    # Start Flask app
    host = host_port_args.host
    port = host_port_args.port
    print(f"\n🐍 Snake Game Web Replay starting at http://{host}:{port}")
    print("\nOpen the link in your browser to view the replay.")
    print("\nControls:")
    print("  • Play/Pause: Space bar or button")
    print("  • Navigate games: Left/Right arrow keys or buttons")
    print(
        "  • Adjust move pause: Up arrow (increase by 0.1s) / Down arrow (decrease by 0.1s)"
    )
    print("  • Restart game: R key or button")
    print("  • Exit: Ctrl+C in terminal\n")

    # Run Flask app
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
