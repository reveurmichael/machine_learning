"""
Snake Game – Web Live Mode
--------------------------
Run a live LLM-controlled Snake session and stream board states to a tiny
Flask front-end exactly like `replay_web.py` / `human_play_web.py` do.
The game logic itself is untouched – we simply run `GameManager` in a
background thread with `--no-gui` and expose JSON snapshots at `/api/state`.

Usage (same CLI as main.py plus host/port):

    python main_web.py --max-games 5 --no-gui --host 0.0.0.0 --port 8000
"""

# Ensure pygame does not attempt to open an X11 window — must be done before pygame import
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import threading
import sys
import logging
from flask import Flask, render_template, jsonify, request
from utils.network_utils import find_free_port
logging.getLogger('werkzeug').setLevel(logging.WARNING)  # Suppress per-request logs

# Local imports from the project (after dummy driver is set)
from core.game_manager import GameManager
from main import parse_arguments  # Re-use the full CLI from main.py
from config.ui_constants import GRID_SIZE
from utils.web_utils import build_color_map, translate_end_reason

# ----------------------------------------
# Flask setup (identical static/template folders to replay_web)
# ----------------------------------------
app = Flask(__name__, static_folder='web/static', template_folder='web/templates')

# Global objects filled after thread starts
manager = None            # type: GameManager | None
manager_thread = None

# ----------------------------------------
# Helper: translate live game state into a front-end friendly dict
# ----------------------------------------

def build_state_dict(gm: GameManager):
    game = gm.game
    # Convert numpy arrays to plain lists for JSON
    snake = game.snake_positions.tolist() if hasattr(game.snake_positions, 'tolist') else game.snake_positions
    apple = game.apple_position.tolist() if hasattr(game.apple_position, 'tolist') else game.apple_position

    reason_code = getattr(game.game_state, "game_end_reason", None)
    end_reason_readable = translate_end_reason(reason_code)

    return {
        'snake_positions': snake,
        'apple_position': apple,
        'score': game.score,
        'steps': game.steps,
        'game_number': gm.game_count + 1,
        'round_count': gm.round_count,
        'grid_size': GRID_SIZE,
        'colors': build_color_map(),
        'running': gm.running,
        'game_active': gm.game_active,
        'planned_moves': game.planned_moves,
        'llm_response': getattr(game, 'processed_response', ''),
        'move_pause': gm.get_pause_between_moves(),
        'game_end_reason': end_reason_readable,
    }

# ----------------------------------------
# Background thread target
# ----------------------------------------

def manager_thread_fn(gm: GameManager, args):
    """Background worker for running the game (new or continuation)."""
    try:
        # Continuation mode – resume from previous directory
        cont_dir = getattr(args, "continue_with_game_in_dir", None)
        if cont_dir:
            try:
                # ------------------------------------
                # Load experiment configuration from summary.json to
                # overwrite CLI defaults so the resumed session uses the
                # original provider/model and limits.
                # -------------------------------------
                import json
                import os
                summary_path = os.path.join(cont_dir, "summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    original_cfg = summary.get("configuration", {})
                    for k in (
                        "provider",
                        "model",
                        "parser_provider",
                        "parser_model",
                        "move_pause",
                        "max_steps",
                        "max_consecutive_empty_moves_allowed",
                        "max_consecutive_something_is_wrong_allowed",
                        "max_consecutive_invalid_reversals_allowed",
                        "no_gui",
                    ):
                        if k in original_cfg:
                            setattr(gm.args, k, original_cfg[k])

                # Determine next game
                from utils.file_utils import get_next_game_number
                next_game = get_next_game_number(cont_dir)

                # Flag continuation
                gm.args.is_continuation = True

                # Run continuation
                gm.continue_from_session(cont_dir, next_game)
            except Exception as e:
                print(f"[main_web] Continuation session crashed: {e}")
                return
        else:
            gm.run()
    except Exception as e:
        print(f"[main_web] GameManager thread crashed: {e}")

# ----------------------------------------
# Flask routes
# ----------------------------------------

@app.route('/')
def index():
    return render_template('main.html')  # New live template

@app.route('/api/state')
def api_state():
    if manager is None or manager.game is None:
        return jsonify({'error': 'game not started'})
    return jsonify(build_state_dict(manager))

@app.route('/api/control', methods=['POST'])
def api_control():
    """Minimal play/pause toggle. Accept {"command": "pause"|"play"}."""
    if manager is None:
        return jsonify({'status': 'error', 'msg': 'no manager'})
    cmd = request.json.get('command') if request.is_json else None
    if cmd == 'pause':
        manager.running = False  # stops outer while loops – this effectively pauses
        return jsonify({'status': 'paused'})
    if cmd == 'play':
        # Not trivial to resume once gm.run() has returned, so just acknowledge
        return jsonify({'status': 'unpause-not-supported'})
    return jsonify({'status': 'error', 'msg': 'unknown command'})

# ----------------------------------------
# Entry-point
# ----------------------------------------

def main():
    """Entry point for the web live mode.

    We need to recognise --host/--port *before* calling `parse_arguments()` from
    main.py, otherwise that function will raise an "unrecognised arguments"
    error.  The strategy is:

    1.  Parse --host/--port with a lightweight ArgumentParser using
        `parse_known_args()` so we can keep the remaining CLI untouched.
    2.  Temporarily patch `sys.argv` to exclude those two flags and delegate
        the rest of the CLI to `parse_arguments()` (which contains the full
        game-related options).
    3.  Restore `sys.argv` afterwards and start the GameManager / Flask app.
    """

    # -------------------------------
    # Step 1 – extract host / port, leave the rest intact
    # -------------------------------
    host_port_parser = argparse.ArgumentParser(add_help=False)
    host_port_parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP')
    host_port_parser.add_argument('--port', type=int, default=find_free_port(8000), help='Port number')

    host_port_args, remaining_argv = host_port_parser.parse_known_args()

    # -------------------------------
    # Step 2 – delegate remaining args to the main CLI parser
    # -------------------------------
    argv_backup = sys.argv.copy()
    # Preserve argv[0] (script name) + remaining CLI parts
    sys.argv = [sys.argv[0]] + remaining_argv
    try:
        game_args = parse_arguments()
    finally:
        # Restore original argv regardless of success
        sys.argv = argv_backup

    # Ensure GUI timing code runs (SDL dummy driver prevents a real window)
    game_args.no_gui = False

    # -------------------------------
    # Step 3 – create GameManager (handle continuation vs new session)
    # -------------------------------
    global manager, manager_thread
    manager = GameManager(game_args)
    manager_thread = threading.Thread(target=manager_thread_fn, args=(manager, game_args), daemon=True)
    manager_thread.start()

    # -------------------------------
    # Step 4 – start the Flask app (blocking)
    # -------------------------------
    host = host_port_args.host
    port = host_port_args.port
    print(f"🔌 Serving live game at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main() 