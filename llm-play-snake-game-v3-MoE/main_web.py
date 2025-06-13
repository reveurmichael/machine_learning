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
import time
import sys
from flask import Flask, render_template, jsonify, request

# Local imports from the project (after dummy driver is set)
from core.game_manager import GameManager
from main import parse_arguments  # Re-use the full CLI from main.py
from config import COLORS, GRID_SIZE

# ---------------------------------------------------------------------------
# Flask setup (identical static/template folders to replay_web)
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder='web/static', template_folder='web/templates')

# Global objects filled after thread starts
manager = None            # type: GameManager | None
manager_thread = None

# ---------------------------------------------------------------------------
# Helper: translate live game state into a front-end friendly dict
# ---------------------------------------------------------------------------

def build_state_dict(gm: GameManager):
    game = gm.game
    # Convert numpy arrays to plain lists for JSON
    snake = game.snake_positions.tolist() if hasattr(game.snake_positions, 'tolist') else game.snake_positions
    apple = game.apple_position.tolist() if hasattr(game.apple_position, 'tolist') else game.apple_position

    return {
        'snake_positions': snake,
        'apple_position': apple,
        'score': game.score,
        'steps': game.steps,
        'game_number': gm.game_count + 1,
        'round_count': gm.round_count,
        'grid_size': GRID_SIZE,
        'colors': {
            'snake_head': COLORS['SNAKE_HEAD'],
            'snake_body': COLORS['SNAKE_BODY'],
            'apple': COLORS['APPLE'],
            'background': COLORS['BACKGROUND'],
            'grid': COLORS['GRID'],
        },
        'running': gm.running,
        'game_active': gm.game_active,
        'planned_moves': game.planned_moves,
        'llm_response': getattr(game, 'processed_response', ''),
        'move_pause': gm.get_pause_between_moves(),
    }

# ---------------------------------------------------------------------------
# Background thread target
# ---------------------------------------------------------------------------

def manager_thread_fn(gm: GameManager):
    """Run the GameManager until completes. We set use_gui=False so it is headless."""
    try:
        gm.run()
    except Exception as e:
        print(f"[main_web] GameManager thread crashed: {e}")

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('main.html')  # New live template

@app.route('/api/state')
def api_state():
    global manager
    if manager is None or manager.game is None:
        return jsonify({'error': 'game not started'})
    return jsonify(build_state_dict(manager))

@app.route('/api/control', methods=['POST'])
def api_control():
    """Minimal play/pause toggle. Accept {"command": "pause"|"play"}."""
    global manager
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

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Web live Snake game session.')
    # Let parse_arguments supply the main game args
    args, unknown = parser.parse_known_args([])  # placeholder to create object
    # Re-parse with full set
    full_args = parse_arguments()
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    # Parse again to include host/port extras
    final_args = parser.parse_args(namespace=full_args)

    host = final_args.host
    port = final_args.port

    # Keep GUI logic enabled for correct timing, but SDL dummy driver prevents a real window
    final_args.no_gui = False
    

    # Start GameManager in background thread
    global manager, manager_thread
    manager = GameManager(final_args)
    manager_thread = threading.Thread(target=manager_thread_fn, args=(manager,), daemon=True)
    manager_thread.start()

    # Launch flask app (blocking)
    print(f"🔌 Serving live game at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main() 