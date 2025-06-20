"""Snake Game – Human Play Web mode (scripts version).

This whole module is NOT Task0 specific. But no need to make it generic anyway.

Run with:
    python scripts/human_play_web.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations


from utils.path_utils import ensure_repo_root

_repo_root = ensure_repo_root()

# --------------------------
# Original implementation from root human_play_web.py starts here
# --------------------------

import argparse
import threading
import time
import logging
from flask import Flask, render_template, request, jsonify

from core.game_controller import GameController
from utils.network_utils import find_free_port
from utils.web_utils import build_state_dict, translate_end_reason

app = Flask(__name__, 
           static_folder=str(_repo_root / "web" / "static"), 
           template_folder=str(_repo_root / "web" / "templates"))

# Global state
_game_controller: GameController | None = None
_game_thread: threading.Thread | None = None
_running = True

logging.getLogger("werkzeug").setLevel(logging.WARNING)


class WebGameController(GameController):
    """GameController adapted for web-based human play."""

    def __init__(self, grid_size: int = 10):
        super().__init__(grid_size=grid_size, use_gui=False)
        self.grid_size = grid_size
        self.game_over = False
        self.game_end_reason = None

    def get_current_state(self):
        return build_state_dict(
            self.snake_positions,
            self.apple_position,
            self.score,
            self.steps,
            self.grid_size,
            extra={
                "game_over": self.game_over,
                "game_end_reason": translate_end_reason(self.game_end_reason),
            },
        )

    def make_move(self, direction_key):
        game_active, apple_eaten = super().make_move(direction_key)
        if not game_active:
            self.game_over = True
            if self.last_collision_type == "wall":
                self.game_end_reason = "WALL"
            elif self.last_collision_type == "self":
                self.game_end_reason = "SELF"
        return game_active, apple_eaten

    def reset(self):
        super().reset()
        self.game_over = False
        self.game_end_reason = None


def _background_loop():
    while _running:
        time.sleep(0.1)


@app.route("/")
def index():
    return render_template("human_play.html")


@app.route("/api/state")
def api_state():
    if _game_controller is None:
        return jsonify({"error": "Game controller not initialized"})
    return jsonify(_game_controller.get_current_state())


@app.route("/api/move", methods=["POST"])
def api_move():
    if _game_controller is None:
        return jsonify({"error": "Game controller not initialized"})
    data = request.json
    direction = data.get("direction")
    if direction not in ["UP", "DOWN", "LEFT", "RIGHT"]:
        return jsonify({"status": "error", "message": "Invalid direction"})
    active, apple = _game_controller.make_move(direction)
    return jsonify({
        "status": "ok",
        "game_active": active,
        "apple_eaten": apple,
        "score": _game_controller.score,
        "steps": _game_controller.steps,
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    if _game_controller is None:
        return jsonify({"error": "Game controller not initialized"})
    _game_controller.reset()
    return jsonify({"status": "ok"})


def main():
    global _game_controller, _game_thread, _running

    parser = argparse.ArgumentParser(description="Web-based human play mode for Snake game.")
    parser.add_argument("--port", type=int, default=find_free_port(8000), help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host bind address")
    args = parser.parse_args()

    _game_controller = WebGameController(grid_size=10)

    _game_thread = threading.Thread(target=_background_loop, daemon=True)
    _game_thread.start()

    print(f"\n🐍 Snake Game Web Human Play at http://{args.host}:{args.port}\n")
    print("Controls: Arrow Keys/WASD, R to reset")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main() 
