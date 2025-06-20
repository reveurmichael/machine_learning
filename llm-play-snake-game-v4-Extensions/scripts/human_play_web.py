"""Snake Game – Human Play (Flask Web UI).

Provide a REST-driven web front-end that mirrors the historic PyGame window
controls so users can play via WASD / arrow keys in their browsers.  No
WebSocket dependency – plain HTTP keeps requirements minimal.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Path/bootstrap – same two-liner used across scripts/
# ---------------------------------------------------------------------------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Must come *after* sys.path tweak so the import resolves even when launched
# from inside /scripts.
from utils.path_utils import ensure_repo_root  # noqa: E402
ensure_repo_root()

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import argparse
import threading
import time
import logging
from typing import Any, Dict

from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Third-party – PyGame is still required for core logic (GameController)
# ---------------------------------------------------------------------------
from core.game_controller import GameController
from utils.network_utils import find_free_port
from utils.web_utils import build_state_dict, translate_end_reason

# ---------------------------------------------------------------------------
# Flask application setup
# ---------------------------------------------------------------------------

# Absolute paths so Flask/Jinja resolves templates regardless of cwd.
_repo_root = pathlib.Path(__file__).resolve().parent.parent
_static_folder = str(_repo_root / "web" / "static")
_template_folder = str(_repo_root / "web" / "templates")

app = Flask(__name__, static_folder=_static_folder, template_folder=_template_folder)

logging.getLogger("werkzeug").setLevel(logging.WARNING)  # Quiet logs

# ---------------------------------------------------------------------------
# Runtime globals – kept minimal & documented
# ---------------------------------------------------------------------------
_game_controller: GameController | None = None  # singleton per process
_game_thread: threading.Thread | None = None    # keeps background loop alive
_running: bool = True                           # stop-flag for thread

# ---------------------------------------------------------------------------
# Thin wrapper around GameController to expose web-friendly helpers
# ---------------------------------------------------------------------------
class WebGameController(GameController):
    """GameController adapted for the Flask web UI.

    Changes compared to the base class:
    * always initialised with ``use_gui=False`` (rendering done client-side)
    * exposes :py:meth:`get_current_state` returning JSON-serialisable dict.
    """

    def __init__(self, grid_size: int = 10):
        super().__init__(grid_size=grid_size, use_gui=False)
        self.grid_size = grid_size  # invariant – template relies on 10×10
        self.game_over: bool = False
        self.game_end_reason: str | None = None

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------
    def get_current_state(self) -> Dict[str, Any]:
        """Return lightweight state for AJAX polling."""
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

    # ------------------------------------------------------------------
    # Overrides – we need to track *game_over* & *reason*
    # ------------------------------------------------------------------
    def make_move(self, direction_key: str):  # noqa: D401 – keep signature
        active, apple_eaten = super().make_move(direction_key)
        if not active:
            self.game_over = True
            if self.last_collision_type == "WALL":
                self.game_end_reason = "WALL"
            elif self.last_collision_type == "SELF":
                self.game_end_reason = "SELF"
        return active, apple_eaten

    def reset(self):  # noqa: D401 – simple override
        super().reset()
        self.game_over = False
        self.game_end_reason = None

# ---------------------------------------------------------------------------
# Background thread – keeps the interpreter alive so PyGame clock doesnt exit
# ---------------------------------------------------------------------------

def _background_loop():
    while _running:
        time.sleep(0.1)

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("human_play.html")

@app.route("/api/state")
def api_state():  # noqa: D401 – route
    if _game_controller is None:
        return jsonify({"error": "Game not initialised"})
    return jsonify(_game_controller.get_current_state())

@app.route("/api/move", methods=["POST"])
def api_move():
    if _game_controller is None:
        return jsonify({"error": "Game not initialised"})
    data = request.get_json(silent=True) or {}
    direction = data.get("direction")
    if direction not in {"UP", "DOWN", "LEFT", "RIGHT"}:  # minimal validation
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
        return jsonify({"error": "Game not initialised"})
    _game_controller.reset()
    return jsonify({"status": "ok"})

# ---------------------------------------------------------------------------
# CLI / entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – simple wrapper
    """Run the Flask web server for human-play mode."""

    global _game_controller, _game_thread

    parser = argparse.ArgumentParser(description="Web human-play mode for Snake.")
    parser.add_argument("--port", type=int, default=find_free_port(8000), help="TCP port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address")
    args = parser.parse_args()

    # ----- Initialise core game logic ----------------------------------
    _game_controller = WebGameController(grid_size=10)

    # Background heartbeat so PyGame time module doesn't complain.
    _game_thread = threading.Thread(target=_background_loop, daemon=True)
    _game_thread.start()

    print(f"\n🐍 Snake Game – Human Play at http://{args.host}:{args.port}\n")
    print("Controls: Arrow Keys / WASD, R to reset (via UI buttons)")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":  # pragma: no cover
    main() 
