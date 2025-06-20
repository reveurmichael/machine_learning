"""Snake Game – Web Live Mode (scripts variant)

Run a live LLM-controlled Snake session and expose JSON snapshots at
`/api/state` for a lightweight Flask front-end.  This is a one-file move of
``main_web.py`` into the ``scripts`` package; functionality is unchanged.

This whole module is Task0 specific.
"""

from __future__ import annotations

# ---------------------------------------------------------------
# Guarantee repo root is on sys.path before importing project modules.
# ---------------------------------------------------------------

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Now safe to import helpers
from utils.path_utils import ensure_repo_root, enable_headless_pygame  # noqa: E402

ensure_repo_root()
enable_headless_pygame()

# --------------------------
# Standard library imports (identical to original)
# --------------------------
import argparse
import threading
import logging
from flask import Flask, render_template, jsonify, request

from utils.network_utils import find_free_port

logging.getLogger("werkzeug").setLevel(logging.WARNING)  # Suppress per-request logs

# --------------------------
# Project-internal imports
# --------------------------
from core.game_manager import GameManager
from scripts.main import parse_arguments  # Re-use full CLI from scripts/main.py
from config.ui_constants import GRID_SIZE
from utils.web_utils import build_color_map, translate_end_reason
from llm.agent_llm import LLMSnakeAgent

# --------------------------
# Flask setup (static/template folders)
# --------------------------
_static_folder = str(_repo_root / "web" / "static")
_template_folder = str(_repo_root / "web" / "templates")

app = Flask(__name__, static_folder=_static_folder, template_folder=_template_folder)

# Global objects filled after the worker thread starts
manager: GameManager | None = None
manager_thread: threading.Thread | None = None

# --------------------------
# Helper: translate live game state into a front-end-friendly dict
# --------------------------

def build_state_dict(gm: GameManager):
    game = gm.game
    snake = game.snake_positions.tolist() if hasattr(game.snake_positions, "tolist") else game.snake_positions
    apple = game.apple_position.tolist() if hasattr(game.apple_position, "tolist") else game.apple_position

    reason_code = getattr(game.game_state, "game_end_reason", None)
    end_reason_readable = translate_end_reason(reason_code)

    return {
        "snake_positions": snake,
        "apple_position": apple,
        "score": game.score,
        "steps": game.steps,
        "game_number": gm.game_count + 1,
        "round_count": gm.round_count,
        "grid_size": GRID_SIZE,
        "colors": build_color_map(),
        "running": gm.running,
        "game_active": gm.game_active,
        "planned_moves": game.planned_moves,
        "llm_response": getattr(game, "processed_response", ""),
        "move_pause": gm.get_pause_between_moves(),
        "game_end_reason": end_reason_readable,
    }

# --------------------------
# Background worker thread that runs GameManager
# --------------------------

def _manager_thread_fn(gm: GameManager, args):
    """Run new or continuation session in background thread."""
    try:
        cont_dir = getattr(args, "continue_with_game_in_dir", None)
        if cont_dir:
            try:
                import json

                summary_path = Path(cont_dir) / "summary.json"
                if summary_path.exists():
                    with summary_path.open("r", encoding="utf-8") as f:
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
                        "max_consecutive_no_path_found_allowed",
                        "sleep_after_empty_step",
                        "no_gui",
                    ):
                        if k in original_cfg:
                            setattr(gm.args, k, original_cfg[k])

                from utils.file_utils import get_next_game_number

                next_game = get_next_game_number(cont_dir)
                gm.args.is_continuation = True
                gm.continue_from_session(cont_dir, next_game)
            except Exception as exc:
                print(f"[main_web] Continuation crashed: {exc}")
                return
        else:
            gm.run()
    except Exception as exc:
        print(f"[main_web] GameManager thread crashed: {exc}")

# --------------------------
# Flask routes
# --------------------------

@app.route("/")
def index():
    return render_template("main.html")

@app.route("/api/state")
def api_state():  # noqa: F401 – used via Flask routing
    if manager is None or manager.game is None:
        return jsonify({"error": "game not started"})
    return jsonify(build_state_dict(manager))

@app.route("/api/control", methods=["POST"])
def api_control():  # noqa: F401 – used via Flask routing
    if manager is None:
        return jsonify({"status": "error", "msg": "no manager"})
    cmd = request.json.get("command") if request.is_json else None
    if cmd == "pause":
        manager.running = False
        return jsonify({"status": "paused"})
    if cmd == "play":
        return jsonify({"status": "unpause-not-supported"})
    return jsonify({"status": "error", "msg": "unknown command"})

# --------------------------
# Entry point
# --------------------------

def main():
    """CLI front-end identical to the legacy root `main_web.py`."""

    # 1. Extract --host / --port first
    host_port_parser = argparse.ArgumentParser(add_help=False)
    host_port_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    host_port_parser.add_argument("--port", type=int, default=find_free_port(8000), help="Port number")
    host_port_args, remaining_argv = host_port_parser.parse_known_args()

    # 2. Delegate remaining CLI args to scripts.main.parse_arguments()
    argv_backup = sys.argv.copy()
    sys.argv = [sys.argv[0]] + remaining_argv
    try:
        game_args = parse_arguments()
    finally:
        sys.argv = argv_backup

    global manager, manager_thread
    manager = GameManager(game_args)

    # Inject Task-0 agent
    manager.agent = LLMSnakeAgent(manager, provider=game_args.provider, model=game_args.model)

    manager_thread = threading.Thread(target=_manager_thread_fn, args=(manager, game_args), daemon=True)
    manager_thread.start()

    host = host_port_args.host
    port = host_port_args.port
    print(f"🔌 Serving live game at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main() 
