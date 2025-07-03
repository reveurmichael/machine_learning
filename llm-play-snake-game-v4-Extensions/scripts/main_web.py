"""
Main Web Script – Full GameManager Integration
=============================================

Launches the LLM Snake game in web mode.  All Flask routes and the
background GameManager thread live inside `web.main_app.MainWebApp`,
so this file is as small as the other web launchers.

Usage examples
--------------
python scripts/main_web.py
python scripts/main_web.py --provider deepseek
python scripts/main_web.py --model gpt-4 --port 8080
python scripts/main_web.py --continue-with-game-in-dir logs/exp_20250702_123456
"""

from __future__ import annotations

import argparse
import os
import sys
import pathlib
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. Ensure repository root on sys.path and find template/static directories  #
# --------------------------------------------------------------------------- #
# Ensure project root in sys.path for imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Now we can import from utils
from utils.path_utils import ensure_project_root

REPO_ROOT = ensure_project_root()

from utils.web_utils import get_web_dirs  # noqa: E402

TEMPLATE_DIR, STATIC_DIR = get_web_dirs()

# --------------------------------------------------------------------------- #
# 2. CLI parsing – reuse Task-0 CLI plus two web-specific flags               #
# --------------------------------------------------------------------------- #
from scripts.main import parse_arguments  # noqa: E402
from utils.validation_utils import validate_port  # noqa: E402
from config.ui_constants import GRID_SIZE  # noqa: E402


def _parse_web_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    """Parse web-specific flags first, then feed the rest to Task-0 parser."""
    web_parser = argparse.ArgumentParser(description="Snake Game – LLM Web UI")
    web_parser.add_argument(
        "--port", type=int, default=None, help="Port (auto if omitted)"
    )
    web_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Bind address"
    )
    web_args, remaining = web_parser.parse_known_args()

    # Re-invoke Task-0 CLI parser on the remaining args
    backup_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining
    try:
        game_args = parse_arguments()
    finally:
        sys.argv = backup_argv
    return web_args, game_args


# --------------------------------------------------------------------------- #
# 3. Main entry point                                                         #
# --------------------------------------------------------------------------- #
from web.main_app import MainWebApp  # noqa: E402


def main() -> int:
    web_args, game_args = _parse_web_args()

    host = web_args.host
    port = validate_port(web_args.port) if web_args.port else None
    if port is None:  # auto
        from utils.network_utils import random_free_port  # noqa: E402

        port = random_free_port()

    print("=" * 60)
    print(f"[MainWeb] Host            : {host}")
    print(f"[MainWeb] Port            : {port}")
    print(f"[MainWeb] LLM Provider    : {game_args.provider}/{game_args.model}")
    print(f"[MainWeb] Max Games       : {game_args.max_games}")
    if game_args.continue_with_game_in_dir:
        print(f"[MainWeb] Continue From   : {game_args.continue_with_game_in_dir}")
    print("=" * 60)

    # Headless SDL so no macOS NSWindow crash
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    app = MainWebApp(
        provider=game_args.provider,
        model=game_args.model,
        grid_size=GRID_SIZE,
        max_games=game_args.max_games,
        port=port,
        continue_from_folder=game_args.continue_with_game_in_dir,
        no_gui=game_args.no_gui,  # honour --no-gui flag
        game_args=game_args,  # pass full namespace for completeness
    )

    print("[MainWeb] Server starting – open the URL below in your browser")
    app.run(host=host, port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())


