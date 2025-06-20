"""Wrapper script delegating to the original project-root `main.py`.

It guarantees the working directory is the repository root so that relative
paths (log folders, Flask template dirs, …) behave exactly like when the
script is launched from the root.

This whole module is Task0 specific.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# ------------------
# Ensure current working directory == repository root
# ------------------
_repo_root = Path(__file__).resolve().parent.parent
if Path.cwd() != _repo_root:
    os.chdir(_repo_root)

# Ensure repo root is on sys.path so we can import top-level modules
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ------------------
# Replace wrapper delegation with the *actual* Task-0 implementation so the
# project no longer relies on a root-level `main.py`.
# ------------------

# ---- Standard library imports (duplicated harmlessly) ----
import argparse
import pygame
from colorama import Fore, init as init_colorama

# ---- Project imports ----
from config import (
    PAUSE_BETWEEN_MOVES_SECONDS,
    MAX_STEPS_ALLOWED,
    MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
    MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
    MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
    MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED,
    MAX_GAMES_ALLOWED,
    AVAILABLE_PROVIDERS,
    SLEEP_AFTER_EMPTY_STEP,
)
from core.game_manager import GameManager
from llm.setup_utils import check_env_setup
from llm.providers import get_available_models

# Initialise colour output early (no-op if already called)
init_colorama(autoreset=True)


# ------------------
# CLI parsing (verbatim from original main.py)
# ------------------

def parse_arguments():
    """Parse command-line arguments (same behaviour as legacy main.py)."""
    parser = argparse.ArgumentParser(description="LLM-guided Snake game")
    provider_help = (
        "LLM provider to use for primary LLM. Available: " + ", ".join(AVAILABLE_PROVIDERS)
    )
    parser.add_argument("--provider", "--p1", type=str, default="hunyuan", help=provider_help)

    # Build example string for --model help text
    examples: list[str] = []
    for _prov in AVAILABLE_PROVIDERS:
        _models = get_available_models(_prov)
        if _models:
            examples.append(f"{_prov}: {', '.join(_models[:2])}{'…' if len(_models) > 2 else ''}")
    model_help = "Model name to use for primary LLM. Examples – " + "; ".join(examples)
    parser.add_argument("--model", "--m1", type=str, default=None, help=model_help)

    parser.add_argument("--parser-provider", "--p2", type=str, default=None,
                        help='LLM provider for secondary LLM. Default "none" for single-LLM mode.')
    parser.add_argument("--parser-model", "--m2", type=str, default=None,
                        help="Model name for secondary LLM (uses provider default if omitted)")

    parser.add_argument("--max-games", "-g", type=int, default=MAX_GAMES_ALLOWED,
                        help=f"Maximum games to play (default {MAX_GAMES_ALLOWED})")
    parser.add_argument("--move-pause", type=float, default=PAUSE_BETWEEN_MOVES_SECONDS,
                        help=f"Pause between moves in seconds (default {PAUSE_BETWEEN_MOVES_SECONDS})")
    parser.add_argument("--sleep-before-launching", "-s", type=float, default=0.0,
                        help="Time (minutes, fractions allowed) to sleep before launching")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_ALLOWED,
                        help=f"Max steps per game (default {MAX_STEPS_ALLOWED})")

    # --- Safety limits --------------------------
    parser.add_argument("--max-consecutive-empty-moves-allowed", type=int,
                        default=MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
                        help=f"Max consecutive EMPTY ticks (default {MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED})")
    parser.add_argument("--max-consecutive-something-is-wrong-allowed", type=int,
                        default=MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
                        help=f"Max consecutive SOMETHING_IS_WRONG ticks (default {MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED})")
    parser.add_argument("--max-consecutive-invalid-reversals-allowed", type=int,
                        default=MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
                        help=f"Max consecutive invalid reversals (default {MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED})")
    parser.add_argument("--max-consecutive-no-path-found-allowed", type=int,
                        default=MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED,
                        help=f"Max consecutive NO_PATH_FOUND ticks (default {MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED})")

    parser.add_argument("--sleep-after-empty-step", type=float, default=SLEEP_AFTER_EMPTY_STEP,
                        help="Minutes to sleep after every EMPTY tick. 0 disables.")

    parser.add_argument("--no-gui", "-n", action="store_true", help="Run headless (no PyGame window)")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory to store logs")
    parser.add_argument("--continue-with-game-in-dir", "-c", type=str, default=None,
                        help="Resume from existing experiment directory")

    args = parser.parse_args()

    # --- Post-processing defaults --------------------------
    if args.provider and args.model is None:
        default_models = {
            "hunyuan": "hunyuan-turbos-latest",
            "mistral": "mistral-medium-latest",
            "deepseek": "deepseek-chat",
        }
        args.model = default_models.get(args.provider.lower(), args.model)

    if args.parser_provider is None:
        args.parser_provider = "none"

    if args.parser_provider.lower() != "none" and args.parser_model is None:
        defaults_parser = {
            "hunyuan": "hunyuan-turbos-latest",
            "mistral": "mistral-medium-latest",
            "deepseek": "deepseek-chat",
        }
        args.parser_model = defaults_parser.get(args.parser_provider.lower(), args.model)

    # --- Continuation mode checks --------------------------
    if args.continue_with_game_in_dir:
        raw_args = " ".join(sys.argv[1:])
        restricted = [
            "--provider", "--p1", "--model", "--m1",
            "--parser-provider", "--p2", "--parser-model",
            "--move-pause", "--max-steps",
            "--max-consecutive-empty-moves-allowed",
            "--max-consecutive-something-is-wrong-allowed",
            "--max-consecutive-invalid-reversals-allowed",
            "--max-consecutive-no-path-found-allowed",
            "--sleep-after-empty-step", "--log-dir",
        ]
        for flag in restricted:
            if flag in raw_args:
                raise ValueError(
                    f"Cannot use {flag} with --continue-with-game-in-dir. "
                    "Only --max-games, --no-gui, and --sleep-before-launching are allowed."
                )
    return args


# ------------------
# Main entry (unchanged)
# ------------------

def main():
    """Initialize and run the LLM Snake game (identical logic)."""
    try:
        try:
            args = parse_arguments()
        except ValueError as e:
            print(Fore.RED + f"Command-line error: {e}")
            print(Fore.YELLOW + "For help, use: python scripts/main.py --help")
            sys.exit(1)

        if args.continue_with_game_in_dir:
            print(Fore.GREEN + f"🔄 Continuing from {args.continue_with_game_in_dir}")
            GameManager.continue_from_directory(args)
            return

        primary_env_ok = check_env_setup(args.provider)
        if args.parser_provider and args.parser_provider.lower() != "none":
            _ = check_env_setup(args.parser_provider)

        if not primary_env_ok:
            print(Fore.RED + "Primary LLM environment not ready.")
            sys.exit(1)

        gm = GameManager(args)
        gm.run()
    except KeyboardInterrupt:
        print("\nExiting…")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main() 
