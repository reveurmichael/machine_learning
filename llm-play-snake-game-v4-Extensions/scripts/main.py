"""Wrapper script delegating to the original project-root `main.py`.

It guarantees the working directory is the repository root so that relative
paths (log folders, Flask template dirs, …) behave exactly like when the
script is launched from the root.

This whole module is Task0 specific.
"""

from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from utils.path_utils import ensure_project_root

# ------------------
# Ensure current working directory == repository root
# ------------------
REPO_ROOT = ensure_project_root()

# ------------------
# Replace wrapper delegation with the *actual* Task-0 implementation so the
# project no longer relies on a root-level `main.py`.
# ------------------

# ---- Standard library imports (duplicated harmlessly) ----
import argparse
import logging
import sys
import time
import pygame
from colorama import Fore, init as init_colorama

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Project imports ----
from config import (
    PAUSE_BETWEEN_MOVES_SECONDS,
    MAX_STEPS_ALLOWED,
    MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
    MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
    MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
    MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED,
    MAX_GAMES_ALLOWED,
    SLEEP_AFTER_EMPTY_STEP,
)
from config.game_constants import list_available_providers

AVAILABLE_PROVIDERS = list_available_providers()

from core.game_manager import GameManager
from llm.setup_utils import check_env_setup
from llm.providers import get_available_models
from llm.agent_llm import LLMSnakeAgent

# Initialise colour output early (no-op if already called)
init_colorama(autoreset=True)


# ------------------
# CLI parsing (verbatim from original main.py)
# ------------------

def get_parser() -> argparse.ArgumentParser:
    """Creates and returns the argument parser for the main game CLI.

    This function is separate from `parse_arguments` to allow other scripts
    (e.g., main_web.py) to reuse the parser configuration.

    Returns:
        An argparse.ArgumentParser instance.
    """
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
    parser.add_argument("--pause-between-moves", type=float, default=PAUSE_BETWEEN_MOVES_SECONDS,
                        help=f"Pause between moves in seconds (default {PAUSE_BETWEEN_MOVES_SECONDS})")
    parser.add_argument("--sleep-before-launching", "-s", type=float, default=0.0,
                        help="Time (minutes, fractions allowed) to sleep before launching")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_ALLOWED,
                        help=f"Max steps per game (default {MAX_STEPS_ALLOWED})")

    # --- Safety limits ---------------------
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

    return parser


def parse_arguments():
    """Parse command line arguments."""
    parser = get_parser()
    args = parser.parse_args()

    # --- Post-processing defaults ---------------------
    if args.provider and args.model is None:
        # Import here to get the actual default models from each provider
        from llm.providers import get_default_model
        
        # First try hardcoded defaults for known providers
        default_models = {
            "hunyuan": "hunyuan-turbos-latest",
            "mistral": "mistral-medium-latest",
            "deepseek": "deepseek-chat",
            "ollama": "deepseek-r1:7b",
        }
        args.model = default_models.get(args.provider.lower())
        
        # If no hardcoded default found, use the provider's own default
        if args.model is None:
            try:
                args.model = get_default_model(args.provider)
            except Exception:
                print(Fore.YELLOW + f"Warning: Could not get default model for provider {args.provider}")

    if args.parser_provider is None:
        args.parser_provider = "none"

    if args.parser_provider.lower() != "none" and args.parser_model is None:
        # Import here to get the actual default models from each provider
        from llm.providers import get_default_model
        
        # First try hardcoded defaults for known providers
        defaults_parser = {
            "hunyuan": "hunyuan-turbos-latest",
            "mistral": "mistral-medium-latest",
            "deepseek": "deepseek-chat",
            "ollama": "deepseek-r1:7b",
        }
        args.parser_model = defaults_parser.get(args.parser_provider.lower())
        
        # If no hardcoded default found, use the provider's own default
        if args.parser_model is None:
            # If parser provider has no default, fall back to primary model
            args.parser_model = args.model

    # --- Continuation mode checks ---------------------
    if args.continue_with_game_in_dir:
        raw_args = " ".join(sys.argv[1:])
        restricted = [
            "--provider", "--p1", "--model", "--m1",
            "--parser-provider", "--p2", "--parser-model",
            "--pause-between-moves", "--max-steps",
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

class MainApplication:
    """
    OOP wrapper for the main Snake game application.
    
    This class encapsulates the main game logic in an object-oriented way,
    following the naming conventions and design patterns used throughout
    the project. It replaces the procedural main() function.
    
    Design Pattern: Facade Pattern
    - Provides a simplified interface to the complex game subsystem
    - Encapsulates initialization, execution, and cleanup logic
    - Makes the application easier to test and extend
    """
    
    def __init__(self, args=None):
        """
        Initialize the main application.
        
        Args:
            args: Parsed command line arguments (optional, will parse if None)
        """
        self.args = args or parse_arguments()
        self.game_manager = None
        self.controller = None
        
        logger.info("Initialized MainApplication")
    
    def setup_environment(self) -> None:
        """Set up the environment and validate configuration."""
        # Environment setup
        primary_env_ok = check_env_setup(self.args.provider)
        if self.args.parser_provider and self.args.parser_provider.lower() != "none":
            _ = check_env_setup(self.args.parser_provider)

        if not primary_env_ok:
            print(Fore.RED + "Primary LLM environment not ready.")
            sys.exit(1)
        
        # Sleep before launching if requested
        if self.args.sleep_before_launching > 0:
            sleep_time = self.args.sleep_before_launching * 60
            print(f"Sleeping for {self.args.sleep_before_launching} minutes before launching...")
            time.sleep(sleep_time)
    
    def create_game_components(self) -> None:
        """Create and configure game components."""
        from core.game_controller import CLIGameController
        
        # Create GameManager
        self.game_manager = GameManager(self.args)
        
        # Create OOP controller
        self.controller = CLIGameController(
            self.game_manager, 
            use_gui=not self.args.no_gui
        )
        
        logger.info("Created game components")
    
    def run_application(self) -> None:
        """Run the complete application using OOP controller."""
        try:
            # Parse arguments and handle continuation mode
            try:
                if self.args.continue_with_game_in_dir:
                    print(Fore.GREEN + f"🔄 Continuing from {self.args.continue_with_game_in_dir}")
                    # Create and configure game manager for continuation
                    self.game_manager = GameManager.continue_from_directory(self.args)
                    
                    # Set up LLM agent for Task-0 continuation
                    self.game_manager.agent = LLMSnakeAgent(
                        self.game_manager, 
                        provider=self.args.provider, 
                        model=self.args.model
                    )
                    
                    # Create OOP controller for continuation
                    from core.game_controller import CLIGameController
                    self.controller = CLIGameController(
                        self.game_manager, 
                        use_gui=not self.args.no_gui
                    )
                    
                    # Run the continuation session
                    self.controller.run_game_session()
                    return
                    
            except ValueError as e:
                print(Fore.RED + f"Command-line error: {e}")
                print(Fore.YELLOW + "For help, use: python scripts/main.py --help")
                return
            
            self.setup_environment()
            self.create_game_components()
            
            # Set up LLM agent for Task-0
            self.game_manager.agent = LLMSnakeAgent(
                self.game_manager, 
                provider=self.args.provider, 
                model=self.args.model
            )
            
            # Use the OOP controller's template method
            self.controller.run_game_session()
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠️ Game interrupted by user")
        except Exception as e:
            print(f"{Fore.RED}❌ Fatal error: {e}")
            logger.error(f"Fatal error in main application: {e}", exc_info=True)
            raise
        finally:
            pygame.quit()


def main():
    """Initialize and run the LLM Snake game using OOP approach."""
    app = MainApplication()
    app.run_application()


if __name__ == "__main__":
    main() 
