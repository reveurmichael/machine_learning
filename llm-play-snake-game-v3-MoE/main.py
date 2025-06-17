"""
Main entry point for the LLM-controlled Snake game.
This script parses command line arguments and launches the game.
"""

import sys
import argparse
import pygame
from colorama import Fore, init as init_colorama
from config import PAUSE_BETWEEN_MOVES_SECONDS, MAX_STEPS_ALLOWED, MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED, MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED, MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED, MAX_GAMES_ALLOWED, AVAILABLE_PROVIDERS
from core.game_manager import GameManager
from llm.setup_utils import check_env_setup
from llm.providers import get_available_models

# Initialize colorama for colored terminal output
init_colorama(autoreset=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM-guided Snake game")
    provider_help = (
        "LLM provider to use for primary LLM. Available: " + ", ".join(AVAILABLE_PROVIDERS)
    )
    parser.add_argument(
        "--provider",
        "--p1",
        type=str,
        default="hunyuan",
        help=provider_help,
    )
    examples: list[str] = []
    for _prov in AVAILABLE_PROVIDERS:
        _models = get_available_models(_prov)
        if _models:
            examples.append(f"{_prov}: {', '.join(_models[:2])}{'…' if len(_models) > 2 else ''}")

    model_help = (
        "Model name to use for primary LLM. Examples – " + "; ".join(examples)
    )

    parser.add_argument(
        "--model",
        "--m1",
        type=str,
        default=None,
        help=model_help,
    )
    parser.add_argument(
        "--parser-provider",
        "--p2",
        type=str,
        default=None,
        help='LLM provider to use for secondary LLM. Default: "none" (single LLM mode). Set to same as --provider for dual LLM mode.',
    )
    parser.add_argument(
        "--parser-model",
        "--m2",
        type=str,
        default=None,
        help="Model name to use for secondary LLM (if not specified, uses the default for the secondary provider)",
    )
    parser.add_argument(
        "--max-games",
        "-g",
        type=int,
        default=MAX_GAMES_ALLOWED,
        help=f"Maximum number of games to play (default: {MAX_GAMES_ALLOWED})",
    )
    parser.add_argument(
        "--move-pause",
        type=float,
        default=PAUSE_BETWEEN_MOVES_SECONDS,
        help=f"Pause between moves in seconds (default: {PAUSE_BETWEEN_MOVES_SECONDS})",
    )
    parser.add_argument(
        "--sleep-before-launching",
        "-s",
        type=float,
        default=0.0,
        help="Time to sleep (in minutes; fractions allowed) before launching the program",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS_ALLOWED,
        help=f"Maximum steps a snake can take in a single game (default: {MAX_STEPS_ALLOWED})",
    )
    parser.add_argument(
        "--max-consecutive-empty-moves-allowed",
        type=int,
        default=MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
        help=f"Maximum consecutive empty moves before game over (default: {MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED})",
    )
    parser.add_argument(
        "--max-consecutive-something-is-wrong-allowed",
        type=int,
        default=MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
        help=f"Maximum consecutive errors allowed before game over (default: {MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED})",
    )
    parser.add_argument(
        "--max-consecutive-invalid-reversals-allowed",
        type=int,
        default=MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
        help=f"Maximum consecutive invalid reversals allowed before game over (default: {MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED})",
    )
    parser.add_argument(
        "--no-gui", "-n", action="store_true", help="Run without GUI (text-only mode)"
    )
    parser.add_argument(
        "--log-dir", type=str, default=None, help="Directory to store logs"
    )
    parser.add_argument(
        "--continue-with-game-in-dir",
        "-c",
        type=str,
        default=None,
        help="Continue an experiment from a directory containing previous games",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Set default model names based on provider at the earliest stage
    # This ensures model names are consistently set before any checks or logging
    if args.provider:
        # Set default model for primary LLM if not specified
        if args.model is None:
            if args.provider.lower() == "hunyuan":
                args.model = "hunyuan-turbos-latest"
            elif args.provider.lower() == "mistral":
                args.model = "mistral-medium-latest"
            elif args.provider.lower() == "deepseek":
                args.model = "deepseek-chat"

    # Set parser provider and model if not specified
    if args.parser_provider is None:
        # Default to 'none' instead of using primary provider
        # This makes single-LLM mode the default
        args.parser_provider = "none"

    # Set default model for parser LLM if not specified but parser is enabled
    if (
        args.parser_provider
        and args.parser_provider.lower() != "none"
        and args.parser_model is None
    ):
        if args.parser_provider.lower() == "hunyuan":
            args.parser_model = "hunyuan-turbos-latest"
        elif args.parser_provider.lower() == "mistral":
            args.parser_model = "mistral-medium-latest"
        elif args.parser_provider.lower() == "deepseek":
            args.parser_model = "deepseek-chat"
        # If no specific default, use the same as primary model
        elif args.model:
            args.parser_model = args.model

    # Validate continue mode restrictions
    if args.continue_with_game_in_dir:
        # Get all command line arguments
        raw_args = " ".join(sys.argv[1:])

        # List of arguments not allowed with continue mode
        restricted_args = [
            "--provider",
            "--p1",
            "--model",
            "--m1",
            "--parser-provider",
            "--p2",
            "--parser-model",
            "--move-pause",
            "--max-steps",
            "--max-consecutive-empty-moves-allowed",
            "--max-consecutive-something-is-wrong-allowed",
            "--max-consecutive-invalid-reversals-allowed",
            "--log-dir",
        ]

        # Check for any restricted arguments
        for arg in restricted_args:
            if arg in raw_args:
                raise ValueError(
                    f"Cannot use {arg} with --continue-with-game-in-dir. "
                    f"Only --max-games, --no-gui, and --sleep-before-launching are allowed in continuation mode."
                )
    return args


def main():
    """Initialize and run the LLM Snake game."""
    try:
        # Parse command line arguments
        try:
            args = parse_arguments()
        except ValueError as e:
            # Handle command line argument errors
            print(Fore.RED + f"Command-line error: {e}")
            print(Fore.YELLOW + "For help, use: python main.py --help")
            sys.exit(1)
            
        # Default assumption: environment is fine.  We may overwrite this
        # in the new-session branch below; the variable must always exist so
        # later checks don't raise an UnboundLocalError.
        primary_env_ok = True

        # Check if we're continuing from a previous session
        if args.continue_with_game_in_dir:
            # Continue from existing directory
            print(Fore.GREEN + f"🔄 Continuing from existing session: {args.continue_with_game_in_dir}")
            GameManager.continue_from_directory(args)
            # Exit early so we don't start a brand-new session after the continuation run finishes.
            # Without this return the code below would create a second GameManager instance which,
            # in continuation mode, is *not* fully initialised (``args.is_continuation`` is set),
            # leading to an endless idle loop and the process appearing to "not exit".
            return
        else:
            # Check environment setup for new session
            primary_env_ok = check_env_setup(args.provider)
        
        # Check secondary LLM environment if specified
        if args.parser_provider and args.parser_provider.lower() != 'none':
            secondary_env_ok = check_env_setup(args.parser_provider)
            if not secondary_env_ok:
                print(Fore.YELLOW + f"⚠️ Warning: Secondary LLM ({args.parser_provider}) environment setup issues detected")
        
        if not primary_env_ok:
            user_choice = input(Fore.YELLOW + "Environment setup issues detected. Continue anyway? (y/n): ")
            if user_choice.lower() != 'y':
                print(Fore.RED + "Exiting due to environment setup issues.")
                sys.exit(1)
        
        # Create and run the game manager
        game_manager = GameManager(args)
        game_manager.run()
        
    except Exception as e:
        print(Fore.RED + f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up pygame
        if pygame.get_init():
            pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
