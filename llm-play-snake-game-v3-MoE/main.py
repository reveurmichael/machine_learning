"""
Main entry point for the LLM-controlled Snake game.
This script parses command line arguments and launches the game.
"""

import sys
import argparse
import pygame
from colorama import Fore, init as init_colorama
from config import PAUSE_BETWEEN_MOVES_SECONDS, MAX_CONSECUTIVE_EMPTY_MOVES, MAX_CONSECUTIVE_ERRORS_ALLOWED
from core.game_manager import GameManager
from llm.setup_utils import check_env_setup

# Initialize colorama for colored terminal output
init_colorama(autoreset=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM-guided Snake game')
    parser.add_argument('--provider', type=str, default='hunyuan',
                      help='LLM provider to use for primary LLM (hunyuan, ollama, deepseek, or mistral)')
    parser.add_argument('--model', type=str, default=None,
                      help='Model name to use for primary LLM. For Ollama: check first what\'s available on the server. For DeepSeek: "deepseek-chat" or "deepseek-reasoner". For Mistral: "mistral-medium-latest" (default) or "mistral-large-latest"')
    parser.add_argument('--parser-provider', type=str, default=None,
                      help='LLM provider to use for secondary LLM (if not specified, uses the same as --provider). Use "none" to skip using a parser LLM and use primary LLM output directly.')
    parser.add_argument('--parser-model', type=str, default=None,
                      help='Model name to use for secondary LLM (if not specified, uses the default for the secondary provider)')
    parser.add_argument('--max-game', type=int, default=6,
                      help='Maximum number of games to play')
    parser.add_argument('--move-pause', type=float, default=PAUSE_BETWEEN_MOVES_SECONDS,
                      help=f'Pause between moves in seconds (default: {PAUSE_BETWEEN_MOVES_SECONDS})')
    parser.add_argument('--sleep-before-launching', type=int, default=0,
                      help='Time to sleep (in minutes) before launching the program')
    parser.add_argument('--max-steps', type=int, default=400,
                      help='Maximum steps a snake can take in a single game (default: 400)')
    parser.add_argument('--max-empty-moves', type=int, default=MAX_CONSECUTIVE_EMPTY_MOVES,
                      help=f'Maximum consecutive empty moves before game over (default: {MAX_CONSECUTIVE_EMPTY_MOVES})')
    parser.add_argument('--max-consecutive-errors-allowed', type=int, default=MAX_CONSECUTIVE_ERRORS_ALLOWED,
                      help=f'Maximum consecutive errors allowed before game over (default: {MAX_CONSECUTIVE_ERRORS_ALLOWED})')
    parser.add_argument('--no-gui', action='store_true',
                      help='Run without GUI (text-only mode)')
    parser.add_argument('--log-dir', type=str, default=None,
                      help='Directory to store logs')
    parser.add_argument('--continue-with-game-in-dir', type=str, default=None,
                      help='Continue an experiment from a directory containing previous games')

    # Parse the arguments
    args = parser.parse_args()
    
    # Validate continue mode restrictions
    if args.continue_with_game_in_dir:
        # Get all command line arguments
        raw_args = ' '.join(sys.argv[1:])
        
        # List of arguments not allowed with continue mode
        restricted_args = [
            '--provider', 
            '--model', 
            '--parser-provider', 
            '--parser-model', 
            '--move-pause', 
            '--max-steps', 
            '--max-empty-moves', 
            '--max-consecutive-errors-allowed',
            '--log-dir'
        ]
        
        # Check for any restricted arguments
        for arg in restricted_args:
            if arg in raw_args:
                raise ValueError(f"Cannot use {arg} with --continue-with-game-in-dir. "
                                 f"Only --max-game, --no-gui, and --sleep-before-launching are allowed.")
    
    # Set current game count to 0 for new sessions
    args.current_game_count = 0

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
            
        # Check if we're continuing from a previous session
        if args.continue_with_game_in_dir:
            # Continue from existing directory
            print(Fore.GREEN + f"🔄 Continuing from existing session: {args.continue_with_game_in_dir}")
            GameManager.continue_from_directory(args)
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
