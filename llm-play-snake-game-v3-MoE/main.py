"""
Main entry point for the LLM-controlled Snake game.
This script parses command line arguments and launches the game.
"""

import sys
import os
import json
import argparse
import pygame
from colorama import Fore, init as init_colorama
from config import PAUSE_BETWEEN_MOVES_SECONDS, MAX_CONSECUTIVE_EMPTY_MOVES, MAX_CONSECUTIVE_ERRORS_ALLOWED
from core.game_manager import GameManager

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
    parser.add_argument('--session-dir', type=str, default=None,
                      help='Directory to store session data')

    # Parse the arguments
    args = parser.parse_args()
    
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
