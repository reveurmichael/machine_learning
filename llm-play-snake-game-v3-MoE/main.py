"""
Main entry point for the LLM-controlled Snake game.
This script parses command line arguments and launches the game.
"""

import sys
import argparse
import pygame
from colorama import Fore, init as init_colorama
from config import MOVE_PAUSE, MAX_CONSECUTIVE_EMPTY_MOVES
from game_manager import GameManager

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
    parser.add_argument('--max-games', type=int, default=6,
                      help='Maximum number of games to play')
    parser.add_argument('--move-pause', type=float, default=MOVE_PAUSE,
                      help='Pause between sequential moves in seconds (default: 1.0)')
    parser.add_argument('--sleep-before-launching', type=int, default=0,
                      help='Time to sleep (in minutes) before launching the program')
    parser.add_argument('--max-steps', type=int, default=400,
                      help='Maximum steps a snake can take in a single game (default: 400)')
    parser.add_argument('--max-empty-moves', type=int, default=MAX_CONSECUTIVE_EMPTY_MOVES,
                      help=f'Maximum consecutive empty moves before game termination (default: {MAX_CONSECUTIVE_EMPTY_MOVES})')
    parser.add_argument('--no-gui', action='store_true',
                      help='Run the game without GUI for headless environments')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Validate the command-line arguments to detect duplicate or invalid arguments
    raw_args = ' '.join(sys.argv[1:])
    
    # Check for duplicate --model arguments (which would silently overwrite each other)
    model_count = raw_args.count('--model')
    if model_count > 1:
        raise ValueError(f"Error: '--model' argument appears {model_count} times. Use '--model' for the primary LLM and '--parser-model' for the secondary LLM.")
    
    # Check for duplicate --provider arguments
    provider_count = raw_args.count('--provider')
    if provider_count > 1:
        raise ValueError(f"Error: '--provider' argument appears {provider_count} times. Use '--provider' for the primary LLM and '--parser-provider' for the secondary LLM.")
    
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