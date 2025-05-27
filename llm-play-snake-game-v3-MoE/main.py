"""
Main entry point for the LLM-controlled Snake game.
This script parses command line arguments and launches the game.
"""

import sys
import os
import json
import glob
import argparse
import pygame
from colorama import Fore, init as init_colorama
from config import PAUSE_BETWEEN_MOVES_SECONDS, MAX_CONSECUTIVE_EMPTY_MOVES
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
    parser.add_argument('--move-pause', type=float, default=PAUSE_BETWEEN_MOVES_SECONDS,
                      help=f'Pause between moves in seconds (default: {PAUSE_BETWEEN_MOVES_SECONDS})')
    parser.add_argument('--sleep-before-launching', type=int, default=0,
                      help='Time to sleep (in minutes) before launching the program')
    parser.add_argument('--max-steps', type=int, default=400,
                      help='Maximum steps a snake can take in a single game (default: 400)')
    parser.add_argument('--max-empty-moves', type=int, default=MAX_CONSECUTIVE_EMPTY_MOVES,
                      help=f'Maximum consecutive empty moves before game over (default: {MAX_CONSECUTIVE_EMPTY_MOVES})')
    parser.add_argument('--no-gui', action='store_true',
                      help='Run without GUI (text-only mode)')
    parser.add_argument('--continue-with-game-in-dir', type=str, default=None,
                      help='Continue from a previous game session in the specified directory')

    # Parse the arguments
    args = parser.parse_args()

    # Check if --continue-with-game-in-dir is used
    if args.continue_with_game_in_dir:
        # Validate the directory
        if not os.path.isdir(args.continue_with_game_in_dir):
            raise ValueError(f"Directory '{args.continue_with_game_in_dir}' does not exist")

        # Check for summary.json
        summary_path = os.path.join(args.continue_with_game_in_dir, "summary.json")
        if not os.path.isfile(summary_path):
            raise ValueError(f"Missing summary.json in '{args.continue_with_game_in_dir}'")

        # Check for prompts directory
        prompts_dir = os.path.join(args.continue_with_game_in_dir, "prompts")
        if not os.path.isdir(prompts_dir):
            raise ValueError(f"Missing 'prompts' directory in '{args.continue_with_game_in_dir}'")

        # Ensure no other arguments are provided
        raw_args = ' '.join(sys.argv[1:])
        disallowed_args = [
            "--provider",
            "--model",
            "--parser-provider",
            "--parser-model",
            "--max-games",
            "--move-pause",
            "--max-steps",
            "--max-empty-moves",
        ]  # only "--no-gui" and "--sleep-before-launching" are allowed when on "--continue-with-game-in-dir" mode

        for arg in disallowed_args:
            if arg in raw_args and not raw_args.startswith(f"--continue-with-game-in-dir {args.continue_with_game_in_dir}"):
                raise ValueError(f"Cannot use {arg} with --continue-with-game-in-dir")

        # Load configuration from summary.json
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)

            # Set configuration from summary.json
            if 'primary_llm' in summary_data:
                args.provider = summary_data['primary_llm'].get('provider', 'hunyuan')
                args.model = summary_data['primary_llm'].get('model', None)

            if 'secondary_llm' in summary_data:
                args.parser_provider = summary_data['secondary_llm'].get('provider', None)
                args.parser_model = summary_data['secondary_llm'].get('model', None)

            if 'game_configuration' in summary_data:
                args.max_steps = summary_data['game_configuration'].get('max_steps_per_game', 400)
                args.max_empty_moves = summary_data['game_configuration'].get('max_consecutive_empty_moves', MAX_CONSECUTIVE_EMPTY_MOVES)
                args.max_games = summary_data['game_configuration'].get('max_games', 6)

        except Exception as e:
            raise ValueError(f"Error loading summary.json: {e}")
    else:
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

def get_next_game_number(log_dir):
    """Determine the next game number to start from.
    
    Args:
        log_dir: The log directory to check
        
    Returns:
        The next game number to use
    """
    # Check for existing game files
    game_files = glob.glob(os.path.join(log_dir, "game*.json"))
    
    if not game_files:
        return 1  # Start from game 1 if no games exist
    
    # Extract game numbers from filenames
    game_numbers = []
    for file in game_files:
        filename = os.path.basename(file)
        try:
            game_number = int(filename.replace("game", "").replace(".json", ""))
            game_numbers.append(game_number)
        except ValueError:
            continue
    
    if not game_numbers:
        return 1
        
    return max(game_numbers) + 1

def clean_prompt_files(log_dir, start_game):
    """Clean prompt and response files for games >= start_game.
    
    Args:
        log_dir: The log directory
        start_game: The starting game number
    """
    prompts_dir = os.path.join(log_dir, "prompts")
    responses_dir = os.path.join(log_dir, "responses")
    
    # Clean prompt files
    if os.path.exists(prompts_dir):
        for file in os.listdir(prompts_dir):
            if file.startswith(f"game{start_game}_") or any(file.startswith(f"game{i}_") for i in range(start_game, 100)):
                os.remove(os.path.join(prompts_dir, file))
    
    # Clean response files
    if os.path.exists(responses_dir):
        for file in os.listdir(responses_dir):
            if file.startswith(f"game{start_game}_") or any(file.startswith(f"game{i}_") for i in range(start_game, 100)):
                os.remove(os.path.join(responses_dir, file))

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
        
        # Handle continuing from a previous session
        if args.continue_with_game_in_dir:
            print(Fore.GREEN + f"🔄 Continuing from previous session in '{args.continue_with_game_in_dir}'")
            
            # Determine the next game number
            next_game = get_next_game_number(args.continue_with_game_in_dir)
            print(Fore.GREEN + f"✅ Starting from game {next_game}")
            
            # Clean existing prompt and response files for games >= next_game
            clean_prompt_files(args.continue_with_game_in_dir, next_game)
            
            # Create and run the game manager with continuation settings
            game_manager = GameManager(args)
            game_manager.continue_from_session(args.continue_with_game_in_dir, next_game)
        else:
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
