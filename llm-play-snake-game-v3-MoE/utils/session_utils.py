"""
Utility module for session management in Snake game.
Handles reading and writing game data to persistent storage.
"""

import os
import sys
import json
from colorama import Fore

def read_session_data(log_dir, game_count):
    """Read session data from game files.
    
    Args:
        log_dir: Path to the log directory
        game_count: The number of games to read
        
    Returns:
        Tuple of (total_score, total_steps, game_scores, empty_steps, error_steps, parser_usage_count)
    """
    # Initialize counters
    total_score = 0
    total_steps = 0
    empty_steps = 0
    error_steps = 0
    parser_usage_count = 0
    game_scores = []
    missing_games = []
    corrupted_games = []
    
    print(Fore.GREEN + f"🔍 Reading data from {game_count} games...")
    
    # Read game data from each file up to game_count
    for game_num in range(1, game_count + 1):
        # Construct game file path
        game_file_path = os.path.join(log_dir, f"game_{game_num}.json")
                
        if os.path.exists(game_file_path):
            try:
                with open(game_file_path, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
                    
                    # Basic game stats
                    score = game_data.get('score', 0)
                    steps = game_data.get('steps', 0)
                    game_scores.append(score)
                    total_score += score
                    total_steps += steps
                    
                    # Track step types if available
                    if 'step_stats' in game_data:
                        step_stats = game_data.get('step_stats', {})
                        empty_steps += step_stats.get('empty_steps', 0)
                        error_steps += step_stats.get('error_steps', 0)
                    
                    # Track parser usage
                    parser_usage_count += game_data.get('parser_usage_count', 0)
            except json.JSONDecodeError as e:
                corrupted_games.append(game_num)
                print(Fore.YELLOW + f"⚠️ Warning: Game file {game_file_path} is corrupted: {e}")
            except Exception as e:
                corrupted_games.append(game_num)
                print(Fore.YELLOW + f"⚠️ Warning: Could not load data from {game_file_path}: {e}")
        else:
            missing_games.append(game_num)
    
    # Report any issues with game files
    if missing_games:
        print(Fore.YELLOW + f"⚠️ Warning: {len(missing_games)} game files missing: {missing_games}")
        
    if corrupted_games:
        print(Fore.YELLOW + f"⚠️ Warning: {len(corrupted_games)} game files corrupted: {corrupted_games}")
        
    # Successfully loaded games
    successful_games = game_count - len(missing_games) - len(corrupted_games)
    print(Fore.GREEN + f"✅ Successfully loaded {successful_games} game files")
    
    return total_score, total_steps, game_scores, empty_steps, error_steps, parser_usage_count

def setup_session_directories(game_manager):
    """Set up session directories for logging game data.
    
    Args:
        game_manager: The GameManager instance
    """
    # Create session directory if specified
    if game_manager.args.session_dir:
        game_manager.log_dir = game_manager.args.session_dir
    else:
        # Create timestamped log directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_manager.log_dir = os.path.join("logs", f"session_{timestamp}")
    
    # Create subdirectories for detailed logs
    game_manager.prompts_dir = os.path.join(game_manager.log_dir, "prompts")
    game_manager.responses_dir = os.path.join(game_manager.log_dir, "responses")
    
    # Create all directories
    os.makedirs(game_manager.log_dir, exist_ok=True)
    os.makedirs(game_manager.prompts_dir, exist_ok=True)
    os.makedirs(game_manager.responses_dir, exist_ok=True)
    
    print(Fore.GREEN + f"📁 Session directory created: {game_manager.log_dir}")

def setup_llm_clients(game_manager):
    """Set up the LLM clients for the game session.
    
    Args:
        game_manager: The GameManager instance
    """
    from utils.llm_utils import check_llm_health
    
    # Set up primary LLM client
    provider = game_manager.args.provider
    model = game_manager.args.model
    game_manager.llm_client = game_manager.create_llm_client(provider, model)
    print(Fore.GREEN + f"🤖 Primary LLM: {provider}" + (f" ({model})" if model else ""))
    
    # Check if primary LLM is operational
    check_llm_health(game_manager.llm_client)
    
    # Set up parser LLM client if needed
    if game_manager.args.parser_provider and game_manager.args.parser_provider.lower() != "none":
        game_manager.parser_provider = game_manager.args.parser_provider
        game_manager.parser_model = game_manager.args.parser_model
        parser_client = game_manager.create_llm_client(game_manager.parser_provider, game_manager.parser_model)
        print(Fore.GREEN + f"🤖 Parser LLM: {game_manager.parser_provider}" + 
              (f" ({game_manager.parser_model})" if game_manager.parser_model else ""))
        
        # Check if parser LLM is operational
        check_llm_health(parser_client)
    else:
        print(Fore.GREEN + "🤖 No separate parser LLM will be used")

def initialize_game_state(game_manager):
    """Initialize the game state.
    
    Args:
        game_manager: The GameManager instance
    """
    import pygame
    
    # Initialize pygame if using GUI
    if game_manager.use_gui:
        pygame.init()
        pygame.font.init()
    
    # Set up the game
    game_manager.setup_game()
    
    # Display configuration info
    print(Fore.GREEN + f"⏱️ Pause between moves: {game_manager.get_pause_between_moves()} seconds")
    print(Fore.GREEN + f"⏱️ Maximum steps per game: {game_manager.args.max_steps}")
    print(Fore.GREEN + f"📊 Running games: {game_manager.args.max_game}") 