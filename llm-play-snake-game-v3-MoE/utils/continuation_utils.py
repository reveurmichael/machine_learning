"""
Utility module for continuation functionality in Snake game.
Handles reading existing game data and continuing sessions.
"""

import os
import sys
import json
import traceback
from colorama import Fore
from datetime import datetime

def read_existing_game_data(log_dir, start_game_number):
    """Read existing game data from game files.
    
    Args:
        log_dir: Path to the log directory
        start_game_number: The game number to start from
        
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
    
    print(Fore.GREEN + f"🔍 Reading data from existing {start_game_number-1} games...")
    
    # Read game data from each file up to start_game_number - 1
    for game_num in range(1, start_game_number):
        # Construct game file paths (checking both naming conventions)
        game_file_path = os.path.join(log_dir, f"game_{game_num}.json")
        alt_game_file_path = os.path.join(log_dir, f"game{game_num}.json")
        
        # Find the first existing file
        game_files = [path for path in [game_file_path, alt_game_file_path] if os.path.exists(path)]
        
        # Process the first available file
        game_file = None
        for file in game_files:
            if os.path.exists(file):
                game_file = file
                break
                
        if game_file:
            try:
                with open(game_file, 'r', encoding='utf-8') as f:
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
                    
                    # Extract game duration
                    if 'time_stats' in game_data:
                        time_stats = game_data.get('time_stats', {})
                        # Note: we don't need to track game_durations as it's not used
            except json.JSONDecodeError as e:
                corrupted_games.append(game_num)
                print(Fore.YELLOW + f"⚠️ Warning: Game file {game_file} is corrupted: {e}")
            except Exception as e:
                corrupted_games.append(game_num)
                print(Fore.YELLOW + f"⚠️ Warning: Could not load data from {game_file}: {e}")
        else:
            missing_games.append(game_num)
    
    # Report any issues with game files
    if missing_games:
        print(Fore.YELLOW + f"⚠️ Warning: {len(missing_games)} game files missing: {missing_games}")
        
    if corrupted_games:
        print(Fore.YELLOW + f"⚠️ Warning: {len(corrupted_games)} game files corrupted: {corrupted_games}")
        
    # Successfully loaded games
    successful_games = start_game_number - 1 - len(missing_games) - len(corrupted_games)
    print(Fore.GREEN + f"✅ Successfully loaded {successful_games} game files")
    
    return total_score, total_steps, game_scores, empty_steps, error_steps, parser_usage_count

def setup_continuation_session(game_manager, log_dir, start_game_number):
    """Set up a game session for continuation.
    
    Args:
        game_manager: The GameManager instance
        log_dir: Path to the log directory to continue from
        start_game_number: The game number to start from
    """
    from utils.json_utils import reset_json_error_stats
    
    # Verify log directory exists and is valid
    if not os.path.isdir(log_dir):
        print(Fore.RED + f"❌ Log directory does not exist: {log_dir}")
        sys.exit(1)
        
    # Check if summary.json exists
    summary_path = os.path.join(log_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(Fore.RED + f"❌ Missing summary.json in '{log_dir}'")
        sys.exit(1)
    
    # Load the original experiment's summary to preserve configuration
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            
        # Update the summary with continuation info if it doesn't exist
        if 'continuation_info' not in summary_data:
            summary_data['continuation_info'] = {
                'is_continuation': True,
                'continuation_count': 1,
                'continuation_timestamps': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'original_timestamp': summary_data.get('timestamp')
            }
        else:
            # Update existing continuation info
            continuation_info = summary_data['continuation_info']
            continuation_info['continuation_count'] = continuation_info.get('continuation_count', 0) + 1
            if 'continuation_timestamps' not in continuation_info:
                continuation_info['continuation_timestamps'] = []
            continuation_info['continuation_timestamps'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        # Save the updated summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
            
        print(Fore.GREEN + "📝 Updated continuation info in summary.json")
    except Exception as e:
        print(Fore.YELLOW + f"⚠️ Warning: Could not update continuation info in summary.json: {e}")
        
    # Set the log directory
    game_manager.log_dir = log_dir
    game_manager.prompts_dir = os.path.join(log_dir, "prompts")
    game_manager.responses_dir = os.path.join(log_dir, "responses")
    
    # Create directories if they don't exist
    os.makedirs(game_manager.prompts_dir, exist_ok=True)
    os.makedirs(game_manager.responses_dir, exist_ok=True)
    
    # Reset JSON error statistics
    reset_json_error_stats()
    
    # Load and validate the previous game number
    if start_game_number < 1:
        print(Fore.RED + f"❌ Invalid starting game number: {start_game_number}")
        sys.exit(1)
        
    # Check if the previous game files exist
    game_file_path = os.path.join(log_dir, f"game_{start_game_number-1}.json")
    alt_game_file_path = os.path.join(log_dir, f"game{start_game_number-1}.json")
    
    if start_game_number > 1 and not (os.path.exists(game_file_path) or os.path.exists(alt_game_file_path)):
        print(Fore.RED + f"❌ Previous game file not found for game {start_game_number-1}")
        sys.exit(1)
    
    # Load statistics from existing games
    (game_manager.total_score, 
     game_manager.total_steps, 
     game_manager.game_scores, 
     game_manager.empty_steps, 
     game_manager.error_steps, 
     game_manager.parser_usage_count) = read_existing_game_data(log_dir, start_game_number)
    
    # Set game count to continue from the next game
    game_manager.game_count = start_game_number - 1

def setup_llm_clients(game_manager):
    """Set up the LLM clients for continuation.
    
    Args:
        game_manager: The GameManager instance
    """
    from utils.initialization_utils import setup_llm_clients as common_setup_llm_clients
    
    # Print configuration being used
    print(Fore.GREEN + "🔄 Setting up LLM clients for continuation mode")
    print(Fore.GREEN + f"🤖 Using Primary LLM: {game_manager.args.provider}" + 
          (f" ({game_manager.args.model})" if game_manager.args.model else ""))
    
    if game_manager.args.parser_provider and game_manager.args.parser_provider.lower() != 'none':
        print(Fore.GREEN + f"🤖 Using Parser LLM: {game_manager.args.parser_provider}" + 
              (f" ({game_manager.args.parser_model})" if game_manager.args.parser_model else ""))
    
    # Use the common utility function to set up LLM clients
    common_setup_llm_clients(game_manager)

def handle_continuation_game_state(game_manager):
    """Handle game state for continuation mode.
    
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
    
    # Mark this as a continuation in the game data
    game_manager.game.game_state.record_continuation()
    print(Fore.GREEN + f"📝 Marked session as continuation ({game_manager.game.game_state.continuation_count})")
    
    # Load summary.json to get information about the previous session
    try:
        summary_path = os.path.join(game_manager.log_dir, "summary.json")
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            
        # Synchronize game state with summary.json data
        game_manager.game.game_state.synchronize_with_summary_json(summary_data)
        
        # Record this continuation with the previous session data
        game_manager.game.game_state.record_continuation(summary_data)
        
        # Display continuation info
        prev_count = game_manager.game.game_state.continuation_count
        print(Fore.GREEN + f"ℹ️ This is continuation #{prev_count} of this experiment")
        
    except Exception as e:
        print(Fore.YELLOW + f"⚠️ Warning: Could not load continuation info from summary.json: {e}")
        # Still record continuation even if synchronization fails
        game_manager.game.game_state.record_continuation()
    
    print(Fore.GREEN + f"⏱️ Pause between moves: {game_manager.get_pause_between_moves()} seconds")
    print(Fore.GREEN + f"⏱️ Maximum steps per game: {game_manager.args.max_steps}")
    print(Fore.GREEN + f"📊 Continuing from game {game_manager.game_count + 1}, with {game_manager.total_score} total score so far")

def continue_from_directory(game_manager_class, args):
    """Factory method to create a GameManager instance for continuation.
    
    Args:
        game_manager_class: The GameManager class
        args: Command-line arguments with continue_with_game_in_dir set
        
    Returns:
        GameManager instance set up for continuation
    """
    from utils.file_utils import get_next_game_number, clean_prompt_files
    
    log_dir = args.continue_with_game_in_dir
    
    # Validate the continuation directory
    if not os.path.isdir(log_dir):
        print(Fore.RED + f"❌ Continuation directory does not exist: '{log_dir}'")
        sys.exit(1)
        
    # Check if summary.json exists
    summary_path = os.path.join(log_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(Fore.RED + f"❌ Missing summary.json in '{log_dir}'")
        sys.exit(1)
    
    # Load the original experiment configuration from summary.json
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            
        # Check if configuration exists in the summary
        if 'configuration' in summary_data:
            original_config = summary_data['configuration']
            
            # Apply the original experiment's configuration
            print(Fore.GREEN + "📝 Loading original experiment configuration from summary.json")
            
            # Copy the original provider and model
            args.provider = original_config.get('provider')
            args.model = original_config.get('model')
            
            # Copy the original parser settings
            args.parser_provider = original_config.get('parser_provider')
            args.parser_model = original_config.get('parser_model')
            
            # Copy other important configuration parameters
            args.move_pause = original_config.get('move_pause', args.move_pause)
            args.max_steps = original_config.get('max_steps', args.max_steps)
            args.max_empty_moves = original_config.get('max_empty_moves', args.max_empty_moves)
            args.max_consecutive_errors_allowed = original_config.get('max_consecutive_errors_allowed', args.max_consecutive_errors_allowed)
            
            # Log the applied configuration
            print(Fore.GREEN + f"🤖 Primary LLM: {args.provider}" + (f" ({args.model})" if args.model else ""))
            if args.parser_provider and args.parser_provider.lower() != 'none':
                print(Fore.GREEN + f"🤖 Parser LLM: {args.parser_provider}" + (f" ({args.parser_model})" if args.parser_model else ""))
            print(Fore.GREEN + f"⏱️ Move pause: {args.move_pause} seconds")
            print(Fore.GREEN + f"⏱️ Max steps: {args.max_steps}")
            print(Fore.GREEN + f"⏱️ Max empty moves: {args.max_empty_moves}")
            print(Fore.GREEN + f"⏱️ Max consecutive errors: {args.max_consecutive_errors_allowed}")
    except Exception as e:
        print(Fore.YELLOW + f"⚠️ Warning: Could not load configuration from summary.json: {e}")
        print(Fore.YELLOW + "⚠️ Continuing with command-line arguments")
        
    # Check if any game files exist
    game_files = []
    for file in os.listdir(log_dir):
        if file.startswith("game_") or (file.startswith("game") and not file.startswith("game_")):
            if file.endswith(".json"):
                game_files.append(file)
    
    if not game_files:
        print(Fore.YELLOW + f"⚠️ Warning: No game files found in '{log_dir}'")
        print(Fore.YELLOW + "⚠️ Starting from game 1 but in continuation mode")
        next_game = 1
    else:
        # Determine the next game number
        next_game = get_next_game_number(log_dir)
    
    print(Fore.GREEN + f"🔄 Continuing from previous session in '{log_dir}'")
    print(Fore.GREEN + f"✅ Starting from game {next_game}")
    
    # Clean existing prompt and response files for games >= next_game
    clean_prompt_files(log_dir, next_game)
    
    # Create and run the game manager with continuation settings
    game_manager = game_manager_class(args)
    
    # Set the is_continuation flag explicitly
    args.is_continuation = True
    
    # Continue from the session
    try:
        game_manager.continue_from_session(log_dir, next_game)
    except Exception as e:
        print(Fore.RED + f"❌ Error continuing from session: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    return game_manager 