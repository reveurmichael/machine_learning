"""
File and storage management system.
Comprehensive functions for handling game logs, statistics extraction, and
session management in the Snake game environment.
"""

import os
import glob
import json
from pathlib import Path

def find_log_folders(base_dir='.', max_depth=4):
    """Find all session folders containing game data.
    
    Searches the given directory for folders that contain game data files
    and session logs, up to the specified maximum depth.
    
    Args:
        base_dir: Base directory to start searching from
        max_depth: Maximum directory depth to search
        
    Returns:
        List of paths to session folders containing game data
    """
    log_folders = []
    
    # Search for log directories in the specified base directory
    for root, dirs, files in os.walk(base_dir):
        # Skip if we've exceeded max depth
        rel_path = os.path.relpath(root, base_dir)
        if rel_path == '.':
            depth = 0
        else:
            depth = rel_path.count(os.sep) + 1
            
        if depth > max_depth:
            continue
        
        # Check for session folders with game data
        pattern = os.path.join(root, "logs/session_*")
        potential_folders = glob.glob(pattern)
        
        for folder in potential_folders:
            folder_path = Path(folder)
            # Check if folder contains required files and directories
            has_summary_json = (folder_path / 'summary.json').exists()
            has_game_files = bool(glob.glob(str(folder_path / 'game*.json')))
            has_prompts_dir = (folder_path / 'prompts').is_dir()
            has_responses_dir = (folder_path / 'responses').is_dir()
            
            # If it has all required components, it's a valid session folder
            if has_summary_json and has_game_files and has_prompts_dir and has_responses_dir:
                log_folders.append(folder)
    
    return log_folders

def extract_game_stats(log_folder):
    """Extract game statistics from a log folder.
    
    Args:
        log_folder: Path to the log folder
        
    Returns:
        Dictionary with game statistics
    """
    stats = {
        'folder': log_folder,
        'date': None,
        'total_games': 0,
        'total_score': 0,
        'total_steps': 0,
        'max_score': 0,
        'min_score': 0,
        'mean_score': 0,
        'primary_llm': 'Unknown',
        'secondary_llm': 'Unknown',
        'avg_response_time': 0,
        'avg_secondary_response_time': 0,
        'steps_per_apple': 0,
        'json_success_rate': 0,
        'apples_per_step': 0,
        'providers': [],
        'models': [],
        'game_data': {}  # Store per-game data
    }
    
    # Load data from summary.json
    info_path = Path(log_folder) / 'summary.json'
    if info_path.exists():
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
            
            # Extract experiment information
            stats['date'] = info_data.get('date')
            
            # Extract LLM information
            primary_llm_info = info_data.get('primary_llm', {})
            if primary_llm_info:
                primary_provider = primary_llm_info.get('provider')
                primary_model = primary_llm_info.get('model')
                
                if primary_provider:
                    stats['primary_llm'] = primary_provider
                    stats['providers'].append(primary_provider)
                
                if primary_model:
                    stats['primary_llm'] += f" - {primary_model}"
                    stats['models'].append(primary_model)
            
            secondary_llm_info = info_data.get('secondary_llm', {})
            if secondary_llm_info:
                secondary_provider = secondary_llm_info.get('provider')
                secondary_model = secondary_llm_info.get('model')
                
                if secondary_provider:
                    stats['secondary_llm'] = secondary_provider
                    if secondary_provider not in stats['providers']:
                        stats['providers'].append(secondary_provider)
                
                if secondary_model:
                    stats['secondary_llm'] += f" - {secondary_model}"
                    if secondary_model not in stats['models']:
                        stats['models'].append(secondary_model)
            
            # Extract game statistics
            game_stats = info_data.get('game_statistics', {})
            stats['total_games'] = game_stats.get('total_games', 0)
            stats['total_score'] = game_stats.get('total_score', 0)
            stats['total_steps'] = game_stats.get('total_steps', 0)
            stats['max_score'] = game_stats.get('max_score', 0)
            stats['min_score'] = game_stats.get('min_score', 0)
            stats['mean_score'] = game_stats.get('mean_score', 0.0)
            stats['steps_per_apple'] = game_stats.get('steps_per_apple', 0.0)
            stats['apples_per_step'] = game_stats.get('apples_per_step', 0.0)
            
            # Extract response time statistics if available
            if 'response_time_stats' in info_data:
                stats['response_time_stats'] = info_data.get('response_time_stats', {})
            
            # Extract token usage statistics if available
            if 'token_usage_stats' in info_data:
                stats['token_usage_stats'] = info_data.get('token_usage_stats', {})
            
            # Extract step statistics if available
            if 'step_stats' in info_data:
                stats['step_stats'] = info_data.get('step_stats', {})
            
            # Copy efficiency metrics if available
            if 'efficiency_metrics' in info_data:
                stats['efficiency_metrics'] = info_data.get('efficiency_metrics', {})
            
            # Store game scores if available
            if 'game_scores' in info_data:
                stats['game_scores'] = info_data.get('game_scores', [])
            
        except Exception as e:
            print(f"Error reading summary.json: {e}")
    
    # Extract per-game data from JSON summary files
    total_response_time = 0
    total_secondary_response_time = 0
    total_steps_per_apple = 0
    game_count = 0
    
    for i in range(1, 7):  # Assuming max 6 games
        json_game_file = Path(log_folder) / f"game_{i}.json"
        if json_game_file.exists():
            game_data = extract_game_summary(json_game_file)
            if game_data:
                stats['game_data'][i] = game_data
                game_count += 1
                
                # Accumulate response times and steps per apple
                if 'avg_primary_response_time' in game_data:
                    total_response_time += game_data['avg_primary_response_time']
                
                if 'avg_secondary_response_time' in game_data:
                    total_secondary_response_time += game_data['avg_secondary_response_time']
                
                if 'steps_per_apple' in game_data:
                    total_steps_per_apple += game_data['steps_per_apple']
    
    # Calculate averages across games
    if game_count > 0:
        stats['avg_response_time'] = total_response_time / game_count
        stats['avg_secondary_response_time'] = total_secondary_response_time / game_count
        stats['steps_per_apple'] = total_steps_per_apple / game_count
    
    # Set total_games from game data if not already set
    if stats['total_games'] == 0:
        stats['total_games'] = len(stats['game_data'])
    
    return stats

def extract_game_summary(summary_file):
    """Extract game summary from a summary file.
    
    Args:
        summary_file: Path to the summary file
        
    Returns:
        Dictionary with game summary information
    """
    summary = {}
    
    try:
        if not os.path.exists(summary_file):
            return summary
            
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract basic stats
        summary['date'] = data.get('date', 'Unknown')
        summary['game_count'] = data.get('game_count', 0)
        summary['total_score'] = data.get('total_score', 0)
        summary['total_steps'] = data.get('total_steps', 0)
        summary['avg_score'] = summary['total_score'] / max(1, summary['game_count'])
        summary['avg_steps'] = summary['total_steps'] / max(1, summary['game_count'])
        
        # Extract LLM information
        if 'primary_llm' in data:
            llm_info = data['primary_llm']
            summary['primary_provider'] = llm_info.get('provider', 'Unknown')
            summary['primary_model'] = llm_info.get('model', 'Unknown')
            
        if 'secondary_llm' in data:
            llm_info = data['secondary_llm']
            summary['secondary_provider'] = llm_info.get('provider', 'None')
            summary['secondary_model'] = llm_info.get('model', 'None')
            
        # Extract response time metrics
        if 'prompt_response_stats' in data:
            prompt_stats = data.get('prompt_response_stats', {})
            summary['avg_primary_response_time'] = prompt_stats.get('avg_primary_response_time', 0)
            summary['avg_secondary_response_time'] = prompt_stats.get('avg_secondary_response_time', 0)
            summary['min_primary_response_time'] = prompt_stats.get('min_primary_response_time', 0)
            summary['max_primary_response_time'] = prompt_stats.get('max_primary_response_time', 0)
            summary['min_secondary_response_time'] = prompt_stats.get('min_secondary_response_time', 0)
            summary['max_secondary_response_time'] = prompt_stats.get('max_secondary_response_time', 0)
        
        # Extract performance metrics
        if 'efficiency_metrics' in data:
            eff_metrics = data.get('efficiency_metrics', {})
            summary['apples_per_step'] = eff_metrics.get('apples_per_step', 0)
            summary['steps_per_game'] = eff_metrics.get('steps_per_game', 0)
            summary['valid_move_ratio'] = eff_metrics.get('valid_move_ratio', 0)
        elif 'performance_metrics' in data:
            perf_metrics = data.get('performance_metrics', {})
            summary['steps_per_apple'] = perf_metrics.get('steps_per_apple', 0)
        
        # Extract token statistics
        if 'token_stats' in data:
            token_stats = data.get('token_stats', {})
            summary['token_stats'] = token_stats
            
    except Exception as e:
        print(f"Error extracting summary: {e}")
        
    return summary

def get_next_game_number(log_dir):
    """Determine the next game number to start from.
    
    Args:
        log_dir: The log directory to check
        
    Returns:
        The next game number to use
    """
    # Check for existing game files
    game_files = glob.glob(os.path.join(log_dir, "game_*.json"))
    
    if not game_files:
        return 1  # Start from game 1 if no games exist
    
    # Extract game numbers from filenames
    game_numbers = []
    for file in game_files:
        filename = os.path.basename(file)
        try:
            game_number = int(filename.replace("game_", "").replace(".json", ""))
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
            if (file.startswith(f"game_{start_game}_") or 
                any(file.startswith(f"game_{i}_") for i in range(start_game, 100))):
                os.remove(os.path.join(prompts_dir, file))
    
    # Clean response files
    if os.path.exists(responses_dir):
        for file in os.listdir(responses_dir):
            if (file.startswith(f"game_{start_game}_") or 
                any(file.startswith(f"game_{i}_") for i in range(start_game, 100))):
                os.remove(os.path.join(responses_dir, file))

def save_to_file(content, directory, filename, metadata=None):
    """Save content to a file in the specified directory.
    
    Args:
        content: The content to save
        directory: The directory to save the file in
        filename: The name of the file
        metadata: Optional dictionary of metadata to include at the top of the file
        
    Returns:
        The path to the saved file
    """
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create the full path
    file_path = os.path.join(directory, filename)
    
    # If metadata is provided, format it for inclusion
    formatted_content = ""
    if metadata:
        from datetime import datetime
        
        # Add timestamp if not provided
        if 'timestamp' not in metadata and 'Timestamp' not in metadata:
            metadata['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        # Format metadata as key-value pairs
        for key, value in metadata.items():
            # Skip lowercase timestamp as we prefer the capitalized version
            if key == 'timestamp' and 'Timestamp' in metadata:
                continue
                
            formatted_content += f"{key}: {value}\n"
            
        # Add section header based on the file type
        if "prompt" in filename.lower():
            if "parser" in filename.lower():
                formatted_content += "\n\n========== SECONDARY LLM PROMPT ==========\n\n"
            else:
                formatted_content += "\n\n========== PRIMARY LLM PROMPT ==========\n\n"
        elif "response" in filename.lower():
            if "parsed" in filename.lower():
                formatted_content += "\n\n========== SECONDARY LLM RESPONSE ==========\n\n"
            elif "raw" in filename.lower():
                formatted_content += "\n\n========== PRIMARY LLM RESPONSE (GAME STRATEGY) ==========\n\n"
    
    # Append the main content
    formatted_content += content
    
    # Write the content to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_content)
    
    return file_path

def get_game_json_filename(game_number):
    """Get the standardized filename for a game's JSON summary file.
    
    Args:
        game_number: The game number (1-based)
        
    Returns:
        String with the standardized filename
    """
    return f"game_{game_number}.json"

def get_prompt_filename(game_number, round_number, file_type="prompt"):
    """Get the standardized filename for a prompt or response file.
    
    Args:
        game_number: The game number (1-based)
        round_number: The round number (1-based)
        file_type: Type of file ("prompt", "raw_response", "parser_prompt", "parsed_response")
        
    Returns:
        String with the standardized filename
    """
    valid_types = ["prompt", "raw_response", "parser_prompt", "parsed_response"]
    if file_type not in valid_types:
        raise ValueError(f"Invalid file type '{file_type}'. Must be one of: {valid_types}")
        
    return f"game_{game_number}_round_{round_number}_{file_type}.txt"

def join_log_path(log_dir, filename):
    """Join the log directory with a filename.
    
    Args:
        log_dir: The log directory path
        filename: The filename to join
        
    Returns:
        String with the full path
    """
    return os.path.join(log_dir, filename) 