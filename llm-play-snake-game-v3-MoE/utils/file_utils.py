"""
Utility module for file and directory operations.
Handles log folder detection, game statistics extraction, and other file-related operations.
"""

import os
import glob
import json
from pathlib import Path

def find_log_folders(base_dir='.', max_depth=4):
    """Find all log folders in the given directory and its subdirectories.
    
    Args:
        base_dir: Base directory to start search from
        max_depth: Maximum depth of subdirectories to search
        
    Returns:
        List of paths to log folders containing game data
    """
    log_folders = []
    base_path = Path(base_dir)
    
    for depth in range(max_depth + 1):
        # Create pattern for current depth
        # Use wildcards for each directory level up to the current depth
        if depth == 0:
            # Just look in the base directory
            patterns = [str(base_path / "*")]
        else:
            # Look in subdirectories up to depth
            parts = ['*'] * depth
            patterns = [str(base_path.joinpath(*parts) / "*")]
        
        # Find all potential log folders for current patterns
        for pattern in patterns:
            potential_folders = glob.glob(pattern)
            
            for folder in potential_folders:
                folder_path = Path(folder)
                # Check if folder contains required files and directories
                has_info_json = (folder_path / 'info.json').exists()
                has_summary_files = bool(glob.glob(str(folder_path / 'game*_summary.json')))
                has_prompts_dir = (folder_path / 'prompts').is_dir()
                has_responses_dir = (folder_path / 'responses').is_dir()
                
                # If it has all required components, it's a log folder
                if has_info_json and has_summary_files and has_prompts_dir and has_responses_dir:
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
        'providers': [],
        'models': [],
        'game_data': {}  # Store per-game data
    }
    
    # Load data from info.json
    info_path = Path(log_folder) / 'info.json'
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
            
            # Extract JSON success rate
            json_stats = info_data.get('json_parsing_stats', {})
            stats['json_success_rate'] = json_stats.get('success_rate', 0.0)
            
        except Exception as e:
            print(f"Error reading info.json: {e}")
    
    # Extract per-game data from JSON summary files
    total_response_time = 0
    total_secondary_response_time = 0
    total_steps_per_apple = 0
    game_count = 0
    
    for i in range(1, 7):  # Assuming max 6 games
        json_summary_file = Path(log_folder) / f"game{i}_summary.json"
        if json_summary_file.exists():
            game_data = extract_game_summary(json_summary_file)
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
    
    # Update total_games if not set from info.json
    if stats['total_games'] == 0:
        stats['total_games'] = len(stats['game_data'])
    
    return stats

def extract_game_summary(summary_file):
    """Extract game summary from a summary file.
    
    Args:
        summary_file: Path to the game summary file
        
    Returns:
        Dictionary with game summary information
    """
    summary = {
        'file': str(summary_file),
        'score': 0,
        'steps': 0,
        'game_end_reason': None,
        'has_apple_positions': False,
        'has_moves': False,
        'move_count': 0,
        'avg_primary_response_time': 0,
        'avg_secondary_response_time': 0,
        'steps_per_apple': 0,
        'json_success_rate': 0
    }
    
    try:
        # Parse JSON file
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract fields from JSON
        summary['score'] = data.get('score', 0)
        summary['steps'] = data.get('steps', 0)
        summary['game_end_reason'] = data.get('game_end_reason', 'Unknown')
        
        # Check if the file has apple positions
        if 'apple_positions' in data and data['apple_positions']:
            summary['has_apple_positions'] = True
            
        # Check if the file has moves
        if 'moves' in data and data['moves']:
            summary['has_moves'] = True
            summary['move_count'] = len(data['moves'])
        
        # Extract response time metrics if available
        if 'prompt_response_stats' in data:
            prompt_stats = data.get('prompt_response_stats', {})
            summary['avg_primary_response_time'] = prompt_stats.get('avg_primary_response_time', 0)
            summary['avg_secondary_response_time'] = prompt_stats.get('avg_secondary_response_time', 0)
        
        # Extract performance metrics if available
        if 'performance_metrics' in data:
            perf_metrics = data.get('performance_metrics', {})
            summary['steps_per_apple'] = perf_metrics.get('steps_per_apple', 0)
        
        # Extract JSON parsing success rate if available
        if 'json_parsing_stats' in data:
            json_stats = data.get('json_parsing_stats', {})
            summary['json_success_rate'] = json_stats.get('success_rate', 0)
            
    except Exception as e:
        print(f"Error extracting summary from {summary_file}: {e}")
    
    return summary 