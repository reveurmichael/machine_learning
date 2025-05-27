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
                has_summary_json = (folder_path / 'summary.json').exists()
                has_game_files = bool(glob.glob(str(folder_path / 'game*.json')))
                has_prompts_dir = (folder_path / 'prompts').is_dir()
                has_responses_dir = (folder_path / 'responses').is_dir()
                
                # If it has all required components, it's a log folder
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
            
            # Extract JSON success rate
            json_stats = info_data.get('json_parsing_stats', {})
            stats['json_success_rate'] = json_stats.get('success_rate', 0.0)
            
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
        json_game_file = Path(log_folder) / f"game{i}.json"
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
    
    # Update total_games if not set from summary.json
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
        'apples_per_step': 0,
        'json_success_rate': 0,
        'valid_step_percentage': 0,
        'parser_usage_count': 0
    }
    
    try:
        # Parse JSON file
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract fields from JSON
        summary['score'] = data.get('score', 0)
        summary['steps'] = data.get('steps', 0)
        summary['game_end_reason'] = data.get('game_end_reason', 'Unknown')
        summary['parser_usage_count'] = data.get('parser_usage_count', 0)
        
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
            summary['min_primary_response_time'] = prompt_stats.get('min_primary_response_time', 0)
            summary['max_primary_response_time'] = prompt_stats.get('max_primary_response_time', 0)
            summary['min_secondary_response_time'] = prompt_stats.get('min_secondary_response_time', 0)
            summary['max_secondary_response_time'] = prompt_stats.get('max_secondary_response_time', 0)
        
        # Extract efficiency metrics if available
        if 'efficiency_metrics' in data:
            eff_metrics = data.get('efficiency_metrics', {})
            summary['apples_per_step'] = eff_metrics.get('apples_per_step', 0)
            summary['steps_per_game'] = eff_metrics.get('steps_per_game', 0)
            summary['valid_move_ratio'] = eff_metrics.get('valid_move_ratio', 0)
        elif 'performance_metrics' in data:
            # For backward compatibility
            perf_metrics = data.get('performance_metrics', {})
            summary['steps_per_apple'] = perf_metrics.get('steps_per_apple', 0)
        
        # Extract token statistics if available
        if 'token_stats' in data:
            token_stats = data.get('token_stats', {})
            summary['token_stats'] = token_stats
        
        # Extract step statistics if available
        if 'step_stats' in data:
            step_stats = data.get('step_stats', {})
            summary['valid_steps'] = step_stats.get('valid_steps', 0)
            summary['valid_step_percentage'] = step_stats.get('valid_step_percentage', 0)
            summary['empty_steps'] = step_stats.get('empty_steps', 0)
            summary['empty_step_percentage'] = step_stats.get('empty_step_percentage', 0)
            summary['error_steps'] = step_stats.get('error_steps', 0)
            summary['error_step_percentage'] = step_stats.get('error_step_percentage', 0)
            summary['max_consecutive_empty_moves'] = step_stats.get('max_consecutive_empty_moves', 0)
        
        # Extract JSON parsing success rate if available
        if 'json_parsing_stats' in data:
            json_stats = data.get('json_parsing_stats', {})
            summary['json_success_rate'] = json_stats.get('success_rate', 0)
            summary['json_extraction_attempts'] = json_stats.get('total_extraction_attempts', 0)
            summary['json_successful_extractions'] = json_stats.get('successful_extractions', 0)
            summary['json_failed_extractions'] = json_stats.get('failed_extractions', 0)
            
    except Exception as e:
        print(f"Error extracting summary from {summary_file}: {e}")
    
    return summary 