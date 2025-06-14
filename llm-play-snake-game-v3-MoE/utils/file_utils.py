"""
File and storage management system.
Comprehensive functions for handling game logs, statistics extraction, and
session management in the Snake game environment.
"""

import os
import glob
import json
from pathlib import Path

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


def get_folder_display_name(path: str) -> str:
    """Return basename of a log folder (used by dashboard)."""
    return os.path.basename(path)


def load_summary_data(folder_path: str):
    """Load *summary.json* from *folder_path* and return dict or None."""
    summary_path = os.path.join(folder_path, "summary.json")
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_game_data(folder_path: str):
    """Return dict {game_number: game_json} for each *game_*.json* in folder."""
    games = {}
    for file in os.listdir(folder_path):
        if file.startswith("game_") and file.endswith(".json"):
            try:
                with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                num = int(file.replace("game_", "").replace(".json", ""))
                games[num] = data
            except Exception:
                continue
    return games


def find_valid_log_folders(root_dir: str = "logs", max_depth: int = 4):
    """Return experiment folders that contain the expected artefacts.

    A *valid* experiment folder must contain:
    • summary.json
    • at least one game_*.json
    • prompts/ sub-directory
    • responses/ sub-directory
    The search walks the directory tree (depth-limited) so experiments can live
    at arbitrary nesting levels.
    """
    valid_folders: list[str] = []

    # Walk through directories up to *max_depth*
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Compute depth relative to *root_dir*
        rel_path = os.path.relpath(dirpath, root_dir)
        depth = 0 if rel_path == "." else len(rel_path.split(os.sep))

        if depth > max_depth:
            # Skip deeper directories by clearing dirnames (no recursion)
            dirnames.clear()
            continue

        # Determine whether current dir qualifies as an experiment folder
        has_summary = "summary.json" in filenames
        has_game_files = any(f.startswith("game_") and f.endswith(".json") for f in filenames)
        has_prompts_dir = "prompts" in dirnames
        has_responses_dir = "responses" in dirnames

        if has_summary and has_game_files and has_prompts_dir and has_responses_dir:
            valid_folders.append(dirpath)

    return valid_folders 