"""
Replay system for the Snake game.
Functions for replaying recorded games and extracting replay data for analysis.
Enables visualization of past gameplay and integration with analytics tools.
"""

import os
import subprocess
import json
from pathlib import Path

def run_replay(log_dir, game_number=None, move_pause=1.0):
    """Run a replay of a specific game.
    
    Args:
        log_dir: Directory containing game logs
        game_number: Specific game number to replay
        move_pause: Pause between moves in seconds
        
    Returns:
        Process return code
    """
    cmd = ["python", "replay.py", "--log-dir", log_dir, "--move-pause", str(move_pause)]
    
    if game_number is not None:
        cmd.extend(["--game", str(game_number)])
    
    return subprocess.call(cmd)

def check_game_summary_for_moves(log_dir, game_number):
    """Check if a game summary file contains moves.
    
    This function is used by the Streamlit analytics dashboard when run separately.
    
    Args:
        log_dir: Path to the log directory
        game_number: Game number to check
        
    Returns:
        Boolean indicating if the game has moves
    """
    json_summary_file = os.path.join(log_dir, f"game_{game_number}.json")
    
    if not os.path.exists(json_summary_file):
        return False
    
    try:
        with open(json_summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Check if the file has moves
        return 'moves' in data and isinstance(data['moves'], list) and len(data['moves']) > 0
    except Exception:
        return False

def extract_apple_positions(log_dir, game_number):
    """Extract apple positions from a game summary file.
    
    This function is used by the Streamlit analytics dashboard when run separately.
    
    Args:
        log_dir: Path to the log directory
        game_number: Game number to extract from
        
    Returns:
        List of apple positions or None if not found
    """
    log_dir_path = Path(log_dir)
    json_summary_file = log_dir_path / f"game_{game_number}.json"
    
    if not json_summary_file.exists():
        return None
    
    try:
        with open(json_summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract apple positions from standard location
        if 'apple_positions' in data and isinstance(data['apple_positions'], list):
            return data['apple_positions']
        
        # Check in game history structure
        if 'detailed_history' in data and 'apple_positions' in data['detailed_history']:
            return data['detailed_history']['apple_positions']
            
    except Exception:
        pass
        
    return None

def find_valid_log_folders(root_dir, max_depth=4):
    """Find valid log folders for replay.
    
    A folder is considered valid if it has:
    - A summary.json file
    - At least one game_*.json file
    - A prompts directory
    - A responses directory
    
    Args:
        root_dir: Root directory to start search from
        max_depth: Maximum directory depth to search
        
    Returns:
        List of valid log folder paths
    """
    valid_folders = []
    
    # Walk through directories up to max_depth
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check current depth
        relative_path = os.path.relpath(dirpath, root_dir)
        depth = 0 if relative_path == '.' else len(relative_path.split(os.sep))
        
        if depth > max_depth:
            # Skip deeper directories
            dirnames.clear()  # Don't recurse further from this point
            continue
        
        # Check if this directory has the required files
        has_summary = 'summary.json' in filenames
        has_game_files = any(f.startswith('game_') and f.endswith('.json') for f in filenames)
        has_prompts_dir = 'prompts' in dirnames
        has_responses_dir = 'responses' in dirnames
        
        if has_summary and has_game_files and has_prompts_dir and has_responses_dir:
            valid_folders.append(dirpath)
    
    return valid_folders 