"""
Replay system for the Snake game.
Functions for replaying recorded games and extracting replay data for analysis.
Enables visualization of past gameplay and integration with analytics tools.
"""

import os
import json
from pathlib import Path
from utils.file_utils import get_game_json_filename, join_log_path

def check_game_summary_for_moves(log_dir, game_number):
    """Check if a game summary file contains moves.
    
    This function is used by the Streamlit analytics dashboard when run separately.
    
    Args:
        log_dir: Path to the log directory
        game_number: Game number to check
        
    Returns:
        Boolean indicating if the game has moves
    """
    game_filename = get_game_json_filename(game_number)
    json_summary_file = join_log_path(log_dir, game_filename)
    
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
    game_filename = get_game_json_filename(game_number)
    json_summary_file = log_dir_path / game_filename
    
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
