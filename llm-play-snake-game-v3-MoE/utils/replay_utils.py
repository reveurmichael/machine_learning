"""
Utility functions for replaying recorded games.
Provides functionality for running replays and extracting data.
"""

import os
import subprocess
import time
import json
import traceback
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
            
        # Extract apple positions
        if 'apple_positions' in data and isinstance(data['apple_positions'], list):
            return data['apple_positions']
        
        # Check in detailed history
        if ('detailed_history' in data and 
            'apple_positions' in data['detailed_history'] and 
            isinstance(data['detailed_history']['apple_positions'], list)):
            return data['detailed_history']['apple_positions']
            
    except Exception:
        pass
        
    return None 