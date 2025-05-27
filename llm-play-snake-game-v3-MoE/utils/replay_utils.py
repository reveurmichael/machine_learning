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
    """Check if the game summary contains move data for replay.
    
    Args:
        log_dir: Path to the log directory
        game_number: Game number to check
        
    Returns:
        Boolean indicating if the game has move data
    """
    try:
        # Try to load the game summary file
        json_summary_file = os.path.join(log_dir, f"game{game_number}.json")
        
        if not os.path.exists(json_summary_file):
            print(f"Game {game_number} summary file not found: {json_summary_file}")
            return False
        
        with open(json_summary_file, 'r') as f:
            data = json.load(f)
            
        # Check if the file has moves
        if 'detailed_history' in data and 'moves' in data['detailed_history'] and data['detailed_history']['moves']:
            return True
        elif 'moves' in data and data['moves']:
            return True
            
        return False
    except Exception as e:
        print(f"Error checking game summary: {e}")
        return False

def extract_apple_positions(log_dir_path, game_number):
    """Extract apple positions from a game summary file.
    
    Args:
        log_dir_path: Path to the log directory
        game_number: Game number to extract apple positions from
        
    Returns:
        List of apple positions or empty list if not found
    """
    try:
        # Convert to Path object if not already
        if not isinstance(log_dir_path, Path):
            log_dir_path = Path(log_dir_path)
        
        # Get the game summary file
        json_summary_file = log_dir_path / f"game{game_number}.json"
        
        if not json_summary_file.exists():
            print(f"Game {game_number} summary file not found: {json_summary_file}")
            return []
        
        # Load the JSON data
        with open(json_summary_file, 'r') as f:
            data = json.load(f)
        
        # Extract apple positions
        if 'detailed_history' in data and 'apple_positions' in data['detailed_history']:
            return data['detailed_history']['apple_positions']
        elif 'apple_positions' in data:
            return data['apple_positions']
        
        return []
    except Exception as e:
        print(f"Error extracting apple positions: {e}")
        traceback.print_exc()
        return [] 