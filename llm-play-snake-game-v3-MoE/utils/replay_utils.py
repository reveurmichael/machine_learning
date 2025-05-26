"""
Utility module for game replay functionality.
Handles replay of snake games based on stored moves and game state.
"""

import os
import subprocess
import time

def run_replay(log_dir, game_number=None, move_pause=1.0):
    """Run the game replay using subprocess.
    
    Args:
        log_dir: Path to the log directory
        game_number: Specific game number to replay (None means all games)
        move_pause: Pause between moves in seconds
    """
    cmd = ["python", "replay.py", "--log-dir", log_dir, "--move-pause", str(move_pause)]
    
    if game_number is not None:
        cmd.extend(["--game", str(game_number)])
        
    try:
        process = subprocess.Popen(cmd)
        print(f"Started replay of {os.path.basename(log_dir)}")
        if game_number is not None:
            print(f"Game {game_number}")
        else:
            print("All games")
        print("Note: The replay window opens in a separate window.")
        
        # Keep the application running while the subprocess is active
        return process
        
    except Exception as e:
        print(f"Error running replay: {e}")
        return None

def check_game_summary_for_moves(log_dir, game_number):
    """Check if a game summary contains stored moves.
    
    Args:
        log_dir: Path to the log directory
        game_number: Game number to check
        
    Returns:
        Tuple of (has_moves, move_count)
    """
    json_summary_file = os.path.join(log_dir, f"game{game_number}_summary.json")
    if os.path.exists(json_summary_file):
        try:
            import json
            with open(json_summary_file, 'r') as f:
                summary_data = json.load(f)
            
            if 'moves' in summary_data and summary_data['moves']:
                move_count = len(summary_data['moves'])
                return True, move_count
        except Exception as e:
            print(f"Error checking for moves in summary file: {e}")
    
    return False, 0 