"""
Game Runner utilities – run heuristic games & load raw logs.

This module provides utilities to:
1. Launch heuristic game sessions via subprocess
2. Load and parse game log files
3. Return structured game data for dataset generation

Design Philosophy:
- Single responsibility: Only handles game execution and log loading
- Reusable: Can be used by supervised/RL extensions
- Simple logging: Uses print() statements for all operations
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

__all__ = ["run_heuristic_games", "load_game_logs"]


def run_heuristic_games(
    algorithm: str,
    max_games: int,
    max_steps: int,
    grid_size: int,
    verbose: bool = False
) -> List[str]:
    """
    Launch one HeuristicGameManager session that plays *max_games* games.
    Return the single log directory produced by v0.04 GameManager.
    
    Args:
        algorithm: Algorithm name (BFS, ASTAR, etc.)
        max_games: Number of games to play
        max_steps: Maximum steps per game
        grid_size: Grid size for the game
        verbose: Enable verbose output
        
    Returns:
        List of log directory paths (typically single path for v0.04)
    """
    print(f"[GameRunner] Starting single run for {max_games} games with {algorithm}...")
    
    cmd = [
        sys.executable, "scripts/main.py",
        "--algorithm", algorithm,
        "--max-games", str(max_games),
        "--max-steps", str(max_steps),
        "--grid-size", str(grid_size)
    ]
    
    if verbose:
        cmd.append("--verbose")

    # Get heuristics-v0.04 directory path
    heuristics_dir = Path(__file__).resolve().parents[2] / "heuristics-v0.04"
    
    result = subprocess.run(
        cmd,
        cwd=str(heuristics_dir),
        capture_output=True,
        text=True,
        timeout=max(300, max_games * 30)  # Rough timeout: 30s per game
    )

    log_dirs: List[str] = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "📂 Logs:" in line:
                log_path = line.split("📂 Logs: ")[1].strip()
                log_dirs.append(log_path)
                print(f"[GameRunner] Found log directory: {log_path}")
                break  # Only one directory expected for v0.04
        else:
            print("[GameRunner] Warning: Could not parse log directory from output")
    else:
        print(f"[GameRunner] Error running games: {result.stderr}")

    print(f"[GameRunner] Completed run – log directory count: {len(log_dirs)}")
    return log_dirs


def load_game_logs(log_dirs: List[str], verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Parse *game_N.json* files under given directories and return parsed dicts.
    
    Args:
        log_dirs: List of log directory paths
        verbose: Enable verbose output
        
    Returns:
        List of game data dictionaries
    """
    games: List[Dict[str, Any]] = []
    
    for i, log_dir in enumerate(log_dirs, 1):
        if verbose:
            print(f"[LogLoader] Loading games from directory {i}/{len(log_dirs)}: {log_dir}")
        
        try:
            log_path = Path(log_dir)
            
            # Look for game JSON files
            game_files = list(log_path.glob("game_*.json"))
            
            for game_file in game_files:
                try:
                    with open(game_file, 'r') as f:
                        game_data = json.load(f)
                    
                    # Add metadata including the log directory path
                    game_data['log_path'] = str(log_dir)
                    game_data['log_file'] = str(game_file)
                    game_data['log_directory'] = str(log_path)
                    
                    games.append(game_data)
                    
                    if verbose:
                        rounds_count = len(game_data.get('rounds', []))
                        score = game_data.get('final_score', 0)
                        print(f"[LogLoader] Loaded game: {rounds_count} rounds, score {score}")
                        
                except Exception as e:
                    if verbose:
                        print(f"[LogLoader] Failed to read {game_file}: {e}")
            
            if not game_files:
                if verbose:
                    print(f"[LogLoader] Warning: No game files found in {log_dir}")
                    
        except Exception as e:
            print(f"[LogLoader] Error loading {log_dir}: {e}")
    
    print(f"[LogLoader] Successfully loaded {len(games)} games")
    return games 