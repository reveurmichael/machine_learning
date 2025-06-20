"""
File and storage management system.
Comprehensive functions for handling game logs, statistics extraction, and
session management in the Snake game environment.
"""

from __future__ import annotations

import os
import glob
import json
from pathlib import Path
from typing import Any, Dict, Union, Optional

# This function is NOT Task0 specific. Or, at least, we should make it generic, in some way.
def extract_game_summary(summary_file: Union[str, Path]) -> Dict[str, Any]:
    """Extract game summary from a summary file.
    
    Args:
        summary_file: Path to the summary file
        
    Returns:
        Dictionary with game summary information
    """
    summary = {}

    try:
        summary_path = Path(summary_file)
        if not summary_path.exists():
            return summary

        data = json.loads(summary_path.read_text(encoding="utf-8"))

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

# This function is NOT Task0 specific.
def get_next_game_number(log_dir: Union[str, Path]) -> int:
    """Determine the next game number to start from.
    
    Args:
        log_dir: The log directory to check
        
    Returns:
        The next game number to use
    """
    # Check for existing game files
    game_files = glob.glob(os.path.join(str(log_dir), "game_*.json"))
    
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

# This function is Task0 specific.
def clean_prompt_files(log_dir: Union[str, Path], start_game: int) -> None:
    """Clean prompt and response files for games >= start_game.
    
    Args:
        log_dir: The log directory
        start_game: The starting game number
    """
    prompts_dir = Path(log_dir) / "prompts"
    responses_dir = Path(log_dir) / "responses"
    
    # Clean prompt files
    if os.path.exists(prompts_dir):
        for file in os.listdir(prompts_dir):
            if file.startswith(f"game_{start_game}_"):
                (prompts_dir / file).unlink(missing_ok=True)
    
    # Clean response files
    if os.path.exists(responses_dir):
        for file in os.listdir(responses_dir):
            if file.startswith(f"game_{start_game}_"):
                (responses_dir / file).unlink(missing_ok=True)

# This function is Task0 specific.
def save_to_file(
    content: str,
    directory: Union[str, Path],
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save content to a file in the specified directory.
    
    Args:
        content: The content to save
        directory: The directory to save the file in
        filename: The name of the file
        metadata: Optional dictionary of metadata to include at the top of the file
        
    Returns:
        The path to the saved file
    """
    path_dir = Path(directory)
    path_dir.mkdir(parents=True, exist_ok=True)

    file_path = path_dir / filename
    
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
    file_path.write_text(formatted_content, encoding="utf-8")
    
    return str(file_path)


# This function is NOT Task0 specific.
def get_game_json_filename(game_number: int) -> str:
    """Get the standardized filename for a game's JSON summary file.
    
    Args:
        game_number: The game number (1-based)
        
    Returns:
        String with the standardized filename
    """
    return f"game_{game_number}.json"

# This function is Task0 specific.
def get_prompt_filename(
    game_number: int,
    round_number: int,
    file_type: str = "prompt",
) -> str:
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

# This function is NOT Task0 specific.
def join_log_path(log_dir: Union[str, Path], filename: str) -> str:
    """Return an absolute path inside *log_dir* for *filename*."""
    return str(Path(log_dir) / filename)


# This function is NOT Task0 specific.
def get_folder_display_name(path: Union[str, Path]) -> str:
    """Return basename of a log folder (used by dashboard)."""
    return os.path.basename(path)


# This function is NOT Task0 specific. Or, at least, we should make it generic, in some way.
def load_summary_data(folder_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load *summary.json* from *folder_path* and return dict or None."""
    summary_path = Path(folder_path) / "summary.json"
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


# This function is NOT Task0 specific.
def load_game_data(folder_path: str):
    """
    1. Scans the whole folder.
    2. Loads every game_*.json file into a dict {number: dict}.
    3. Does no validation or per-game initialisation.
    4. Return dict {game_number: game_json} for each *game_*.json* in folder.
    """
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


# This function is Task0 specific.
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


# This function is NOT Task0 specific.
def get_total_games(log_dir: str) -> int:
    """Return the total number of games present in *log_dir*.

    The function first tries to read *summary.json* (preferred – cheap and
    reflects the *authoritative* number recorded at runtime).  If the file is
    missing or malformed it falls back to counting the existing *game_*.json*
    artefacts on disk.

    Parameters
    ----------
    log_dir
        Path to a directory that contains *summary.json* and/or the per-game
        JSON files.

    Returns
    -------
    int
        The total number of games recorded. Guarantees to return at least 1
        so consumer code can safely divide by the value without having to
        handle *0* special-cases.
    """

    summary_file = os.path.join(log_dir, "summary.json")
    if os.path.isfile(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            total = int(data.get("game_statistics", {}).get("total_games", 0))
            if total > 0:
                return total
        except Exception:
            pass  # fall back to disk scan below

    # Disk fallback – count *game_*.json* files
    game_files = glob.glob(os.path.join(log_dir, "game_*.json"))
    return max(1, len(game_files)) 
