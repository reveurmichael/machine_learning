"""
Game State Utilities - Common game state extraction functions
----------------

This module contains utility functions for extracting and manipulating
game state data that are used across multiple extensions.

These functions were moved from agent_bfs.py to follow SSOT principles
and make them available to all extensions.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from utils.path_utils import ensure_project_root
ensure_project_root()

from typing import List, Dict, Any
import numpy as np


def extract_head_position(game_state: dict) -> List[int]:
    """
    SSOT: Extract head position from game state.
    
    Single source of truth for head position extraction.
    All other files must use this method instead of calculating head position.
    """
    snake_positions = game_state.get('snake_positions', [])
    if not snake_positions:
        return [0, 0]
    return list(snake_positions[-1])  # Head is always the last element


def extract_body_positions(game_state: dict) -> List[List[int]]:
    """
    Extracts the snake's body positions from the game state (excluding the head).
    SSOT: Ensures body positions are consistently derived.

    Args:
        game_state: The current game state dictionary.

    Returns:
        A list of lists, where each inner list is an [x, y] position of a body segment.
    """
    snake_positions = game_state.get('snake_positions', [])
    head_pos = game_state.get('head_position', [0, 0])

    # The last element in snake_positions is the head, the second to last is the first body segment, etc.
    # So we reverse the list and exclude the head (which is now the first element).
    body_positions = [pos for pos in snake_positions if pos != head_pos][::-1]

    return body_positions


def extract_apple_position(game_state: dict) -> List[int]:
    """
    Extracts the apple's position from the game state.

    Args:
        game_state: The current game state dictionary.

    Returns:
        A list representing the apple's [x, y] position.
    """
    return list(game_state.get('apple_position', [0, 0]))


def extract_grid_size(game_state: dict) -> int:
    """
    Extracts the grid size from the game state.

    Args:
        game_state: The current game state dictionary.

    Returns:
        The integer grid size.
    """
    return game_state.get('grid_size', 10)  # Default to 10 for safety if not found


def to_serializable(obj):
    """
    Convert numpy types to Python types for JSON serialization.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


def format_metrics_for_jsonl(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format metrics for JSONL completion, ensuring all values are JSON serializable.
    
    Args:
        metrics: Raw metrics dictionary
        
    Returns:
        Formatted metrics dictionary with consistent naming
    """
    if not metrics:
        return {}
    
    formatted_metrics = {}
    
    for key, value in metrics.items():
        # Convert numpy types to Python types for JSON serialization
        if hasattr(value, 'item'):  # numpy scalar
            formatted_metrics[key] = value.item()
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples that might contain numpy types
            formatted_metrics[key] = [
                item.item() if hasattr(item, 'item') else item 
                for item in value
            ]
        elif isinstance(value, dict):
            # Handle nested dictionaries
            formatted_metrics[key] = format_metrics_for_jsonl(value)
        else:
            formatted_metrics[key] = value
    
    return formatted_metrics


def flatten_explanation_for_jsonl(explanation: Any) -> str:
    """
    Convert a structured explanation dict to a rich, human-readable string for JSONL output.
    If already a string, return as-is. If dict, use 'natural_language_summary' and 'explanation_steps'.
    
    Args:
        explanation: The explanation object to flatten
        
    Returns:
        Rich, human-readable explanation string
    """
    if isinstance(explanation, str):
        return explanation
    if isinstance(explanation, dict):
        # Prefer natural_language_summary + explanation_steps
        summary = explanation.get('natural_language_summary', '')
        steps = explanation.get('explanation_steps', [])
        if steps and isinstance(steps, list):
            steps_text = '\n'.join(steps)
        else:
            steps_text = ''
        # Compose
        if summary and steps_text:
            return f"{steps_text}\n\n{summary}"
        elif steps_text:
            return steps_text
        elif summary:
            return summary
        # Fallback: join all string fields in the dict
        return '\n'.join(str(v) for v in explanation.values() if isinstance(v, str))
    # Fallback: just str()
    return str(explanation) 