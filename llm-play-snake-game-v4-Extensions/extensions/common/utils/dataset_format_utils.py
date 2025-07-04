"""
Dataset Format Utilities for Heuristics Extensions

This module provides centralized dataset extraction and formatting utilities
for generating rich JSONL and CSV entries from game states and agent explanations.

Design Philosophy:
- Agent-agnostic: Works with any agent that provides explanations
- Rich content: Extracts full explanations and metrics for fine-tuning
- Consistent schema: Standardized output format across all agents
- Single responsibility: Only handles dataset orchestration, delegates to specialized modules

Usage:
    from extensions.common.utils.dataset_format_utils import extract_dataset_records
    
    # Extract records from a completed game
    jsonl_records, csv_records = extract_dataset_records(game_data, agent_name)
"""

from typing import Dict, List, Any, Tuple
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from utils.print_utils import print_info, print_warning

# Import specialized modules
from .jsonl_utils import create_jsonl_record
from .csv_utils import create_csv_record


def extract_dataset_records(game_data: Dict[str, Any], agent_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract JSONL and CSV records from a completed game.
    For heuristics-v0.04 and future extensions, always require authoritative per-round game states (e.g., dataset_game_states).
    Fail fast if not present or incomplete.
    """
    jsonl_records = []
    csv_records = []
    
    # Use authoritative per-round game states
    dataset_game_states = game_data.get('dataset_game_states', {})
    if not dataset_game_states or not isinstance(dataset_game_states, dict):
        raise ValueError("[DatasetFormatUtils] Missing or invalid 'dataset_game_states' in game data. Cannot extract records without SSOT.")

    moves_history = game_data.get('detailed_history', {}).get('moves', [])
    if not moves_history:
        print_warning("[DatasetFormatUtils] No moves found in game data")
        return jsonl_records, csv_records

    explanations, metrics_list = _extract_explanations_and_metrics(game_data, len(moves_history))

    for i, move in enumerate(moves_history):
        # Game states are keyed by round number (1-indexed), moves are 0-indexed
        # So for move index i, we need game state from round i+1
        round_key = str(i + 1)
        game_state = dataset_game_states.get(round_key)
        if not game_state:
            raise ValueError(f"[DatasetFormatUtils] Missing game state for round {round_key} (move index {i}) in 'dataset_game_states'. SSOT violation.")
        jsonl_record = create_jsonl_record(
            game_state=game_state,
            move=move,
            explanation=explanations[i] if i < len(explanations) else "No explanation provided.",
            metrics=metrics_list[i] if i < len(metrics_list) else {},
            agent_name=agent_name
        )
        jsonl_records.append(jsonl_record)
        csv_record = create_csv_record(
            game_state=game_state,
            move=move,
            step_number=i + 1,
            metrics=metrics_list[i] if i < len(metrics_list) else {}
        )
        csv_records.append(csv_record)

    print_info(f"[DatasetFormatUtils] Extracted {len(jsonl_records)} JSONL records and {len(csv_records)} CSV records")
    return jsonl_records, csv_records


def _extract_explanations_and_metrics(game_data: Dict[str, Any], num_moves: int) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Extract explanations and metrics from game data.
    
    Args:
        game_data: Game data dictionary
        num_moves: Number of moves in the game
        
    Returns:
        Tuple of (explanations, metrics_list)
    """
    explanations = []
    metrics_list = []
    
    # Extract explanations and metrics from the game data
    move_explanations = game_data.get('move_explanations', [])
    move_metrics = game_data.get('move_metrics', [])
    
    # Add explanations and metrics
    explanations.extend(move_explanations)
    metrics_list.extend(move_metrics)
    
    # If we don't have enough explanations, create placeholders
    while len(explanations) < num_moves:
        explanations.append("No explanation provided.")
    while len(metrics_list) < num_moves:
        metrics_list.append({})
    
    return explanations, metrics_list 