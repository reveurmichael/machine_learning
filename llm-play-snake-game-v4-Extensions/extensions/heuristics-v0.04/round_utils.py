"""
Round Management Utilities for Heuristics v0.04

This module provides utilities to manage rounds properly following Task-0 patterns.
It eliminates the ugly +2 offset by ensuring rounds start from 1 with actual moves,
not from round 0 with initial state.

Design Philosophy:
- Follow Task-0 round numbering: Round 1 = first move
- Provide clean utilities for dataset generation
- Single source of truth for round/move alignment
- Elegant and KISS-compliant implementation
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.print_utils import print_info, print_warning, print_error

__all__ = [
    "create_round_move_mapping",
    "get_move_for_round", 
    "get_round_for_move_index",
    "validate_round_consistency",
    "extract_dataset_records"
]


def create_round_move_mapping(moves_history: List[str], rounds_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Create a clean mapping between round numbers and moves.
    
    This function follows Task-0 patterns where:
    - Round 1 contains the first move
    - Round N contains the Nth move (or group of moves)
    - No Round 0 with initial state
    
    The function automatically shifts rounds to start from 1 for Task-0 compatibility.
    
    Args:
        moves_history: List of all moves in chronological order
        rounds_data: Raw rounds data from game JSON
        
    Returns:
        Dictionary mapping round number to move data (shifted to start from 1)
        
    Raises:
        RuntimeError: If round consistency validation fails
    """
    round_mapping = {}
    
    # Convert string keys to integers and sort
    round_numbers = sorted([int(k) for k in rounds_data.keys() if str(k).isdigit()])
    
    if not round_numbers:
        print_warning("[RoundUtils] No valid round numbers found in rounds_data")
        return round_mapping
    
    # Extract rounds that have moves (skip empty initial rounds)
    rounds_with_moves = []
    for round_num in round_numbers:
        round_key = str(round_num)
        round_data = rounds_data.get(round_key) or rounds_data.get(round_num)
        
        if not round_data:
            continue
            
        # Get moves from this round
        round_moves = round_data.get("moves", [])
        
        if round_moves:
            rounds_with_moves.append((round_num, round_data, round_moves))
    
    # Create mapping with Task-0 compatible numbering (start from 1)
    move_index = 0
    for i, (original_round_num, round_data, round_moves) in enumerate(rounds_with_moves):
        # Shift to Task-0 pattern: first round with moves becomes Round 1
        task0_round_num = i + 1
        
        round_mapping[task0_round_num] = {
            "moves": round_moves,
            "apple_position": round_data.get("apple_position"),
            "planned_moves": round_data.get("planned_moves", round_moves),
            "move_start_index": move_index,
            "move_count": len(round_moves),
            "original_round_num": original_round_num  # Keep track of original for dataset states
        }
        move_index += len(round_moves)
        
        print_info(f"[RoundUtils] Mapped original round {original_round_num} -> Task-0 round {task0_round_num}")
    
    return round_mapping


def get_move_for_round(round_num: int, round_mapping: Dict[int, Dict[str, Any]]) -> Optional[str]:
    """
    Get the first move for a specific round.
    
    For heuristics, each round typically contains exactly one move.
    
    Args:
        round_num: Round number
        round_mapping: Round mapping from create_round_move_mapping
        
    Returns:
        First move in the round, or None if round not found
    """
    round_data = round_mapping.get(round_num)
    if not round_data:
        return None
        
    moves = round_data.get("moves", [])
    return moves[0] if moves else None


def get_round_for_move_index(move_index: int, round_mapping: Dict[int, Dict[str, Any]]) -> Optional[int]:
    """
    Get the round number for a specific move index.
    
    Args:
        move_index: 0-based move index in chronological order
        round_mapping: Round mapping from create_round_move_mapping
        
    Returns:
        Round number containing this move, or None if not found
    """
    for round_num, round_data in round_mapping.items():
        start_idx = round_data["move_start_index"]
        count = round_data["move_count"]
        
        if start_idx <= move_index < start_idx + count:
            return round_num
    
    return None


def validate_round_consistency(moves_history: List[str], round_mapping: Dict[int, Dict[str, Any]]) -> bool:
    """
    Validate that round mapping is consistent with moves history.
    
    Args:
        moves_history: List of all moves in chronological order
        round_mapping: Round mapping from create_round_move_mapping
        
    Returns:
        True if consistent, False otherwise
    """
    # Count total moves in rounds
    total_round_moves = sum(rd["move_count"] for rd in round_mapping.values())
    
    if total_round_moves != len(moves_history):
        print_error(f"[RoundUtils] Move count mismatch: rounds={total_round_moves}, history={len(moves_history)}")
        return False
    
    # Verify move sequence
    for i, move in enumerate(moves_history):
        round_num = get_round_for_move_index(i, round_mapping)
        if round_num is None:
            print_error(f"[RoundUtils] Move {i} ({move}) not found in any round")
            return False
    
    return True


def extract_dataset_records(
    game_data: Dict[str, Any],
    moves_history: List[str],
    explanations: List[Any],
    metrics_list: List[Dict[str, Any]]
) -> List[Tuple[int, str, Any, Dict[str, Any], Dict[str, Any]]]:
    """
    Extract dataset records with proper round alignment.
    
    This function eliminates the +2 offset by using proper round mapping.
    
    Args:
        game_data: Complete game data dictionary
        moves_history: List of moves in chronological order
        explanations: List of move explanations
        metrics_list: List of move metrics
        
    Returns:
        List of tuples: (round_num, move, explanation, metrics, game_state)
        
    Raises:
        RuntimeError: If data consistency validation fails
    """
    records = []
    
    # Extract required data
    rounds_data = game_data.get("detailed_history", {}).get("rounds_data", {})
    dataset_game_states = game_data.get("dataset_game_states", {})
    
    # Create clean round mapping
    round_mapping = create_round_move_mapping(moves_history, rounds_data)
    
    # Validate consistency
    if not validate_round_consistency(moves_history, round_mapping):
        raise RuntimeError("[RoundUtils] Round consistency validation failed")
    
    # Process each move with proper round alignment
    n_records = min(len(moves_history), len(explanations), len(metrics_list))
    
    for i in range(n_records):
        move = moves_history[i]
        explanation = explanations[i]
        metrics = metrics_list[i]
        
        # Get round number for this move (no +2 offset!)
        round_num = get_round_for_move_index(i, round_mapping)
        
        if round_num is None:
            print_warning(f"[RoundUtils] Move {i} ({move}) not found in round mapping, skipping")
            continue
        
        # Get original round number for dataset states lookup
        round_data = round_mapping.get(round_num, {})
        original_round_num = round_data.get("original_round_num", round_num)
        
        # Get game state using original round number
        game_state = dataset_game_states.get(str(original_round_num)) or dataset_game_states.get(original_round_num)
        
        if not game_state:
            print_warning(f"[RoundUtils] Game state for original round {original_round_num} (Task-0 round {round_num}) missing, skipping move {i}")
            continue
        
        # Add the record using Task-0 round numbering
        records.append((round_num, move, explanation, metrics, game_state))
        
        print_info(f"[RoundUtils] Processed move {i}: round={round_num}, move={move}")
    
    return records 