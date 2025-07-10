"""
Heuristic Game Rounds - Round management for heuristic algorithms
----------------

This module provides access to the base round management system.
The base architecture is already perfectly prepared for all tasks.

Design Philosophy:
- Use BaseRoundManager directly (inherits all generic round functionality)
- No extension needed - base class is already perfect for heuristics
- Maintains compatibility with the base round management system
- Keeps heuristics extension self-contained and standalone
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path_utils import ensure_project_root
ensure_project_root()

from core.game_rounds import BaseRoundManager
from typing import List, Dict, Any, Optional, Tuple
from utils.print_utils import print_info, print_warning, print_error

# Use BaseRoundManager directly - no extension needed
# The base class is already perfectly prepared for all tasks per round.md guidelines
HeuristicRoundManager = BaseRoundManager 


def create_round_move_mapping(moves_history: List[str], rounds_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Generate mapping of Task-0 round numbers to their move data without any offset."""
    round_mapping: Dict[int, Dict[str, Any]] = {}
    round_numbers = sorted([int(k) for k in rounds_data.keys() if str(k).isdigit()])
    if not round_numbers:
        print_warning("[HeuristicRoundManager] No valid round numbers found in rounds_data")
        return round_mapping

    rounds_with_moves: List[Tuple[int, Dict[str, Any], List[str]]] = []
    for round_num in round_numbers:
        round_key = str(round_num)
        round_data = rounds_data.get(round_key) or rounds_data.get(round_num)
        if not round_data:
            continue
        round_moves = round_data.get("moves", [])
        if round_moves:
            rounds_with_moves.append((round_num, round_data, round_moves))

    move_index = 0
    for i, (original_round_num, round_data, round_moves) in enumerate(rounds_with_moves):
        task0_round_num = i + 1  # first round containing moves becomes Round 1
        round_mapping[task0_round_num] = {
            "moves": round_moves,
            "apple_position": round_data.get("apple_position"),
            "planned_moves": round_data.get("planned_moves", round_moves),
            "move_start_index": move_index,
            "move_count": len(round_moves),
            "original_round_num": original_round_num,
        }
        move_index += len(round_moves)
        print_info(f"[HeuristicRoundManager] Mapped original round {original_round_num} -> Task-0 round {task0_round_num}")

    return round_mapping


def get_round_for_move_index(move_index: int, round_mapping: Dict[int, Dict[str, Any]]) -> Optional[int]:
    """Return the Task-0 round number corresponding to *move_index*."""
    for round_num, round_data in round_mapping.items():
        start_idx = round_data["move_start_index"]
        count = round_data["move_count"]
        if start_idx <= move_index < start_idx + count:
            return round_num
    return None


def validate_round_consistency(moves_history: List[str], round_mapping: Dict[int, Dict[str, Any]]) -> bool:
    """Ensure that *round_mapping* perfectly covers *moves_history*."""
    total_round_moves = sum(rd["move_count"] for rd in round_mapping.values())
    if total_round_moves != len(moves_history):
        print_error(f"[HeuristicRoundManager] Move count mismatch: rounds={total_round_moves}, history={len(moves_history)}")
        return False
    for i in range(len(moves_history)):
        if get_round_for_move_index(i, round_mapping) is None:
            print_error(f"[HeuristicRoundManager] Move {i} not represented in round mapping")
            return False
    return True


def extract_dataset_records(
    game_data: Dict[str, Any],
    moves_history: List[str],
    explanations: List[Any],
    metrics_list: List[Dict[str, Any]],
) -> List[Tuple[int, str, Any, Dict[str, Any], Dict[str, Any]]]:
    """Return dataset records with correct round alignment (no +2 offset)."""
    records: List[Tuple[int, str, Any, Dict[str, Any], Dict[str, Any]]] = []

    rounds_data = game_data.get("detailed_history", {}).get("rounds_data", {})
    dataset_game_states = game_data.get("dataset_game_states", {})

    round_mapping = create_round_move_mapping(moves_history, rounds_data)
    if not validate_round_consistency(moves_history, round_mapping):
        raise RuntimeError("[HeuristicRoundManager] Round consistency validation failed")

    n_records = min(len(moves_history), len(explanations), len(metrics_list))
    for i in range(n_records):
        move = moves_history[i]
        explanation = explanations[i]
        metrics = metrics_list[i]
        round_num = get_round_for_move_index(i, round_mapping)
        if round_num is None:
            print_warning(f"[HeuristicRoundManager] Move {i} ({move}) not mapped to any round — skipping")
            continue
        round_data = round_mapping.get(round_num, {})
        original_round_num = round_data.get("original_round_num", round_num)
        game_state = dataset_game_states.get(str(original_round_num)) or dataset_game_states.get(original_round_num)
        if not game_state:
            print_warning(f"[HeuristicRoundManager] Missing game_state for original round {original_round_num} — skipping move {i}")
            continue
        records.append((round_num, move, explanation, metrics, game_state))
        print_info(f"[HeuristicRoundManager] Processed move {i}: round={round_num}, move={move}")

    return records

# Re-export for convenience so callers can `from game_rounds import extract_dataset_records`
__all__ = [
    *globals().get("__all__", []),
    "create_round_move_mapping",
    "get_round_for_move_index",
    "validate_round_consistency",
    "extract_dataset_records",
] 