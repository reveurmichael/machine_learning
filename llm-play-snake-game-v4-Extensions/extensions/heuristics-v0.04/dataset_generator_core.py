"""
Core dataset generator – convert raw logs into CSV / JSONL.

This module provides the core DatasetGenerator class that converts
heuristic game logs into structured datasets for machine learning.

Design Philosophy:
- Algorithm-agnostic: Can be reused by supervised/RL extensions
- Single responsibility: Only handles dataset conversion
- Standardized logging: Uses print_utils functions for all operations
- Generic: Uses common utilities for CSV feature extraction
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import csv
import json
import sys
import os
import traceback

# Fix UTF-8 encoding issues on Windows
# This ensures that all subprocesses and file operations use UTF-8
# All file operations (CSV, JSONL, JSON) in v0.04 use UTF-8 encoding for cross-platform compatibility
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Add project root to path to allow absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.game_constants import DIRECTIONS

from utils.print_utils import print_info, print_warning, print_success, print_error
from agents.agent_bfs import BFSAgent

# Import common CSV utilities for SSOT compliance
from extensions.common.utils.csv_utils import CSVFeatureExtractor, create_csv_record_with_explanation
from extensions.common.utils.game_analysis_utils import calculate_danger_assessment, calculate_apple_direction

__all__ = ["DatasetGenerator"]


class DatasetGenerator:
    """
    Convert raw heuristic game logs to datasets (CSV / JSONL).
    Designed to be algorithm-agnostic so supervised / RL can reuse it.
    
    This generator reads heuristic algorithm game logs and converts them
    into structured datasets suitable for machine learning tasks.
    
    Design Philosophy:
    - Uses common CSV utilities for feature extraction (SSOT compliance)
    - Maintains JSONL functionality for language-rich datasets
    - Generic and extensible for all tasks 1-5
    """

    def __init__(self, algorithm: str, output_dir: Path):
        """
        Initialize the dataset generator.
        
        Args:
            algorithm: The algorithm name (e.g., 'bfs', 'dfs')
            output_dir: Output directory for datasets
        """
        self.algorithm = algorithm
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV feature extractor from common utilities
        self.csv_extractor = CSVFeatureExtractor()
        
        # File handles
        self._csv_writer = None
        self._jsonl_fh = None
        
        print_info(f"Initialized for {algorithm} (output: {output_dir})", "DatasetGenerator")

    # ---------------- CSV
    def _open_csv(self):
        """Open CSV file for writing."""
        csv_path = self.output_dir / f"{self.algorithm}_dataset.csv"
        fh = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=self.csv_extractor.feature_names)
        writer.writeheader()
        self._csv_writer = (writer, fh)
        print_info(f"Opened CSV file: {csv_path}", "DatasetGenerator")

    # ---------------- JSONL
    def _open_jsonl(self):
        """Open JSONL file for writing."""
        jsonl_path = self.output_dir / f"{self.algorithm}_dataset.jsonl"
        self._jsonl_fh = jsonl_path.open("w", encoding="utf-8")
        print_info(f"Opened JSONL file: {jsonl_path}", "DatasetGenerator")

    # ---------------- PUBLIC
    def generate_games_and_write_datasets(self, max_games: int, max_steps: int, grid_size: int, formats: list = ["csv", "jsonl"], verbose: bool = False):
        """
        Run games in memory and generate datasets directly, without loading from disk.
        Args:
            max_games: Number of games to play
            max_steps: Maximum steps per game
            grid_size: Grid size for the game
            formats: List of formats to generate ("csv", "jsonl", or both)
            verbose: Enable verbose output
        """
        from game_manager import HeuristicGameManager
        import argparse
        
        # Build args namespace for HeuristicGameManager
        args = argparse.Namespace(
            algorithm=self.algorithm,
            max_games=max_games,
            max_steps=max_steps,
            grid_size=grid_size,
            verbose=verbose,
            no_gui=True
        )
        
        # Run games in memory using the proper game manager workflow
        game_manager = HeuristicGameManager(args)
        game_manager.initialize()
        
        # Open output files for dataset generation
        if "csv" in formats:
            self._open_csv()
        if "jsonl" in formats:
            self._open_jsonl()

        # Use the proper game manager run method which handles JSON file saving
        game_manager.run()

        # Close handles
        if self._csv_writer:
            self._csv_writer[1].close()
            print_success("CSV dataset saved")
        if self._jsonl_fh:
            self._jsonl_fh.close()
            print_success("JSONL dataset saved")
            
        if verbose:
            print_success(f"[DatasetGenerator] Dataset generation complete for {self.algorithm}")
            print_info(f"📁 Game files and summary saved in: {self.output_dir}")

    # ---------------- INTERNAL
    def _process_single_game(self, game_data: Dict[str, Any]) -> None:
        """Process a single game and generate dataset entries."""
        try:
            # Extract moves and explanations
            moves_history = game_data.get("detailed_history", {}).get("moves", [])
            explanations = game_data.get("move_explanations", [])
            metrics_list = game_data.get("move_metrics", [])

            if not moves_history:
                print_warning("[DatasetGenerator] No moves found in game data")
                return

            # Extract dataset game states
            dataset_game_states = game_data.get("dataset_game_states", {})
            # SSOT: Round 0 is only the initial state, not used for moves
            # All moves and planned_moves start from round 1
            rounds_data = game_data.get("detailed_history", {}).get("rounds_data", {})
            available_rounds = set(int(k) for k in rounds_data.keys())
            required_rounds = set(range(1, len(moves_history) + 1))
            missing_rounds = required_rounds - available_rounds
            if missing_rounds:
                raise RuntimeError(f"[SSOT] Missing required rounds {sorted(missing_rounds)} for {len(moves_history)} moves. Available: {sorted(available_rounds)}")

            # Only process up to the minimum length of all three lists
            n_records = min(len(moves_history), len(explanations), len(metrics_list))
            print_info(f"[DEBUG] moves_history: {len(moves_history)}, explanations: {len(explanations)}, metrics_list: {len(metrics_list)}, n_records: {n_records}")

            # For each move, use round N+1 (starting from 2)
            for i in range(n_records):
                move = moves_history[i]
                try:
                    round_num = i + 2  # rounds start from 2 for moves (skip initial state)

                    # SSOT: Check that all required data exists for this round
                    round_data = rounds_data.get(str(round_num)) or rounds_data.get(round_num)
                    if not round_data:
                        print_warning(f"[SSOT] Round {round_num} missing in rounds_data. Stopping at move {i}.")
                        break

                    game_state = dataset_game_states.get(str(round_num)) or dataset_game_states.get(round_num)
                    if not game_state:
                        print_warning(f"[SSOT] Game state for round {round_num} missing in dataset_game_states. Stopping at move {i}.")
                        break
                    
                    # SSOT: Use centralized utilities for all position extractions
                    head_pos_for_check = BFSAgent.extract_head_position(game_state)
                    body_positions_for_check = BFSAgent.extract_body_positions(game_state)

                    current_explanation = explanations[i]
                    current_metrics = metrics_list[i]

                    record = self._extract_jsonl_record({
                        "game_state": game_state,
                        "move": move,
                        "explanation": current_explanation,
                        "metrics": current_metrics,
                        "game_id": game_data.get('game_number', game_data.get('metadata', {}).get('game_number', 1)),
                        "round_num": round_num
                    })
                    
                    # Write to JSONL
                    if self._jsonl_fh:
                        self._jsonl_fh.write(json.dumps(record) + '\n')
                    # Write to CSV using common utilities
                    if self._csv_writer:
                        csv_record = self._extract_csv_features(record, step_number=round_num)  # CSV step numbers are 1-indexed
                        self._csv_writer[0].writerow(csv_record)
                        
                except Exception as e:
                    print_error(f"[DatasetGenerator] Error processing move {i} (round {round_num}): {e}")
                    print_error(f"[DatasetGenerator] Move: {move if 'move' in locals() else 'N/A'}")
                    print_error(f"[DatasetGenerator] Round data exists: {'round_data' in locals() and round_data is not None}")
                    print_error(f"[DatasetGenerator] Game state exists: {'game_state' in locals() and game_state is not None}")
                    traceback.print_exc()
                    raise
        except Exception as e:
            print_error(f"[DatasetGenerator] Error processing game {game_data.get('game_number', 'unknown')}: {str(e)}")
            raise

    def _extract_csv_features(self, record: dict, step_number: int = None) -> dict:
        """
        Extract CSV features from a single game record using common utilities.
        SSOT Compliance: Use the common CSVFeatureExtractor for consistency.
        
        PRE-EXECUTION: All game_state values are from BEFORE the move is executed.
        This ensures consistency with the JSONL format and the prompt state.
        """
        game_state = record.get('game_state', {})
        explanation = record.get('explanation', {})
        game_id = record.get('game_id', 1)
        
        # Use common CSV utilities for feature extraction
        # This ensures consistency across all extensions and follows SSOT principles
        csv_record = create_csv_record_with_explanation(game_state, explanation, step_number, game_id)
        
        return csv_record

    def _extract_jsonl_record(self, record: dict) -> Dict[str, Any]:
        """
        Extracts and formats a single JSONL record from game data.
        SSOT Compliance: Uses the agent's actual move and explanation as the source of truth.
        
        PRE-EXECUTION: All game_state values are from BEFORE the move is executed.
        This ensures consistency between the prompt (which shows pre-move state) and
        the completion metrics (which should match the prompt's state).
        
        Args:
            record: Game record containing state and move information
        Returns:
            Dictionary with prompt and completion for JSONL format
        """
        game_state = record.get('game_state', {})
        explanation = record.get('explanation', {})
        move_chosen = record.get('move') # The chosen move from moves_history
        game_id = record.get('game_id', 1)
        round_num = record.get('round_num')
        
        # SSOT: Use centralized utilities from BFSAgent for all position and calculation extractions
        head_pos = BFSAgent.extract_head_position(game_state)
        body_positions = BFSAgent.extract_body_positions(game_state)
        apple_position = game_state.get('apple_position') # Directly extract apple_position from game_state
        grid_size = BFSAgent.extract_grid_size(game_state)

        # SSOT: body_positions can be empty for initial moves when snake is only 1 segment long
        # This is valid behavior and should be included in the dataset

        manhattan_distance = BFSAgent.calculate_manhattan_distance(game_state)
        valid_moves = BFSAgent.calculate_valid_moves_ssot(game_state)

        # Ensure imports are within the method for clarity and to avoid circular dependencies if moved later
        # Removed redundant import: from extensions.common.utils.game_analysis_utils import calculate_danger_assessment, calculate_apple_direction
        
        # KISS: Use agent's explanation directly - no fallbacks needed
        # SSOT: The explanation comes from the agent and is already properly formatted
        if isinstance(explanation, dict) and 'explanation_steps' in explanation:
            explanation_text = '\n'.join(explanation['explanation_steps'])
        else:
            raise RuntimeError(f"SSOT violation: Agent explanation missing 'explanation_steps' for record {game_id}")

        # Extract the move direction from the explanation metrics (SSOT)
        move_direction = 'UNKNOWN'
        if isinstance(explanation, dict) and 'metrics' in explanation:
            agent_metrics = explanation['metrics']
            if 'final_chosen_direction' in agent_metrics:
                move_direction = agent_metrics['final_chosen_direction']
            elif 'move' in agent_metrics: # Fallback for older formats if needed, but prefer final_chosen_direction
                move_direction = agent_metrics['move']
        
        if move_direction == 'UNKNOWN':
            raise RuntimeError(f"SSOT violation: No valid move direction found in agent metrics for record {game_id}")
        
        # SSOT FAIL-FAST: Ensure the move_chosen passed from moves_history matches the agent's chosen direction
        if move_chosen != move_direction:
             raise RuntimeError(f"[SSOT] FAIL-FAST: Mismatch between recorded move {move_chosen} and agent's chosen direction {move_direction} for record {game_id}")

        # Validate head position before processing (redundant with initial check in _process_single_game, but for robustness)
        # SSOT: Use centralized utilities for all position extractions
        pre_move_head = BFSAgent.extract_head_position(game_state)
        if pre_move_head is None or not isinstance(pre_move_head, (list, tuple)) or len(pre_move_head) != 2:
            raise RuntimeError(f"[SSOT] Invalid head position in game state for round {round_num}: {pre_move_head}")
        
        # --- SSOT FAIL-FAST: Explanation head must match prompt head ---
        explanation_head = explanation.get('metrics', {}).get('head_position') if isinstance(explanation, dict) else None
        if explanation_head and tuple(explanation_head) != tuple(pre_move_head):
            print_error(f"[SSOT] FAIL-FAST: JSONL explanation head {explanation_head} != prompt head {pre_move_head} for game {game_id} round {round_num}")
            print_error(f"[SSOT] Game state: {game_state}")
            print_error(f"[SSOT] Explanation: {explanation}")
            raise RuntimeError(f"[SSOT] FAIL-FAST: JSONL explanation head {explanation_head} != prompt head {pre_move_head} for game {game_id} round {round_num}")

        # Calculate danger features using centralized utility
        danger_assessment = calculate_danger_assessment(head_pos, body_positions, grid_size, move_direction)
        danger_straight = danger_assessment['straight']
        danger_left = danger_assessment['left']
        danger_right = danger_assessment['right']

        # Calculate apple direction features using centralized utility
        apple_direction_info = calculate_apple_direction(head_pos, apple_position)
        apple_dir_up = apple_direction_info.get('up', False)
        apple_dir_down = apple_direction_info.get('down', False)
        apple_dir_left = apple_direction_info.get('left', False)
        apple_dir_right = apple_direction_info.get('right', False)
        
        # Calculate free space features
        free_space_up = BFSAgent.count_free_space_in_direction(game_state, "UP")
        free_space_down = BFSAgent.count_free_space_in_direction(game_state, "DOWN")
        free_space_left = BFSAgent.count_free_space_in_direction(game_state, "LEFT")
        free_space_right = BFSAgent.count_free_space_in_direction(game_state, "RIGHT")

        # Check if the move is valid (SSOT validation - agent should only choose valid moves)
        if move_direction not in valid_moves:
            raise RuntimeError(f"SSOT violation: JSONL target_move '{move_direction}' is not in valid moves {valid_moves} for head {head_pos} in game {game_id} round {round_num}.")

        # Format prompt using the game state
        prompt = self._format_prompt(game_state)

        # Format completion with the move and explanation
        completion = self._format_completion(move_direction, explanation_text, {
            'valid_moves': valid_moves,
            'manhattan_distance': manhattan_distance,
            'apple_direction': {
                'up': apple_dir_up,
                'down': apple_dir_down,
                'left': apple_dir_left,
                'right': apple_dir_right
            },
            'danger_assessment': {
                'straight': danger_straight,
                'left': danger_left,
                'right': danger_right
            },
            'free_space': {
                'up': free_space_up,
                'down': free_space_down,
                'left': free_space_left,
                'right': free_space_right
            }
        })
        
        # Return the complete JSONL record
        # PRE-EXECUTION: All features are from pre-move state
        return {
            "prompt": prompt,
            "completion": completion,
            "game_id": game_id,
            "round_num": round_num
        }

    def _convert_coordinates_to_tuples(self, coordinates):
        """
        Convert coordinate lists to tuple format for consistent representation.
        
        Args:
            coordinates: Can be a single coordinate [x, y] or list of coordinates [[x1, y1], [x2, y2], ...]
        Returns:
            Tuple format: (x, y) for single coordinate or [(x1, y1), (x2, y2), ...] for list
        """
        if not coordinates:
            return coordinates
        
        # Handle single coordinate [x, y]
        if isinstance(coordinates, list) and len(coordinates) == 2 and all(isinstance(x, (int, float)) for x in coordinates):
            return tuple(coordinates)
        
        # Handle list of coordinates [[x1, y1], [x2, y2], ...]
        if isinstance(coordinates, list) and all(isinstance(pos, list) and len(pos) == 2 for pos in coordinates):
            return [tuple(pos) for pos in coordinates]
        
        # Return as-is if not a coordinate format
        return coordinates

    def _format_prompt(self, game_state: dict) -> str:
        """
        Format the game state into a language-rich and structured prompt for fine-tuning.
        Args:
            game_state: Game state dictionary
        Returns:
            Formatted prompt string
        """
        if not game_state:
            return "Current game state is not available."

        grid_size = game_state.get('grid_size', 10)
        snake_positions = game_state.get('snake_positions', [])
        apple_position = game_state.get('apple_position', [])
        score = game_state.get('score', 0)
        steps = game_state.get('steps', 0)
        algorithm = game_state.get('algorithm', self.algorithm) 
        
        # SSOT: Use centralized utilities from BFSAgent for position extractions
        head_pos = BFSAgent.extract_head_position(game_state)
        if not snake_positions:
            return "Invalid game state: Snake has no positions."

        # SSOT: Use centralized body positions calculation from BFSAgent
        body_positions = BFSAgent.extract_body_positions(game_state)
        # SSOT: body_positions can be empty for initial moves when snake is only 1 segment long
        # This is valid behavior and should be included in the dataset
        manhattan_distance = BFSAgent.calculate_manhattan_distance(game_state)
        valid_moves = BFSAgent.calculate_valid_moves_ssot(game_state)
        # Remove any direct calculations of these values below

        # Validate positions
        if not isinstance(head_pos, (list, tuple)) or len(head_pos) != 2:
            head_pos = [0, 0]
        
        head_x, head_y = head_pos[0], head_pos[1]  # PRE-MOVE: current head coordinates
        
        # Calculate apple direction features using centralized utility
        # PRE-EXECUTION: Apple direction relative to current head position
        apple_direction = calculate_apple_direction(head_pos, apple_position)
        apple_dir_up = apple_direction['up']
        apple_dir_down = apple_direction['down']
        apple_dir_left = apple_direction['left']
        apple_dir_right = apple_direction['right']
        
        # Calculate free space features
        # PRE-EXECUTION: Free space in each direction from current head position
        free_space_up = BFSAgent.count_free_space_in_direction(game_state, "UP")
        free_space_down = BFSAgent.count_free_space_in_direction(game_state, "DOWN")
        free_space_left = BFSAgent.count_free_space_in_direction(game_state, "LEFT")
        free_space_right = BFSAgent.count_free_space_in_direction(game_state, "RIGHT")
        
        # Convert coordinates to tuple format for consistent representation
        head_pos_tuple = self._convert_coordinates_to_tuples(head_pos)
        apple_position_tuple = self._convert_coordinates_to_tuples(apple_position)
        body_positions_tuple = self._convert_coordinates_to_tuples(body_positions)
        
        # Format prompt using the game state with tuple coordinates
        prompt = f"""You are playing Snake on a {grid_size}x{grid_size} grid. The coordinate system is (0,0) at bottom-left to ({grid_size-1},{grid_size-1}) at top-right. Movement: UP=y+1, DOWN=y-1, RIGHT=x+1, LEFT=x-1.

Current game state:
- Score: {score}
- Steps: {steps}
- Algorithm: {algorithm}
- Snake head position: {head_pos_tuple}
- Apple position: {apple_position_tuple}
- Snake body positions: {body_positions_tuple}
- Snake length: {len(snake_positions)}

What is the best move to make? Consider:
1. Path to the apple
2. Avoiding collisions with walls and snake body
3. Maximizing score and survival

Choose from: UP, DOWN, LEFT, RIGHT

Move:"""

        return prompt 

    def _format_completion(self, move: str, explanation: str, metrics: dict) -> str:
        """
        Format the completion with move and explanation.
        Args:
            move: Agent chosen move
            explanation: Agent explanation
            metrics: Calculated metrics
        Returns:
            Formatted completion string
        """
        # Format the completion
        completion = f""" {explanation}

Metrics:
- Valid moves: {metrics.get('valid_moves', [])}
- Manhattan distance to apple: {metrics.get('manhattan_distance', 0)}
- Apple direction: {metrics.get('apple_direction', {})}
- Danger assessment: {metrics.get('danger_assessment', {})}
- Free space: {metrics.get('free_space', {})}

Conclusion: The move {move.lower()}"""

        return completion 