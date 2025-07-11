"""
Core dataset generator – generate CSV / JSONL from in-memory game data.

This module provides the core DatasetGenerator class that processes
live heuristic game data during execution and generates structured datasets for machine learning.

Design Philosophy:
- Algorithm-agnostic: Can be reused by supervised/RL extensions
- Single responsibility: Only handles dataset generation from live data
- Standardized logging: Uses print_utils functions for all operations
- Generic: Uses common utilities for CSV feature extraction
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import csv
import json
import sys
import os
import traceback

# Fix UTF-8 encoding issues on Windows
# This ensures that all subprocesses and file operations use UTF-8
# All file operations (CSV, JSONL, JSON) use UTF-8 encoding for cross-platform compatibility
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Add project root to path to allow absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.print_utils import print_info, print_warning, print_success, print_error
from extensions.common.utils.game_state_utils import (
    extract_head_position,
    extract_body_positions,
    extract_grid_size,
)
from heuristics_utils import (
    calculate_manhattan_distance,
    calculate_valid_moves_ssot,
    count_free_space_in_direction,
)

# Import common CSV utilities for SSOT compliance
from extensions.common.utils.csv_utils import (
    CSVFeatureExtractor,
    create_csv_record_with_explanation,
)
# Import common utilities for game analysis (only needed for fallback)
from extensions.common.utils.game_analysis_utils import (
    calculate_danger_assessment,
    calculate_apple_direction,
)

# Import round utilities for clean round management
from game_rounds import create_dataset_records

# Import common utilities for coordinate conversion
from extensions.common.utils.game_state_utils import convert_coordinates_to_tuples

__all__ = ["DatasetGenerator"]


class DatasetGenerator:
    """
    Generate datasets (CSV / JSONL) from in-memory heuristic game data.
    Designed to be algorithm-agnostic so supervised / RL can reuse it.

    This class processes live game data during execution and generates structured datasets
    suitable for machine learning training. It supports both CSV and JSONL formats
    and can handle multiple algorithms and grid sizes.
    """

    def __init__(self, algorithm: str, output_dir: Path, agent: Any):
        """
        Initialize the dataset generator.

        Args:
            algorithm: The algorithm name (e.g., 'bfs', 'dfs')
            output_dir: Output directory for datasets
            agent: Optional agent instance for custom prompt/completion formatting
        """
        self.algorithm = algorithm
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV feature extractor from common utilities
        self.csv_extractor = CSVFeatureExtractor()

        # File handles
        self._csv_writer = None
        self._jsonl_fh = None

        if agent is None:
            raise RuntimeError("DatasetGenerator requires a non-null agent instance (SSOT enforcement).")

        # Single Source of Truth: keep a single shared agent reference
        self.agent = agent

        print_info(
            f"Initialized for {algorithm} (output: {output_dir})", "DatasetGenerator"
        )

    # ---------------- CSV
    def _open_csv(self):
        """Open CSV file for writing."""
        csv_path = self.output_dir / f"{self.algorithm}_dataset.csv"
        fh = csv_path.open("w", newline="", encoding="utf-8")
        # Use complete column list including metadata and target columns
        from extensions.common.config.csv_formats import CSV_ALL_COLUMNS

        writer = csv.DictWriter(fh, fieldnames=CSV_ALL_COLUMNS)
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
    def generate_games_and_write_datasets(
        self,
        max_games: int,
        max_steps: int,
        grid_size: int,
        formats: list = ["csv", "jsonl"],
        verbose: bool = False,
    ):
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
            no_gui=True,
        )

        # Run games in memory using the proper game manager workflow with shared agent (SSOT)
        game_manager = HeuristicGameManager(args, agent=self.agent)
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
            print_success(
                f"[DatasetGenerator] Dataset generation complete for {self.algorithm}"
            )
            print_info(f"📁 Game files and summary saved in: {self.output_dir}")

    # ---------------- INTERNAL
    def _process_single_game(self, game_data: Dict[str, Any]) -> None:
        """Process a single game and generate dataset entries using clean round management."""
        try:
            # Extract moves and explanations
            moves_history = game_data.get("detailed_history", {}).get("moves", [])
            explanations = game_data.get("move_explanations", [])
            metrics_list = game_data.get("move_metrics", [])

            if not moves_history:
                print_warning("[DatasetGenerator] No moves found in game data")
                return

            print_info(
                f"[DatasetGenerator] Processing {len(moves_history)} moves with {len(explanations)} explanations"
            )

            # Use clean round utilities to extract dataset records (eliminates +2 offset!)
            try:
                dataset_records = create_dataset_records(
                    game_data, moves_history, explanations, metrics_list
                )
            except RuntimeError as e:
                print_error(f"[DatasetGenerator] Round extraction failed: {e}")
                raise

            # Process each record with proper round alignment
            for round_num, move, explanation, metrics, game_state in dataset_records:
                try:
                    # SSOT: Use centralized utilities for all position extractions
                    head_pos_for_check = extract_head_position(game_state)
                    body_positions_for_check = extract_body_positions(game_state)

                    record = self._create_jsonl_record(
                        {
                            "game_state": game_state,
                            "move": move,
                            "explanation": explanation,
                            "metrics": metrics,
                            "game_id": game_data.get(
                                "game_number",
                                game_data.get("metadata", {}).get("game_number", 1),
                            ),
                            "round_num": round_num,
                        }
                    )

                    # Write to JSONL
                    if self._jsonl_fh:
                        self._jsonl_fh.write(json.dumps(record) + "\n")
                        self._jsonl_fh.flush()  # Ensure immediate write on Windows

                    # Write to CSV using common utilities
                    if self._csv_writer:
                        csv_record = self._create_csv_record(
                            record, step_number=round_num
                        )
                        self._csv_writer[0].writerow(csv_record)
                        self._csv_writer[1].flush()  # Ensure immediate write on Windows

                except Exception as e:
                    print_error(
                        f"[DatasetGenerator] Error processing round {round_num}, move {move}: {e}"
                    )
                    print_error(
                        f"[DatasetGenerator] Game state exists: {game_state is not None}"
                    )
                    traceback.print_exc()
                    raise

        except Exception as e:
            print_error(
                f"[DatasetGenerator] Error processing game {game_data.get('game_number', 'unknown')}: {str(e)}"
            )
            raise

    def _create_csv_record(self, record: dict, step_number: int = None) -> dict:
        """Create a CSV record from the dataset record."""
        game_state = record.get("game_state", {})
        explanation = record.get("explanation", {})
        game_id = record.get("game_id", 1)

        # Use common CSV utilities for feature extraction
        # This ensures consistency across all extensions and follows SSOT principles
        csv_record = create_csv_record_with_explanation(
            game_state, explanation, step_number, game_id
        )

        return csv_record

    def _create_jsonl_record(self, record: dict) -> Dict[str, Any]:
        """Create a JSONL record from the dataset record using centralized agent method."""
        game_state = record.get("game_state", {})
        explanation = record.get("explanation", {})
        move_chosen = record.get("move")
        game_id = record.get("game_id", 1)
        round_num = record.get("round_num", 1)

        # SSOT: Use agent's centralized JSONL generation method
        # This eliminates code duplication and ensures consistency
        if not self.agent:
            raise RuntimeError("Agent is required for JSONL generation")
        
        # Check if agent has the centralized method
        if hasattr(self.agent, 'generate_jsonl_record'):
            return self.agent.generate_jsonl_record(
                game_state=game_state,
                move=move_chosen,
                explanation=explanation,
                game_id=game_id,
                round_num=round_num
            )
        
        # Fallback for agents without centralized method (backwards compatibility)
        return self._create_jsonl_record_fallback(record)

    def _create_jsonl_record_fallback(self, record: dict) -> Dict[str, Any]:
        """Fallback JSONL record creation for agents without centralized method."""
        game_state = record.get("game_state", {})
        explanation = record.get("explanation", {})
        move_chosen = record.get("move")
        game_id = record.get("game_id", 1)
        round_num = record.get("round_num")

        head_pos = extract_head_position(game_state)
        body_positions = extract_body_positions(game_state)
        apple_position = game_state.get("apple_position")
        grid_size = extract_grid_size(game_state)

        manhattan_distance = calculate_manhattan_distance(game_state)
        valid_moves = calculate_valid_moves_ssot(game_state)

        # KISS: Use agent's explanation directly - no fallbacks needed
        if isinstance(explanation, dict) and "explanation_steps" in explanation:
            explanation_text = "\n".join(explanation["explanation_steps"])
        else:
            raise RuntimeError(
                f"SSOT violation: Agent explanation missing 'explanation_steps' for record {game_id}"
            )

        # Extract move direction from explanation metrics (SSOT)
        if not isinstance(explanation, dict) or "metrics" not in explanation:
            raise RuntimeError(
                f"SSOT violation: Agent explanation missing 'metrics' for record {game_id}"
            )

        agent_metrics = explanation["metrics"]
        if "final_chosen_direction" in agent_metrics:
            move_direction = agent_metrics["final_chosen_direction"]
        elif "move" in agent_metrics:
            move_direction = agent_metrics["move"]
        else:
            raise RuntimeError(
                f"SSOT violation: No valid move direction found in agent metrics for record {game_id}"
            )

        if move_chosen != move_direction:
            raise RuntimeError(
                f"[SSOT] FAIL-FAST: Mismatch between recorded move {move_chosen} and agent's chosen direction {move_direction} for record {game_id}"
            )

        # Validate head position
        pre_move_head = extract_head_position(game_state)
        if (
            pre_move_head is None
            or not isinstance(pre_move_head, (list, tuple))
            or len(pre_move_head) != 2
        ):
            raise RuntimeError(
                f"[SSOT] Invalid head position in game state for round {round_num}: {pre_move_head}"
            )

        # SSOT FAIL-FAST: Explanation head must match prompt head
        explanation_head = (
            explanation.get("metrics", {}).get("head_position")
            if isinstance(explanation, dict)
            else None
        )
        if explanation_head and tuple(explanation_head) != tuple(pre_move_head):
            print_error(
                f"[SSOT] FAIL-FAST: JSONL explanation head {explanation_head} != prompt head {pre_move_head} for game {game_id} round {round_num}"
            )
            raise RuntimeError(
                f"[SSOT] FAIL-FAST: JSONL explanation head {explanation_head} != prompt head {pre_move_head} for game {game_id} round {round_num}"
            )

        # Check if the move is valid
        if move_direction not in valid_moves:
            raise RuntimeError(
                f"SSOT violation: JSONL target_move '{move_direction}' is not in valid moves {valid_moves} for head {head_pos} in game {game_id} round {round_num}."
            )

        # Calculate optional metrics based on agent switches
        metrics_dict = {
            "valid_moves": valid_moves,
            "manhattan_distance": manhattan_distance,
        }

        if getattr(self.agent, "include_apple_direction", False):
            apple_direction_info = calculate_apple_direction(head_pos, apple_position)
            metrics_dict["apple_direction"] = apple_direction_info

        if getattr(self.agent, "include_danger_assessment", False):
            danger_assessment = calculate_danger_assessment(
                head_pos, body_positions, grid_size, move_direction
            )
            metrics_dict["danger_assessment"] = danger_assessment

        if getattr(self.agent, "include_free_space", False):
            metrics_dict["free_space"] = {
                "up": count_free_space_in_direction(game_state, "UP"),
                "down": count_free_space_in_direction(game_state, "DOWN"),
                "left": count_free_space_in_direction(game_state, "LEFT"),
                "right": count_free_space_in_direction(game_state, "RIGHT"),
            }

        # Format prompt and completion using agent hooks
        if not self.agent:
            raise RuntimeError("Agent is required for JSONL generation")
        
        prompt = self.agent.format_prompt(game_state)
        completion = self.agent.format_completion(
            move_direction,
            explanation_text,
            metrics_dict,
        )

        return {
            "prompt": prompt,
            "completion": completion,
        }
