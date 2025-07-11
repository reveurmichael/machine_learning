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
# No additional imports needed - agents handle all validation and extraction

# Import common CSV utilities for SSOT compliance
from extensions.common.utils.csv_utils import (
    CSVFeatureExtractor,
    create_csv_record_with_explanation,
)

# Import round utilities for clean round management
from game_rounds import create_dataset_records

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
                    # Delegate all validation and processing to agent
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
        """Create a JSONL record using agent's centralized generation method."""
        # Generic data extraction (centralized in dataset_generator)
        game_state = record.get("game_state", {})
        explanation = record.get("explanation", {})
        move_chosen = record.get("move")
        game_id = record.get("game_id", 1)
        round_num = record.get("round_num", 1)

        # SSOT: Agent is required and must have centralized method
        if not self.agent:
            raise RuntimeError("Agent is required for JSONL generation")
        
        if not hasattr(self.agent, 'generate_jsonl_record'):
            raise RuntimeError(f"Agent {self.agent.__class__.__name__} must implement generate_jsonl_record() method")

        # Delegate all validation, formatting, and record generation to agent
        return self.agent.generate_jsonl_record(
            game_state=game_state,
            move=move_chosen,
            explanation=explanation,
            game_id=game_id,
            round_num=round_num
        )


