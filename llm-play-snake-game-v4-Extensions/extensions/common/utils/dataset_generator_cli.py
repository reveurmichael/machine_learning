"""
Unified Dataset Generation CLI for Snake Game AI Extensions.

This module provides a command-line interface for generating datasets
in multiple formats (CSV, JSONL, NPZ) from heuristic algorithm gameplay.

Design Philosophy:
- Format-agnostic generation pipeline
- Educational value through rich explanations
- Extensible for different algorithm types
- Single source of truth for dataset generation

Reference: docs/extensions-guideline/final-decision-10.md
"""

import argparse
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
import tempfile

from ..config import (
    EXTENSIONS_LOGS_DIR,
    HEURISTICS_LOG_PREFIX,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_GAMES,
    DEFAULT_MAX_STEPS
)
from ..config.dataset_formats import (
    CSV_BASIC_COLUMNS,
    JSONL_REQUIRED_FIELDS,
    JSONL_OPTIONAL_FIELDS
)
from .dataset_utils import save_csv_dataset, save_jsonl_dataset
from .path_utils import setup_extension_paths

# =============================================================================
# Game Session Runner Integration
# =============================================================================

def run_heuristic_games(algorithm: str, max_games: int, max_steps: int, grid_size: int, verbose: bool = False) -> List[str]:
    """
    Run actual heuristic games and return list of log directories.
    
    This function runs the actual heuristic games using the existing
    game infrastructure and returns the paths to generated log files.
    """
    print(f"[GameRunner] Running {max_games} games with {algorithm} algorithm...")
    
    log_paths = []
    
    for game_num in range(max_games):
        if verbose:
            print(f"[GameRunner] Starting game {game_num + 1}/{max_games}")
        
        try:
            # Run a single game using the heuristics main script
            cmd = [
                sys.executable, "scripts/main.py",
                "--algorithm", algorithm,
                "--max-games", "1",
                "--max-steps", str(max_steps),
                "--grid-size", str(grid_size)
            ]
            
            if verbose:
                cmd.append("--verbose")
            
            # Change to heuristics-v0.04 directory for execution
            heuristics_dir = Path(__file__).parent.parent.parent / "heuristics-v0.04"
            
            # Run the game
            result = subprocess.run(
                cmd,
                cwd=str(heuristics_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per game
            )
            
            if result.returncode == 0:
                # Parse output to find log directory
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "📂 Logs:" in line:
                        log_path = line.split("📂 Logs: ")[1].strip()
                        log_paths.append(log_path)
                        if verbose:
                            print(f"[GameRunner] Game {game_num + 1} completed: {log_path}")
                        break
                else:
                    if verbose:
                        print(f"[GameRunner] Warning: Could not parse log path from game {game_num + 1}")
            else:
                if verbose:
                    print(f"[GameRunner] Error in game {game_num + 1}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"[GameRunner] Game {game_num + 1} timed out")
        except Exception as e:
            print(f"[GameRunner] Error running game {game_num + 1}: {e}")
    
    print(f"[GameRunner] Completed: {len(log_paths)} successful games out of {max_games}")
    return log_paths


def load_game_logs(log_paths: List[str], verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Load game data from log directories.
    
    Args:
        log_paths: List of paths to game log directories
        verbose: Enable verbose logging
        
    Returns:
        List of game data dictionaries
    """
    games_data = []
    
    for i, log_path in enumerate(log_paths, 1):
        if verbose:
            print(f"[LogLoader] Loading game {i}/{len(log_paths)}: {log_path}")
        
        try:
            log_dir = Path(log_path)
            
            # Look for game JSON files
            game_files = list(log_dir.glob("game_*.json"))
            
            for game_file in game_files:
                with open(game_file, 'r') as f:
                    game_data = json.load(f)
                    
                # Add metadata
                game_data['log_path'] = str(log_path)
                game_data['log_file'] = str(game_file)
                
                games_data.append(game_data)
                
                if verbose:
                    rounds_count = len(game_data.get('rounds', []))
                    score = game_data.get('final_score', 0)
                    print(f"[LogLoader] Loaded game: {rounds_count} rounds, score {score}")
            
            if not game_files:
                if verbose:
                    print(f"[LogLoader] Warning: No game files found in {log_path}")
                
        except Exception as e:
            print(f"[LogLoader] Error loading {log_path}: {e}")
    
    print(f"[LogLoader] Successfully loaded {len(games_data)} games")
    return games_data

# =============================================================================
# Dataset Generation Core
# =============================================================================

class DatasetGenerator:
    """
    Core dataset generation logic that can produce multiple formats.
    
    Design Pattern: Strategy Pattern
    Purpose: Generate datasets in different formats from the same game data
    
    This generator reads heuristic algorithm game logs and converts them
    into structured datasets suitable for machine learning tasks.
    """
    
    def __init__(self, algorithm: str, grid_size: int = DEFAULT_GRID_SIZE):
        """Initialize dataset generator for specific algorithm."""
        self.algorithm = algorithm
        self.grid_size = grid_size
        self.csv_rows = []
        self.jsonl_records = []
        self.output_dir = None
        
        print(f"[DatasetGenerator] Initialized for {algorithm} (grid size: {grid_size})")
    
    def generate_from_logs(self, log_files: List[str], formats: List[str]) -> None:
        """
        Generates datasets by processing a list of log files.

        This is the main entry point for the generator logic. It sets up file
        writers based on the requested formats and then iterates through each
        log file to process it.
        """
        # Create a temporary directory for output files
        with tempfile.TemporaryDirectory() as temp_dir:
            writers = {}
            if 'jsonl' in formats:
                jsonl_path = Path(temp_dir) / "data.jsonl"
                writers['jsonl'] = open(jsonl_path, 'w', encoding='utf-8')
            if 'csv' in formats:
                csv_path = Path(temp_dir) / "data.csv"
                # Setup CSV writer
                csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
                self.csv_headers = CSV_BASIC_COLUMNS
                writers['csv'] = csv.DictWriter(csv_file, fieldnames=self.csv_headers)
                writers['csv'].writeheader()

            self.jsonl_writer = writers.get('jsonl')
            self.csv_writer = writers.get('csv')

            print(f"[DatasetGenerator] Processing {len(log_files)} game log files...")
            for game_file in log_files:
                self._process_single_game(game_file)
            
            # Close file handlers
            for writer in writers.values():
                writer.close()

            # Move completed files to final destination
            if self.output_dir:
                if 'jsonl' in formats:
                    final_jsonl_path = self.output_dir / f"{self.algorithm.lower()}_dataset.jsonl"
                    Path(jsonl_path).rename(final_jsonl_path)
                    print(f"[DatasetGenerator] JSONL dataset saved to {final_jsonl_path}")
                if 'csv' in formats:
                    final_csv_path = self.output_dir / f"{self.algorithm.lower()}_dataset.csv"
                    Path(csv_path).rename(final_csv_path)
                    print(f"[DatasetGenerator] CSV dataset saved to {final_csv_path}")

    def _process_single_game(self, game_file: str) -> None:
        """Processes a single game log file (game_N.json)."""
        try:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not read or parse {game_file}. Skipping.")
            return

        print(f"Processing game file: {game_file}")
        
        rounds_data_dict = game_data.get('detailed_history', {}).get('rounds_data', {})
        
        if not rounds_data_dict:
            print(f"Warning: No rounds_data found in {game_file}. Skipping.")
            return

        moves_history = game_data.get('detailed_history', {}).get('moves', [])
        explanations = game_data.get('detailed_history', {}).get('move_explanations', [])
        
        while len(explanations) < len(moves_history):
            explanations.append("No explanation provided.")

        for i, move in enumerate(moves_history):
            round_number_str = str(i + 1)
            round_data = rounds_data_dict.get(round_number_str, {})
            
            game_state = round_data.get('game_state')
            if not game_state:
                print(f"Warning: No game_state found for round {round_number_str} in {game_file}. Skipping step.")
                continue

            record = {
                "game_state": game_state,
                "move": move,
                "explanation": explanations[i]
            }

            if self.jsonl_writer:
                jsonl_record = self._extract_jsonl_record(record)
                self.jsonl_writer.write(json.dumps(jsonl_record) + '\\n')
            
            if self.csv_writer:
                csv_record = self._extract_csv_features(record)
                self.csv_writer.writerow(csv_record)

    def _extract_jsonl_record(self, record: dict) -> dict:
        """Creates a JSONL record with a natural language prompt and completion."""
        game_state = record['game_state']
        prompt = self._format_prompt(game_state)
        
        completion = {
            "move": record['move'],
            "explanation": record['explanation'],
            "algorithm": game_state.get('algorithm', 'unknown')
        }
        
        return {"prompt": prompt, "completion": json.dumps(completion)}

    def _extract_csv_features(self, record: dict) -> dict:
        """
        Extracts features for a CSV record from the game state.
        
        Note: This is a placeholder. For full functionality, this should be
        updated to extract the 16 grid-agnostic features.
        """
        game_state = record.get('game_state', {})
        if not game_state:
            return {header: None for header in self.csv_headers}

        return {
            "move": record.get('move'),
            "score": game_state.get('score'),
            "steps": game_state.get('steps'),
            "snake_length": len(game_state.get('snake_positions', [])),
            "apple_x": game_state.get('apple_position', [None, None])[0],
            "apple_y": game_state.get('apple_position', [None, None])[1],
            # Placeholder for remaining CSV columns
            **{k: None for k in self.csv_headers if k not in ['move', 'score', 'steps', 'snake_length', 'apple_x', 'apple_y']}
        }

    def _format_prompt(self, game_state: dict) -> str:
        """Formats the game state into a language-rich prompt."""
        if not game_state:
            return "Current game state is not available."

        grid_size = game_state.get('grid_size', 10)
        snake_positions = game_state.get('snake_positions', [])
        apple_position = game_state.get('apple_position', [])
        score = game_state.get('score', 0)
        steps = game_state.get('steps', 0)

        board = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
        if apple_position:
            board[apple_position[1]][apple_position[0]] = 'A'
        for i, pos in enumerate(snake_positions):
            if i == 0:
                board[pos[1]][pos[0]] = 'H'
            else:
                board[pos[1]][pos[0]] = 'S'
        
        board_str = "\n".join(" ".join(row) for row in board)

        return f"""You are an expert snake game AI. Analyze the current game state and decide the next optimal move.

Current Board ({grid_size}x{grid_size}):
{board_str}

Game Status:
- Score: {score}
- Steps Taken: {steps}
- Snake Length: {len(snake_positions)}

Your task is to determine the single best move ('UP', 'DOWN', 'LEFT', or 'RIGHT').
Provide the move and a brief explanation of your strategy.
"""

    def close(self) -> None:
        """Closes any open file writers."""
        # ... existing code ...

# =============================================================================
# Command Line Interface
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate datasets from heuristic algorithm gameplay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate JSONL dataset for BFS algorithm
    python generate_datasets.py --algorithm BFS --format jsonl --max-games 100
    
    # Generate both CSV and JSONL for all algorithms
    python generate_datasets.py --all-algorithms --format both --max-games 50
    
    # Generate CSV dataset with specific grid size
    python generate_datasets.py --algorithm ASTAR --format csv --grid-size 12
        """
    )
    
    # Algorithm selection
    algorithm_group = parser.add_mutually_exclusive_group(required=True)
    algorithm_group.add_argument(
        "--algorithm",
        type=str,
        help="Specific algorithm to generate dataset for (e.g., BFS, ASTAR, HAMILTONIAN)"
    )
    algorithm_group.add_argument(
        "--all-algorithms",
        action="store_true",
        help="Generate datasets for all available algorithms"
    )
    
    # Format selection
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "jsonl", "both"],
        default="both",
        help="Dataset format to generate (default: both)"
    )
    
    # Game parameters
    parser.add_argument(
        "--max-games",
        type=int,
        default=DEFAULT_MAX_GAMES,
        help=f"Maximum number of games to play per algorithm (default: {DEFAULT_MAX_GAMES})"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum steps per game (default: {DEFAULT_MAX_STEPS})"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Grid size for the game (default: {DEFAULT_GRID_SIZE})"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory (default: auto-generated in logs/extensions)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def find_available_algorithms() -> List[str]:
    """Find available heuristic algorithms from existing game logs."""
    # This is a simplified implementation
    # In a real scenario, this would scan existing algorithms or import them dynamically
    return ["BFS", "ASTAR", "DFS", "HAMILTONIAN", "BFS-SAFE-GREEDY", "ASTAR-HAMILTONIAN", "BFS-HAMILTONIAN"]


def main() -> None:
    """Main entry point for dataset generation CLI."""
    setup_extension_paths()
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if args.verbose:
        print("Dataset Generation CLI v0.04 (Real Game Integration)")
        print("=" * 50)
        print(f"Format: {args.format}")
        print(f"Max games: {args.max_games}")
        print(f"Grid size: {args.grid_size}")
        print()
    
    # Determine algorithms to process
    if args.all_algorithms:
        algorithms = find_available_algorithms()
        print(f"[CLI] Processing all algorithms: {algorithms}")
    else:
        algorithms = [args.algorithm]
        print(f"[CLI] Processing algorithm: {args.algorithm}")
    
    # Generate datasets for each algorithm
    for algorithm in algorithms:
        print(f"\n[CLI] Starting dataset generation for {algorithm}")
        
        try:
            # Run actual games to generate data
            log_paths = run_heuristic_games(
                algorithm, args.max_games, args.max_steps, args.grid_size, args.verbose
            )
            
            if not log_paths:
                print(f"[CLI] ⚠️  No successful games for {algorithm}, skipping...")
                continue
            
            # Load game data from logs
            games_data = load_game_logs(log_paths, args.verbose)
            
            if not games_data:
                print(f"[CLI] ⚠️  No game data loaded for {algorithm}, skipping...")
                continue
            
            # Create dataset generator and define output directory
            generator = DatasetGenerator(algorithm, args.grid_size)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                output_dir = Path(EXTENSIONS_LOGS_DIR) / "datasets" / f"grid-size-{args.grid_size}" / f"heuristics-{algorithm.lower()}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            generator.output_dir = output_dir

            # Get the list of actual game log files to process
            log_files = [game['log_file'] for game in games_data]
            
            # Determine formats to generate
            formats_to_generate = []
            if args.format in ["csv", "both"]:
                formats_to_generate.append("csv")
            if args.format in ["jsonl", "both"]:
                formats_to_generate.append("jsonl")

            # Generate datasets from the logs
            generator.generate_from_logs(log_files, formats_to_generate)
            
            print(f"[CLI] ✅ Completed dataset generation for {algorithm}")
            print(f"[CLI] 📁 Output directory: {output_dir}")
            
        except Exception as e:
            print(f"[CLI] ❌ Failed to generate dataset for {algorithm}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print("\n[CLI] ✅ Dataset generation completed!")


if __name__ == "__main__":
    main() 