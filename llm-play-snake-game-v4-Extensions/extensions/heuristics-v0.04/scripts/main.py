"""
Dataset Generation CLI – parameter parsing & orchestration only.

This module provides the command-line interface for dataset generation,
orchestrating the game running and dataset generation processes.

Design Philosophy:
- Single responsibility: Only handles CLI parsing and orchestration
- Delegates actual work to dataset_game_runner and dataset_generator_core
- Standardized logging: Uses print_utils functions for all operations

Example usage:
    python main.py --algorithm BFS --format jsonl --max-games 2
    python main.py --all-algorithms --format both --max-games 3
"""

import sys
import os
from pathlib import Path

# Fix UTF-8 encoding issues on Windows
# This ensures that all subprocesses and file operations use UTF-8
# All file operations (CSV, JSONL, JSON) in v0.04 use UTF-8 encoding for cross-platform compatibility
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Add the heuristics-v0.04 directory to sys.path for imports
current_dir = Path(__file__).resolve().parent
heuristics_dir = current_dir.parent
sys.path.insert(0, str(heuristics_dir))

# Add the project root for utils imports
project_root = heuristics_dir.parent.parent
sys.path.insert(0, str(project_root))

from dataset_generator_core import DatasetGenerator
import argparse
from typing import List

from utils.path_utils import ensure_project_root

# Unified CLI logging helpers (Emoji + Color)
from utils.print_utils import print_info, print_success, print_warning, print_error

# Default parameters for CLI
DEFAULT_GRID_SIZE = 10
DEFAULT_MAX_GAMES = 100
DEFAULT_MAX_STEPS = 500

__all__ = ["create_argument_parser", "find_available_algorithms", "main"]


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate datasets from heuristic algorithm gameplay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate JSONL dataset for BFS algorithm
    python main.py --algorithm BFS --format jsonl --max-games 100
    
    # Generate both CSV and JSONL for all algorithms
    python main.py --all-algorithms --format both --max-games 50
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
    """Find available heuristic algorithms."""
    # This is a simplified implementation
    # In a real scenario, this would scan existing algorithms or import them dynamically
    return ["BFS",  "BFS-SAFE-GREEDY"]


def main() -> None:
    """Main entry point for dataset generation CLI."""
    ensure_project_root()
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if args.verbose:
        print_info("Dataset Generation CLI v0.04 (Modular Architecture)")
        print_info("=" * 50)
        print_info(f"Format: {args.format}")
        print_info(f"Max games: {args.max_games}")
        print_info(f"Grid size: {args.grid_size}")
        print_info("")
    
    # Determine algorithms to process
    if args.all_algorithms:
        algorithms = find_available_algorithms()
        print_info(f"Processing all algorithms: {algorithms}")
    else:
        algorithms = [args.algorithm]
        print_info(f"Processing algorithm: {args.algorithm}")
    
    # Generate datasets for each algorithm
    for algorithm in algorithms:
        print_info(f"Starting dataset generation for {algorithm}")
        
        try:
            # Step 1: Create output directory
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                print_info(f"📁 Using custom output directory: {output_dir}")
            else:
                # Use a default log directory for unified output
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(f"logs/extensions/datasets/grid-size-{args.grid_size}/heuristics_v0.04_{timestamp}/{algorithm.lower()}")
                output_dir.mkdir(parents=True, exist_ok=True)
                print_info(f"📁 Using auto-generated output directory: {output_dir}")

            # Step 2: Run games in memory and generate datasets
            generator = DatasetGenerator(algorithm, output_dir)
            generator.generate_games_and_write_datasets(
                max_games=args.max_games,
                max_steps=args.max_steps,
                grid_size=args.grid_size,
                formats=[args.format] if args.format != "both" else ["csv", "jsonl"],
                verbose=args.verbose
            )
            print_success(f"Completed dataset generation for {algorithm}")
            print_info(f"📁 Dataset files created in: {output_dir}")
            
        except Exception as e:
            print_error(f"Failed to generate dataset for {algorithm}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print_success("Dataset generation completed!")


if __name__ == "__main__":
    main() 
