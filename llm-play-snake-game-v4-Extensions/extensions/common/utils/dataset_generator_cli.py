"""
Dataset Generation CLI – parameter parsing & orchestration only.

This module provides the command-line interface for dataset generation,
orchestrating the game running and dataset generation processes.

Design Philosophy:
- Single responsibility: Only handles CLI parsing and orchestration
- Delegates actual work to dataset_game_runner and dataset_generator_core
- Simple logging: Uses print() statements for all operations
"""

import argparse
from pathlib import Path
from typing import List

from .dataset_game_runner import run_heuristic_games, load_game_logs
from .dataset_generator_core import DatasetGenerator
from utils.path_utils import ensure_project_root
from ..config import DEFAULT_GRID_SIZE, DEFAULT_MAX_GAMES, DEFAULT_MAX_STEPS

# Unified CLI logging helpers (Emoji + Color)
from utils.print_utils import print_info, print_success, print_warning, print_error

__all__ = ["create_argument_parser", "find_available_algorithms", "main"]


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
    """Find available heuristic algorithms."""
    # This is a simplified implementation
    # In a real scenario, this would scan existing algorithms or import them dynamically
    return ["BFS", "ASTAR", "DFS", "HAMILTONIAN", "BFS-SAFE-GREEDY", "ASTAR-HAMILTONIAN", "BFS-HAMILTONIAN"]


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
            # Step 1: Run games to generate logs
            log_dirs = run_heuristic_games(
                algorithm, args.max_games, args.max_steps, args.grid_size, args.verbose
            )
            
            if not log_dirs:
                print_warning(f"No successful games for {algorithm}, skipping…")
                continue
            
            # Step 2: Load game data from logs
            games = load_game_logs(log_dirs, args.verbose)
            
            if not games:
                print_warning(f"No game data loaded for {algorithm}, skipping…")
                continue
            
            # Step 3: Determine output directory
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                print_info(f"📁 Using custom output directory: {output_dir}")
            else:
                # Use the shared log directory for unified output
                output_dir = Path(log_dirs[0])
                print_info(f"📁 Using shared log directory: {output_dir}")

            # Step 4: Generate datasets
            generator = DatasetGenerator(algorithm, output_dir)
            formats = {"csv", "jsonl"} if args.format == "both" else {args.format}
            generator.generate(games, list(formats))
            
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