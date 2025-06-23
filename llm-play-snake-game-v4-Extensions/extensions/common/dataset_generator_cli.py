#!/usr/bin/env python3
"""
Unified Dataset Generator CLI for Heuristic Snake Agents

This CLI provides a clean, unified interface for generating both CSV and JSONL
datasets from heuristic game logs, replacing the confusing multiple script
approach with a single, well-organized command-line tool.

Key Features:
- Unified interface for both CSV (v0.03) and JSONL (v0.04) generation
- Auto-discovery of log directories
- Batch processing for multiple algorithms
- Progress tracking and statistics
- Output validation and verification

Usage Examples:
    # Generate CSV dataset for BFS algorithm
    python dataset_generator_cli.py csv --algorithm BFS --log-dir logs/extensions/heuristics-bfs_20231201_120000/

    # Generate JSONL dataset for all algorithms
    python dataset_generator_cli.py jsonl --all-algorithms --output-dir datasets/jsonl/
    
    # Generate both formats for A* algorithm with custom settings
    python dataset_generator_cli.py both --algorithm ASTAR --max-games 500 --verbose

Design Philosophy:
This CLI follows the Unix philosophy of "do one thing well" while providing
comprehensive functionality for dataset generation. It replaces multiple
confusingly-named scripts with a single, clear interface.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add project root to Python path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import our generators
try:
    from extensions.common.heuristic_csv_generator import generate_heuristic_csv
    from extensions.common.heuristic_jsonl_generator import generate_heuristic_jsonl
except ImportError as e:
    print(f"Error importing generators: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetGeneratorCLI:
    """
    Unified CLI for Heuristic Dataset Generation.
    
    This class provides a comprehensive command-line interface for generating
    both CSV and JSONL datasets from heuristic game logs, with support for
    batch processing, auto-discovery, and validation.
    
    Features:
    - Format selection (CSV, JSONL, or both)
    - Algorithm filtering and batch processing
    - Log directory auto-discovery
    - Progress tracking and statistics
    - Output validation and verification
    
    Design Patterns:
    - Command Pattern: CLI commands as objects
    - Strategy Pattern: Different generation strategies
    - Observer Pattern: Progress reporting
    """
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.supported_algorithms = [
            "BFS", "BFS_SAFE_GREEDY", "BFS_HAMILTONIAN",
            "DFS", "ASTAR", "ASTAR_HAMILTONIAN", "HAMILTONIAN"
        ]
        self.generated_files: List[str] = []
        self.generation_stats: List[Dict[str, Any]] = []
    
    def run(self, args: argparse.Namespace) -> None:
        """
        Main entry point for CLI execution.
        
        Args:
            args: Parsed command-line arguments
        """
        try:
            logger.info(f"Starting dataset generation: format={args.format}")
            
            # Validate arguments
            self._validate_args(args)
            
            # Discover log directories if needed
            log_dirs = self._discover_log_directories(args)
            
            # Generate datasets
            if args.format == "csv":
                self._generate_csv_datasets(log_dirs, args)
            elif args.format == "jsonl":
                self._generate_jsonl_datasets(log_dirs, args)
            elif args.format == "both":
                self._generate_csv_datasets(log_dirs, args)
                self._generate_jsonl_datasets(log_dirs, args)
            
            # Report results
            self._report_results(args)
            
        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    def _validate_args(self, args: argparse.Namespace) -> None:
        """Validate command-line arguments."""
        if args.algorithm and args.algorithm.upper() not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {args.algorithm}. "
                           f"Supported: {', '.join(self.supported_algorithms)}")
        
        if args.log_dir and not Path(args.log_dir).exists():
            raise ValueError(f"Log directory does not exist: {args.log_dir}")
        
        if args.max_games and args.max_games <= 0:
            raise ValueError(f"max_games must be positive: {args.max_games}")
    
    def _discover_log_directories(self, args: argparse.Namespace) -> List[Dict[str, str]]:
        """
        Discover log directories for processing.
        
        Returns:
            List of dictionaries with 'algorithm' and 'path' keys
        """
        log_dirs = []
        
        if args.log_dir:
            # Single log directory specified
            algorithm = args.algorithm or self._detect_algorithm_from_path(args.log_dir)
            log_dirs.append({"algorithm": algorithm, "path": args.log_dir})
        
        elif args.all_algorithms:
            # Auto-discover all algorithm log directories
            log_dirs = self._auto_discover_log_directories()
        
        else:
            raise ValueError("Must specify either --log-dir or --all-algorithms")
        
        if not log_dirs:
            raise ValueError("No log directories found for processing")
        
        logger.info(f"Found {len(log_dirs)} log directories for processing")
        return log_dirs
    
    def _detect_algorithm_from_path(self, log_path: str) -> str:
        """Detect algorithm name from log directory path."""
        path_name = Path(log_path).name.lower()
        
        for algorithm in self.supported_algorithms:
            if algorithm.lower().replace("_", "-") in path_name:
                return algorithm
        
        # Default to BFS if can't detect
        logger.warning(f"Could not detect algorithm from path {log_path}, defaulting to BFS")
        return "BFS"
    
    def _auto_discover_log_directories(self) -> List[Dict[str, str]]:
        """Auto-discover heuristic log directories."""
        log_dirs = []
        extensions_logs_path = Path("logs/extensions")
        
        if not extensions_logs_path.exists():
            logger.warning("No logs/extensions directory found")
            return log_dirs
        
        # Look for heuristic log directories
        for log_dir in extensions_logs_path.iterdir():
            if log_dir.is_dir() and "heuristics" in log_dir.name.lower():
                algorithm = self._detect_algorithm_from_path(str(log_dir))
                log_dirs.append({"algorithm": algorithm, "path": str(log_dir)})
        
        return sorted(log_dirs, key=lambda x: x["algorithm"])
    
    def _generate_csv_datasets(self, log_dirs: List[Dict[str, str]], args: argparse.Namespace) -> None:
        """Generate CSV datasets for all log directories."""
        logger.info("Generating CSV datasets...")
        
        for log_info in log_dirs:
            algorithm = log_info["algorithm"]
            log_path = log_info["path"]
            
            try:
                logger.info(f"Generating CSV for {algorithm} from {log_path}")
                
                stats = generate_heuristic_csv(
                    algorithm=algorithm,
                    log_directory=log_path,
                    output_directory=args.output_dir,
                    max_games=args.max_games,
                    verbose=args.verbose
                )
                
                stats["format"] = "CSV"
                self.generation_stats.append(stats)
                self.generated_files.append(stats["output_file"])
                
                logger.info(f"CSV generation completed: {stats['total_rows']} rows")
                
            except Exception as e:
                logger.error(f"Failed to generate CSV for {algorithm}: {e}")
                if not args.continue_on_error:
                    raise
    
    def _generate_jsonl_datasets(self, log_dirs: List[Dict[str, str]], args: argparse.Namespace) -> None:
        """Generate JSONL datasets for all log directories."""
        logger.info("Generating JSONL datasets...")
        
        for log_info in log_dirs:
            algorithm = log_info["algorithm"]
            log_path = log_info["path"]
            
            try:
                logger.info(f"Generating JSONL for {algorithm} from {log_path}")
                
                stats = generate_heuristic_jsonl(
                    algorithm=algorithm,
                    log_directory=log_path,
                    output_directory=args.output_dir,
                    max_games=args.max_games,
                    prompt_format=args.prompt_format,
                    include_metadata=args.include_metadata,
                    verbose=args.verbose
                )
                
                stats["format"] = "JSONL"
                self.generation_stats.append(stats)
                self.generated_files.extend(stats["output_files"])
                
                logger.info(f"JSONL generation completed: {stats['total_entries']} entries")
                
            except Exception as e:
                logger.error(f"Failed to generate JSONL for {algorithm}: {e}")
                if not args.continue_on_error:
                    raise
    
    def _report_results(self, args: argparse.Namespace) -> None:
        """Report generation results and statistics."""
        logger.info("\n" + "="*60)
        logger.info("Dataset Generation Results")
        logger.info("="*60)
        
        if not self.generation_stats:
            logger.warning("No datasets were generated")
            return
        
        # Summary statistics
        total_csv_rows = sum(s["total_rows"] for s in self.generation_stats if s["format"] == "CSV")
        total_jsonl_entries = sum(s["total_entries"] for s in self.generation_stats if s["format"] == "JSONL")
        total_time = sum(s["generation_time_seconds"] for s in self.generation_stats)
        
        logger.info(f"Total CSV rows generated: {total_csv_rows}")
        logger.info(f"Total JSONL entries generated: {total_jsonl_entries}")
        logger.info(f"Total generation time: {total_time:.2f} seconds")
        logger.info(f"Total files created: {len(self.generated_files)}")
        
        # Detailed statistics
        if args.verbose:
            logger.info("\nDetailed Statistics:")
            for stats in self.generation_stats:
                logger.info(f"  {stats['algorithm']} ({stats['format']}): "
                          f"{stats.get('total_rows', stats.get('total_entries', 0))} items "
                          f"in {stats['generation_time_seconds']:.2f}s")
        
        # Generated files
        logger.info("\nGenerated Files:")
        for file_path in self.generated_files:
            file_size = Path(file_path).stat().st_size / 1024 / 1024  # MB
            logger.info(f"  {file_path} ({file_size:.1f} MB)")
        
        # Save generation report
        if args.save_report:
            self._save_generation_report(args)
    
    def _save_generation_report(self, args: argparse.Namespace) -> None:
        """Save generation report to JSON file."""
        report = {
            "generation_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "format": args.format,
                "output_directory": args.output_dir,
                "total_files_generated": len(self.generated_files)
            },
            "statistics": self.generation_stats,
            "generated_files": self.generated_files,
            "command_line_args": vars(args)
        }
        
        report_path = Path(args.output_dir) / f"generation_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generation report saved: {report_path}")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified Dataset Generator for Heuristic Snake Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate CSV dataset for BFS algorithm
  python dataset_generator_cli.py csv --algorithm BFS --log-dir logs/extensions/heuristics-bfs_20231201_120000/

  # Generate JSONL dataset for all algorithms
  python dataset_generator_cli.py jsonl --all-algorithms --output-dir datasets/jsonl/
  
  # Generate both formats for A* algorithm
  python dataset_generator_cli.py both --algorithm ASTAR --max-games 500 --verbose
        """
    )
    
    # Format selection (positional argument)
    parser.add_argument(
        "format",
        choices=["csv", "jsonl", "both"],
        help="Dataset format to generate"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--log-dir",
        help="Path to log directory containing game_*.json files"
    )
    input_group.add_argument(
        "--all-algorithms",
        action="store_true",
        help="Auto-discover and process all heuristic log directories"
    )
    
    # Algorithm filtering
    parser.add_argument(
        "--algorithm",
        help="Specific algorithm to process (auto-detected from path if not specified)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Output directory for generated datasets (default: datasets)"
    )
    
    # Processing options
    parser.add_argument(
        "--max-games",
        type=int,
        help="Maximum number of games to process per algorithm"
    )
    
    # JSONL-specific options
    parser.add_argument(
        "--prompt-format",
        choices=["simple", "detailed", "instruction"],
        default="detailed",
        help="Prompt template format for JSONL generation (default: detailed)"
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        default=True,
        help="Include metadata in JSONL entries (default: True)"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_false",
        dest="include_metadata",
        help="Exclude metadata from JSONL entries"
    )
    
    # Control options
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other algorithms if one fails"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save generation report to JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the CLI
    cli = DatasetGeneratorCLI()
    cli.run(args)


if __name__ == "__main__":
    main() 