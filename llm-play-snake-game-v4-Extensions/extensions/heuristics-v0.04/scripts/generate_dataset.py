#!/usr/bin/env python3
"""
JSONL Dataset Generation Script for Heuristics v0.04
===================================================

Generates JSONL training datasets from heuristic game logs for LLM fine-tuning.
This script specifically creates language-rich datasets with natural language
explanations for each move decision.

v0.04 Focus: LLM Fine-tuning Dataset Generation
- Generates JSONL files (not CSV/NPZ/Parquet)
- Includes rich natural language explanations
- Structured for prompt-completion fine-tuning
- Optimized for Task 4 LLM training

Usage:
    # Generate JSONL dataset for LLM fine-tuning
    python scripts/generate_dataset.py --algorithm BFS --games 1000 \
        --output-dir ../../logs/extensions/datasets/grid-size-10/ \
        --log-path ../../logs/extensions/heuristics-bfs_20250623_102805

    # Generate dataset with multiple algorithms  
    python scripts/generate_dataset.py --algorithm mixed --games 500 \
        --log-path ../../logs/extensions/heuristics-bfs_20250623_102805 \
        --log-path ../../logs/extensions/heuristics-astar_20250623_103000

Features:
- JSONL format for LLM training pipelines
- Rich natural language move explanations
- Prompt-completion pairs for supervised fine-tuning
- Support for multiple algorithms in single dataset
- Configurable prompt templates
- Extensive metadata preservation for analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))

from extensions.common.path_utils import ensure_datasets_dir, DEFAULT_GRID_SIZE, validate_grid_size


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for JSONL dataset generation.
    
    Design Pattern: Command Pattern
    - Encapsulates dataset generation requests as objects
    - Allows parameterization of different prompt formats
    - Supports queuing and logging of generation operations
    """
    parser = argparse.ArgumentParser(
        description="Generate JSONL training datasets from heuristic game logs for LLM fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single algorithm JSONL dataset
  python scripts/generate_dataset.py --algorithm BFS --games 1000 \\
      --log-path ../../logs/extensions/heuristics-bfs_20250623_102805

  # Mixed algorithms for diverse training data
  python scripts/generate_dataset.py --algorithm mixed --games 2000 \\
      --log-path ../../logs/extensions/heuristics-bfs_20250623_102805 \\
      --log-path ../../logs/extensions/heuristics-astar_20250623_103000 \\
      --log-path ../../logs/extensions/heuristics-hamiltonian_20250623_103500

  # Custom prompt format
  python scripts/generate_dataset.py --algorithm ASTAR --games 500 \\
      --prompt-format detailed --include-metadata \\
      --log-path ../../logs/extensions/heuristics-astar_20250623_103000
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--algorithm",
        required=True,
        help="Algorithm name for dataset (BFS, ASTAR, HAMILTONIAN, mixed, etc.)"
    )
    
    parser.add_argument(
        "--log-path",
        action="append",
        required=True,
        help="Path to heuristic log directory (can be specified multiple times for mixed datasets)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Target number of games to process (default: 1000)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for JSONL files (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Grid size for the dataset (default: {DEFAULT_GRID_SIZE})"
    )
    
    parser.add_argument(
        "--prompt-format",
        choices=["simple", "detailed", "instruction"],
        default="detailed",
        help="Format for prompt generation (default: detailed)"
    )
    
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include additional metadata in JSONL entries"
    )
    
    parser.add_argument(
        "--min-moves",
        type=int,
        default=5,
        help="Minimum number of moves required per game (default: 5)"
    )
    
    parser.add_argument(
        "--max-moves",
        type=int,
        default=500,
        help="Maximum number of moves per game to include (default: 500)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually generating the dataset"
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if arguments are valid, False otherwise
    """
    # Validate grid size
    if not validate_grid_size(args.grid_size):
        from extensions.common.config import SUPPORTED_GRID_SIZES
        print(f"❌ Unsupported grid size: {args.grid_size}")
        print(f"Supported sizes: {SUPPORTED_GRID_SIZES}")
        return False
    
    # Auto-generate output directory if not provided
    if not args.output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"logs/extensions/datasets/grid-size-{args.grid_size}/jsonl_{args.algorithm}_{timestamp}"
        print(f"📁 Auto-generated output directory: {args.output_dir}")
    
    # Ensure output directory exists
    output_path = Path(args.output_dir)
    if not args.dry_run:
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            if args.verbose:
                print(f"✅ Created output directory: {output_path}")
        except Exception as e:
            print(f"❌ Cannot create output directory: {e}")
            return False
    
    # Check if log paths exist
    for log_path in args.log_path:
        path = Path(log_path)
        if not path.exists():
            print(f"❌ Log path does not exist: {log_path}")
            return False
        
        # Check if it contains game JSON files
        game_files = list(path.glob("game_*.json"))
        if not game_files:
            print(f"❌ No game_*.json files found in: {log_path}")
            return False
        
        if args.verbose:
            print(f"✅ Found {len(game_files)} game files in: {log_path}")
    
    return True


def load_game_data(log_path: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Load game data from a log directory.
    
    Args:
        log_path: Path to the log directory
        verbose: Enable verbose output
        
    Returns:
        List of game data dictionaries
    """
    games = []
    log_dir = Path(log_path)
    
    game_files = sorted(log_dir.glob("game_*.json"))
    
    for game_file in game_files:
        try:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
                games.append(game_data)
                
        except Exception as e:
            if verbose:
                print(f"⚠️  Error loading {game_file}: {e}")
    
    return games


def generate_prompt_completion_pair(
    game_data: Dict[str, Any], 
    move_index: int, 
    prompt_format: str = "detailed",
    include_metadata: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Generate a prompt-completion pair for a specific move.
    
    Args:
        game_data: Game data dictionary
        move_index: Index of the move to generate pair for
        prompt_format: Format for prompt generation
        include_metadata: Include additional metadata
        
    Returns:
        Dictionary with prompt, completion, and optional metadata
    """
    try:
        # Extract game information
        detailed_history = game_data.get("detailed_history", {})
        moves = detailed_history.get("moves", [])
        apple_positions = detailed_history.get("apple_positions", [])
        rounds_data = detailed_history.get("rounds_data", {})
        
        # Validate move index
        if move_index >= len(moves):
            return None
            
        # Get current move and explanation
        current_move = moves[move_index]
        
        # Try to get explanation from rounds data
        explanation = ""
        for round_key, round_data in rounds_data.items():
            round_moves = round_data.get("moves", [])
            if move_index < len(round_moves) and round_moves[move_index] == current_move:
                explanation = round_data.get("explanation", "")
                break
        
        # Fallback explanation if not found
        if not explanation:
            algorithm = game_data.get("heuristic_info", {}).get("algorithm", "Unknown")
            explanation = f"{algorithm} algorithm chose to move {current_move}."
        
        # Get game state at this move
        head_pos = None
        apple_pos = None
        snake_length = game_data.get("snake_length", 3)
        
        if move_index < len(apple_positions):
            apple_pos = apple_positions[move_index]
        
        # Generate prompt based on format
        if prompt_format == "simple":
            prompt = f"What move should the snake make next?"
            
        elif prompt_format == "detailed":
            state_description = f"Game state: Move {move_index + 1}"
            if apple_pos:
                state_description += f", apple at {apple_pos}"
            state_description += f", snake length {snake_length}."
            prompt = f"{state_description} What move should the snake make and why?"
            
        elif prompt_format == "instruction":
            instruction = "You are an expert Snake game player. Analyze the current game state and choose the best move."
            state_description = f"Current move: {move_index + 1}"
            if apple_pos:
                state_description += f", Apple position: {apple_pos}"
            state_description += f", Snake length: {snake_length}"
            prompt = f"{instruction}\n\n{state_description}\n\nProvide your move and reasoning:"
        
        # Generate completion (move + explanation)
        completion = f"Move {current_move}. {explanation}"
        
        # Create the pair
        pair = {
            "prompt": prompt,
            "completion": completion
        }
        
        # Add metadata if requested
        if include_metadata:
            pair.update({
                "game_id": game_data.get("metadata", {}).get("game_number", 0),
                "move_index": move_index,
                "algorithm": game_data.get("heuristic_info", {}).get("algorithm", "Unknown"),
                "score": game_data.get("score", 0),
                "steps": game_data.get("steps", 0),
                "apple_position": apple_pos,
                "snake_length": snake_length
            })
        
        return pair
        
    except Exception as e:
        print(f"⚠️  Error generating prompt-completion pair: {e}")
        return None


def generate_jsonl_dataset(
    log_paths: List[str], 
    algorithm: str,
    output_dir: str,
    args: argparse.Namespace
) -> bool:
    """
    Generate JSONL dataset from game logs.
    
    Args:
        log_paths: List of log directory paths
        algorithm: Algorithm name for output file
        output_dir: Output directory path
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Collect all games from all log paths
        all_games = []
        
        for log_path in log_paths:
            games = load_game_data(log_path, args.verbose)
            all_games.extend(games)
            
            if args.verbose:
                print(f"✅ Loaded {len(games)} games from {log_path}")
        
        if not all_games:
            print("❌ No games loaded from any log path")
            return False
        
        # Filter games by move count
        filtered_games = []
        for game in all_games:
            moves = game.get("detailed_history", {}).get("moves", [])
            if args.min_moves <= len(moves) <= args.max_moves:
                filtered_games.append(game)
        
        if args.verbose:
            print(f"✅ Filtered to {len(filtered_games)} games (moves: {args.min_moves}-{args.max_moves})")
        
        # Limit number of games if specified
        if args.games > 0:
            filtered_games = filtered_games[:args.games]
            if args.verbose:
                print(f"✅ Limited to {len(filtered_games)} games")
        
        # Generate JSONL entries
        jsonl_entries = []
        
        for game in filtered_games:
            moves = game.get("detailed_history", {}).get("moves", [])
            
            # Generate prompt-completion pairs for each move
            for move_index in range(len(moves)):
                pair = generate_prompt_completion_pair(
                    game, 
                    move_index, 
                    args.prompt_format, 
                    args.include_metadata
                )
                
                if pair:
                    jsonl_entries.append(pair)
        
        if not jsonl_entries:
            print("❌ No valid prompt-completion pairs generated")
            return False
        
        # Write JSONL file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(output_dir) / f"heuristic_{algorithm}_{timestamp}.jsonl"
        
        if args.dry_run:
            print(f"🔍 DRY RUN: Would write {len(jsonl_entries)} entries to {output_file}")
            # Show sample entries
            for i, entry in enumerate(jsonl_entries[:3]):
                print(f"\nSample entry {i+1}:")
                print(json.dumps(entry, indent=2))
            return True
        
        with open(output_file, 'w') as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"✅ Generated JSONL dataset: {output_file}")
        print(f"📊 Total entries: {len(jsonl_entries)}")
        print(f"📊 Games processed: {len(filtered_games)}")
        
        # Generate metadata file
        metadata = {
            "dataset_info": {
                "format": "jsonl",
                "purpose": "LLM fine-tuning",
                "algorithm": algorithm,
                "total_entries": len(jsonl_entries),
                "games_processed": len(filtered_games),
                "prompt_format": args.prompt_format,
                "include_metadata": args.include_metadata
            },
            "generation_params": {
                "min_moves": args.min_moves,
                "max_moves": args.max_moves,
                "grid_size": args.grid_size,
                "target_games": args.games
            },
            "source_logs": log_paths,
            "generated_at": datetime.datetime.now().isoformat()
        }
        
        metadata_file = Path(output_dir) / f"metadata_{algorithm}_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Generated metadata: {metadata_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating JSONL dataset: {e}")
        return False


def print_dataset_info(args: argparse.Namespace) -> None:
    """Print information about the dataset generation."""
    print("🐍 Heuristics v0.04 - JSONL Dataset Generation")
    print("=" * 50)
    print(f"Algorithm: {args.algorithm}")
    print(f"Target games: {args.games}")
    print(f"Grid size: {args.grid_size}")
    print(f"Prompt format: {args.prompt_format}")
    print(f"Include metadata: {args.include_metadata}")
    print(f"Move range: {args.min_moves}-{args.max_moves}")
    print(f"Log paths: {len(args.log_path)}")
    for i, path in enumerate(args.log_path, 1):
        print(f"  {i}. {path}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    print()


def main() -> None:
    """Main function for JSONL dataset generation."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print information
        print_dataset_info(args)
        
        # Validate arguments
        if not validate_arguments(args):
            print("❌ Argument validation failed")
            sys.exit(1)
        
        if args.verbose:
            print("✅ Arguments validated successfully")
        
        # Generate JSONL dataset
        success = generate_jsonl_dataset(
            args.log_path,
            args.algorithm,
            args.output_dir,
            args
        )
        
        if success:
            print("\n✅ JSONL dataset generation completed successfully!")
            print("\n💡 Next steps:")
            print("   1. Review the generated JSONL file")
            print("   2. Use the dataset for LLM fine-tuning (Task 4)")
            print("   3. Validate the prompt-completion pairs")
        else:
            print("\n❌ JSONL dataset generation failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 