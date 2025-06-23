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
    python scripts/generate_jsonl_dataset.py --algorithm BFS --games 1000

    # Generate dataset with multiple algorithms  
    python scripts/generate_jsonl_dataset.py --algorithm mixed --games 500
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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for JSONL dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate JSONL training datasets from heuristic game logs for LLM fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--algorithm",
        required=True,
        help="Algorithm name for dataset (BFS, ASTAR, HAMILTONIAN, mixed, etc.)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Target number of games to process (default: 100)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for JSONL files (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Grid size for the dataset (default: 10)"
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


def find_latest_log_dir(algorithm: str) -> Optional[str]:
    """Find the latest log directory for the given algorithm."""
    logs_dir = Path("../../logs/extensions/")
    # Fixed pattern to match actual log directory naming: heuristicsbfs_* (no dash)
    pattern = f"heuristics{algorithm.lower()}_*"
    
    if not logs_dir.exists():
        return None
    
    matching_dirs = list(logs_dir.glob(pattern))
    if not matching_dirs:
        return None
    
    # Return the most recent directory
    latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
    return str(latest_dir)


def load_game_data(log_path: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """Load game data from a log directory."""
    games = []
    log_dir = Path(log_path)
    
    if not log_dir.exists():
        if verbose:
            print(f"⚠️  Log directory does not exist: {log_path}")
        return games
    
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


def extract_state_from_move(game_data: Dict[str, Any], move_index: int) -> Dict[str, Any]:
    """Extract game state information for a specific move."""
    detailed_history = game_data.get("detailed_history", {})
    moves = detailed_history.get("moves", [])
    apple_positions = detailed_history.get("apple_positions", [])
    
    if move_index >= len(moves):
        return {}
    
    state = {
        "move_number": move_index + 1,
        "total_moves": len(moves),
        "score": game_data.get("score", 0),
        "steps": game_data.get("steps", 0)
    }
    
    # Add apple position if available
    if move_index < len(apple_positions):
        apple_pos = apple_positions[move_index]
        state["apple_position"] = apple_pos
    
    return state


def get_explanation_from_logs(game_data: Dict[str, Any], move_index: int, algorithm: str, move: str, state: Dict[str, Any]) -> str:
    """Extract explanation from game logs or generate fallback explanation."""
    
    # Try to extract real explanation from v0.04 logs
    detailed_history = game_data.get("detailed_history", {})
    move_explanations = detailed_history.get("move_explanations", [])
    
    if move_index < len(move_explanations) and move_explanations[move_index]:
        # Use real explanation from agent
        return move_explanations[move_index]
    
    # Fallback to generated explanation for older logs or missing explanations
    explanations = {
        "BFS": f"BFS algorithm found the shortest path and chose to move {move}. This move brings us closer to the apple while avoiding obstacles.",
        "ASTAR": f"A* pathfinding with Manhattan distance heuristic selected {move} as the optimal next step. This balances path cost with distance to target.",
        "DFS": f"DFS exploration chose to move {move}, following a depth-first search strategy to reach the apple.",
        "HAMILTONIAN": f"Hamiltonian cycle algorithm selected {move} to maintain the cycle while efficiently collecting the apple.",
        "BFS_SAFE_GREEDY": f"BFS with safety checks chose {move} after verifying it leads to safe territory and progresses toward the apple.",
        "BFS_HAMILTONIAN": f"BFS enhanced with Hamiltonian concepts selected {move} to balance shortest path with cycle preservation.",
        "ASTAR_HAMILTONIAN": f"A* with Hamiltonian optimization chose {move} to maintain optimal pathfinding while preserving cycle integrity."
    }
    
    base_explanation = explanations.get(algorithm, f"{algorithm} algorithm chose to move {move}.")
    
    # Add context if available
    if "apple_position" in state:
        apple_pos = state["apple_position"]
        base_explanation += f" Apple is currently at position {apple_pos}."
    
    if "move_number" in state:
        move_num = state["move_number"]
        total_moves = state.get("total_moves", move_num)
        base_explanation += f" This is move {move_num} of {total_moves} in the current game."
    
    return base_explanation


def generate_prompt_completion_pair(
    game_data: Dict[str, Any], 
    move_index: int, 
    algorithm: str,
    prompt_format: str = "detailed",
    include_metadata: bool = False
) -> Optional[Dict[str, Any]]:
    """Generate a prompt-completion pair for a specific move."""
    try:
        # Extract move information
        detailed_history = game_data.get("detailed_history", {})
        moves = detailed_history.get("moves", [])
        
        if move_index >= len(moves):
            return None
            
        current_move = moves[move_index]
        state = extract_state_from_move(game_data, move_index)
        
        # Generate prompt based on format
        if prompt_format == "simple":
            prompt = "What move should the snake make next?"
            
        elif prompt_format == "detailed":
            move_num = state.get("move_number", move_index + 1)
            score = state.get("score", 0)
            prompt = f"Snake game state: Move {move_num}, Score {score}. "
            
            if "apple_position" in state:
                prompt += f"Apple at {state['apple_position']}. "
            
            prompt += "What move should the snake make and why?"
            
        elif prompt_format == "instruction":
            instruction = "You are an expert Snake game player. Analyze the current game state and choose the best move."
            move_num = state.get("move_number", move_index + 1)
            score = state.get("score", 0)
            context = f"Move {move_num}, Score {score}"
            
            if "apple_position" in state:
                context += f", Apple at {state['apple_position']}"
            
            prompt = f"{instruction}\n\nGame state: {context}\n\nProvide your move and reasoning:"
        
        # Generate explanation and completion
        explanation = get_explanation_from_logs(game_data, move_index, algorithm, current_move, state)
        completion = f"{current_move}. {explanation}"
        
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
                "algorithm": algorithm,
                "score": state.get("score", 0),
                "move_number": state.get("move_number", move_index + 1),
                "apple_position": state.get("apple_position"),
                "grid_size": game_data.get("grid_size", 10)
            })
        
        return pair
        
    except Exception as e:
        print(f"⚠️  Error generating prompt-completion pair: {e}")
        return None


def generate_jsonl_dataset(args: argparse.Namespace) -> bool:
    """Generate JSONL dataset from game logs."""
    try:
        # Find log directory for the algorithm
        log_path = find_latest_log_dir(args.algorithm)
        if not log_path:
            print(f"❌ No log directory found for algorithm: {args.algorithm}")
            print(f"Expected pattern: ../../logs/extensions/heuristics-{args.algorithm.lower()}_*")
            return False
        
        if args.verbose:
            print(f"✅ Found log directory: {log_path}")
        
        # Load game data
        games = load_game_data(log_path, args.verbose)
        if not games:
            print(f"❌ No games found in {log_path}")
            return False
        
        if args.verbose:
            print(f"✅ Loaded {len(games)} games")
        
        # Filter and limit games
        valid_games = []
        for game in games:
            moves = game.get("detailed_history", {}).get("moves", [])
            if len(moves) >= 5:  # Minimum 5 moves
                valid_games.append(game)
        
        if args.games > 0:
            valid_games = valid_games[:args.games]
        
        if args.verbose:
            print(f"✅ Processing {len(valid_games)} valid games")
        
        # Generate JSONL entries
        jsonl_entries = []
        
        for game in valid_games:
            moves = game.get("detailed_history", {}).get("moves", [])
            
            # Generate prompt-completion pairs for each move
            for move_index in range(len(moves)):
                pair = generate_prompt_completion_pair(
                    game, 
                    move_index, 
                    args.algorithm,
                    args.prompt_format, 
                    args.include_metadata
                )
                
                if pair:
                    jsonl_entries.append(pair)
        
        if not jsonl_entries:
            print("❌ No valid prompt-completion pairs generated")
            return False
        
        # Set up output directory
        if not args.output_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = f"../../logs/extensions/datasets/grid-size-{args.grid_size}/jsonl_{args.algorithm}_{timestamp}"
        
        output_path = Path(args.output_dir)
        
        if args.dry_run:
            print(f"🔍 DRY RUN: Would create {len(jsonl_entries)} entries in {output_path}")
            # Show sample entries
            for i, entry in enumerate(jsonl_entries[:3]):
                print(f"\nSample entry {i+1}:")
                print(json.dumps(entry, indent=2))
            return True
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write JSONL file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"heuristic_{args.algorithm}_{timestamp}.jsonl"
        
        with open(output_file, 'w') as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"✅ Generated JSONL dataset: {output_file}")
        print(f"📊 Total entries: {len(jsonl_entries)}")
        print(f"📊 Games processed: {len(valid_games)}")
        
        # Generate metadata file
        metadata = {
            "dataset_info": {
                "format": "jsonl",
                "purpose": "LLM fine-tuning",
                "algorithm": args.algorithm,
                "total_entries": len(jsonl_entries),
                "games_processed": len(valid_games),
                "prompt_format": args.prompt_format,
                "include_metadata": args.include_metadata
            },
            "generation_params": {
                "grid_size": args.grid_size,
                "target_games": args.games
            },
            "source_log": log_path,
            "generated_at": datetime.datetime.now().isoformat()
        }
        
        metadata_file = output_path / f"metadata_{args.algorithm}_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Generated metadata: {metadata_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating JSONL dataset: {e}")
        return False


def main() -> None:
    """Main function for JSONL dataset generation."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        print("🐍 Heuristics v0.04 - JSONL Dataset Generation")
        print("=" * 50)
        print(f"Algorithm: {args.algorithm}")
        print(f"Target games: {args.games}")
        print(f"Grid size: {args.grid_size}")
        print(f"Prompt format: {args.prompt_format}")
        print()
        
        # Generate JSONL dataset
        success = generate_jsonl_dataset(args)
        
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