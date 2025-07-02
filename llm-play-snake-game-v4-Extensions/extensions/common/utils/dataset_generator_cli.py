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
        
        print(f"[DatasetGenerator] Initialized for {algorithm} (grid size: {grid_size})")
    
    def generate_from_games(self, games_data: List[Dict[str, Any]]) -> None:
        """
        Generate dataset records from game session data.
        
        Args:
            games_data: List of game dictionaries with moves and states
        """
        print(f"[DatasetGenerator] Processing {len(games_data)} games...")
        
        for game_idx, game_data in enumerate(games_data, 1):
            self._process_single_game(game_data, game_idx)
        
        print(f"[DatasetGenerator] Generated {len(self.csv_rows)} CSV rows")
        print(f"[DatasetGenerator] Generated {len(self.jsonl_records)} JSONL records")
    
    def _process_single_game(self, game_data: Dict[str, Any], game_id: int) -> None:
        """Process a single game and extract features/explanations."""
        rounds = game_data.get('rounds', [])
        
        for round_data in rounds:
            # Extract CSV features (16-feature format)
            csv_row = self._extract_csv_features(round_data, game_id)
            if csv_row:
                self.csv_rows.append(csv_row)
            
            # Extract JSONL language-rich data
            jsonl_record = self._extract_jsonl_record(round_data, game_id)
            if jsonl_record:
                self.jsonl_records.append(jsonl_record)
    
    def _extract_csv_features(self, round_data: Dict[str, Any], game_id: int) -> Optional[Dict[str, Any]]:
        """Extract 16 standardized CSV features from round data."""
        try:
            game_state = round_data.get('game_state', {})
            snake = game_state.get('snake', [])
            apple = game_state.get('apple', [0, 0])
            move = round_data.get('move')
            round_num = round_data.get('round', 0)
            
            if not snake or not move:
                return None
            
            head = snake[0]
            head_x, head_y = head[0], head[1]
            apple_x, apple_y = apple[0], apple[1]
            snake_length = len(snake)
            
            # Calculate apple direction features (binary)
            apple_dir_up = 1 if apple_y > head_y else 0
            apple_dir_down = 1 if apple_y < head_y else 0
            apple_dir_right = 1 if apple_x > head_x else 0
            apple_dir_left = 1 if apple_x < head_x else 0
            
            # Calculate danger detection features (simplified)
            # This would ideally be more sophisticated based on the actual game logic
            danger_straight = 1 if self._would_collide_straight(head, snake[1:]) else 0
            danger_left = 1 if self._would_collide_left(head, snake[1:]) else 0
            danger_right = 1 if self._would_collide_right(head, snake[1:]) else 0
            
            # Calculate free space features (simplified count)
            free_space_up = self._count_free_space_direction(head, snake, (0, 1))
            free_space_down = self._count_free_space_direction(head, snake, (0, -1))
            free_space_left = self._count_free_space_direction(head, snake, (-1, 0))
            free_space_right = self._count_free_space_direction(head, snake, (1, 0))
            
            return {
                # Metadata
                'game_id': game_id,
                'step_in_game': round_num,
                
                # Position features
                'head_x': head_x,
                'head_y': head_y,
                'apple_x': apple_x,
                'apple_y': apple_y,
                
                # Game state
                'snake_length': snake_length,
                
                # Apple direction features
                'apple_dir_up': apple_dir_up,
                'apple_dir_down': apple_dir_down,
                'apple_dir_left': apple_dir_left,
                'apple_dir_right': apple_dir_right,
                
                # Danger features
                'danger_straight': danger_straight,
                'danger_left': danger_left,
                'danger_right': danger_right,
                
                # Free space features
                'free_space_up': free_space_up,
                'free_space_down': free_space_down,
                'free_space_left': free_space_left,
                'free_space_right': free_space_right,
                
                # Target
                'target_move': move
            }
            
        except Exception as e:
            print(f"[DatasetGenerator] Error extracting CSV features: {e}")
            return None
    
    def _extract_jsonl_record(self, round_data: Dict[str, Any], game_id: int) -> Optional[Dict[str, Any]]:
        """Extract JSONL record with natural language explanations."""
        try:
            game_state = round_data.get('game_state', {})
            move = round_data.get('move')
            round_num = round_data.get('round', 0)
            reasoning = round_data.get('reasoning', '')
            
            if not move:
                return None
            
            # Generate prompt describing the game state
            prompt = self._generate_state_prompt(game_state)
            
            # Generate completion with algorithm reasoning
            completion = self._generate_reasoning_completion(move, reasoning, game_state)
            
            record = {
                'prompt': prompt,
                'completion': completion,
                'game_id': game_id,
                'step_in_game': round_num,
                'algorithm': self.algorithm,
                'move': move
            }
            
            # Add metadata if available
            if reasoning:
                record['metadata'] = {'original_reasoning': reasoning}
            
            return record
            
        except Exception as e:
            print(f"[DatasetGenerator] Error extracting JSONL record: {e}")
            return None
    
    def _generate_state_prompt(self, game_state: Dict[str, Any]) -> str:
        """Generate natural language description of game state."""
        snake = game_state.get('snake', [])
        apple = game_state.get('apple', [0, 0])
        score = game_state.get('score', 0)
        
        if not snake:
            return "Game state unavailable."
        
        head = snake[0]
        
        prompt = f"""You are playing Snake on a {self.grid_size}x{self.grid_size} grid.

Current game state:
- Snake head position: ({head[0]}, {head[1]})
- Apple position: ({apple[0]}, {apple[1]})
- Snake length: {len(snake)}
- Current score: {score}

The snake body occupies positions: {snake[1:] if len(snake) > 1 else 'None (just head)'}

What move should the snake make next? Consider:
1. Moving toward the apple
2. Avoiding collisions with walls and snake body
3. Maintaining a safe path for future moves

Choose from: UP, DOWN, LEFT, RIGHT"""
        
        return prompt
    
    def _generate_reasoning_completion(self, move: str, reasoning: str, game_state: Dict[str, Any]) -> str:
        """Generate natural language explanation for the chosen move."""
        snake = game_state.get('snake', [])
        apple = game_state.get('apple', [0, 0])
        
        if not snake:
            return f"I choose {move}."
        
        head = snake[0]
        
        # Start with the algorithm's reasoning if available
        if reasoning:
            completion = f"Algorithm reasoning: {reasoning}\n\n"
        else:
            completion = ""
        
        # Add strategic explanation
        completion += f"I choose to move {move}. "
        
        # Explain the strategic reasoning
        if move == "UP":
            if apple[1] > head[1]:
                completion += "This moves toward the apple which is above the snake head. "
            else:
                completion += "This moves away from potential dangers below. "
        elif move == "DOWN":
            if apple[1] < head[1]:
                completion += "This moves toward the apple which is below the snake head. "
            else:
                completion += "This moves away from potential dangers above. "
        elif move == "LEFT":
            if apple[0] < head[0]:
                completion += "This moves toward the apple which is to the left of the snake head. "
            else:
                completion += "This moves away from potential dangers on the right. "
        elif move == "RIGHT":
            if apple[0] > head[0]:
                completion += "This moves toward the apple which is to the right of the snake head. "
            else:
                completion += "This moves away from potential dangers on the left. "
        
        completion += f"This {self.algorithm} algorithm ensures safe pathfinding while maximizing score."
        
        return completion
    
    # Simplified collision and space calculation helpers
    def _would_collide_straight(self, head: List[int], body: List[List[int]]) -> bool:
        """Simplified collision detection for straight movement."""
        # This is a placeholder - real implementation would consider current direction
        return False  # Simplified for now
    
    def _would_collide_left(self, head: List[int], body: List[List[int]]) -> bool:
        """Simplified collision detection for left turn."""
        return False  # Simplified for now
    
    def _would_collide_right(self, head: List[int], body: List[List[int]]) -> bool:
        """Simplified collision detection for right turn."""
        return False  # Simplified for now
    
    def _count_free_space_direction(self, head: List[int], snake: List[List[int]], direction: tuple) -> int:
        """Count free spaces in a given direction."""
        # Simplified implementation - count until hitting boundary or snake
        x, y = head[0], head[1]
        dx, dy = direction
        count = 0
        snake_set = set(tuple(pos) for pos in snake)
        
        while True:
            x += dx
            y += dy
            
            # Check boundaries
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                break
            
            # Check snake collision
            if (x, y) in snake_set:
                break
            
            count += 1
            
            # Limit count to prevent infinite loops
            if count > self.grid_size * 2:
                break
        
        return count
    
    def save_csv(self, filepath: str) -> None:
        """Save CSV dataset to file."""
        if not self.csv_rows:
            print("[DatasetGenerator] No CSV data to save")
            return
        
        import pandas as pd
        df = pd.DataFrame(self.csv_rows)
        
        # Ensure columns are in correct order
        df = df.reindex(columns=CSV_BASIC_COLUMNS)
        
        save_csv_dataset(df, filepath)
        print(f"[DatasetGenerator] CSV saved: {filepath} ({len(df)} rows)")
    
    def save_jsonl(self, filepath: str) -> None:
        """Save JSONL dataset to file."""
        if not self.jsonl_records:
            print("[DatasetGenerator] No JSONL data to save")
            return
        
        save_jsonl_dataset(self.jsonl_records, filepath)
        print(f"[DatasetGenerator] JSONL saved: {filepath} ({len(self.jsonl_records)} records)")

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


def generate_games_for_algorithm(algorithm: str, max_games: int, max_steps: int, grid_size: int, verbose: bool) -> List[Dict[str, Any]]:
    """
    Generate game data for a specific algorithm.
    
    This function would ideally integrate with the actual game runner,
    but for now it provides a placeholder that shows the expected structure.
    """
    print(f"[CLI] Generating {max_games} games for {algorithm}...")
    
    # Placeholder: In real implementation, this would run the actual games
    # For now, return mock data structure
    games_data = []
    
    for game_num in range(1, max_games + 1):
        # Mock game data structure
        game_data = {
            'game_id': game_num,
            'algorithm': algorithm,
            'grid_size': grid_size,
            'final_score': 5,  # Mock score
            'total_rounds': 25,  # Mock rounds
            'rounds': []
        }
        
        # Mock some rounds
        for round_num in range(1, min(26, max_steps + 1)):
            round_data = {
                'round': round_num,
                'game_state': {
                    'snake': [[5, 5], [5, 4], [5, 3]],  # Mock snake
                    'apple': [7, 8],  # Mock apple
                    'score': round_num // 5,  # Mock score
                },
                'move': ['UP', 'DOWN', 'LEFT', 'RIGHT'][round_num % 4],
                'reasoning': f"{algorithm} pathfinding from ({5}, {5}) to ({7}, {8})"
            }
            game_data['rounds'].append(round_data)
        
        games_data.append(game_data)
        
        if verbose:
            print(f"[CLI] Generated game {game_num}/{max_games}")
    
    return games_data


def main() -> None:
    """Main entry point for dataset generation CLI."""
    setup_extension_paths()
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if args.verbose:
        print("Dataset Generation CLI v0.04")
        print("=" * 40)
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
            # Generate games data
            games_data = generate_games_for_algorithm(
                algorithm, args.max_games, args.max_steps, args.grid_size, args.verbose
            )
            
            # Create dataset generator
            generator = DatasetGenerator(algorithm, args.grid_size)
            generator.generate_from_games(games_data)
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                output_dir = Path(EXTENSIONS_LOGS_DIR) / f"datasets" / f"grid-size-{args.grid_size}" / f"heuristics-{algorithm.lower()}_{timestamp}"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save datasets
            if args.format in ["csv", "both"]:
                csv_path = output_dir / f"{algorithm.lower()}_dataset.csv"
                generator.save_csv(str(csv_path))
            
            if args.format in ["jsonl", "both"]:
                jsonl_path = output_dir / f"{algorithm.lower()}_dataset.jsonl"
                generator.save_jsonl(str(jsonl_path))
            
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