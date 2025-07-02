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
        
        if rounds:
            for round_data in rounds:
                # Extract CSV features (16-feature format)
                csv_row = self._extract_csv_features(round_data, game_id)
                if csv_row:
                    self.csv_rows.append(csv_row)
                
                # Extract JSONL language-rich data
                jsonl_record = self._extract_jsonl_record(round_data, game_id)
                if jsonl_record:
                    self.jsonl_records.append(jsonl_record)
        else:
            # Fallback: use moves + move_explanations from detailed_history
            dh = game_data.get('detailed_history', {})
            moves = dh.get('moves', [])
            explanations = dh.get('move_explanations', [])
            apple_positions = dh.get('apple_positions', [])
            for idx, move in enumerate(moves):
                explanation = explanations[idx] if idx < len(explanations) else ""
                apple_pos = apple_positions[idx//len(apple_positions)] if apple_positions else [0,0]
                round_data = {
                    'round': idx+1,
                    'move': move,
                    'game_state': {
                        'snake': [],
                        'apple': apple_pos,
                        'score': game_data.get('score',0)
                    },
                    'algorithm_decision': {'reasoning': explanation}
                }
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
            
            # Calculate danger detection features (based on actual game logic)
            danger_straight = self._detect_danger_direction(head, snake[1:], self._get_current_direction(round_data))
            danger_left = self._detect_danger_direction(head, snake[1:], self._turn_left(self._get_current_direction(round_data)))
            danger_right = self._detect_danger_direction(head, snake[1:], self._turn_right(self._get_current_direction(round_data)))
            
            # Calculate free space features
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
            
            # Get algorithm reasoning if available
            reasoning = round_data.get('algorithm_decision', {}).get('reasoning', '')
            path_found = round_data.get('algorithm_decision', {}).get('path_found', True)
            
            if not move:
                return None
            
            # Generate prompt describing the game state
            prompt = self._generate_state_prompt(game_state, round_num)
            
            # Generate completion with algorithm reasoning
            completion = self._generate_reasoning_completion(move, reasoning, game_state, path_found)
            
            record = {
                'prompt': prompt,
                'completion': completion,
                'game_id': game_id,
                'step_in_game': round_num,
                'algorithm': self.algorithm,
                'move': move
            }
            
            # Add metadata if available
            metadata = {}
            if reasoning:
                metadata['original_reasoning'] = reasoning
            if 'algorithm_decision' in round_data:
                metadata['algorithm_decision'] = round_data['algorithm_decision']
            
            if metadata:
                record['metadata'] = metadata
            
            return record
            
        except Exception as e:
            print(f"[DatasetGenerator] Error extracting JSONL record: {e}")
            return None
    
    def _generate_state_prompt(self, game_state: Dict[str, Any], round_num: int) -> str:
        """Generate natural language description of game state."""
        snake = game_state.get('snake', [])
        apple = game_state.get('apple', [0, 0])
        score = game_state.get('score', 0)
        
        if not snake:
            return "Game state unavailable."
        
        head = snake[0]
        body = snake[1:] if len(snake) > 1 else []
        
        prompt = f"""You are an AI playing Snake on a {self.grid_size}x{self.grid_size} grid. This is step {round_num} of the game.

Current game state:
- Snake head position: ({head[0]}, {head[1]})
- Apple position: ({apple[0]}, {apple[1]})
- Snake length: {len(snake)}
- Current score: {score}"""

        if body:
            prompt += f"\n- Snake body positions: {body}"
        else:
            prompt += "\n- Snake body: None (just the head)"

        # Add strategic context
        distance_to_apple = abs(head[0] - apple[0]) + abs(head[1] - apple[1])
        prompt += f"\n- Manhattan distance to apple: {distance_to_apple}"
        
        # Add constraints
        prompt += f"""

Constraints:
- Grid boundaries: (0,0) to ({self.grid_size-1},{self.grid_size-1})
- Cannot move into walls or snake body
- Goal: Reach the apple safely while planning future moves

What move should you choose next? Analyze the situation and select from: UP, DOWN, LEFT, RIGHT"""
        
        return prompt
    
    def _generate_reasoning_completion(self, move: str, reasoning: str, game_state: Dict[str, Any], path_found: bool = True) -> str:
        """Generate natural language explanation for the chosen move."""
        snake = game_state.get('snake', [])
        apple = game_state.get('apple', [0, 0])
        score = game_state.get('score', 0)
        
        if not snake:
            return f"I choose {move}."
        
        head = snake[0]
        
        # Start with the algorithm's reasoning if available
        completion = ""
        
        if reasoning:
            completion += f"Algorithm analysis: {reasoning}\n\n"
        elif not path_found:
            completion += f"Algorithm analysis: No direct path found to apple. Making safety move.\n\n"
        else:
            completion += f"Algorithm analysis: {self.algorithm} pathfinding from ({head[0]}, {head[1]}) to ({apple[0]}, {apple[1]})\n\n"
        
        # Add strategic explanation
        completion += f"Decision: I choose to move {move}. "
        
        # Explain the strategic reasoning based on the actual game state
        apple_direction = self._get_apple_direction(head, apple)
        move_towards_apple = move in apple_direction
        
        if move_towards_apple:
            completion += f"This move brings me closer to the apple at ({apple[0]}, {apple[1]}). "
        else:
            completion += f"Although this doesn't directly approach the apple, it's necessary for safe pathfinding. "
        
        # Add safety considerations
        if not path_found:
            completion += "The algorithm detected potential risks with direct apple approaches, so this move prioritizes safety. "
        
        # Add algorithm-specific context
        if self.algorithm == "BFS":
            completion += "The BFS algorithm ensures optimal pathfinding by exploring all possible safe routes systematically."
        elif self.algorithm == "ASTAR":
            completion += "The A* algorithm uses heuristic guidance to find efficient paths while avoiding obstacles."
        elif self.algorithm == "DFS":
            completion += "The DFS algorithm explores deep paths to find creative solutions to complex board states."
        elif "HAMILTONIAN" in self.algorithm:
            completion += "The Hamiltonian approach seeks to create cycles that cover the entire board systematically."
        else:
            completion += f"The {self.algorithm} algorithm balances efficiency and safety in pathfinding decisions."
        
        return completion
    
    # Helper methods for better feature extraction
    def _get_current_direction(self, round_data: Dict[str, Any]) -> str:
        """Get current movement direction from round data."""
        move = round_data.get('move', 'UP')
        return move
    
    def _turn_left(self, direction: str) -> str:
        """Get the left turn direction."""
        turns = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
        return turns.get(direction, "UP")
    
    def _turn_right(self, direction: str) -> str:
        """Get the right turn direction."""
        turns = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}
        return turns.get(direction, "UP")
    
    def _detect_danger_direction(self, head: List[int], body: List[List[int]], direction: str) -> int:
        """Detect if moving in a direction would cause collision."""
        direction_map = {
            "UP": (0, 1),
            "DOWN": (0, -1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }
        
        dx, dy = direction_map.get(direction, (0, 0))
        new_x, new_y = head[0] + dx, head[1] + dy
        
        # Check wall collision
        if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
            return 1
        
        # Check body collision
        if [new_x, new_y] in body:
            return 1
        
        return 0
    
    def _count_free_space_direction(self, head: List[int], snake: List[List[int]], direction: tuple) -> int:
        """Count free spaces in a given direction."""
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
    
    def _get_apple_direction(self, head: List[int], apple: List[int]) -> List[str]:
        """Get list of directions that lead toward the apple."""
        directions = []
        
        if apple[0] > head[0]:  # Apple is to the right
            directions.append("RIGHT")
        elif apple[0] < head[0]:  # Apple is to the left
            directions.append("LEFT")
        
        if apple[1] > head[1]:  # Apple is above
            directions.append("UP")
        elif apple[1] < head[1]:  # Apple is below
            directions.append("DOWN")
        
        return directions
    
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