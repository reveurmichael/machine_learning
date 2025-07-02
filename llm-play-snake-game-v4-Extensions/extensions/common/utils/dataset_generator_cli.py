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

# Add project root to path to allow absolute imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.game_constants import DIRECTIONS

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
                    
                # Add metadata including the log directory path
                game_data['log_path'] = str(log_path)
                game_data['log_file'] = str(game_file)
                game_data['log_directory'] = str(log_dir)  # Add this for dataset generation
                
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
    
    Evolution v0.04: Stores dataset files in the same directory as game logs
    for unified output structure following forward-looking architecture principles.
    """
    
    def __init__(self, algorithm: str, grid_size: int = DEFAULT_GRID_SIZE):
        """Initialize dataset generator for specific algorithm."""
        self.algorithm = algorithm
        self.grid_size = grid_size
        self.csv_rows = []
        self.jsonl_records = []
        self.output_dir = None
        self.shared_log_directory = None  # Store log directory for unified output
        
        print(f"[DatasetGenerator] Initialized for {algorithm} (grid size: {grid_size})")
    
    def generate_from_logs(self, log_files: List[str], formats: List[str], games_data: List[Dict[str, Any]] = None) -> None:
        """
        Generates datasets by processing a list of log files.

        This method now uses the standardized dataset directory structure:
        logs/extensions/datasets/grid-size-N/heuristics-v0.04_timestamp/algorithm/
        
        Evolution v0.04: Following the forward-looking architecture with
        algorithm-specific subdirectories and unified game logs + datasets.
        """
        from ..config.path_constants import (
            ALGORITHM_DATASET_PATH_TEMPLATE, 
            ROOT_DIR_NAME, 
            EXTENSIONS_DIR_NAME, 
            DATASETS_DIR_NAME
        )
        
        # Create standardized dataset directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension_type = "heuristics-v0.04"
        
        # Build the standardized path
        standardized_path = ALGORITHM_DATASET_PATH_TEMPLATE.format(
            root_dir=ROOT_DIR_NAME,
            extensions_dir=EXTENSIONS_DIR_NAME,
            datasets_dir=DATASETS_DIR_NAME,
            grid_size=self.grid_size,
            extension_type=extension_type,
            version="0.04",
            timestamp=timestamp,
            algorithm=self.algorithm.lower()
        )
        
        # Get project root (3 levels up from this file)
        project_root = Path(__file__).resolve().parents[3]
        algorithm_output_dir = project_root / standardized_path
        
        # Create the directory structure
        algorithm_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DatasetGenerator] Created standardized dataset directory: {algorithm_output_dir}")
        
        # Copy game logs to the new algorithm directory
        if games_data:
            print(f"[DatasetGenerator] Copying game logs to algorithm directory...")
            game_counter = 1
            for game_data in games_data:
                original_log_file = game_data.get('log_file')
                if original_log_file:
                    original_path = Path(original_log_file)
                    if original_path.exists():
                        # Copy game log with proper numbering
                        target_filename = f"game_{game_counter}.json"
                        target_file = algorithm_output_dir / target_filename
                        import shutil
                        shutil.copy2(original_path, target_file)
                        print(f"[DatasetGenerator] Copied {original_path.name} as {target_filename}")
                        game_counter += 1
            
            # Also copy summary.json if it exists (use the last game's directory for summary)
            if games_data:
                # Get summary from the last game's log directory
                last_log_dir = Path(games_data[-1].get('log_directory', ''))
                summary_file = last_log_dir / "summary.json"
                if summary_file.exists():
                    target_summary = algorithm_output_dir / "summary.json"
                    import shutil
                    shutil.copy2(summary_file, target_summary)
                    print(f"[DatasetGenerator] Copied summary.json to algorithm directory")
        
        # Setup dataset file writers in the algorithm directory
        writers = {}
        if 'jsonl' in formats:
            jsonl_path = algorithm_output_dir / f"{self.algorithm.lower()}_dataset.jsonl"
            writers['jsonl'] = open(jsonl_path, 'w', encoding='utf-8')
        if 'csv' in formats:
            csv_path = algorithm_output_dir / f"{self.algorithm.lower()}_dataset.csv"
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
        for name, writer in writers.items():
            if name == 'jsonl':
                writer.close()
            elif name == 'csv':
                # For CSV DictWriter, we need to close the underlying file object
                csv_file.close()

        # Report final locations
        if 'jsonl' in formats:
            final_jsonl_path = algorithm_output_dir / f"{self.algorithm.lower()}_dataset.jsonl"
            print(f"[DatasetGenerator] ✅ JSONL dataset saved to {final_jsonl_path}")
        if 'csv' in formats:
            final_csv_path = algorithm_output_dir / f"{self.algorithm.lower()}_dataset.csv"
            print(f"[DatasetGenerator] ✅ CSV dataset saved to {final_csv_path}")
        
        print(f"[DatasetGenerator] ✅ Standardized dataset directory: {algorithm_output_dir}")
        print(f"[DatasetGenerator] ✅ All files (game logs + datasets) now in same location!")

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
                self.jsonl_writer.write(json.dumps(jsonl_record) + '\n')
            
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
        Extracts the 16 standard features for CSV record from the game state.
        
        This method implements the grid-size agnostic feature extraction
        following the CSV format specification.
        """
        game_state = record.get('game_state', {})
        move = record.get('move', 'UNKNOWN')
        
        if not game_state:
            # Return default values for all required columns
            return {col: 0 for col in self.csv_headers if col != 'target_move'} | {'target_move': move}

        # Extract basic game state information
        snake_positions = game_state.get('snake_positions', [])
        apple_position = game_state.get('apple_position', [0, 0])
        grid_size = game_state.get('grid_size', 10)
        game_id = game_state.get('game_number', 1)
        step_in_game = game_state.get('steps', 0)
        
        if not snake_positions:
            # Invalid game state - return defaults
            return {col: 0 for col in self.csv_headers if col != 'target_move'} | {'target_move': move}
        
        head_pos = snake_positions[0]
        head_x, head_y = head_pos[0], head_pos[1]
        apple_x, apple_y = apple_position[0], apple_position[1]
        
        # Calculate apple direction features (binary)
        apple_dir_up = 1 if apple_y < head_y else 0
        apple_dir_down = 1 if apple_y > head_y else 0
        apple_dir_left = 1 if apple_x < head_x else 0
        apple_dir_right = 1 if apple_x > head_x else 0
        
        # Calculate danger detection features
        snake_body_set = set(tuple(pos) for pos in snake_positions)
        
        # Check danger in each direction
        directions = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0)
        }
        
        # Determine current direction (simplified - assume last move direction)
        current_direction = move if move in directions else 'UP'
        
        # Check danger straight ahead
        dx, dy = directions[current_direction]
        straight_pos = (head_x + dx, head_y + dy)
        danger_straight = 1 if (straight_pos in snake_body_set or 
                              straight_pos[0] < 0 or straight_pos[0] >= grid_size or
                              straight_pos[1] < 0 or straight_pos[1] >= grid_size) else 0
        
        # Calculate relative left and right based on current direction
        if current_direction == 'UP':
            left_dir, right_dir = 'LEFT', 'RIGHT'
        elif current_direction == 'DOWN':
            left_dir, right_dir = 'RIGHT', 'LEFT'
        elif current_direction == 'LEFT':
            left_dir, right_dir = 'DOWN', 'UP'
        else:  # RIGHT
            left_dir, right_dir = 'UP', 'DOWN'
        
        # Check danger left and right
        left_dx, left_dy = directions[left_dir]
        right_dx, right_dy = directions[right_dir]
        
        left_pos = (head_x + left_dx, head_y + left_dy)
        right_pos = (head_x + right_dx, head_y + right_dy)
        
        danger_left = 1 if (left_pos in snake_body_set or 
                           left_pos[0] < 0 or left_pos[0] >= grid_size or
                           left_pos[1] < 0 or left_pos[1] >= grid_size) else 0
        
        danger_right = 1 if (right_pos in snake_body_set or 
                            right_pos[0] < 0 or right_pos[0] >= grid_size or
                            right_pos[1] < 0 or right_pos[1] >= grid_size) else 0
        
        # Calculate free space features (simplified count of reachable cells)
        def count_free_space_in_direction(start_pos, direction):
            """Count free spaces in a given direction"""
            dx, dy = directions[direction]
            count = 0
            current_x, current_y = start_pos[0] + dx, start_pos[1] + dy
            
            while (0 <= current_x < grid_size and 0 <= current_y < grid_size and
                   (current_x, current_y) not in snake_body_set):
                count += 1
                current_x += dx
                current_y += dy
                # Limit count to avoid infinite loops in open areas
                if count >= grid_size:
                    break
            
            return count
        
        free_space_up = count_free_space_in_direction(head_pos, 'UP')
        free_space_down = count_free_space_in_direction(head_pos, 'DOWN')
        free_space_left = count_free_space_in_direction(head_pos, 'LEFT')
        free_space_right = count_free_space_in_direction(head_pos, 'RIGHT')
        
        # Return the complete CSV record
        return {
            # Metadata
            'game_id': game_id,
            'step_in_game': step_in_game,
            
            # Position features
            'head_x': head_x,
            'head_y': head_y,
            'apple_x': apple_x,
            'apple_y': apple_y,
            
            # Game state
            'snake_length': len(snake_positions),
            
            # Apple direction features
            'apple_dir_up': apple_dir_up,
            'apple_dir_down': apple_dir_down,
            'apple_dir_left': apple_dir_left,
            'apple_dir_right': apple_dir_right,
            
            # Danger detection features
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

    def _format_prompt(self, game_state: dict) -> str:
        """Formats the game state into a language-rich and structured prompt for fine-tuning."""
        if not game_state:
            return "Current game state is not available."

        # --- Extract Data ---
        grid_size = game_state.get('grid_size', 10)
        snake_positions = game_state.get('snake_positions', [])
        apple_position = game_state.get('apple_position', [])
        score = game_state.get('score', 0)
        steps = game_state.get('steps', 0)
        # The 'algorithm' key is added during completion extraction, so get it from the record.
        # Let's assume the game_state might have it for prompts.
        algorithm = game_state.get('algorithm', self.algorithm) 
        
        if not snake_positions:
            return "Invalid game state: Snake has no positions."

        head_pos = snake_positions[0]
        
        # --- Board Representation ---
        board = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
        if apple_position:
            board[apple_position[1]][apple_position[0]] = 'A'
        for i, pos in enumerate(snake_positions):
            board[pos[1]][pos[0]] = 'S'
        board[head_pos[1]][head_pos[0]] = 'H'
        board_str = "\\n".join(" ".join(row) for row in board)

        # --- Strategic Analysis ---
        manhattan_distance = abs(head_pos[0] - apple_position[0]) + abs(head_pos[1] - apple_position[1]) if apple_position else -1
        
        # Determine valid moves
        valid_moves = []
        for move, (dx, dy) in DIRECTIONS.items():
            next_pos = (head_pos[0] + dx, head_pos[1] + dy)
            if (0 <= next_pos[0] < grid_size and
                0 <= next_pos[1] < grid_size and
                next_pos not in snake_positions):
                valid_moves.append(move)

        # --- Structured Prompt ---
        prompt = f"""### Instruction:
You are an expert Snake game AI. Your task is to analyze the provided game state and determine the single best move from the list of valid moves. Your decision should be based on the logic of the specified heuristic algorithm.

### Input:
**Algorithm:** {algorithm}
**Game State:**
- Grid Size: {grid_size}x{grid_size}
- Score: {score}
- Steps: {steps}
- Snake Length: {len(snake_positions)}
- Head Position: {head_pos}
- Apple Position: {apple_position}

**Board:**
```
{board_str}
```

**Strategic Context:**
- Manhattan Distance to Apple: {manhattan_distance}
- Valid Moves: {valid_moves}

### Task:
Based on the `{algorithm}` logic, what is the optimal next move? Provide the move and a detailed, step-by-step explanation of the reasoning.
"""
        return prompt

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
            
            # Create dataset generator - no need for separate output directory
            # as datasets will be stored in the same directory as game logs
            generator = DatasetGenerator(algorithm, args.grid_size)
            
            # Use custom output directory only if explicitly specified
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                generator.output_dir = output_dir
                print(f"[CLI] 📁 Using custom output directory: {output_dir}")
            else:
                # Use the game log directory (unified output structure)
                log_dir = Path(games_data[0]['log_directory']) if games_data else None
                print(f"[CLI] 📁 Using shared log directory: {log_dir}")

            # Get the list of actual game log files to process
            log_files = [game['log_file'] for game in games_data]
            
            # Determine formats to generate
            formats_to_generate = []
            if args.format in ["csv", "both"]:
                formats_to_generate.append("csv")
            if args.format in ["jsonl", "both"]:
                formats_to_generate.append("jsonl")

            # Generate datasets from the logs (will use shared log directory)
            generator.generate_from_logs(log_files, formats_to_generate, games_data)
            
            print(f"[CLI] ✅ Completed dataset generation for {algorithm}")
            final_dir = generator.shared_log_directory or generator.output_dir
            print(f"[CLI] 📁 Dataset files created in: {final_dir}")
            
        except Exception as e:
            print(f"[CLI] ❌ Failed to generate dataset for {algorithm}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print("\n[CLI] ✅ Dataset generation completed!")


if __name__ == "__main__":
    main() 