"""
Core dataset generator – convert raw logs into CSV / JSONL.

This module provides the core DatasetGenerator class that converts
heuristic game logs into structured datasets for machine learning.

Design Philosophy:
- Algorithm-agnostic: Can be reused by supervised/RL extensions
- Single responsibility: Only handles dataset conversion
- Simple logging: Uses print() statements for all operations
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import csv
import json

# Add project root to path to allow absolute imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.game_constants import DIRECTIONS

from extensions.common.config.dataset_formats import CSV_BASIC_COLUMNS
from utils.print_utils import print_info, print_warning, print_success

__all__ = ["DatasetGenerator"]


class DatasetGenerator:
    """
    Convert raw heuristic game logs to datasets (CSV / JSONL).
    Designed to be algorithm-agnostic so supervised / RL can reuse it.
    
    This generator reads heuristic algorithm game logs and converts them
    into structured datasets suitable for machine learning tasks.
    """

    def __init__(self, algorithm: str, output_dir: Path):
        """
        Initialize dataset generator for specific algorithm.
        
        Args:
            algorithm: Algorithm name (BFS, ASTAR, etc.)
            output_dir: Directory to save generated datasets
        """
        self.algorithm = algorithm.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._csv_writer = None
        self._jsonl_fh = None
        self.csv_headers = CSV_BASIC_COLUMNS
        
        print_info(f"Initialized for {algorithm} (output: {output_dir})", "DatasetGenerator")

    # ------------------------------------------------------------------ CSV
    def _open_csv(self):
        """Open CSV file for writing."""
        csv_path = self.output_dir / f"{self.algorithm}_dataset.csv"
        fh = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=self.csv_headers)
        writer.writeheader()
        self._csv_writer = (writer, fh)
        print_info(f"Opened CSV file: {csv_path}", "DatasetGenerator")

    # -------------------------------------------------------------- JSONL
    def _open_jsonl(self):
        """Open JSONL file for writing."""
        jsonl_path = self.output_dir / f"{self.algorithm}_dataset.jsonl"
        self._jsonl_fh = jsonl_path.open("w", encoding="utf-8")
        print_info(f"Opened JSONL file: {jsonl_path}", "DatasetGenerator")

    # ------------------------------------------------------------ PUBLIC
    def generate(self, games: List[Dict[str, Any]], formats: List[str] = ["csv", "jsonl"]):
        """
        Generate datasets from game data.
        
        Args:
            games: List of game data dictionaries
            formats: List of formats to generate ("csv", "jsonl", or both)
        """
        print_info(f"Processing {len(games)} games...", "DatasetGenerator")
        
        # Open output files
        if "csv" in formats:
            self._open_csv()
        if "jsonl" in formats:
            self._open_jsonl()

        # Process each game
        for game in games:
            self._process_single_game(game)

        # Close handles
        if self._csv_writer:
            self._csv_writer[1].close()
            print_success("CSV dataset saved")
        if self._jsonl_fh:
            self._jsonl_fh.close()
            print_success("JSONL dataset saved")

    # ---------------------------------------------------------- INTERNAL
    def _process_single_game(self, game_data: Dict[str, Any]) -> None:
        """
        Process a single game and extract features for CSV/JSONL.
        
        Args:
            game_data: Game data dictionary from load_game_logs
        """
        rounds_data_dict = game_data.get('detailed_history', {}).get('rounds_data', {})
        
        if not rounds_data_dict:
            print_warning("No rounds_data found in game. Skipping.")
            return

        moves_history = game_data.get('detailed_history', {}).get('moves', [])
        explanations = game_data.get('detailed_history', {}).get('move_explanations', [])
        metrics_list = game_data.get('detailed_history', {}).get('move_metrics', [])
        
        # Pad explanations if needed
        while len(explanations) < len(moves_history):
            explanations.append("No explanation provided.")

        # Pad metrics list if needed
        while len(metrics_list) < len(moves_history):
            metrics_list.append({})

        print_info(f"Processing game with {len(moves_history)} moves...", "DatasetGenerator")

        for i, move in enumerate(moves_history):
            round_number_str = str(i + 1)
            round_data = rounds_data_dict.get(round_number_str, {})
            
            game_state = round_data.get('game_state')
            if not game_state:
                print_warning(f"No game_state found for round {round_number_str}. Skipping step.")
                continue

            record = {
                "game_state": game_state,
                "move": move,
                "explanation": explanations[i],
                "metrics": metrics_list[i]
            }

            # Write to JSONL
            if self._jsonl_fh:
                jsonl_record = self._extract_jsonl_record(record)
                self._jsonl_fh.write(json.dumps(jsonl_record) + '\n')
            
            # Write to CSV
            if self._csv_writer:
                csv_record = self._extract_csv_features(record)
                self._csv_writer[0].writerow(csv_record)

    def _extract_jsonl_record(self, record: dict) -> dict:
        """
        Create a JSONL record with natural language prompt and completion.
        
        Args:
            record: Record containing game_state, move, and explanation
            
        Returns:
            JSONL record with prompt and completion
        """
        game_state = record['game_state']
        prompt = self._format_prompt(game_state)
        
        explanation = record.get('explanation', "")
        algorithm = game_state.get('algorithm', self.algorithm)

        completion = {
            "move": record['move'],
            "algorithm": algorithm,
            "metrics": record.get('metrics', {}),
            "explanation": explanation
        }

        return {"prompt": prompt, "completion": json.dumps(completion, ensure_ascii=False)}

    def _extract_csv_features(self, record: dict) -> dict:
        """
        Extract the 16 standard features for CSV record from the game state.
        
        This method implements the grid-size agnostic feature extraction
        following the CSV format specification.
        
        Args:
            record: Record containing game_state and move
            
        Returns:
            Dictionary with CSV features
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
        """
        Format the game state into a language-rich and structured prompt for fine-tuning.
        
        Args:
            game_state: Game state dictionary
            
        Returns:
            Formatted prompt string
        """
        if not game_state:
            return "Current game state is not available."

        # Extract data
        grid_size = game_state.get('grid_size', 10)
        snake_positions = game_state.get('snake_positions', [])
        apple_position = game_state.get('apple_position', [])
        score = game_state.get('score', 0)
        steps = game_state.get('steps', 0)
        algorithm = game_state.get('algorithm', self.algorithm) 
        
        if not snake_positions:
            return "Invalid game state: Snake has no positions."

        head_pos = snake_positions[0]
        
        # Board representation
        board = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
        if apple_position:
            board[apple_position[1]][apple_position[0]] = 'A'
        for i, pos in enumerate(snake_positions):
            board[pos[1]][pos[0]] = 'S'
        board[head_pos[1]][head_pos[0]] = 'H'
        board_str = "\n".join(" ".join(row) for row in board)

        # Strategic analysis
        manhattan_distance = abs(head_pos[0] - apple_position[0]) + abs(head_pos[1] - apple_position[1]) if apple_position else -1
        
        # Determine valid moves
        valid_moves = []
        for move, (dx, dy) in DIRECTIONS.items():
            next_pos = (head_pos[0] + dx, head_pos[1] + dy)
            if (0 <= next_pos[0] < grid_size and
                0 <= next_pos[1] < grid_size and
                next_pos not in snake_positions):
                valid_moves.append(move)

        # Structured prompt
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