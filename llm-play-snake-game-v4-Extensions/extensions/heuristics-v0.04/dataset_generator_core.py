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
from typing import List, Dict, Any, Optional, Tuple
import csv
import json
import re
import os
import sys
from datetime import datetime

# Add project root to path to allow absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.game_constants import DIRECTIONS

from extensions.common.config.dataset_formats import CSV_BASIC_COLUMNS
from utils.print_utils import print_info, print_warning, print_success, print_error
from jsonl_utils import flatten_explanation_for_jsonl  # NEW IMPORT

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
        Initialize the dataset generator.
        
        Args:
            algorithm: The algorithm name (e.g., 'bfs', 'dfs')
            output_dir: Output directory for datasets
        """
        self.algorithm = algorithm
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV headers for the standard features including tail_path_length
        self.csv_headers = [
            'game_id', 'step_in_game', 'head_x', 'head_y', 'apple_x', 'apple_y', 'snake_length',
            'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',
            'danger_straight', 'danger_left', 'danger_right',
            'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right',
            'tail_path_length', 'target_move'
        ]
        
        # File handles
        self._csv_writer = None
        self._jsonl_fh = None
        
        print_info(f"Initialized for {algorithm} (output: {output_dir})", "DatasetGenerator")

    # ---------------- CSV
    def _open_csv(self):
        """Open CSV file for writing."""
        csv_path = self.output_dir / f"{self.algorithm}_dataset.csv"
        fh = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=self.csv_headers)
        writer.writeheader()
        self._csv_writer = (writer, fh)
        print_info(f"Opened CSV file: {csv_path}", "DatasetGenerator")

    # ---------------- JSONL
    def _open_jsonl(self):
        """Open JSONL file for writing."""
        jsonl_path = self.output_dir / f"{self.algorithm}_dataset.jsonl"
        self._jsonl_fh = jsonl_path.open("w", encoding="utf-8")
        print_info(f"Opened JSONL file: {jsonl_path}", "DatasetGenerator")

    # ---------------- PUBLIC
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

    # ---------------- INTERNAL
    def _process_single_game(self, game_data: Dict[str, Any]) -> None:
        """
        Process a single game and extract features for CSV/JSONL.
        For heuristics-v0.04, always require and use dataset_game_states. Fail fast if missing or incomplete.
        """
        # Ensure file handles are opened on first use
        if self._csv_writer is None:
            self._open_csv()
        if self._jsonl_fh is None:
            self._open_jsonl()
            
        detailed_history = game_data.get('detailed_history', {})
        moves_history = detailed_history.get('moves', [])
        explanations = game_data.get('move_explanations', [])
        metrics_list = game_data.get('move_metrics', [])
        grid_size = game_data.get('grid_size', 10)
        game_number = game_data.get('game_number', 1)  # Get game_number from game_data

        if not moves_history:
            print_warning(f"[DatasetGenerator] No moves found in game {game_number}")
            return

        # Pad explanations/metrics if needed to match moves count
        if not explanations:
            explanations = ["No explanation provided."] * len(moves_history)
        else:
            while len(explanations) < len(moves_history):
                explanations.append("No explanation provided.")
            if len(explanations) > len(moves_history):
                explanations = explanations[:len(moves_history)]
                
        if not metrics_list:
            metrics_list = [{}] * len(moves_history)
        else:
            while len(metrics_list) < len(moves_history):
                metrics_list.append({})
            if len(metrics_list) > len(moves_history):
                metrics_list = metrics_list[:len(moves_history)]

        dataset_game_states = game_data.get('dataset_game_states', {})
        if not dataset_game_states or not isinstance(dataset_game_states, dict):
            raise RuntimeError("[DatasetGenerator] dataset_game_states missing or not a dict. This is a critical error.")

        # Set game_number in all game states for SSOT compliance
        for round_key, game_state in dataset_game_states.items():
            game_state['game_number'] = game_number

        # Use only the rounds for which we have both a move and a game state
        for i in range(1, len(moves_history) + 1):
            round_key = str(i)
            if round_key not in dataset_game_states:
                raise RuntimeError(f"[DatasetGenerator] dataset_game_states missing round {round_key}. This is a critical error.")
            game_state = dataset_game_states[round_key]
            move = moves_history[i - 1]
            explanation = explanations[i - 1]
            metrics = metrics_list[i - 1]
            record = {
                "game_state": game_state,
                "move": move,
                "explanation": explanation,
                "metrics": metrics
            }
            # Write to JSONL
            if self._jsonl_fh:
                jsonl_record = self._extract_jsonl_record(record)
                self._jsonl_fh.write(json.dumps(jsonl_record) + '\n')
            # Write to CSV
            if self._csv_writer:
                csv_record = self._extract_csv_features(record, step_number=i)
                self._csv_writer[0].writerow(csv_record)

    def _extract_jsonl_record(self, record: dict) -> dict:
        """
        Extract a single JSONL record from game data.
        
        SSOT Compliance: Both prompt and metrics use the same recorded game state,
        ensuring perfect consistency and preventing coordinate mismatches.
        
        Args:
            record: Game record containing state and move information
            
        Returns:
            Dictionary with prompt and completion for JSONL format
        """
        game_state = record.get('game_state', {})
        move = record.get('move', 'UNKNOWN')
        explanation = record.get('explanation', {})
        
        # Format the prompt using the recorded game state
        prompt = self._format_prompt(game_state)
        
        # Extract metrics from explanation but override ALL position-related metrics
        # with values from the recorded game state to ensure SSOT compliance
        explanation_metrics = explanation.get("metrics", {}) if isinstance(explanation, dict) else {}
        
        # Pull out agent-computed metrics, but drop all position fields
        ssot_metrics = {
            k: v for k, v in explanation_metrics.items()
            if k not in (
                "head_position","apple_position","manhattan_distance",
                "grid_size","snake_length","valid_moves",
                "apple_path_length","path_length","tail_path_length","final_chosen_direction"
            )
        }

        # Now derive everything else from game_state alone
        if game_state:
            pre_move_head = game_state.get('head_position', [0, 0])
            apple_pos = game_state.get('apple_position', [0, 0])
            grid_size = game_state.get('grid_size', 10)
            snake_positions = game_state.get('snake_positions', [])

            # 1) valid_moves
            body_set = set(tuple(p) for p in snake_positions)
            valid_moves = [
                d for d,(dx,dy) in DIRECTIONS.items()
                if 0 <= pre_move_head[0]+dx < grid_size
                and 0 <= pre_move_head[1]+dy < grid_size
                and (pre_move_head[0]+dx, pre_move_head[1]+dy) not in body_set
            ]

            # 2) manhattan_distance
            manhattan = abs(pre_move_head[0]-apple_pos[0]) + abs(pre_move_head[1]-apple_pos[1])

            # 3) apple_path_length (real BFS)
            path = self._bfs_pathfind(pre_move_head, apple_pos, body_set - {tuple(snake_positions[-1])}, grid_size)
            apple_path_len = len(path)-1 if path else None

            # 4) tail_path_length - use agent's calculation if available, otherwise calculate
            tail_path_len = explanation_metrics.get('tail_path_length')
            if tail_path_len is None and len(snake_positions) > 1:
                tail = snake_positions[-1]
                obstacles = set(tuple(p) for p in snake_positions[:-1])
                tail_path = self._bfs_pathfind(pre_move_head, tail, obstacles, grid_size)
                tail_path_len = len(tail_path)-1 if tail_path else None

            # 5) pack SSOT metrics
            ssot_metrics.update({
                "head_position": pre_move_head,
                "apple_position": apple_pos,
                "grid_size": grid_size,
                "snake_length": len(snake_positions),
                "valid_moves": valid_moves,
                "manhattan_distance": manhattan,
                "apple_path_length": apple_path_len,
                "path_length": apple_path_len,
                "tail_path_length": tail_path_len,
                "final_chosen_direction": move,
            })
        
        # Ensure tail_path_length is always present in metrics, even if None
        if 'tail_path_length' not in ssot_metrics:
            ssot_metrics['tail_path_length'] = None
        
        # ----------------
        # Ensure the *explanation* field is a flattened, human-readable string
        # so that it never contains stale nested metrics.
        # ----------------
        explanation_text = flatten_explanation_for_jsonl(explanation)
        
        # Update explanation text to use SSOT metrics for consistency
        explanation_text = self._update_explanation_with_ssot_metrics(explanation_text, ssot_metrics)
        
        # Fix coordinate references in explanation to match prompt exactly
        if game_state:
            pre_move_head = game_state.get('head_position', [0, 0])
            apple_pos = game_state.get('apple_position', [0, 0])
            
            # Replace any coordinate references in explanation with correct ones from prompt
            # Using named groups for more robust replacements and easier debugging
            
            # Apple position patterns with named groups
            apple_patterns = [
                r'apple at \((?P<x>\d+), (?P<y>\d+)\)',
                r'apple at position \((?P<x>\d+), (?P<y>\d+)\)',
                r'apple \((?P<x>\d+), (?P<y>\d+)\)',
                r'BFS to the apple at \((?P<x>\d+), (?P<y>\d+)\)',
                r'target at \((?P<x>\d+), (?P<y>\d+)\)'
            ]
            
            # Head position patterns with named groups
            head_patterns = [
                r'from \((?P<x>\d+), (?P<y>\d+)\)',
                r'head at \((?P<x>\d+), (?P<y>\d+)\)',
                r'position \((?P<x>\d+), (?P<y>\d+)\)',
                r'BFS discovered a shortest path of length \d+ from \((?P<x>\d+), (?P<y>\d+)\)',
                r'at \((?P<x>\d+), (?P<y>\d+)\)'
            ]
            
            # Replace apple coordinate references
            for pattern in apple_patterns:
                explanation_text = re.sub(
                    pattern,
                    f"apple at ({apple_pos[0]}, {apple_pos[1]})",
                    explanation_text
                )
            
            # Replace head coordinate references
            for pattern in head_patterns:
                explanation_text = re.sub(
                    pattern,
                    f"from ({pre_move_head[0]}, {pre_move_head[1]})",
                    explanation_text
                )
        
        # Add tail_path_length to explanation if available
        if ssot_metrics.get('tail_path_length') is not None:
            explanation_text += f"\n- Tail path length (BFS to tail): {ssot_metrics['tail_path_length']}"

        # SSOT-compliant completion with consistent metrics
        completion = {
            "move": move,
            "algorithm": self.algorithm,
            "metrics": ssot_metrics,
            "explanation": explanation_text,
        }
        
        return {
            "prompt": prompt,
            "completion": json.dumps(completion, ensure_ascii=False)
        }

    def _bfs_pathfind(self, start: List[int], goal: List[int], obstacles: Set[Tuple[int, int]], grid_size: int) -> Optional[List[List[int]]]:
        """
        BFS pathfinding from start to goal, avoiding obstacles.
        
        Args:
            start: Starting position [x, y]
            goal: Goal position [x, y]
            obstacles: Set of obstacle positions as tuples (x, y)
            grid_size: Size of the game grid
            
        Returns:
            List of positions forming the path from start to goal, or None if no path exists
        """
        if start == goal:
            return [start]
        
        # Convert to tuples for set operations
        start_tuple = tuple(start)
        goal_tuple = tuple(goal)
        
        if start_tuple in obstacles or goal_tuple in obstacles:
            return None
        
        # BFS queue: (position, path)
        queue = [(start_tuple, [start])]
        visited = {start_tuple}
        
        # Direction vectors for BFS
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        
        while queue:
            current_pos, path = queue.pop(0)
            
            for dx, dy in directions:
                next_x = current_pos[0] + dx
                next_y = current_pos[1] + dy
                next_pos = (next_x, next_y)
                
                # Check bounds
                if not (0 <= next_x < grid_size and 0 <= next_y < grid_size):
                    continue
                
                # Check if visited or obstacle
                if next_pos in visited or next_pos in obstacles:
                    continue
                
                # Check if goal reached
                if next_pos == goal_tuple:
                    return path + [[next_x, next_y]]
                
                # Add to queue
                visited.add(next_pos)
                queue.append((next_pos, path + [[next_x, next_y]]))
        
        return None

    def _get_next_position(self, current_pos: List[int], move: str, grid_size: int) -> List[int]:
        """
        Calculate the next position based on current position and move.
        
        Args:
            current_pos: Current position [x, y]
            move: Move direction (UP, DOWN, LEFT, RIGHT)
            grid_size: Size of the game grid
            
        Returns:
            Next position [x, y] if within bounds, None otherwise
            
        Note: 
            Uses universal coordinate system from docs/extensions-guideline/coordinate-system.md:
            - UP: (0, 1) - Move up (increase Y)
            - DOWN: (0, -1) - Move down (decrease Y)
            - LEFT: (-1, 0) - Move left (decrease X)
            - RIGHT: (1, 0) - Move right (increase X)
        """
        x, y = current_pos
        
        if move == 'UP':
            new_pos = [x, y + 1]  # UP increases Y (bottom-left origin)
        elif move == 'DOWN':
            new_pos = [x, y - 1]  # DOWN decreases Y
        elif move == 'LEFT':
            new_pos = [x - 1, y]  # LEFT decreases X
        elif move == 'RIGHT':
            new_pos = [x + 1, y]  # RIGHT increases X
        else:
            return None
        
        # Check bounds
        if 0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size:
            return new_pos
        else:
            return None

    def _extract_csv_features(self, record: dict, step_number: int = None) -> dict:
        """
        Extract the 16 standard features for CSV record from the game state.
        
        This method implements the grid-size agnostic feature extraction
        following the CSV format specification.
        
        Args:
            record: Record containing game_state and move
            step_number: Current step number (1-based). If None, uses game_state.steps
            
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
        # Use provided step_number if available, otherwise fall back to game_state.steps
        step_in_game = step_number if step_number is not None else game_state.get('steps', 0)
        
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
        
        # Check danger in each direction using universal coordinate system
        from config.game_constants import DIRECTIONS
        directions = DIRECTIONS
        
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
        
        # Extract tail_path_length from metrics if available
        metrics = record.get('metrics', {})
        tail_path_length = metrics.get('tail_path_length', 0)
        
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
            
            # Tail path length (safety metric)
            'tail_path_length': tail_path_length,
            
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
        # Flip vertically so that top row is y = grid_size-1 (bottom-left origin)
        board_str = "\n".join(" ".join(row) for row in reversed(board))

        # Strategic analysis
        manhattan_distance = abs(head_pos[0] - apple_position[0]) + abs(head_pos[1] - apple_position[1]) if apple_position else -1
        
        # Determine valid moves using universal coordinate system
        from config.game_constants import DIRECTIONS
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
{board_str}

**Strategic Context:**
- Manhattan Distance to Apple: {manhattan_distance}
- Valid Moves: {valid_moves}

### Task:
Based on the `{algorithm}` logic, what is the optimal next move? Provide the move and a detailed, step-by-step explanation of the reasoning.
"""
        return prompt 

    def _update_explanation_with_ssot_metrics(self, explanation_text: str, ssot_metrics: dict) -> str:
        """
        Update explanation text to use SSOT metrics for consistency.
        
        This ensures that any numeric values mentioned in the explanation
        match the metrics that will be used for supervision.
        
        Args:
            explanation_text: Original explanation text
            ssot_metrics: SSOT-compliant metrics dictionary
            
        Returns:
            Updated explanation text with consistent values
        """
        updated_text = explanation_text
        
        # Update apple position references
        if 'apple_position' in ssot_metrics:
            apple_pos = ssot_metrics['apple_position']
            # Replace any "apple at (X, Y)" patterns with SSOT values
            apple_pattern = r'apple at \((\d+), (\d+)\)'
            apple_replacement = f'apple at ({apple_pos[0]}, {apple_pos[1]})'
            updated_text = re.sub(apple_pattern, apple_replacement, updated_text)
        
        # Update head position references in explanations
        if 'head_position' in ssot_metrics:
            head_pos = ssot_metrics['head_position']
            # Replace "from (X, Y)" patterns (BFS path descriptions)
            from_pattern = r'from \((\d+), (\d+)\)'
            from_replacement = f'from ({head_pos[0]}, {head_pos[1]})'
            updated_text = re.sub(from_pattern, from_replacement, updated_text)
            
            # Also replace "at (X, Y)" patterns
            at_pattern = r'at \((\d+), (\d+)\)'
            at_replacement = f'at ({head_pos[0]}, {head_pos[1]})'
            updated_text = re.sub(at_pattern, at_replacement, updated_text)
        
        # Update path length references
        if 'apple_path_length' in ssot_metrics:
            path_length = ssot_metrics['apple_path_length']
            # Replace "path length found: X" patterns
            path_pattern = r'path length found: (\d+)'
            path_replacement = f'path length found: {path_length}'
            updated_text = re.sub(path_pattern, path_replacement, updated_text)
            
            # Also replace "length X" patterns
            length_pattern = r'length (\d+)'
            length_replacement = f'length {path_length}'
            updated_text = re.sub(length_pattern, length_replacement, updated_text)
            
            # Also replace "shortest path of length X" patterns
            shortest_pattern = r'shortest path of length (\d+)'
            shortest_replacement = f'shortest path of length {path_length}'
            updated_text = re.sub(shortest_pattern, shortest_replacement, updated_text)
        
        # Update Manhattan distance references
        if 'manhattan_distance' in ssot_metrics:
            manhattan = ssot_metrics['manhattan_distance']
            # Replace "Manhattan distance: X" patterns
            manhattan_pattern = r'Manhattan distance: (\d+)'
            manhattan_replacement = f'Manhattan distance: {manhattan}'
            updated_text = re.sub(manhattan_pattern, manhattan_replacement, updated_text)
            
            # Also replace "Manhattan distance to apple: X" patterns
            manhattan_to_pattern = r'Manhattan distance to apple: (\d+)'
            manhattan_to_replacement = f'Manhattan distance to apple: {manhattan}'
            updated_text = re.sub(manhattan_to_pattern, manhattan_to_replacement, updated_text)
        
        # Update valid moves references to match prompt
        if 'valid_moves' in ssot_metrics:
            valid_moves = ssot_metrics['valid_moves']
            # Replace "valid immediate moves: [...]" patterns
            moves_pattern = r"valid immediate moves: \[[^\]]+\]"
            moves_replacement = f"valid immediate moves: {valid_moves}"
            updated_text = re.sub(moves_pattern, moves_replacement, updated_text)
            
            # Also replace "Identify valid immediate moves: [...]" patterns
            identify_pattern = r"Identify valid immediate moves: \[[^\]]+\]"
            identify_replacement = f"Identify valid immediate moves: {valid_moves}"
            updated_text = re.sub(identify_pattern, identify_replacement, updated_text)
        
        return updated_text 