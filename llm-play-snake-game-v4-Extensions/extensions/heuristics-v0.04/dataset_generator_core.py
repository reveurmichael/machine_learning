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
# SSOT: Import shared logic from ssot_utils - DO NOT reimplement these functions
from ssot_utils import ssot_bfs_pathfind, ssot_calculate_valid_moves
from agents.agent_bfs import BFSAgent

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
        
        self.csv_headers = [
            'game_id', 'step_in_game', 'head_x', 'head_y', 'apple_x', 'apple_y', 'snake_length',
            'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',
            'danger_straight', 'danger_left', 'danger_right',
            'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right',
            'target_move'
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

        # SSOT Fix: All indexing and format issues resolved

        # Set game_number in all game states for SSOT compliance
        for round_key, game_state in dataset_game_states.items():
            game_state['game_number'] = game_number

        # Use only the rounds for which we have both a move and a game state
        # SSOT Fix: Handle round numbering correctly for multiple games
        available_rounds = sorted(dataset_game_states.keys())
        if not available_rounds:
            raise RuntimeError("[DatasetGenerator] No dataset_game_states available. This is a critical error.")
        
        # Find the minimum round number to handle cases where rounds don't start from 2
        # Convert string keys to integers for proper arithmetic
        try:
            min_round = min(int(round_key) for round_key in available_rounds)
        except (ValueError, TypeError):
            # Fallback: use the first available round as string
            min_round = available_rounds[0]
        
        # Process moves and game states together
        for i in range(len(moves_history)):
            # Calculate the expected round key based on the minimum round
            if isinstance(min_round, int):
                round_key = str(min_round + i)  # Convert back to string for dict lookup
            else:
                # Fallback: use string arithmetic or skip
                round_key = str(int(min_round) + i) if min_round.isdigit() else min_round
            if round_key not in dataset_game_states:
                print(f"[WARNING] Skipping move {i}: dataset_game_states missing round {round_key}.")
                continue
            game_state = dataset_game_states[round_key]
            move = moves_history[i]
            explanation = explanations[i]
            metrics = metrics_list[i]

            # Validate head position before processing
            pre_move_head = game_state.get('head_position', [0, 0])
            if pre_move_head is None or not isinstance(pre_move_head, (list, tuple)) or len(pre_move_head) != 2:
                print(f"[WARNING] Skipping move {i}: Invalid head position in game state for round {round_key}: {pre_move_head}")
                continue

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
                csv_record = self._extract_csv_features(record, step_number=i+1)  # CSV step numbers are 1-indexed
                self._csv_writer[0].writerow(csv_record)

    def _calculate_valid_moves(self, head_pos: List[int], snake_positions: List[List[int]], grid_size: int) -> List[str]:
        """
        SSOT: Calculate valid moves using ssot_calculate_valid_moves from ssot_utils.
        """
        return ssot_calculate_valid_moves(head_pos, snake_positions, grid_size)

    def _extract_jsonl_record(self, record: dict) -> dict:
        """
        Extract a single JSONL record from game data.
        SSOT Compliance: Use the agent's actual move and explanation as the source of truth.
        Args:
            record: Game record containing state and move information
        Returns:
            Dictionary with prompt and completion for JSONL format
        """
        import copy
        from config.game_constants import DIRECTIONS
        
        game_state = record.get('game_state', {})
        move = record.get('move', 'UNKNOWN')
        explanation = record.get('explanation', {})
        
        # Format the prompt using the recorded PRE-MOVE game state
        prompt = self._format_prompt(game_state)
        
        # Extract pre-move state
        pre_move_head = game_state.get('head_position', [0, 0])
        raw_apple_pos = game_state.get('apple_position', [0, 0])
        if isinstance(raw_apple_pos, dict):
            apple_pos = [raw_apple_pos.get('x', 0), raw_apple_pos.get('y', 0)]
        else:
            apple_pos = raw_apple_pos
        grid_size = game_state.get('grid_size', 10)
        snake_positions = game_state.get('snake_positions', [])
        score = game_state.get('score', 0)
        steps = game_state.get('steps', 0)

        # Validate head position
        if pre_move_head is None or not isinstance(pre_move_head, (list, tuple)) or len(pre_move_head) != 2:
            print(f"[WARNING] Invalid head position: {pre_move_head}")
            pre_move_head = [0, 0]

        # Validate apple position
        if apple_pos is None or not isinstance(apple_pos, (list, tuple)) or len(apple_pos) != 2:
            print(f"[WARNING] Invalid apple position: {apple_pos}")
            apple_pos = [0, 0]

        # SSOT: Use pre_move_head as the definitive source of truth for pre-move head position
        # This ensures consistency between prompt and all pre-move calculations
        head = tuple(pre_move_head)  # Use the game state's head position as SSOT
        apple = tuple(apple_pos)
        # Obstacles: all body except head (use snake_positions but exclude the head position)
        obstacles = set(tuple(p) for p in snake_positions if len(p) >= 2 and tuple(p) != head)

        # Compute valid moves from pre-move state (exclude head position)
        valid_moves = self._calculate_valid_moves(head, snake_positions, grid_size)

        # Enforce that the chosen move is in the valid moves list
        if move not in valid_moves:
            print(f"[ERROR] Chosen move '{move}' is not in valid moves {valid_moves} for head {head} and snake {snake_positions}. Skipping entry.")
            return None

        # Compute manhattan distance from head to apple
        manhattan = abs(head[0] - apple[0]) + abs(head[1] - apple[1])

        # Use agent's actual metrics if available, otherwise compute them
        agent_metrics = {}
        agent_chosen_direction = move  # Default to game history move
        if isinstance(explanation, dict):
            agent_metrics = explanation.get('metrics', {})
            # Use agent's actual chosen direction if available
            if 'final_chosen_direction' in agent_metrics:
                agent_chosen_direction = agent_metrics['final_chosen_direction']
        
        # Always compute apple_path_length from SSOT state
        path = ssot_bfs_pathfind(list(head), list(apple), obstacles, grid_size)
        apple_path_len = len(path) - 1 if path else None
        # Sanity check: path length should never be less than Manhattan distance
        if apple_path_len is not None and apple_path_len < manhattan:
            print(f"[BUG] apple_path_length ({apple_path_len}) < manhattan ({manhattan}) from {head} to {apple} -- setting to None!")
            apple_path_len = None

        next_head_prediction = list(head)
        if agent_chosen_direction in DIRECTIONS:
            dx, dy = DIRECTIONS[agent_chosen_direction]
            next_head_prediction = [head[0] + dx, head[1] + dy]
        

        # Compute remaining free cells
        remaining_free_cells = grid_size * grid_size - len(snake_positions)

        # Compute post-move head position for completion metrics
        post_move_head = list(head)
        if agent_chosen_direction in DIRECTIONS:
            dx, dy = DIRECTIONS[agent_chosen_direction]
            post_move_head = [head[0] + dx, head[1] + dy]

        # Compute post-move snake positions (simulate move)
        post_move_snake = snake_positions.copy()
        if post_move_head not in post_move_snake:
            post_move_snake.append(post_move_head)
            post_move_snake = post_move_snake[1:]  # Remove tail (normal move)
        else:
            # If move collides, just append for metrics (should be rare)
            post_move_snake.append(post_move_head)

        # Obstacles for post-move state: all body except head
        post_move_obstacles = set(tuple(p) for p in post_move_snake[:-1] if len(p) >= 2)

        # Compute valid moves from post-move state
        post_move_valid_moves = self._calculate_valid_moves(post_move_head, post_move_snake, grid_size)

        # Compute manhattan distance from post-move head to apple
        post_move_manhattan = abs(post_move_head[0] - apple[0]) + abs(post_move_head[1] - apple[1])

        # Compute apple_path_length from post-move state
        post_move_apple_path_len = None
        path = ssot_bfs_pathfind(list(post_move_head), list(apple), post_move_obstacles, grid_size)
        post_move_apple_path_len = len(path) - 1 if path else None
        if post_move_apple_path_len is not None and post_move_apple_path_len < post_move_manhattan:
            print(f"[BUG] post-move apple_path_length ({post_move_apple_path_len}) < manhattan ({post_move_manhattan}) from {post_move_head} to {apple} -- setting to None!")
            post_move_apple_path_len = None

        # Compute remaining free cells after move
        post_move_remaining_free_cells = grid_size * grid_size - len(post_move_snake)

        # Robust check: ensure all required metrics are present and valid before writing entry
        required_metrics = [
            post_move_head, post_move_valid_moves, post_move_manhattan, post_move_apple_path_len,
        ]
        if (
            post_move_head is None or
            post_move_valid_moves is None or not isinstance(post_move_valid_moves, list) or len(post_move_valid_moves) == 0 or
            post_move_manhattan is None or
            post_move_apple_path_len is None or
            post_move_remaining_free_cells is None or
            agent_chosen_direction is None
        ):
            print(f"[ERROR] Skipping entry due to missing or invalid metrics: head={post_move_head}, valid_moves={post_move_valid_moves}, manhattan={post_move_manhattan}, apple_path_length={post_move_apple_path_len}, remaining_free_cells={post_move_remaining_free_cells}, move={agent_chosen_direction}")
            return None

        # Compose SSOT metrics, all from post-move state
        ssot_metrics = {
            "head_position": post_move_head,  # POST-MOVE head position
            "apple_position": apple,
            "grid_size": grid_size,
            "snake_length": len(post_move_snake),
            "valid_moves": post_move_valid_moves,
            "manhattan_distance": post_move_manhattan,
            "apple_path_length": post_move_apple_path_len,
            "remaining_free_cells": post_move_remaining_free_cells,
            "final_chosen_direction": agent_chosen_direction,
        }
        
        # Add any extra agent-computed metrics (e.g. apple_path_safe, fallback_used)
        for k in ["apple_path_safe", "fallback_used"]:
            if k in agent_metrics:
                ssot_metrics[k] = agent_metrics[k]

        # Use agent's explanation if available, otherwise flatten
        if isinstance(explanation, dict):
            explanation_text = flatten_explanation_for_jsonl(explanation)
        else:
            explanation_text = str(explanation)
        
        explanation_text = self._update_explanation_with_ssot_metrics(explanation_text, ssot_metrics)
        
        return {
            "prompt": prompt,
            "completion": json.dumps({
                "move": agent_chosen_direction,  # Use agent's actual chosen direction
                "algorithm": self.algorithm,
                "metrics": ssot_metrics,
                "explanation": explanation_text
            }, ensure_ascii=False)
        }

    # SSOT: BFS pathfinding is implemented in ssot_utils.py
    # Do not reimplement here - use ssot_bfs_pathfind from ssot_utils

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
        
        head_pos = snake_positions[-1]  # Head is at index -1 (last element)
        head_x, head_y = head_pos[0], head_pos[1]
        
        # Handle both dict and list formats for apple_position
        if isinstance(apple_position, dict):
            apple_x, apple_y = apple_position.get('x', 0), apple_position.get('y', 0)
        else:
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

        head_pos = snake_positions[-1]  # Head is at index -1 (last element)
        
        # Board representation
        board = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
        if apple_position and len(apple_position) >= 2:
            try:
                apple_x, apple_y = apple_position[0], apple_position[1]
                if 0 <= apple_x < grid_size and 0 <= apple_y < grid_size:
                    board[apple_y][apple_x] = 'A'
            except (KeyError, TypeError, IndexError) as e:
                # Try to handle if it's a dict with x,y keys
                if isinstance(apple_position, dict) and 'x' in apple_position and 'y' in apple_position:
                    apple_x, apple_y = apple_position['x'], apple_position['y']
                    if 0 <= apple_x < grid_size and 0 <= apple_y < grid_size:
                        board[apple_y][apple_x] = 'A'
        for i, pos in enumerate(snake_positions):
            if len(pos) >= 2:
                pos_x, pos_y = pos[0], pos[1]
                if 0 <= pos_x < grid_size and 0 <= pos_y < grid_size:
                    board[pos_y][pos_x] = 'S'
        if len(head_pos) >= 2:
            head_x, head_y = head_pos[0], head_pos[1]
            if 0 <= head_x < grid_size and 0 <= head_y < grid_size:
                board[head_y][head_x] = 'H'
        # Flip vertically so that top row is y = grid_size-1 (bottom-left origin)
        board_str = "\n".join(" ".join(row) for row in reversed(board))

        # Strategic analysis  
        if apple_position:
            if isinstance(apple_position, dict):
                apple_x, apple_y = apple_position.get('x', 0), apple_position.get('y', 0)
            else:
                apple_x, apple_y = apple_position[0], apple_position[1]
            manhattan_distance = abs(head_pos[0] - apple_x) + abs(head_pos[1] - apple_y)
        else:
            manhattan_distance = -1
        
        # Determine valid moves using universal coordinate system
        from config.game_constants import DIRECTIONS
        valid_moves = self._calculate_valid_moves(head_pos, snake_positions, grid_size)

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

    def _json_serializer(self, obj):
        """Handle numpy types for JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

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
            apple_replacement = f'apple at ({str(apple_pos[0])}, {str(apple_pos[1])})'
            updated_text = re.sub(apple_pattern, apple_replacement, updated_text)
        
        # Update head position references in explanations
        if 'head_position' in ssot_metrics:
            head_pos = ssot_metrics['head_position']
            # Replace "from (X, Y)" patterns (BFS path descriptions)
            from_pattern = r'from \((\d+), (\d+)\)'
            from_replacement = f'from ({str(head_pos[0])}, {str(head_pos[1])})'
            updated_text = re.sub(from_pattern, from_replacement, updated_text)
            
            # Also replace "at (X, Y)" patterns
            at_pattern = r'at \((\d+), (\d+)\)'
            at_replacement = f'at ({str(head_pos[0])}, {str(head_pos[1])})'
            updated_text = re.sub(at_pattern, at_replacement, updated_text)
        
        # Update path length references
        if 'apple_path_length' in ssot_metrics:
            path_length = ssot_metrics['apple_path_length']
            # Replace "path length found: X" patterns
            path_pattern = r'path length found: (\d+)'
            path_replacement = f'path length found: {str(path_length)}'
            updated_text = re.sub(path_pattern, path_replacement, updated_text)
            
            # Also replace "length X" patterns
            length_pattern = r'length (\d+)'
            length_replacement = f'length {str(path_length)}'
            updated_text = re.sub(length_pattern, length_replacement, updated_text)
            
            # Also replace "shortest path of length X" patterns
            shortest_pattern = r'shortest path of length (\d+)'
            shortest_replacement = f'shortest path of length {str(path_length)}'
            updated_text = re.sub(shortest_pattern, shortest_replacement, updated_text)
        
        # Update Manhattan distance references
        if 'manhattan_distance' in ssot_metrics:
            manhattan = ssot_metrics['manhattan_distance']
            # Replace "Manhattan distance: X" patterns
            manhattan_pattern = r'Manhattan distance: (\d+)'
            manhattan_replacement = f'Manhattan distance: {str(manhattan)}'
            updated_text = re.sub(manhattan_pattern, manhattan_replacement, updated_text)
            
            # Also replace "Manhattan distance to apple: X" patterns
            manhattan_to_pattern = r'Manhattan distance to apple: (\d+)'
            manhattan_to_replacement = f'Manhattan distance to apple: {str(manhattan)}'
            updated_text = re.sub(manhattan_to_pattern, manhattan_to_replacement, updated_text)
        
        # Update valid moves references to match prompt
        if 'valid_moves' in ssot_metrics:
            valid_moves = ssot_metrics['valid_moves']
            # Replace "valid immediate moves: [...]" patterns
            moves_pattern = r"valid immediate moves: \[[^\]]+\]"
            moves_replacement = f"valid immediate moves: {str(valid_moves)}"
            updated_text = re.sub(moves_pattern, moves_replacement, updated_text)
            
            # Also replace "Identify valid immediate moves: [...]" patterns
            identify_pattern = r"Identify valid immediate moves: \[[^\]]+\]"
            identify_replacement = f"Identify valid immediate moves: {str(valid_moves)}"
            updated_text = re.sub(identify_pattern, identify_replacement, updated_text)
        
        return updated_text 