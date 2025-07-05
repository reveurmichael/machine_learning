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
import sys

# Add project root to path to allow absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config.game_constants import DIRECTIONS

from utils.print_utils import print_info, print_warning, print_success, print_error
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
        """Process a single game and generate dataset entries."""
        try:
            # Extract moves and explanations
            moves_history = game_data.get("detailed_history", {}).get("moves", [])
            explanations = game_data.get("move_explanations", [])
            metrics_list = game_data.get("move_metrics", [])
            
            if not moves_history:
                print_warning("[DatasetGenerator] No moves found in game data")
                return
            
            # Extract dataset game states
            dataset_game_states = game_data.get("dataset_game_states", {})
            if not dataset_game_states:
                print_warning("[DatasetGenerator] No dataset game states found")
                return
            
            # SSOT: Fail fast if we don't have enough rounds for the moves
            # Determine the key type (int or str) from the first available key
            available_keys = list(dataset_game_states.keys())
            if not available_keys:
                raise RuntimeError("[SSOT] No dataset_game_states available")
            
            # Use the same key type as the dataset
            first_key = available_keys[0]
            if isinstance(first_key, int):
                required_rounds = list(range(len(moves_history)))
                key_converter = lambda i: i
            else:
                required_rounds = [str(i) for i in range(len(moves_history))]
                key_converter = lambda i: str(i)
            
            available_rounds = set(dataset_game_states.keys())
            
            # Check if all required rounds are available
            missing_rounds = [r for r in required_rounds if r not in available_rounds]
            if missing_rounds:
                raise RuntimeError(f"[SSOT] Missing required rounds {missing_rounds} for {len(moves_history)} moves. Available: {sorted(available_rounds)}")
            
            # Process moves and game states together
            for i in range(len(moves_history)):
                round_key = key_converter(i)  # Move i uses appropriate key type
                game_state = dataset_game_states[round_key]
                move = moves_history[i]
                explanation = explanations[i] if i < len(explanations) else {}
                metrics = metrics_list[i] if i < len(metrics_list) else {}

                # Validate head position before processing
                pre_move_head = game_state.get('head_position', [0, 0])
                if pre_move_head is None or not isinstance(pre_move_head, (list, tuple)) or len(pre_move_head) != 2:
                    raise RuntimeError(f"[SSOT] Invalid head position in game state for round {round_key}: {pre_move_head}")

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
                    
        except Exception as e:
            print_error(f"[DatasetGenerator] Error processing game {game_data.get('game_number', 'unknown')}: {str(e)}")
            raise

    # Valid moves calculation moved to agent_bfs.py for SSOT compliance
    # Use agent's _calculate_valid_moves method

    def _extract_jsonl_record(self, record: dict) -> dict:
        """
        Extract a single JSONL record from game data.
        SSOT Compliance: Use the agent's actual move and explanation as the source of truth.
        
        PRE-EXECUTION: All game_state values are from BEFORE the move is executed.
        This ensures consistency between the prompt (which shows pre-move state) and
        the completion metrics (which should match the prompt's state).
        
        Args:
            record: Game record containing state and move information
        Returns:
            Dictionary with prompt and completion for JSONL format
        """
        
        game_state = record.get('game_state', {})
        move = record.get('move', 'UNKNOWN')
        explanation = record.get('explanation', {})
        
        # Format the prompt using the recorded PRE-MOVE game state
        # PRE-EXECUTION: The prompt shows the game state before the move is executed
        # This includes: head_position, apple_position, snake_positions, score, steps
        prompt = self._format_prompt(game_state)
        
        # Extract pre-move state
        # PRE-EXECUTION: All these values are from the state BEFORE the move
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
            pre_move_head = [0, 0]

        # Validate apple position
        if apple_pos is None or not isinstance(apple_pos, (list, tuple)) or len(apple_pos) != 2:
            apple_pos = [0, 0]

        # SSOT: Use pre_move_head as the definitive source of truth for pre-move head position
        # PRE-EXECUTION: This ensures consistency between prompt and all pre-move calculations
        # This is the head position BEFORE the move is executed
        head = tuple(pre_move_head)  # Use the game state's head position as SSOT
        apple = tuple(apple_pos)
        # Obstacles: all body except head (use snake_positions but exclude the head position)
        # PRE-EXECUTION: These are the obstacles BEFORE the move is executed
        obstacles = set(tuple(p) for p in snake_positions if len(p) >= 2 and tuple(p) != head)

        # Compute valid moves from pre-move state (exclude head position)
        # PRE-EXECUTION: Valid moves calculated from pre-move head and snake positions
        # Use agent's static methods for SSOT compliance
        from agents.agent_bfs import BFSAgent
        valid_moves = BFSAgent._calculate_valid_moves(list(head), snake_positions, grid_size)

        # Compute manhattan distance from head to apple
        # PRE-EXECUTION: Distance from pre-move head to pre-move apple
        manhattan = abs(head[0] - apple[0]) + abs(head[1] - apple[1])

        # Use agent's actual metrics if available, otherwise compute them
        # PRE-EXECUTION: Agent metrics should all be from pre-move state
        agent_metrics = {}
        agent_chosen_direction = move  # Default to game history move
        if isinstance(explanation, dict):
            agent_metrics = explanation.get('metrics', {})
            # Use agent's actual chosen direction if available
            if 'final_chosen_direction' in agent_metrics:
                agent_chosen_direction = agent_metrics['final_chosen_direction']
        
        # SSOT: Use agent's metrics state as authoritative for validation
        # PRE-EXECUTION: Agent's head_position and valid_moves should be from pre-move state
        if agent_metrics and 'head_position' in agent_metrics:
            # Use agent's state for validation instead of recorded game state
            agent_head = agent_metrics['head_position']
            agent_valid_moves = agent_metrics.get('valid_moves', valid_moves)
            
            # Validate against agent's state
            if agent_chosen_direction not in agent_valid_moves:
                raise RuntimeError(f"SSOT violation: Agent's chosen direction '{agent_chosen_direction}' is not in valid moves {agent_valid_moves} for head {agent_head}.")
        else:
            # Fallback: validate against recorded game state
            if agent_chosen_direction not in valid_moves:
                raise RuntimeError(f"SSOT violation: Agent's chosen direction '{agent_chosen_direction}' is not in valid moves {valid_moves} for head {head}.")
        
        # Always compute apple_path_length from agent's BFS pathfinding
        # PRE-EXECUTION: Path length from pre-move head to pre-move apple
        path = BFSAgent._bfs_pathfind(list(head), list(apple), obstacles, grid_size)
        apple_path_len = len(path) - 1 if path else None
        # Sanity check: path length should never be less than Manhattan distance
        if apple_path_len is not None and apple_path_len < manhattan:
            apple_path_len = None

        # PRE-EXECUTION: Calculate where the head will be after the move
        # This is a prediction based on pre-move state and chosen direction
        next_head_prediction = list(head)
        if agent_chosen_direction in DIRECTIONS:
            dx, dy = DIRECTIONS[agent_chosen_direction]
            next_head_prediction = [head[0] + dx, head[1] + dy]
        

        # Compute remaining free cells
        # PRE-EXECUTION: Free cells based on pre-move snake positions
        remaining_free_cells = grid_size * grid_size - len(snake_positions)

        # Robust check: ensure all required metrics are present and valid before writing entry
        # Note: apple_path_length can be None when no path exists to the apple
        if (
            head is None or
            valid_moves is None or not isinstance(valid_moves, list) or
            manhattan is None or
            remaining_free_cells is None or
            agent_chosen_direction is None
        ):
            print(f"[ERROR] Skipping entry due to missing or invalid metrics: head={head}, valid_moves={valid_moves}, manhattan={manhattan}, apple_path_length={apple_path_len}, remaining_free_cells={remaining_free_cells}, move={agent_chosen_direction}")
            return None

        # Compose SSOT metrics - use PRE-MOVE state for consistency with prompt
        # PRE-EXECUTION: All metrics are from pre-move state to match the prompt
        # The prompt shows the pre-move state, so completion metrics should match
        ssot_metrics = {
            "head_position": list(head),  # PRE-MOVE head position (matches prompt)
            "apple_position": list(apple),  # PRE-MOVE apple position (matches prompt)
            "grid_size": grid_size,
            "snake_length": len(snake_positions),  # PRE-MOVE snake length (matches prompt)
            "valid_moves": valid_moves,  # PRE-MOVE valid moves (matches prompt)
            "manhattan_distance": manhattan,  # PRE-MOVE manhattan distance (matches prompt)
            "apple_path_length": apple_path_len,  # PRE-MOVE apple path length (matches prompt)
            "remaining_free_cells": remaining_free_cells,  # PRE-MOVE free cells (matches prompt)
            "final_chosen_direction": agent_chosen_direction,  # The move that will be executed
        }
        
        # Add any extra agent-computed metrics (e.g. apple_path_safe, fallback_used)
        # PRE-EXECUTION: These should also be from pre-move state
        for k in ["apple_path_safe", "fallback_used"]:
            if k in agent_metrics:
                ssot_metrics[k] = agent_metrics[k]

        # Use agent's explanation if available, otherwise flatten
        # PRE-EXECUTION: Explanation should describe the decision based on pre-move state
        if isinstance(explanation, dict):
            explanation_text = BFSAgent.flatten_explanation_for_jsonl(explanation)
        else:
            explanation_text = str(explanation)
        
        # Explanation is already constructed with correct values at source
        # No need to update with SSOT metrics
        
        # Fail-fast: agent's returned move and metrics['final_chosen_direction'] must match
        if 'final_chosen_direction' in agent_metrics and agent_chosen_direction != agent_metrics['final_chosen_direction']:
            raise RuntimeError(f"SSOT violation: agent_chosen_direction '{agent_chosen_direction}' does not match metrics['final_chosen_direction'] '{agent_metrics['final_chosen_direction']}'")
        
        return {
            "prompt": prompt,  # PRE-EXECUTION: Shows pre-move state
            "completion": json.dumps({
                "move": agent_chosen_direction,  # The move that will be executed
                "algorithm": self.algorithm,
                "metrics": ssot_metrics,  # PRE-EXECUTION: All metrics from pre-move state
                "explanation": explanation_text  # PRE-EXECUTION: Explanation based on pre-move state
            }, ensure_ascii=False)
        }

    # BFS pathfinding moved to agent_bfs.py for SSOT compliance
    # Use agent's _bfs_pathfind method

    # Next position calculation moved to agent_bfs.py for SSOT compliance
    # Use agent's position calculation methods

    def _extract_csv_features(self, record: dict, step_number: int = None) -> dict:
        """
        Extract CSV features from a single game record.
        SSOT Compliance: Use the agent's actual chosen direction as the source of truth.
        
        PRE-EXECUTION: All game_state values are from BEFORE the move is executed.
        This ensures consistency with the JSONL format and the prompt state.
        """
        game_state = record.get('game_state', {})
        explanation = record.get('explanation', {})
        
        # KISS: Use agent's actual chosen direction, not game history move
        # PRE-EXECUTION: This is the direction that will be executed
        move = 'UNKNOWN'  # Default
        if isinstance(explanation, dict) and 'metrics' in explanation:
            agent_metrics = explanation['metrics']
            if 'final_chosen_direction' in agent_metrics:
                move = agent_metrics['final_chosen_direction']
        else:
            # Fallback to game history move
            move = record.get('move', 'UNKNOWN')
        
        # Extract game state data
        # PRE-EXECUTION: All these values are from the state BEFORE the move
        game_id = record.get('game_id', 0)
        head_pos = game_state.get('head_position', [0, 0])  # PRE-MOVE: current head position
        apple_pos = game_state.get('apple_position', [0, 0])  # PRE-MOVE: current apple position
        snake_positions = game_state.get('snake_positions', [])  # PRE-MOVE: current snake positions
        grid_size = game_state.get('grid_size', 10)
        
        # Validate positions
        if not isinstance(head_pos, (list, tuple)) or len(head_pos) != 2:
            head_pos = [0, 0]
        if not isinstance(apple_pos, (list, tuple)) or len(apple_pos) != 2:
            apple_pos = [0, 0]
        
        head_x, head_y = head_pos[0], head_pos[1]  # PRE-MOVE: current head coordinates
        apple_x, apple_y = apple_pos[0], apple_pos[1]  # PRE-MOVE: current apple coordinates
        
        # Calculate apple direction features
        # PRE-EXECUTION: Apple direction relative to current head position
        apple_dir_up = 1 if apple_y > head_y else 0
        apple_dir_down = 1 if apple_y < head_y else 0
        apple_dir_left = 1 if apple_x < head_x else 0
        apple_dir_right = 1 if apple_x > head_x else 0
        
        # Calculate danger features
        # PRE-EXECUTION: Danger assessment based on current head position and snake body
        danger_straight = 0
        danger_left = 0
        danger_right = 0
        
        # Check for wall collision
        # PRE-EXECUTION: Check if moving in each direction would hit a wall from current position
        if move == "UP":
            danger_straight = 1 if head_y + 1 >= grid_size else 0
            danger_left = 1 if head_x - 1 < 0 else 0
            danger_right = 1 if head_x + 1 >= grid_size else 0
        elif move == "DOWN":
            danger_straight = 1 if head_y - 1 < 0 else 0
            danger_left = 1 if head_x + 1 >= grid_size else 0
            danger_right = 1 if head_x - 1 < 0 else 0
        elif move == "LEFT":
            danger_straight = 1 if head_x - 1 < 0 else 0
            danger_left = 1 if head_y - 1 < 0 else 0
            danger_right = 1 if head_y + 1 >= grid_size else 0
        elif move == "RIGHT":
            danger_straight = 1 if head_x + 1 >= grid_size else 0
            danger_left = 1 if head_y + 1 >= grid_size else 0
            danger_right = 1 if head_y - 1 < 0 else 0
        
        # Check for body collision
        # PRE-EXECUTION: Check if moving in each direction would hit snake body from current position
        if move == "UP":
            next_pos = [head_x, head_y + 1]
            if next_pos in snake_positions:
                danger_straight = 1
        elif move == "DOWN":
            next_pos = [head_x, head_y - 1]
            if next_pos in snake_positions:
                danger_straight = 1
        elif move == "LEFT":
            next_pos = [head_x - 1, head_y]
            if next_pos in snake_positions:
                danger_straight = 1
        elif move == "RIGHT":
            next_pos = [head_x + 1, head_y]
            if next_pos in snake_positions:
                danger_straight = 1
        
        # Calculate free space features
        # PRE-EXECUTION: Free space in each direction from current head position
        free_space_up = BFSAgent.count_free_space_in_direction(head_pos, "UP", snake_positions, grid_size)
        free_space_down = BFSAgent.count_free_space_in_direction(head_pos, "DOWN", snake_positions, grid_size)
        free_space_left = BFSAgent.count_free_space_in_direction(head_pos, "LEFT", snake_positions, grid_size)
        free_space_right = BFSAgent.count_free_space_in_direction(head_pos, "RIGHT", snake_positions, grid_size)
        
        # Calculate valid moves
        # PRE-EXECUTION: Valid moves from current head position
        valid_moves = BFSAgent._calculate_valid_moves(head_pos, snake_positions, grid_size)
        
        # KISS: Use agent's own valid moves if available, otherwise use computed valid moves
        if isinstance(explanation, dict) and 'metrics' in explanation:
            agent_metrics = explanation['metrics']
            if 'valid_moves' in agent_metrics:
                valid_moves = agent_metrics['valid_moves']
        
        if move not in valid_moves:
            raise RuntimeError(f"SSOT violation: CSV target_move '{move}' is not in valid moves {valid_moves} for head {head_pos}.")
        
        # Return the complete CSV record
        # PRE-EXECUTION: All features are from pre-move state
        return {
            # Metadata
            'game_id': game_id,
            'step_in_game': step_number if step_number is not None else game_state.get('steps', 0),  # PRE-MOVE: current step count
            
            # Position features
            'head_x': head_x,  # PRE-MOVE: current head x coordinate
            'head_y': head_y,  # PRE-MOVE: current head y coordinate
            'apple_x': apple_x,  # PRE-MOVE: current apple x coordinate
            'apple_y': apple_y,  # PRE-MOVE: current apple y coordinate
            
            # Game state
            'snake_length': len(snake_positions),  # PRE-MOVE: current snake length
            
            # Apple direction features
            'apple_dir_up': apple_dir_up,  # PRE-MOVE: apple direction relative to current head
            'apple_dir_down': apple_dir_down,  # PRE-MOVE: apple direction relative to current head
            'apple_dir_left': apple_dir_left,  # PRE-MOVE: apple direction relative to current head
            'apple_dir_right': apple_dir_right,  # PRE-MOVE: apple direction relative to current head
            
            # Danger detection features
            'danger_straight': danger_straight,  # PRE-MOVE: danger assessment from current position
            'danger_left': danger_left,  # PRE-MOVE: danger assessment from current position
            'danger_right': danger_right,  # PRE-MOVE: danger assessment from current position
            
            # Free space features
            'free_space_up': free_space_up,  # PRE-MOVE: free space from current head
            'free_space_down': free_space_down,  # PRE-MOVE: free space from current head
            'free_space_left': free_space_left,  # PRE-MOVE: free space from current head
            'free_space_right': free_space_right,  # PRE-MOVE: free space from current head
            
            # Target
            'target_move': move  # The move that will be executed
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

        grid_size = game_state.get('grid_size', 10)
        snake_positions = game_state.get('snake_positions', [])
        apple_position = game_state.get('apple_position', [])
        score = game_state.get('score', 0)
        steps = game_state.get('steps', 0)
        algorithm = game_state.get('algorithm', self.algorithm)

        if not snake_positions:
            return "Invalid game state: Snake has no positions."

        head_pos = snake_positions[-1]  # Head is at index -1 (last element)
        body_positions = snake_positions[:-1] if len(snake_positions) > 1 else []

        # Board representation
        board = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
        if apple_position and len(apple_position) >= 2:
            try:
                apple_x, apple_y = apple_position[0], apple_position[1]
                if 0 <= apple_x < grid_size and 0 <= apple_y < grid_size:
                    board[apple_y][apple_x] = 'A'
            except (KeyError, TypeError, IndexError):
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
        board_str = "\n".join(" ".join(row) for row in reversed(board))

        if apple_position:
            if isinstance(apple_position, dict):
                apple_x, apple_y = apple_position.get('x', 0), apple_position.get('y', 0)
            else:
                apple_x, apple_y = apple_position[0], apple_position[1]
            manhattan_distance = abs(head_pos[0] - apple_x) + abs(head_pos[1] - apple_y)
        else:
            manhattan_distance = -1
        valid_moves = BFSAgent._calculate_valid_moves(head_pos, snake_positions, grid_size)

        # Add coordinate system and movement example (user's wording)
        coord_system = (
            "Coordinates range from (0, 0) at the **bottom-left** to (GridSize-1,GridSize-1) at the **top-right**.\n"
            "\n## COORDINATE SYSTEM:\n- 'UP' means y+1  \n- 'DOWN' means y-1  \n- 'RIGHT' means x+1  \n- 'LEFT' means x-1  \n"
            "\nExample Moves from (1, 1): \n• UP → (1, 2) \n• DOWN → (1, 0) \n• RIGHT → (2, 1) \n• LEFT → (0, 1)\n"
            "\n### Movement Example (Normal Move):\nIf S = `[HEAD (4, 4), BODY1 (4, 3) ]` and you choose \"LEFT\":  \n- new_head = (4-1, 4) = (3, 4).  \n- No collision if (3, 4) is empty.  \n- Insert old head (4, 4) at front of BODY ⇒ BODY becomes `[(4, 4), (4, 3) ]`.  \n- Remove tail (4, 3) (assuming not eating apple) ⇒ BODY becomes `[(4, 4) ]`.  \n- Update HEAD to (3, 4).  \n- Resulting S = `[(3, 4) (head), (4, 4) (body1) ]`.\n"
        )

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
- Body Positions: {body_positions}

**Board:**
{board_str}

**Strategic Context:**
- Manhattan Distance to Apple: {manhattan_distance}
- Valid Moves: {valid_moves}

{coord_system}

### Task:
Based on the `{algorithm}` logic, what is the optimal next move? Provide the move and a detailed, step-by-step explanation of the reasoning.
"""
        return prompt

    # Explanation update moved to agent_bfs.py for SSOT compliance
    # Use agent's explanation generation methods

    # JSON serialization moved to BFSAgent for SSOT compliance 