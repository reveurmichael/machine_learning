"""
Core dataset generator – convert raw logs into CSV / JSONL.

This module provides the core DatasetGenerator class that converts
heuristic game logs into structured datasets for machine learning.

Design Philosophy:
- Algorithm-agnostic: Can be reused by supervised/RL extensions
- Single responsibility: Only handles dataset conversion
- Standardized logging: Uses print_utils functions for all operations
- Generic: Uses common utilities for CSV feature extraction
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

# Import common CSV utilities for SSOT compliance
from extensions.common.utils.csv_utils import CSVFeatureExtractor, create_csv_record_with_explanation

__all__ = ["DatasetGenerator"]


class DatasetGenerator:
    """
    Convert raw heuristic game logs to datasets (CSV / JSONL).
    Designed to be algorithm-agnostic so supervised / RL can reuse it.
    
    This generator reads heuristic algorithm game logs and converts them
    into structured datasets suitable for machine learning tasks.
    
    Design Philosophy:
    - Uses common CSV utilities for feature extraction (SSOT compliance)
    - Maintains JSONL functionality for language-rich datasets
    - Generic and extensible for all tasks 1-5
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
        
        # Initialize CSV feature extractor from common utilities
        self.csv_extractor = CSVFeatureExtractor()
        
        # File handles
        self._csv_writer = None
        self._jsonl_fh = None
        
        print_info(f"Initialized for {algorithm} (output: {output_dir})", "DatasetGenerator")

    # ---------------- CSV
    def _open_csv(self):
        """Open CSV file for writing."""
        csv_path = self.output_dir / f"{self.algorithm}_dataset.csv"
        fh = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=self.csv_extractor.feature_names)
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
    def generate_games_and_write_datasets(self, max_games: int, max_steps: int, grid_size: int, formats: list = ["csv", "jsonl"], verbose: bool = False):
        """
        Run games in memory and generate datasets directly, without loading from disk.
        Args:
            max_games: Number of games to play
            max_steps: Maximum steps per game
            grid_size: Grid size for the game
            formats: List of formats to generate ("csv", "jsonl", or both)
            verbose: Enable verbose output
        """
        from game_manager import HeuristicGameManager
        import argparse
        
        # Build args namespace for HeuristicGameManager
        args = argparse.Namespace(
            algorithm=self.algorithm,
            max_games=max_games,
            max_steps=max_steps,
            grid_size=grid_size,
            verbose=verbose,
            no_gui=True
        )
        
        # Run games in memory
        game_manager = HeuristicGameManager(args)
        game_manager.initialize()
        
        # Open output files
        if "csv" in formats:
            self._open_csv()
        if "jsonl" in formats:
            self._open_jsonl()
        
        # Process games directly
        for game_id in range(1, max_games + 1):
            if verbose:
                print_info(f"[DatasetGenerator] Running game {game_id}/{max_games}")
            
            # Run single game
            game_duration = game_manager._run_single_game()
            game_manager.game_count += 1
            game_manager.game.game_state.game_number = game_manager.game_count
            game_data = game_manager._generate_game_data(game_duration)
            
            # Process game data directly
            self._process_single_game(game_data)
        
        # Close handles
        if self._csv_writer:
            self._csv_writer[1].close()
            print_success("CSV dataset saved")
        if self._jsonl_fh:
            self._jsonl_fh.close()
            print_success("JSONL dataset saved")
            
        if verbose:
            print_success(f"[DatasetGenerator] Dataset generation complete for {self.algorithm}")

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
            
            # Get game_id from game_data for proper CSV identification
            game_id = game_data.get('game_number', game_data.get('metadata', {}).get('game_number', 1))
            
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
                    "metrics": metrics,
                    "game_id": game_id  # Pass the actual game_id from game_data
                }
                # Write to JSONL
                if self._jsonl_fh:
                    jsonl_record = self._extract_jsonl_record(record)
                    self._jsonl_fh.write(json.dumps(jsonl_record) + '\n')
                # Write to CSV using common utilities
                if self._csv_writer:
                    csv_record = self._extract_csv_features(record, step_number=i+1)  # CSV step numbers are 1-indexed
                    self._csv_writer[0].writerow(csv_record)

        except Exception as e:
            print_error(f"[DatasetGenerator] Error processing game {game_data.get('game_number', 'unknown')}: {str(e)}")
            raise

    def _extract_csv_features(self, record: dict, step_number: int = None) -> dict:
        """
        Extract CSV features from a single game record using common utilities.
        SSOT Compliance: Use the common CSVFeatureExtractor for consistency.
        
        PRE-EXECUTION: All game_state values are from BEFORE the move is executed.
        This ensures consistency with the JSONL format and the prompt state.
        """
        game_state = record.get('game_state', {})
        explanation = record.get('explanation', {})
        game_id = record.get('game_id', 1)
        
        # Use common CSV utilities for feature extraction
        # This ensures consistency across all extensions and follows SSOT principles
        csv_record = create_csv_record_with_explanation(game_state, explanation, step_number, game_id)
        
        return csv_record

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
        explanation = record.get('explanation', {})
        
        # KISS: Use agent's actual chosen direction, not game history move
        # PRE-EXECUTION: This is the direction that will be executed
        move = 'UNKNOWN'  # Default
        if isinstance(explanation, dict) and 'metrics' in explanation:
            agent_metrics = explanation['metrics']
            if 'final_chosen_direction' in agent_metrics:
                move = agent_metrics['final_chosen_direction']
        else:
            # Fail-fast: No valid explanation from agent
            raise RuntimeError(f"SSOT violation: No valid explanation from agent for record {record.get('game_id', 'unknown')}")
        
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
        
        # Calculate apple direction features using centralized utility
        # PRE-EXECUTION: Apple direction relative to current head position
        from extensions.common.utils.game_analysis_utils import calculate_apple_direction, calculate_danger_assessment
        
        apple_direction = calculate_apple_direction(head_pos, apple_pos)
        apple_dir_up = apple_direction['up']
        apple_dir_down = apple_direction['down']
        apple_dir_left = apple_direction['left']
        apple_dir_right = apple_direction['right']
        
        # Calculate danger features using centralized utility
        # PRE-EXECUTION: Danger assessment based on current head position and snake body
        danger_assessment = calculate_danger_assessment(head_pos, snake_positions, grid_size, move)
        danger_straight = danger_assessment['straight']
        danger_left = danger_assessment['left']
        danger_right = danger_assessment['right']
        
        # Calculate free space features
        # PRE-EXECUTION: Free space in each direction from current head position
        free_space_up = BFSAgent.count_free_space_in_direction(game_state, "UP")
        free_space_down = BFSAgent.count_free_space_in_direction(game_state, "DOWN")
        free_space_left = BFSAgent.count_free_space_in_direction(game_state, "LEFT")
        free_space_right = BFSAgent.count_free_space_in_direction(game_state, "RIGHT")
        
        # Calculate valid moves
        # PRE-EXECUTION: Valid moves from current head position
        valid_moves = BFSAgent._calculate_valid_moves(game_state)
        
        # KISS: Use agent's own valid moves if available, otherwise use computed valid moves
        if isinstance(explanation, dict) and 'metrics' in explanation:
            agent_metrics = explanation['metrics']
            if 'valid_moves' in agent_metrics:
                valid_moves = agent_metrics['valid_moves']
        
        if move not in valid_moves:
            raise RuntimeError(f"SSOT violation: JSONL target_move '{move}' is not in valid moves {valid_moves} for head {head_pos}.")
        
        # Format prompt using the game state
        prompt = self._format_prompt(game_state)
        
        # KISS: Use agent's explanation directly - no fallbacks needed
        # SSOT: The explanation comes from the agent and is already properly formatted
        if isinstance(explanation, dict) and 'explanation_steps' in explanation:
            # Use the agent's rich explanation
            explanation_text = '\n'.join(explanation['explanation_steps'])
        else:
            # Fail-fast: Agent must provide proper explanation
            raise RuntimeError(f"SSOT violation: Agent explanation missing 'explanation_steps' for record {record.get('game_id', 'unknown')}")
        
        # Format completion with the move and explanation
        completion = self._format_completion(move, explanation_text, {
            'valid_moves': valid_moves,
            'manhattan_distance': abs(head_x - apple_x) + abs(head_y - apple_y),
            'apple_direction': {
                'up': apple_dir_up,
                'down': apple_dir_down,
                'left': apple_dir_left,
                'right': apple_dir_right
            },
            'danger_assessment': {
                'straight': danger_straight,
                'left': danger_left,
                'right': danger_right
            },
            'free_space': {
                'up': free_space_up,
                'down': free_space_down,
                'left': free_space_left,
                'right': free_space_right
            }
        })
        
        # Return the complete JSONL record
        # PRE-EXECUTION: All features are from pre-move state
        return {
            "prompt": prompt,
            "completion": completion
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
        
        # PRE-EXECUTION: Extract head position from game state
        # The game state has a separate head_position field for clarity
        head_pos = game_state.get('head_position', [0, 0])
        if not snake_positions:
            return "Invalid game state: Snake has no positions."

        # PRE-EXECUTION: Body positions are all snake positions except head
        # This ensures consistency with the agent's obstacle calculation
        body_positions = [pos for pos in snake_positions if pos != head_pos][::-1] # use the reverse order of the body positions, so that the head and the first element of the body_positions are adjacent.

        # Board representation using centralized utility
        from utils.board_utils import create_text_board
        board_str = create_text_board(
            grid_size=grid_size,
            head_position=head_pos,
            body_positions=body_positions,
            apple_position=apple_position
        )

        # Convert coordinate lists to tuples for consistent formatting
        head_pos_tuple = tuple(head_pos) if isinstance(head_pos, (list, tuple)) else (0, 0)
        apple_pos_tuple = tuple(apple_position) if isinstance(apple_position, (list, tuple)) else (0, 0)
        body_positions_tuples = [tuple(pos) if isinstance(pos, (list, tuple)) else (0, 0) for pos in body_positions]

        # Format the prompt
        prompt = f"""You are playing Snake on a {grid_size}x{grid_size} grid. The coordinate system is (0,0) at bottom-left to ({grid_size-1},{grid_size-1}) at top-right. Movement: UP=y+1, DOWN=y-1, RIGHT=x+1, LEFT=x-1.

Current game state:
- Score: {score}
- Steps: {steps}
- Algorithm: {algorithm}
- Snake head position: {head_pos_tuple}
- Apple position: {apple_pos_tuple}
- Snake body positions: {body_positions_tuples}
- Snake length: {len(snake_positions)}

Board representation (H=head, A=apple, S=snake body, .=empty):
{board_str}

What is the best move to make? Consider:
1. Path to the apple
2. Avoiding collisions with walls and snake body
3. Maximizing score and survival

Choose from: UP, DOWN, LEFT, RIGHT

Move:"""

        return prompt 

    def _format_completion(self, move: str, explanation: str, metrics: dict) -> str:
        """
        Format the completion with move and explanation.
        Args:
            move: The chosen move
            explanation: Agent explanation
            metrics: Calculated metrics
        Returns:
            Formatted completion string
        """
        # Format the completion
        completion = f""" {move}

Explanation: {explanation}

Metrics:
- Valid moves: {metrics.get('valid_moves', [])}
- Manhattan distance to apple: {metrics.get('manhattan_distance', 0)}
- Apple direction: {metrics.get('apple_direction', {})}
- Danger assessment: {metrics.get('danger_assessment', {})}
- Free space: {metrics.get('free_space', {})}

Conclusion: The move {move} was chosen because {explanation.lower()}"""

        return completion 