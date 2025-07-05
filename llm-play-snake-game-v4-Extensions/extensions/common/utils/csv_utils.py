"""
CSV Utilities for Snake Game AI Extensions

This module provides CSV-specific formatting utilities for generating
CSV entries from game states and agent explanations.

Design Philosophy:
- Single responsibility: Only handles CSV formatting
- Agent-agnostic: Works with any agent and game state
- Consistent schema: Standardized CSV output format across all agents
- Extensible: Supports inheritance for task-specific customization
- Generic: Works for all tasks 1-5 (heuristics, supervised, RL, etc.)

Usage:
    from extensions.common.utils.csv_utils import create_csv_record, CSVFeatureExtractor
    
    # Create a CSV record
    csv_record = create_csv_record(game_state, move, step_number, metrics)
    
    # Use extensible feature extractor
    extractor = CSVFeatureExtractor()
    features = extractor.extract_features(game_state, move, step_number)
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from utils.print_utils import print_info, print_warning, print_error
from config.game_constants import DIRECTIONS

# Import agent utilities for SSOT compliance
# Note: This follows the standalone principle - we import from core/utils, not other extensions
from utils.moves_utils import normalize_direction, is_reverse


@dataclass
class GameStateForCSV:
    """Standardized game state representation for CSV feature extraction."""
    
    head_position: Tuple[int, int]
    apple_position: Tuple[int, int]
    snake_positions: List[Tuple[int, int]]
    grid_size: int
    score: int
    steps: int
    current_direction: str = "UP"
    game_id: int = 1


class CSVFeatureExtractor:
    """
    Extracts standardized CSV features from game states.
    
    Design Pattern: Strategy Pattern + Template Method Pattern
    Purpose: Provide consistent feature extraction across all tasks 1-5
    
    The 16-feature format works for any grid size by using relative
    positioning and directional indicators rather than absolute coordinates
    that would vary with grid size.
    
    Educational Note (simple logging):
    This class is designed to be extensible for extensions that need
    specialized feature extraction while maintaining compatibility with
    the standard 16-feature format. Extensions can inherit and customize
    specific feature extraction methods.
    
    simple logging Implementation:
    - Base class provides complete 16-feature extraction
    - Protected methods allow selective feature customization
    - Virtual methods enable additional feature extraction
    - Extension-specific extractors can inherit and adapt
    """
    
    def __init__(self):
        """Initialize the CSV feature extractor."""
        self.feature_names = [
            'game_id', 'step_in_game', 'head_x', 'head_y', 'apple_x', 'apple_y', 'snake_length',
            'apple_dir_up', 'apple_dir_down', 'apple_dir_left', 'apple_dir_right',
            'danger_straight', 'danger_left', 'danger_right',
            'free_space_up', 'free_space_down', 'free_space_left', 'free_space_right',
            'target_move'
        ]
        print_info("[CSVFeatureExtractor] Initialized with standard 16-feature format")
    
    def extract_features(self, game_state: Dict[str, Any], move: str, step_number: int = None) -> Dict[str, Any]:
        """
        Extract standardized CSV features from game state.
        
        PRE-EXECUTION: All game_state values are from BEFORE the move is executed.
        This ensures consistency with the JSONL format and the prompt state.
        
        Args:
            game_state: Game state dictionary (PRE-MOVE state)
            move: The move that will be executed
            step_number: Step number in the game (optional)
            
        Returns:
            Dictionary with standardized CSV features
        """
        # Convert to standardized game state
        std_game_state = self._convert_to_standard_state(game_state)
        
        # Extract position features
        position_features = self._extract_position_features(std_game_state)
        
        # Extract apple direction features
        apple_direction_features = self._extract_apple_direction_features(std_game_state)
        
        # Extract danger features
        danger_features = self._extract_danger_features(std_game_state, move)
        
        # Extract free space features
        free_space_features = self._extract_free_space_features(std_game_state)
        
        # Extract metadata features
        metadata_features = self._extract_metadata_features(std_game_state, step_number)
        
        # Allow subclasses to add extension-specific features
        extension_features = self._extract_extension_specific_features(std_game_state, move)
        
        # Combine all features
        features = {}
        features.update(metadata_features)
        features.update(position_features)
        features.update(apple_direction_features)
        features.update(danger_features)
        features.update(free_space_features)
        features.update(extension_features)
        
        # Add target move
        features['target_move'] = move
        
        return features
    
    def _convert_to_standard_state(self, game_state: Dict[str, Any]) -> GameStateForCSV:
        """Convert game state to standardized format."""
        # Extract and validate positions
        head_pos = self._safe_extract_position(game_state.get('head_position', [0, 0]), 'head')
        apple_pos = self._safe_extract_position(game_state.get('apple_position', [0, 0]), 'apple')
        snake_positions = game_state.get('snake_positions', [])
        
        # Validate snake positions
        if not isinstance(snake_positions, list):
            snake_positions = []
        
        # Convert to tuples for consistency
        snake_positions = [tuple(pos) if isinstance(pos, (list, tuple)) else (0, 0) for pos in snake_positions]
        
        return GameStateForCSV(
            head_position=tuple(head_pos),
            apple_position=tuple(apple_pos),
            snake_positions=snake_positions,
            grid_size=game_state.get('grid_size', 10),
            score=game_state.get('score', 0),
            steps=game_state.get('steps', 0),
            current_direction=game_state.get('current_direction', 'UP'),
            game_id=game_state.get('game_id', 1)
        )
    
    def _safe_extract_position(self, pos, name: str) -> List[int]:
        """Safely extract position coordinates."""
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            try:
                return [int(pos[0]), int(pos[1])]
            except (ValueError, TypeError):
                pass
        # If invalid, log warning and return [0, 0]
        print_warning(f"[CSVFeatureExtractor] Invalid {name} position: {pos}, using [0, 0]")
        return [0, 0]
    
    def _extract_position_features(self, game_state: GameStateForCSV) -> Dict[str, int]:
        """Extract absolute position features."""
        head_x, head_y = game_state.head_position
        apple_x, apple_y = game_state.apple_position
        
        return {
            'head_x': head_x,
            'head_y': head_y,
            'apple_x': apple_x,
            'apple_y': apple_y,
            'snake_length': len(game_state.snake_positions)
        }
    
    def _extract_apple_direction_features(self, game_state: GameStateForCSV) -> Dict[str, int]:
        """Extract binary features indicating apple direction relative to head."""
        head_x, head_y = game_state.head_position
        apple_x, apple_y = game_state.apple_position
        
        return {
            'apple_dir_up': 1 if apple_y > head_y else 0,
            'apple_dir_down': 1 if apple_y < head_y else 0,
            'apple_dir_left': 1 if apple_x < head_x else 0,
            'apple_dir_right': 1 if apple_x > head_x else 0
        }
    
    def _extract_danger_features(self, game_state: GameStateForCSV, move: str) -> Dict[str, int]:
        """Extract danger detection features based on current move."""
        head_x, head_y = game_state.head_position
        grid_size = game_state.grid_size
        snake_positions = game_state.snake_positions
        
        # Initialize danger flags
        danger_straight = 0
        danger_left = 0
        danger_right = 0
        
        # Calculate next position based on move
        next_pos = self._calculate_next_position(head_x, head_y, move)
        
        # Check for wall collision
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
        if next_pos in snake_positions:
            danger_straight = 1
        
        return {
            'danger_straight': danger_straight,
            'danger_left': danger_left,
            'danger_right': danger_right
        }
    
    def _extract_free_space_features(self, game_state: GameStateForCSV) -> Dict[str, int]:
        """Extract free space features in each direction."""
        head_x, head_y = game_state.head_position
        grid_size = game_state.grid_size
        snake_positions = set(game_state.snake_positions)
        
        # Calculate free space in each direction
        free_space_up = self._count_free_space_in_direction(head_x, head_y, "UP", snake_positions, grid_size)
        free_space_down = self._count_free_space_in_direction(head_x, head_y, "DOWN", snake_positions, grid_size)
        free_space_left = self._count_free_space_in_direction(head_x, head_y, "LEFT", snake_positions, grid_size)
        free_space_right = self._count_free_space_in_direction(head_x, head_y, "RIGHT", snake_positions, grid_size)
        
        return {
            'free_space_up': free_space_up,
            'free_space_down': free_space_down,
            'free_space_left': free_space_left,
            'free_space_right': free_space_right
        }
    
    def _extract_metadata_features(self, game_state: GameStateForCSV, step_number: Optional[int]) -> Dict[str, int]:
        """Extract metadata features."""
        return {
            'game_id': game_state.game_id,
            'step_in_game': step_number if step_number is not None else game_state.steps
        }
    
    def _extract_extension_specific_features(self, game_state: GameStateForCSV, move: str) -> Dict[str, Any]:
        """
        Extract extension-specific features (simple logging Extension Point).
        
        Override this method in subclasses to add custom features while
        maintaining compatibility with the standard 16-feature format.
        
        Note: This should return an empty dict in the base implementation
        to maintain the standard 16-feature format. Only override if you
        need additional features for specialized extensions.
        
        Example:
            class RLCSVFeatureExtractor(CSVFeatureExtractor):
                def _extract_extension_specific_features(self, game_state, move):
                    return {
                        "reward_signal": self._calculate_reward(game_state, move),
                        "action_value": self._get_action_value(move)
                    }
        """
        return {}
    
    def _calculate_next_position(self, head_x: int, head_y: int, move: str) -> Tuple[int, int]:
        """Calculate the next position based on current move."""
        if move == "UP":
            return (head_x, head_y + 1)
        elif move == "DOWN":
            return (head_x, head_y - 1)
        elif move == "LEFT":
            return (head_x - 1, head_y)
        elif move == "RIGHT":
            return (head_x + 1, head_y)
        else:
            return (head_x, head_y)  # Default to current position
    
    def _count_free_space_in_direction(self, head_x: int, head_y: int, direction: str, 
                                     snake_positions: set, grid_size: int) -> int:
        """Count free space in a specific direction."""
        count = 0
        current_x, current_y = head_x, head_y
        
        while True:
            # Calculate next position
            if direction == "UP":
                next_x, next_y = current_x, current_y + 1
            elif direction == "DOWN":
                next_x, next_y = current_x, current_y - 1
            elif direction == "LEFT":
                next_x, next_y = current_x - 1, current_y
            elif direction == "RIGHT":
                next_x, next_y = current_x + 1, current_y
            else:
                break
            
            # Check bounds
            if not (0 <= next_x < grid_size and 0 <= next_y < grid_size):
                break
            
            # Check if position is occupied by snake
            if (next_x, next_y) in snake_positions:
                break
            
            count += 1
            current_x, current_y = next_x, next_y
        
        return count


def create_csv_record(game_state: Dict[str, Any], move: str, step_number: int, 
                     metrics: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a CSV record from game state and move data.
    
    This function provides a simple interface for creating CSV records
    using the standardized CSVFeatureExtractor.
    
    Args:
        game_state: Game state dictionary
        move: Move made by the agent
        step_number: Step number in the game
        metrics: Agent metrics dictionary (optional)
        
    Returns:
        CSV record dictionary
    """
    extractor = CSVFeatureExtractor()
    return extractor.extract_features(game_state, move, step_number)


def create_csv_record_with_explanation(game_state: Dict[str, Any], explanation: Dict[str, Any], 
                                     step_number: int) -> Dict[str, Any]:
    """
    Create a CSV record from game state and explanation data.
    
    This function extracts the move from the explanation and creates
    a standardized CSV record.
    
    Args:
        game_state: Game state dictionary
        explanation: Agent explanation dictionary containing metrics
        step_number: Step number in the game
        
    Returns:
        CSV record dictionary
    """
    # Extract move from explanation
    move = 'UNKNOWN'  # Default
    if isinstance(explanation, dict) and 'metrics' in explanation:
        agent_metrics = explanation['metrics']
        if 'final_chosen_direction' in agent_metrics:
            move = agent_metrics['final_chosen_direction']
        elif 'move' in agent_metrics:
            move = agent_metrics['move']
    
    # If no move found in explanation, try to get from explanation directly
    if move == 'UNKNOWN' and 'move' in explanation:
        move = explanation['move']
    
    return create_csv_record(game_state, move, step_number, explanation.get('metrics', {})) 