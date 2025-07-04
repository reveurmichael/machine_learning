import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

"""
CSV Schema Utilities for Snake Game AI Extensions.

This module provides utilities for handling the standardized 16-feature CSV
dataset format used across extensions. It implements a grid-size agnostic
approach that works with any board size.

This file follows the principles from final-decision-10.md:
- This file follows the principles from final-decision-10.md.
- All utilities must use simple print logging (simple logging).
- All utilities must be OOP, extensible, and never over-engineered.
- Reference: SimpleFactory in factory_utils.py is the canonical factory pattern for all extensions.
- See also: agents.md, core.md, config.md, factory-design-pattern.md, extension-evolution-rules.md.

Design Philosophy:
- Simple, educational, and extensible CSV schema utilities for all extensions.
- All code examples use print() and create() as canonical patterns.

Design Pattern: Strategy Pattern
- Different feature extraction strategies for different game states
- Pluggable validation strategies
- Consistent interface across all grid sizes

Educational Value:
Demonstrates how to design data format utilities that are both flexible
and maintainable, with proper separation of concerns and validation.

CSV schema utilities for Snake Game AI extensions.

This module follows the principles from final-decision-10.md.
- OOP extensibility is prioritized.
- Logging is always simple (print()).
- No ML/DL/RL/LLM-specific coupling.

Reference: docs/extensions-guideline/final-decision-10.md
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

# Import configuration constants
from ..config.dataset_formats import (
    CSV_COLUMN_NAMES, CSV_FEATURE_NAMES, CSV_METADATA_COLUMNS,
    CSV_TARGET_COLUMN, CSV_BINARY_FEATURES, CSV_COUNT_FEATURES,
    CSV_POSITION_FEATURES, CSV_VALID_MOVES, CSV_EXPECTED_COLUMNS
)
from ..config.validation_rules import CSV_VALIDATION_RULES

# =============================================================================
# Feature Extractor Classes
# =============================================================================

@dataclass
class GameState:
    """Standardized game state representation for feature extraction."""
    
    head_position: Tuple[int, int]
    apple_position: Tuple[int, int]
    snake_positions: List[Tuple[int, int]]
    current_direction: str
    score: int
    steps: int
    snake_length: int
    grid_size: int

class TabularFeatureExtractor:
    """
    Extracts 16 standardized features from Snake game states.
    
    Design Pattern: Strategy Pattern + Template Method Pattern
    Purpose: Provide consistent feature extraction across all grid sizes
    
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
        self.feature_names = CSV_FEATURE_NAMES.copy()
        self._initialize_extractor_settings()
    
    def extract_features(self, game_state: GameState) -> Dict[str, Any]:
        """
        Extract 16 standardized features from game state.
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary with 16 feature values
            
        Features extracted:
        - Position features (4): head_x, head_y, apple_x, apple_y
        - Game state features (1): snake_length
        - Apple direction features (4): binary flags for each direction
        - Danger detection features (3): binary flags for collision risks
        - Free space features (4): count of free cells in each direction
        """
        features = {}
        
        # Extract position features
        features.update(self._extract_position_features(game_state))
        
        # Extract game state features
        features.update(self._extract_game_state_features(game_state))
        
        # Extract apple direction features
        features.update(self._extract_apple_direction_features(game_state))
        
        # Extract danger detection features
        features.update(self._extract_danger_features(game_state))
        
        # Extract free space features
        features.update(self._extract_free_space_features(game_state))
        
        # Allow subclasses to add extension-specific features
        extension_features = self._extract_extension_specific_features(game_state)
        features.update(extension_features)
        
        return features
    
    def _initialize_extractor_settings(self) -> None:
        """
        Initialize extractor-specific settings (simple logging Extension Point).
        
        This method can be overridden by subclasses to set up extension-specific
        feature extraction configurations or custom processing parameters.
        
        Example:
            class RLFeatureExtractor(TabularFeatureExtractor):
                def _initialize_extractor_settings(self):
                    self.include_reward_features = True
                    self.temporal_window = 5
        """
        pass
    
    def _extract_extension_specific_features(self, game_state: GameState) -> Dict[str, Any]:
        """
        Extract extension-specific features (simple logging Extension Point).
        
        Override this method in subclasses to add custom features while
        maintaining compatibility with the standard 16-feature format.
        
        Note: This should return an empty dict in the base implementation
        to maintain the standard 16-feature format. Only override if you
        need additional features for specialized extensions.
        
        Example:
            class EvolutionaryFeatureExtractor(TabularFeatureExtractor):
                def _extract_extension_specific_features(self, game_state):
                    if hasattr(game_state, 'genetic_traits'):
                        return {
                            "fitness_score": game_state.genetic_traits.fitness,
                            "generation": game_state.genetic_traits.generation
                        }
                    return {}
        """
        return {}
    
    def _extract_position_features(self, game_state: GameState) -> Dict[str, int]:
        """Extract absolute position features."""
        head_x, head_y = game_state.head_position
        apple_x, apple_y = game_state.apple_position
        
        return {
            "head_x": head_x,
            "head_y": head_y,
            "apple_x": apple_x,
            "apple_y": apple_y
        }
    
    def _extract_game_state_features(self, game_state: GameState) -> Dict[str, int]:
        """Extract current game state features."""
        return {
            "snake_length": game_state.snake_length
        }
    
    def _extract_apple_direction_features(self, game_state: GameState) -> Dict[str, int]:
        """Extract binary features indicating apple direction relative to head."""
        head_x, head_y = game_state.head_position
        apple_x, apple_y = game_state.apple_position
        
        return {
            "apple_dir_up": 1 if apple_y > head_y else 0,
            "apple_dir_down": 1 if apple_y < head_y else 0,
            "apple_dir_right": 1 if apple_x > head_x else 0,
            "apple_dir_left": 1 if apple_x < head_x else 0
        }
    
    def _extract_danger_features(self, game_state: GameState) -> Dict[str, int]:
        """Extract binary features indicating immediate collision dangers."""
        head_x, head_y = game_state.head_position
        snake_body = set(game_state.snake_positions[1:])  # Exclude head
        grid_size = game_state.grid_size
        
        # Check dangers in each direction relative to current movement
        directions = {
            "UP": (0, 1),
            "DOWN": (0, -1),
            "RIGHT": (1, 0),
            "LEFT": (-1, 0)
        }
        
        current_dir = game_state.current_direction
        
        # Define relative directions based on current movement
        if current_dir == "UP":
            straight = "UP"
            left = "LEFT"
            right = "RIGHT"
        elif current_dir == "DOWN":
            straight = "DOWN"
            left = "RIGHT"
            right = "LEFT"
        elif current_dir == "RIGHT":
            straight = "RIGHT"
            left = "UP"
            right = "DOWN"
        elif current_dir == "LEFT":
            straight = "LEFT"
            left = "DOWN"
            right = "UP"
        else:  # NONE or initial state
            # Default to UP direction
            straight = "UP"
            left = "LEFT"
            right = "RIGHT"
        
        dangers = {}
        for rel_dir, abs_dir in [("straight", straight), ("left", left), ("right", right)]:
            dx, dy = directions[abs_dir]
            next_x, next_y = head_x + dx, head_y + dy
            
            # Check wall collision
            wall_collision = (
                next_x < 0 or next_x >= grid_size or
                next_y < 0 or next_y >= grid_size
            )
            
            # Check body collision
            body_collision = (next_x, next_y) in snake_body
            
            dangers[f"danger_{rel_dir}"] = 1 if (wall_collision or body_collision) else 0
        
        return dangers
    
    def _extract_free_space_features(self, game_state: GameState) -> Dict[str, int]:
        """Extract count features for free space in each direction."""
        head_x, head_y = game_state.head_position
        snake_body = set(game_state.snake_positions)
        grid_size = game_state.grid_size
        
        directions = {
            "up": (0, 1),
            "down": (0, -1),
            "right": (1, 0),
            "left": (-1, 0)
        }
        
        free_space = {}
        
        for direction, (dx, dy) in directions.items():
            count = 0
            x, y = head_x, head_y
            
            # Count free cells in this direction until wall or body
            while True:
                x += dx
                y += dy
                
                # Check bounds
                if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
                    break
                
                # Check body collision
                if (x, y) in snake_body:
                    break
                
                count += 1
            
            free_space[f"free_space_{direction}"] = count
        
        return free_space

# =============================================================================
# CSV Row Creation
# =============================================================================

def create_csv_row(
    game_state: GameState,
    target_move: str,
    game_id: int,
    step_in_game: int
) -> Dict[str, Any]:
    """
    Create a complete CSV row from game state and metadata.
    
    Args:
        game_state: Current game state
        target_move: The move that was taken (UP, DOWN, LEFT, RIGHT)
        game_id: Unique game session identifier  
        step_in_game: Step number within the game
        
    Returns:
        Dictionary with all 19 CSV columns (2 metadata + 16 features + 1 target)
        
    Educational Note:
    This function demonstrates how to combine feature extraction with
    metadata to create complete training examples that can be used
    across different machine learning frameworks.
    """
    from utils.print_utils import print_info
    print_info(f"Creating CSV row for game_id={game_id}, step={step_in_game}", "CSVSchemaUtils")
    # Validate target move
    if target_move not in CSV_VALID_MOVES:
        raise ValueError(f"Invalid target move: {target_move}. Must be one of {CSV_VALID_MOVES}")
    
    # Extract features
    extractor = TabularFeatureExtractor()
    features = extractor.extract_features(game_state)
    
    # Create complete row
    row = {
        # Metadata columns
        "game_id": game_id,
        "step_in_game": step_in_game,
        
        # Feature columns
        **features,
        
        # Target column
        "target_move": target_move
    }
    
    # Validate row has correct number of columns
    if len(row) != len(CSV_EXPECTED_COLUMNS):
        raise ValueError(f"Row has {len(row)} columns, expected {len(CSV_EXPECTED_COLUMNS)}")
    
    return row

# =============================================================================
# CSV Dataset Generation
# =============================================================================

class CSVDatasetGenerator:
    """
    Generates CSV datasets from game session data.
    
    Design Pattern: Builder Pattern
    Purpose: Step-by-step construction of complete datasets
    
    This class handles the conversion from raw game logs to
    standardized CSV datasets that can be used for training.
    
    Educational Note (simple logging):
    This class is designed to be extensible for extensions that need
    specialized dataset generation while maintaining CSV format compatibility.
    
    simple logging Implementation:
    - Base class provides complete CSV generation functionality
    - Pluggable feature extractor allows customization
    - Protected methods enable selective behavior modification
    - Extension-specific generators can inherit and adapt
    """
    
    def __init__(self, feature_extractor: Optional[TabularFeatureExtractor] = None):
        self.extractor = feature_extractor or TabularFeatureExtractor()
        self.rows = []
        self._initialize_generator_settings()
    
    def add_game_session(
        self,
        game_states: List[GameState],
        moves_taken: List[str],
        game_id: int
    ):
        """
        Add a complete game session to the dataset.
        
        Args:
            game_states: List of game states for each step
            moves_taken: List of moves taken at each step
            game_id: Unique identifier for this game
        """
        if len(game_states) != len(moves_taken):
            raise ValueError("Number of game states must match number of moves")
        
        for step, (state, move) in enumerate(zip(game_states, moves_taken)):
            row = create_csv_row(state, move, game_id, step + 1)
            self.rows.append(row)
    
    def generate_dataframe(self) -> pd.DataFrame:
        """
        Generate a pandas DataFrame from collected rows.
        
        Returns:
            DataFrame with standardized column order and types
        """
        if not self.rows:
            raise ValueError("No data rows have been added")
        
        # Create DataFrame with enforced column order
        df = pd.DataFrame(self.rows, columns=CSV_COLUMN_NAMES)
        
        # Ensure correct data types
        df = self._enforce_column_types(df)
        
        return df
    
    def _enforce_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce correct data types for all columns."""
        # Integer columns (most features and metadata)
        int_columns = (
            CSV_METADATA_COLUMNS + 
            list(CSV_POSITION_FEATURES) + 
            list(CSV_BINARY_FEATURES) + 
            list(CSV_COUNT_FEATURES) +
            ["snake_length"]
        )
        
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # String column (target)
        df[CSV_TARGET_COLUMN] = df[CSV_TARGET_COLUMN].astype(str)
        
        return df
    
    def save_to_csv(self, filepath: str, include_index: bool = False):
        """
        Save the dataset to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
            include_index: Whether to include row indices in the CSV
        """
        df = self.generate_dataframe()
        df.to_csv(filepath, index=include_index)
    
    def clear(self):
        """Clear all accumulated rows."""
        self.rows.clear()
    
    def _initialize_generator_settings(self) -> None:
        """
        Initialize generator-specific settings (simple logging Extension Point).
        
        This method can be overridden by subclasses to set up extension-specific
        dataset generation configurations or custom processing parameters.
        
        Example:
            class RLDatasetGenerator(CSVDatasetGenerator):
                def _initialize_generator_settings(self):
                    self.include_reward_data = True
                    self.episode_boundary_markers = True
        """
        pass

# =============================================================================
# CSV Validation
# =============================================================================

class CSVValidator:
    """
    Validates CSV datasets against the standardized schema.
    
    Design Pattern: Validator Pattern
    Purpose: Ensure data quality and format compliance
    
    This validator checks both structural requirements (columns, types)
    and semantic requirements (value ranges, relationships).
    """
    
    def __init__(self):
        self.validation_rules = CSV_VALIDATION_RULES
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate a DataFrame against CSV schema requirements.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check column structure
        errors.extend(self._validate_columns(df))
        
        # Check data types
        errors.extend(self._validate_types(df))
        
        # Check value ranges
        errors.extend(self._validate_values(df))
        
        # Check relationships
        errors.extend(self._validate_relationships(df))
        
        return len(errors) == 0, errors
    
    def _validate_columns(self, df: pd.DataFrame) -> List[str]:
        """Validate column structure."""
        errors = []
        
        # Check column count
        if len(df.columns) != len(CSV_EXPECTED_COLUMNS):
            errors.append(f"Expected {len(CSV_EXPECTED_COLUMNS)} columns, got {len(df.columns)}")
        
        # Check column names
        missing_columns = set(CSV_COLUMN_NAMES) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")
        
        extra_columns = set(df.columns) - set(CSV_COLUMN_NAMES)
        if extra_columns:
            errors.append(f"Extra columns: {extra_columns}")
        
        return errors
    
    def _validate_types(self, df: pd.DataFrame) -> List[str]:
        """Validate data types."""
        errors = []
        
        # Check integer columns
        int_columns = (
            CSV_METADATA_COLUMNS + 
            list(CSV_POSITION_FEATURES) + 
            list(CSV_BINARY_FEATURES) + 
            list(CSV_COUNT_FEATURES) +
            ["snake_length"]
        )
        
        for col in int_columns:
            if col in df.columns and not pd.api.types.is_integer_dtype(df[col]):
                errors.append(f"Column {col} should be integer type")
        
        # Check string column
        if CSV_TARGET_COLUMN in df.columns and not pd.api.types.is_string_dtype(df[CSV_TARGET_COLUMN]):
            errors.append(f"Column {CSV_TARGET_COLUMN} should be string type")
        
        return errors
    
    def _validate_values(self, df: pd.DataFrame) -> List[str]:
        """Validate value ranges."""
        errors = []
        
        # Check binary features (should be 0 or 1)
        for col in CSV_BINARY_FEATURES:
            if col in df.columns:
                invalid_values = df[~df[col].isin([0, 1])][col]
                if not invalid_values.empty:
                    errors.append(f"Column {col} contains non-binary values: {invalid_values.unique()}")
        
        # Check count features (should be non-negative)
        for col in CSV_COUNT_FEATURES:
            if col in df.columns:
                negative_values = df[df[col] < 0][col]
                if not negative_values.empty:
                    errors.append(f"Column {col} contains negative values")
        
        # Check target moves
        if CSV_TARGET_COLUMN in df.columns:
            invalid_moves = df[~df[CSV_TARGET_COLUMN].isin(CSV_VALID_MOVES)][CSV_TARGET_COLUMN]
            if not invalid_moves.empty:
                errors.append(f"Invalid target moves: {invalid_moves.unique()}")
        
        return errors
    
    def _validate_relationships(self, df: pd.DataFrame) -> List[str]:
        """Validate relationships between columns."""
        errors = []
        
        # Check that position coordinates are consistent with grid size
        # (This would require grid_size information, which isn't in the CSV)
        # For now, just check that coordinates are non-negative
        for col in CSV_POSITION_FEATURES:
            if col in df.columns:
                negative_coords = df[df[col] < 0][col]
                if not negative_coords.empty:
                    errors.append(f"Column {col} contains negative coordinates")
        
        return errors

# =============================================================================
# Utility Functions
# =============================================================================

def load_and_validate_csv(filepath: str) -> Tuple[pd.DataFrame, bool, List[str]]:
    """
    Load CSV file and validate it against the schema.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Tuple of (dataframe, is_valid, errors)
    """
    try:
        df = pd.read_csv(filepath)
        validator = CSVValidator()
        is_valid, errors = validator.validate_dataframe(df)
        return df, is_valid, errors
    except Exception as e:
        return pd.DataFrame(), False, [f"Failed to load CSV: {str(e)}"]

def get_feature_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics for all feature columns.
    
    Args:
        df: DataFrame with CSV data
        
    Returns:
        Dictionary with statistics for each feature
    """
    stats = {}
    
    for col in CSV_FEATURE_NAMES:
        if col in df.columns:
            stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "null_count": df[col].isnull().sum(),
                "unique_count": df[col].nunique()
            }
    
    return stats 