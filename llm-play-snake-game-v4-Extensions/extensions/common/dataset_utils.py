"""
Dataset Generation Utilities for Supervised Learning
===================================================

Converts heuristic game data (game_N.json) to training datasets for supervised learning models.
Supports multiple data formats (CSV, NPZ, Parquet) and data structures (tabular, sequential, graph).

This module is designed to be used by all future extensions that need training data,
ensuring consistency across different supervised learning approaches.

Design Pattern: Strategy Pattern
- Different data encoders for different model types
- Pluggable data format writers
- Consistent interface for all dataset generation

Usage:
    from extensions.common.dataset_utils import generate_training_dataset
    
    # Generate CSV dataset for tabular models (XGBoost, LightGBM)
    generate_training_dataset(
        input_dirs=["logs/extensions/heuristics-bfs_20250623_102805"],
        output_path="datasets/tabular_dataset.csv",
        data_structure="tabular",
        data_format="csv"
    )
    
    # Generate NPZ dataset for neural networks
    generate_training_dataset(
        input_dirs=["logs/extensions/heuristics-bfs_20250623_102805"],
        output_path="datasets/sequential_dataset.npz",
        data_structure="sequential",
        data_format="npz"
    )
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import common configuration

# ---------------------
# Data Structure Definitions
# ---------------------

@dataclass
class GameState:
    """
    Represents a single game state for supervised learning.
    
    This is the fundamental unit of training data - each instance represents
    one decision point where the model needs to predict the optimal move.
    """
    # Board representation
    board_state: np.ndarray  # 2D grid representation
    snake_positions: List[Tuple[int, int]]  # Snake body positions
    apple_position: Tuple[int, int]  # Current apple position
    
    # Game context
    score: int  # Current score
    steps: int  # Steps taken so far
    snake_length: int  # Current snake length
    round_number: int  # Current round
    
    # Spatial features
    head_position: Tuple[int, int]  # Snake head position
    head_to_apple_distance: float  # Euclidean distance to apple
    head_to_apple_manhattan: int  # Manhattan distance to apple
    
    # Safety features
    wall_distances: Tuple[int, int, int, int]  # Distance to walls (up, down, left, right)
    body_proximity: List[int]  # Distance to body in each direction
    
    # Target (what we want to predict)
    optimal_move: str  # The move the heuristic algorithm chose
    algorithm: str  # Which algorithm generated this example
    
    # Metadata
    game_id: int  # Which game this state came from
    step_in_game: int  # Which step within the game


# ---------------------
# Data Encoders (Strategy Pattern)
# ---------------------

class DataEncoder(ABC):
    """Abstract base class for different data encoding strategies."""
    
    @abstractmethod
    def encode_game_states(self, game_states: List[GameState]) -> Dict[str, Any]:
        """
        Encode a list of game states into the target data structure.
        
        Args:
            game_states: List of GameState objects
            
        Returns:
            Dictionary containing encoded data ready for the target format
        """
        pass


class TabularEncoder(DataEncoder):
    """
    Encodes game states as tabular data for tree-based models (XGBoost, LightGBM).
    
    Creates a flat feature vector for each game state with engineered features
    optimized for gradient boosting algorithms.
    
    Follows the exact CSV schema specification from the documentation:
    - game_id, step_in_game (metadata, not used as features)
    - head_x, head_y, apple_x, apple_y, snake_length (basic state)
    - apple_dir_up, apple_dir_down, apple_dir_left, apple_dir_right (relative direction)
    - danger_straight, danger_left, danger_right (immediate collision detection)
    - free_space_up, free_space_down, free_space_left, free_space_right (maneuvering space)
    - target_move (supervised learning label)
    """
    
    def encode_game_states(self, game_states: List[GameState]) -> Dict[str, Any]:
        """Encode game states as tabular data with engineered features."""
        features = []
        targets = []
        metadata = []
        
        # Move mapping for classification
        move_to_int = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        
        for state in game_states:
            # Convert GameState to dictionary format for feature extraction
            game_state_dict = {
                "head_position": list(state.head_position),
                "apple_position": list(state.apple_position),
                "snake_positions": [list(pos) for pos in state.snake_positions],
                "current_direction": self._get_current_direction(state),
                "snake_length": state.snake_length,
            }
            
            # Use the exact CSV schema feature extractor
            from .csv_schema import TabularFeatureExtractor
            extractor = TabularFeatureExtractor()
            feature_dict = extractor.extract_features(game_state_dict, 10)  # Assuming 10x10 grid
            
            # Convert to feature vector in the correct order
            feature_vector = [
                feature_dict["head_x"],
                feature_dict["head_y"],
                feature_dict["apple_x"],
                feature_dict["apple_y"],
                feature_dict["snake_length"],
                feature_dict["apple_dir_up"],
                feature_dict["apple_dir_down"],
                feature_dict["apple_dir_left"],
                feature_dict["apple_dir_right"],
                feature_dict["danger_straight"],
                feature_dict["danger_left"],
                feature_dict["danger_right"],
                feature_dict["free_space_up"],
                feature_dict["free_space_down"],
                feature_dict["free_space_left"],
                feature_dict["free_space_right"],
            ]
            
            features.append(feature_vector)
            targets.append(move_to_int[state.optimal_move])
            metadata.append({
                "game_id": state.game_id,
                "step_in_game": state.step_in_game,
                "algorithm": state.algorithm,
                "round_number": state.round_number
            })
        
        return {
            "features": np.array(features, dtype=np.float32),
            "targets": np.array(targets, dtype=np.int32),
            "metadata": metadata,
            "feature_names": self._get_feature_names(),
            "target_names": ["UP", "DOWN", "LEFT", "RIGHT"]
        }
    
    def _get_current_direction(self, state: GameState) -> str:
        """
        Estimate current direction from snake positions.
        
        This is a heuristic since GameState doesn't store current_direction.
        We estimate it from the last two positions in the snake.
        """
        if len(state.snake_positions) < 2:
            return "UP"  # Default direction
        
        head = state.snake_positions[0]
        neck = state.snake_positions[1]
        
        dx = head[0] - neck[0]
        dy = head[1] - neck[1]
        
        if dx == 1:
            return "RIGHT"
        elif dx == -1:
            return "LEFT"
        elif dy == 1:
            return "DOWN"
        elif dy == -1:
            return "UP"
        else:
            return "UP"  # Default
    
    def _get_feature_names(self) -> List[str]:
        """
        Generate feature names for tabular data.
        
        Follows the exact CSV schema specification from the documentation.
        """
        return [
            "head_x", "head_y", "apple_x", "apple_y", "snake_length",
            "apple_dir_up", "apple_dir_down", "apple_dir_left", "apple_dir_right",
            "danger_straight", "danger_left", "danger_right",
            "free_space_up", "free_space_down", "free_space_left", "free_space_right"
        ]


class SequentialEncoder(DataEncoder):
    """
    Encodes game states as sequential data for RNNs and CNNs.
    
    Creates sequences of game states that preserve temporal relationships,
    suitable for models that can learn from game progression patterns.
    """
    
    def __init__(self, sequence_length: int = 10):
        """
        Initialize sequential encoder.
        
        Args:
            sequence_length: Number of consecutive states to include in each sequence
        """
        self.sequence_length = sequence_length
    
    def encode_game_states(self, game_states: List[GameState]) -> Dict[str, Any]:
        """Encode game states as sequential data."""
        # Group states by game
        games = {}
        for state in game_states:
            if state.game_id not in games:
                games[state.game_id] = []
            games[state.game_id].append(state)
        
        # Sort each game by step
        for game_id in games:
            games[game_id].sort(key=lambda x: x.step_in_game)
        
        sequences = []
        targets = []
        metadata = []
        
        move_to_int = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        
        # Create sequences
        for game_id, states in games.items():
            for i in range(len(states) - self.sequence_length + 1):
                sequence = states[i:i + self.sequence_length]
                
                # Encode each state in the sequence
                sequence_data = []
                for state in sequence:
                    state_features = [
                        state.score / 100.0,  # Normalize score
                        state.steps / 1000.0,  # Normalize steps
                        state.snake_length / 100.0,  # Normalize length
                        state.head_position[0] / 10.0,  # Normalize coordinates
                        state.head_position[1] / 10.0,
                        state.apple_position[0] / 10.0,
                        state.apple_position[1] / 10.0,
                        state.head_to_apple_distance / 14.14,  # Normalize distance (max diagonal)
                        state.head_to_apple_manhattan / 18.0,  # Normalize Manhattan distance
                    ]
                    
                    # Add board state as flattened and normalized
                    board_flat = state.board_state.flatten() / 3.0  # Normalize board values
                    state_features.extend(board_flat)
                    
                    sequence_data.append(state_features)
                
                sequences.append(sequence_data)
                targets.append(move_to_int[sequence[-1].optimal_move])  # Predict last move
                metadata.append({
                    "game_id": game_id,
                    "sequence_start": sequence[0].step_in_game,
                    "sequence_end": sequence[-1].step_in_game,
                    "algorithm": sequence[-1].algorithm
                })
        
        return {
            "sequences": np.array(sequences, dtype=np.float32),
            "targets": np.array(targets, dtype=np.int32),
            "metadata": metadata,
            "sequence_length": self.sequence_length,
            "feature_dim": len(sequences[0][0]) if sequences else 0,
            "target_names": ["UP", "DOWN", "LEFT", "RIGHT"]
        }


class GraphEncoder(DataEncoder):
    """
    Encodes game states as graph data for Graph Neural Networks.
    
    Represents the game board as a graph where each cell is a node,
    and edges represent spatial relationships and game dynamics.
    """
    
    def encode_game_states(self, game_states: List[GameState]) -> Dict[str, Any]:
        """Encode game states as graph data."""
        graphs = []
        targets = []
        metadata = []
        
        move_to_int = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        
        for state in game_states:
            # Create node features (one node per board cell)
            grid_size = state.board_state.shape[0]
            num_nodes = grid_size * grid_size
            
            node_features = []
            for i in range(grid_size):
                for j in range(grid_size):
                    # Node features: [is_snake, is_head, is_apple, is_empty, distance_to_apple]
                    is_snake = 1 if state.board_state[i, j] == 1 else 0
                    is_head = 1 if (i, j) == state.head_position else 0
                    is_apple = 1 if (i, j) == state.apple_position else 0
                    is_empty = 1 if state.board_state[i, j] == 0 else 0
                    
                    # Distance to apple
                    dist_to_apple = np.sqrt((i - state.apple_position[0])**2 + (j - state.apple_position[1])**2)
                    
                    node_features.append([is_snake, is_head, is_apple, is_empty, dist_to_apple / 14.14])
            
            # Create edge indices (4-connected grid)
            edge_indices = []
            for i in range(grid_size):
                for j in range(grid_size):
                    node_id = i * grid_size + j
                    
                    # Connect to neighbors
                    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            neighbor_id = ni * grid_size + nj
                            edge_indices.append([node_id, neighbor_id])
            
            graphs.append({
                "node_features": np.array(node_features, dtype=np.float32),
                "edge_indices": np.array(edge_indices, dtype=np.int64).T,  # Shape: [2, num_edges]
                "num_nodes": num_nodes,
                "grid_size": grid_size
            })
            
            targets.append(move_to_int[state.optimal_move])
            metadata.append({
                "game_id": state.game_id,
                "step_in_game": state.step_in_game,
                "algorithm": state.algorithm,
                "round_number": state.round_number
            })
        
        return {
            "graphs": graphs,
            "targets": np.array(targets, dtype=np.int32),
            "metadata": metadata,
            "target_names": ["UP", "DOWN", "LEFT", "RIGHT"]
        }


# ---------------------
# Data Format Writers (Strategy Pattern)
# ---------------------

class DataWriter(ABC):
    """Abstract base class for different data format writers."""
    
    @abstractmethod
    def write_dataset(self, encoded_data: Dict[str, Any], output_path: str) -> None:
        """
        Write encoded data to the specified format.
        
        Args:
            encoded_data: Dictionary containing encoded data
            output_path: Path where to save the dataset
        """
        pass


class CSVWriter(DataWriter):
    """
    Writes tabular data to CSV format.
    
    Follows the exact CSV schema specification from the documentation:
    - game_id, step_in_game (metadata, not used as features)
    - head_x, head_y, apple_x, apple_y, snake_length (basic state)
    - apple_dir_up, apple_dir_down, apple_dir_left, apple_dir_right (relative direction)
    - danger_straight, danger_left, danger_right (immediate collision detection)
    - free_space_up, free_space_down, free_space_left, free_space_right (maneuvering space)
    - target_move (supervised learning label)
    """
    
    def write_dataset(self, encoded_data: Dict[str, Any], output_path: str) -> None:
        """Write tabular data to CSV following the exact schema specification."""
        features = encoded_data["features"]
        targets = encoded_data["targets"]
        metadata = encoded_data["metadata"]
        feature_names = encoded_data["feature_names"]
        
        # Create DataFrame with exact schema specification
        df_data = {}
        
        # Add metadata columns first (exact schema order)
        df_data["game_id"] = [meta["game_id"] for meta in metadata]
        df_data["step_in_game"] = [meta["step_in_game"] for meta in metadata]
        
        # Add feature columns in exact schema order
        for i, name in enumerate(feature_names):
            df_data[name] = features[:, i]
        
        # Add target column last (exact schema order)
        df_data["target_move"] = targets
        
        # Convert target integers back to strings
        target_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        df_data["target_move"] = [target_names[target] for target in targets]
        
        # Create DataFrame with exact column order
        from .csv_schema import generate_csv_schema
        schema = generate_csv_schema(10)  # Assuming 10x10 grid
        column_order = schema.get_column_names()
        
        df = pd.DataFrame(df_data)
        df = df[column_order]  # Ensure correct column order
        
        df.to_csv(output_path, index=False)
        
        print(f"✅ CSV dataset saved to: {output_path}")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")


class NPZWriter(DataWriter):
    """Writes data to NumPy NPZ format."""
    
    def write_dataset(self, encoded_data: Dict[str, Any], output_path: str) -> None:
        """Write data to NPZ format."""
        # Prepare data for saving
        save_dict = {}
        
        for key, value in encoded_data.items():
            if key == "metadata":
                # Convert metadata to structured array
                if value:
                    metadata_arrays = {}
                    for meta_key in value[0].keys():
                        metadata_arrays[f"metadata_{meta_key}"] = [meta[meta_key] for meta in value]
                    save_dict.update(metadata_arrays)
            elif isinstance(value, (list, tuple)):
                save_dict[key] = np.array(value)
            else:
                save_dict[key] = value
        
        np.savez_compressed(output_path, **save_dict)
        
        print(f"✅ NPZ dataset saved to: {output_path}")
        print(f"📊 Keys: {list(save_dict.keys())}")


class ParquetWriter(DataWriter):
    """Writes data to Parquet format for efficient storage and loading."""
    
    def write_dataset(self, encoded_data: Dict[str, Any], output_path: str) -> None:
        """Write data to Parquet format."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for Parquet format. Install with: pip install pyarrow")
        
        if "features" in encoded_data:
            # Tabular data
            features = encoded_data["features"]
            targets = encoded_data["targets"]
            metadata = encoded_data["metadata"]
            feature_names = encoded_data["feature_names"]
            
            # Create DataFrame
            df_data = {}
            
            for i, name in enumerate(feature_names):
                df_data[name] = features[:, i]
            
            df_data["target_move"] = targets
            
            for key in ["game_id", "step_in_game", "algorithm", "round_number"]:
                df_data[key] = [meta[key] for meta in metadata]
            
            df = pd.DataFrame(df_data)
            df.to_parquet(output_path, index=False)
            
        else:
            # For non-tabular data, save as multiple arrays
            table_dict = {}
            for key, value in encoded_data.items():
                if key != "metadata" and isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        table_dict[key] = value
                    else:
                        # Flatten multi-dimensional arrays
                        table_dict[key] = value.flatten()
            
            table = pa.table(table_dict)
            pq.write_table(table, output_path)
        
        print(f"✅ Parquet dataset saved to: {output_path}")


# ---------------------
# Main Dataset Generation Function
# ---------------------

def extract_game_states_from_json(json_file: str, grid_size: int = 10) -> List[GameState]:
    """
    Extract GameState objects from a heuristic game JSON file.
    
    Args:
        json_file: Path to the game JSON file
        grid_size: Size of the game grid (default: 10)
        
    Returns:
        List of GameState objects representing decision points in the game
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        game_data = json.load(f)
    
    game_states = []
    algorithm = game_data.get("algorithm", "unknown")
    game_id = int(Path(json_file).stem.split('_')[1])  # Extract game number
    
    # Get game history
    moves = game_data["detailed_history"]["moves"]
    apple_positions = game_data["detailed_history"]["apple_positions"]
    rounds_data = game_data["detailed_history"]["rounds_data"]
    
    # Initialize game state
    snake_positions = [(5, 5), (4, 5), (3, 5)]  # Initial snake position
    current_apple_idx = 0
    
    for step, move in enumerate(moves):
        # Create board representation
        board = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        # Mark snake positions
        for i, pos in enumerate(snake_positions):
            if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size:
                board[pos[0], pos[1]] = 2 if i == 0 else 1  # Head=2, Body=1
        
        # Mark apple position
        if current_apple_idx < len(apple_positions):
            apple_pos = apple_positions[current_apple_idx]
            apple_x, apple_y = int(apple_pos["x"]), int(apple_pos["y"])
            if 0 <= apple_x < grid_size and 0 <= apple_y < grid_size:
                board[apple_x, apple_y] = 3  # Apple=3
        else:
            apple_x, apple_y = 0, 0
        
        # Calculate features
        head_pos = snake_positions[0]
        apple_distance = np.sqrt((head_pos[0] - apple_x)**2 + (head_pos[1] - apple_y)**2)
        manhattan_distance = abs(head_pos[0] - apple_x) + abs(head_pos[1] - apple_y)
        
        # Wall distances
        wall_distances = (
            grid_size - 1 - head_pos[1],  # Distance to top
            head_pos[1],                   # Distance to bottom
            head_pos[0],                   # Distance to left
            grid_size - 1 - head_pos[0]   # Distance to right
        )
        
        # Body proximity (simplified)
        body_proximity = [grid_size] * 4  # Default to max distance
        for i, (dx, dy) in enumerate([(0, 1), (0, -1), (-1, 0), (1, 0)]):
            check_x, check_y = head_pos[0] + dx, head_pos[1] + dy
            if (check_x, check_y) in snake_positions[1:]:
                body_proximity[i] = 1
        
        # Find corresponding round
        round_number = step + 2  # Rounds start from 2 in the data
        
        game_state = GameState(
            board_state=board,
            snake_positions=snake_positions.copy(),
            apple_position=(apple_x, apple_y),
            score=len(snake_positions) - 3,  # Score is snake length - initial length
            steps=step + 1,
            snake_length=len(snake_positions),
            round_number=round_number,
            head_position=head_pos,
            head_to_apple_distance=apple_distance,
            head_to_apple_manhattan=manhattan_distance,
            wall_distances=wall_distances,
            body_proximity=body_proximity,
            optimal_move=move,
            algorithm=algorithm,
            game_id=game_id,
            step_in_game=step
        )
        
        game_states.append(game_state)
        
        # Update snake position based on move
        dx, dy = {"UP": (0, 1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[move]
        new_head = (head_pos[0] + dx, head_pos[1] + dy)
        
        # Check if apple was eaten
        apple_eaten = new_head == (apple_x, apple_y)
        
        # Update snake
        snake_positions.insert(0, new_head)
        if not apple_eaten:
            snake_positions.pop()  # Remove tail if no apple eaten
        else:
            current_apple_idx += 1  # Move to next apple
    
    return game_states


def generate_training_dataset(
    input_dirs: List[str],
    output_path: str,
    data_structure: str = "tabular",
    data_format: str = "csv",
    grid_size: int = 10,
    **kwargs
) -> None:
    """
    Generate training dataset from heuristic game logs.
    
    Args:
        input_dirs: List of directories containing game JSON files
        output_path: Path where to save the generated dataset
        data_structure: Type of data structure ("tabular", "sequential", "graph")
        data_format: Output format ("csv", "npz", "parquet")
        grid_size: Size of the game grid (default: 10)
        **kwargs: Additional arguments for encoders (e.g., sequence_length for sequential)
    """
    print(f"🚀 Generating {data_structure} dataset in {data_format} format...")
    
    # Collect all game states
    all_game_states = []
    
    for input_dir in input_dirs:
        dir_path = Path(input_dir)
        if not dir_path.exists():
            print(f"⚠️  Directory not found: {input_dir}")
            continue
        
        print(f"📁 Processing directory: {input_dir}")
        
        # Find all game JSON files
        game_files = list(dir_path.glob("game_*.json"))
        
        for game_file in game_files:
            try:
                game_states = extract_game_states_from_json(str(game_file), grid_size)
                all_game_states.extend(game_states)
                print(f"✅ Processed {game_file.name}: {len(game_states)} states")
            except Exception as e:
                print(f"❌ Error processing {game_file}: {e}")
    
    if not all_game_states:
        print("❌ No game states found!")
        return
    
    print(f"📊 Total game states collected: {len(all_game_states)}")
    
    # Choose encoder based on data structure
    if data_structure == "tabular":
        encoder = TabularEncoder()
    elif data_structure == "sequential":
        sequence_length = kwargs.get("sequence_length", 10)
        encoder = SequentialEncoder(sequence_length=sequence_length)
    elif data_structure == "graph":
        encoder = GraphEncoder()
    else:
        raise ValueError(f"Unknown data structure: {data_structure}")
    
    # Encode data
    print(f"🔄 Encoding data with {encoder.__class__.__name__}...")
    encoded_data = encoder.encode_game_states(all_game_states)
    
    # Choose writer based on data format
    if data_format == "csv":
        writer = CSVWriter()
    elif data_format == "npz":
        writer = NPZWriter()
    elif data_format == "parquet":
        writer = ParquetWriter()
    else:
        raise ValueError(f"Unknown data format: {data_format}")
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write dataset
    print(f"💾 Writing dataset with {writer.__class__.__name__}...")
    writer.write_dataset(encoded_data, str(output_path))
    
    print("🎉 Dataset generation completed!")
    print(f"📄 Output: {output_path}")


# ---------------------
# Export Configuration
# ---------------------

__all__ = [
    "GameState",
    "DataEncoder",
    "TabularEncoder", 
    "SequentialEncoder",
    "GraphEncoder",
    "DataWriter",
    "CSVWriter",
    "NPZWriter",
    "ParquetWriter",
    "extract_game_states_from_json",
    "generate_training_dataset",
] 