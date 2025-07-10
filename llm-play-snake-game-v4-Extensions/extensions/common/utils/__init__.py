import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

"""
Common utilities for extensions.

This package provides shared utilities that can be used across different
extension types (heuristics, supervised, reinforcement learning, etc.).

Design Philosophy:
- Forward-looking: No legacy compatibility, clean and self-contained
- Reusable across extensions
- Single responsibility per module
- Clear separation of concerns
"""

# Re-export other utilities
from .path_utils import get_datasets_root, get_dataset_path, get_model_path
from .csv_utils import CSVFeatureExtractor, create_csv_record
from .dataset_utils import save_csv_dataset, save_jsonl_dataset
from .game_state_utils import convert_coordinates_to_tuples
from utils.print_utils import print_info

__all__ = [
    # Other utilities
    "get_datasets_root",
    "get_dataset_path", 
    "get_model_path",

    "CSVFeatureExtractor",
    "create_csv_record", 
    "save_csv_dataset",
    "save_jsonl_dataset",
    "convert_coordinates_to_tuples",
]

# Dataset utilities
from .dataset_utils import (
    load_csv_dataset,
    load_jsonl_dataset,
    load_npz_dataset,
    save_npz_dataset,
    get_dataset_info,
    DatasetIO,
    dataset_io,
    DatasetLoader
)

# CSV utilities
from .csv_utils import (
    create_csv_record_with_explanation,
    GameStateForCSV
)

# Simple utility functions following forward-looking architecture principles
def print_extension_info(extension_name: str, version: str):
    """Simple extension information logging."""
    print_info(f"[CommonUtils] Extension: {extension_name} v{version}")

def get_common_config():
    """Simple common configuration access."""
    print_info("[CommonUtils] Accessing common configuration")
    return {
        "default_grid_size": 10,
        "max_grid_size": 50,
        "min_grid_size": 5
    }

# Additional utility functions referenced in documentation
def load_dataset_for_training(dataset_path: str, format_type: str = "csv"):
    """Load dataset for training using appropriate format."""
    print_info(f"[CommonUtils] Loading dataset: {dataset_path} (format: {format_type})")
    
    if format_type.lower() == "csv":
        return load_csv_dataset(dataset_path)
    elif format_type.lower() == "jsonl": 
        return load_jsonl_dataset(dataset_path)
    elif format_type.lower() == "npz":
        return load_npz_dataset(dataset_path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def save_dataset_standardized(data, dataset_path: str, format_type: str = "csv"):
    """Save dataset using standardized format."""
    print_info(f"[CommonUtils] Saving dataset: {dataset_path} (format: {format_type})")
    
    if format_type.lower() == "csv":
        save_csv_dataset(data, dataset_path)
    elif format_type.lower() == "jsonl":
        save_jsonl_dataset(data, dataset_path)  
    elif format_type.lower() == "npz":
        save_npz_dataset(data, dataset_path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def extract_features_from_game_state(game_state, feature_extractor=None):
    """Extract features from game state using specified extractor."""
    print_info("[CommonUtils] Extracting features from game state")
    
    if feature_extractor is None:
        feature_extractor = CSVFeatureExtractor()
    
    features = feature_extractor.extract_features(game_state, "UNKNOWN", 0)
    print_info(f"[CommonUtils] Extracted {len(features)} features")
    return features

def generate_csv_schema(grid_size: int = 10):
    """Generate CSV schema for specified grid size."""
    print_info(f"[CommonUtils] Generating CSV schema for grid size {grid_size}")
    
    from ..config.csv_formats import CSV_ALL_COLUMNS
    return CSV_ALL_COLUMNS.copy()

def validate_csv_schema(df, expected_columns=None):
    """Simple CSV schema validation."""
    print_info("[CommonUtils] Validating CSV schema")
    
    if expected_columns is None:
        from ..config.csv_formats import CSV_ALL_COLUMNS
        expected_columns = CSV_ALL_COLUMNS
    
    missing = set(expected_columns) - set(df.columns)
    if missing:
        print_info(f"[CommonUtils] Missing columns: {missing}")
        return False
    
    print_info("[CommonUtils] CSV schema validation passed")
    return True

# Dataset generation CLI utilities removed in modular refactor (v0.04) 