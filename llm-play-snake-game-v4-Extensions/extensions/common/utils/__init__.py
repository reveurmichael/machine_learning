"""
Common utilities for extensions.

This package provides shared utilities that can be used across different
extension types (heuristics, supervised, reinforcement learning, etc.).

Design Philosophy:
- Reusable across extensions
- Single responsibility per module
- Clear separation of concerns
"""

# Re-export core dataset generation classes for backward compatibility
from .dataset_generator_core import DatasetGenerator
from .dataset_game_runner import run_heuristic_games, load_game_logs

# Re-export CLI functions
from .dataset_generator_cli import create_argument_parser, find_available_algorithms, main

# Re-export other utilities
from .path_utils import setup_extension_paths
from .csv_schema_utils import CSVValidator, TabularFeatureExtractor
from .dataset_utils import save_csv_dataset, save_jsonl_dataset
from utils.print_utils import print_info

__all__ = [
    # Core dataset generation
    "DatasetGenerator",
    "run_heuristic_games", 
    "load_game_logs",
    
    # CLI functions
    "create_argument_parser",
    "find_available_algorithms", 
    "main",
    
    # Other utilities
    "setup_extension_paths",
    "CSVValidator",
    "TabularFeatureExtractor", 
    "save_csv_dataset",
    "save_jsonl_dataset",
]

print_info("[common.utils] Imported common utilities (final-decision-10.md Guideline 3)")

# Core factory pattern utilities
from .factory_utils import SimpleFactory

# Path management utilities  
from .path_utils import (
    ensure_project_root,
    get_extension_path, 
    get_dataset_path,
    get_model_path,
    validate_path_structure,
    setup_extension_environment,
    ensure_project_root_on_path,
    PathManager,
    path_manager
)

# Dataset utilities
from .dataset_utils import (
    load_csv_dataset,
    save_csv_dataset,
    load_jsonl_dataset,
    save_jsonl_dataset,
    load_npz_dataset,
    save_npz_dataset,
    get_dataset_info,
    DatasetIO,
    dataset_io,
    DatasetLoader
)

# CSV schema utilities
from .csv_schema_utils import (
    create_csv_row,
    TabularFeatureExtractor,
    CSVDatasetGenerator,
    CSVValidator,
    load_and_validate_csv,
    get_feature_statistics
)

# Simple utility functions following SUPREME_RULES from final-decision-10.md
def print_extension_info(extension_name: str, version: str):
    """Simple extension information logging (SUPREME_RULES compliant)."""
    print_info(f"[CommonUtils] Extension: {extension_name} v{version}")

def get_common_config():
    """Simple common configuration access (SUPREME_RULES compliant)."""
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

def validate_dataset_compatibility(dataset_path: str, expected_format: str) -> bool:
    """Simple dataset compatibility validation."""
    print_info(f"[CommonUtils] Validating dataset compatibility: {dataset_path}")
    
    # Basic validation - check file extension matches expected format
    from pathlib import Path
    path = Path(dataset_path)
    
    if not path.exists():
        print_info(f"[CommonUtils] Dataset file not found: {dataset_path}")
        return False
    
    expected_extensions = {
        "csv": ".csv",
        "jsonl": ".jsonl", 
        "npz": ".npz"
    }
    
    expected_ext = expected_extensions.get(expected_format.lower())
    if expected_ext and path.suffix.lower() != expected_ext:
        print_info(f"[CommonUtils] Extension mismatch: expected {expected_ext}, got {path.suffix}")
        return False
    
    print_info("[CommonUtils] Dataset compatibility validated")
    return True

def extract_features_from_game_state(game_state, feature_extractor=None):
    """Extract features from game state using specified extractor."""
    print_info("[CommonUtils] Extracting features from game state")
    
    if feature_extractor is None:
        feature_extractor = TabularFeatureExtractor()
    
    features = feature_extractor.extract_features(game_state)
    print_info(f"[CommonUtils] Extracted {len(features)} features")
    return features

def generate_csv_schema(grid_size: int = 10):
    """Generate CSV schema for specified grid size."""
    print_info(f"[CommonUtils] Generating CSV schema for grid size {grid_size}")
    
    from ..config.dataset_formats import CSV_BASIC_COLUMNS
    return CSV_BASIC_COLUMNS.copy()

def validate_csv_schema(df, expected_columns=None):
    """Simple CSV schema validation."""
    print_info("[CommonUtils] Validating CSV schema")
    
    if expected_columns is None:
        from ..config.dataset_formats import CSV_BASIC_COLUMNS
        expected_columns = CSV_BASIC_COLUMNS
    
    missing = set(expected_columns) - set(df.columns)
    if missing:
        print_info(f"[CommonUtils] Missing columns: {missing}")
        return False
    
    print_info("[CommonUtils] CSV schema validation passed")
    return True

# Dataset generation CLI utilities removed in modular refactor (v0.04) 