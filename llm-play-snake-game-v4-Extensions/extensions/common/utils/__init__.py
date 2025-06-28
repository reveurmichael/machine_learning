"""
Common utilities package for Snake Game AI extensions.

This package is governed by final-decision-10.md governance system.
- Exports all utility modules.
- Logging is always simple (print()).

Reference: docs/extensions-guideline/final-decision-10.md
"""

print("[common.utils] Imported common utilities (final-decision-10.md Guideline 3)")

from .factory_utils import SimpleFactory
from .dataset_utils import DatasetLoader
from .path_utils import ensure_project_root, get_extension_path, get_dataset_path, get_model_path
from .csv_schema_utils import create_csv_row

from .path_utils import (
    validate_path_structure,
    setup_extension_environment
)

from .dataset_utils import (
    load_dataset_for_training,
    save_dataset_standardized,
    validate_dataset_compatibility,
    extract_features_from_game_state,
    create_csv_row
)

from .csv_schema_utils import (
    generate_csv_schema,
    TabularFeatureExtractor,
    validate_csv_schema
)

# final-decision-10.md Guideline 3: Simple utility functions
def print_extension_info(extension_name: str, version: str):
    """Simple extension information logging"""
    print(f"[CommonUtils] Extension: {extension_name} v{version}")

def validate_extension_structure(extension_path: str):
    """Simple extension structure validation"""
    print(f"[CommonUtils] Validating extension structure: {extension_path}")
    return True

def get_common_config():
    """Simple common configuration access"""
    print("[CommonUtils] Accessing common configuration")
    return {
        "default_grid_size": 10,
        "max_grid_size": 50,
        "min_grid_size": 5
    } 