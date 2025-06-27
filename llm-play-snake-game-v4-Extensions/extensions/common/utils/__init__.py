"""
Utility helpers shared by *all* extensions
=========================================

The modules exposed here are intentionally lightweight – each one should be
small enough to understand at a glance and generic enough to work for almost
any extension without modification.  If your extension has specialised needs,
feel free to extend or wrap these helpers inside your own package rather than
changing the common ones.
"""

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
from .dataset_utils import (
    load_csv_dataset,
    load_jsonl_dataset,
    load_npz_dataset,
    save_csv_dataset,
    save_jsonl_dataset,
    save_npz_dataset,
    get_dataset_info,
    guess_dataset_format,
)

# ---------------------------------------------------------------------------
# CSV schema helpers – these are a bit more advanced but still generic.
# ---------------------------------------------------------------------------
from .csv_schema_utils import (
    TabularFeatureExtractor,
    CSVDatasetGenerator,
    CSVValidator,
    load_and_validate_csv,
)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
from .path_utils import (
    ensure_project_root_on_path,
    setup_extension_paths,
    get_extension_path,
    get_dataset_path,
    get_model_path,
    ensure_extension_directories,
    validate_path_structure,
)

__all__ = [
    # dataset
    "load_csv_dataset",
    "load_jsonl_dataset",
    "load_npz_dataset",
    "save_csv_dataset",
    "save_jsonl_dataset",
    "save_npz_dataset",
    "get_dataset_info",
    "guess_dataset_format",

    # csv schema
    "TabularFeatureExtractor",
    "CSVDatasetGenerator",
    "CSVValidator",
    "load_and_validate_csv",

    # path utils
    "ensure_project_root_on_path",
    "setup_extension_paths",
    "get_extension_path",
    "get_dataset_path",
    "get_model_path",
    "ensure_extension_directories",
    "validate_path_structure",
] 