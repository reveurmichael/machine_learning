"""Path utilities for extensions/common/utils/ package.

This module provides path management utilities for the extensions package,
serving as a thin façade to the canonical implementation in the root utils
package to maintain the Single Source of Truth principle.

The public API remains unchanged for backwards compatibility - extensions can
continue to import these functions but the actual logic is single-sourced in
the root utils package.

Key Functions:
    ensure_project_root_on_path: Ensure repository root is in sys.path
    setup_extension_paths: Simple wrapper for path setup
    get_extension_path: Get standardized extension directory paths
    get_dataset_path: Get standardized dataset directory paths
    get_model_path: Get standardized model directory paths
    ensure_extension_directories: Create extension directories if needed
    validate_path_structure: Validate extension path compliance
"""

from __future__ import annotations

# NOTE: The heavy-lifting lives in the root utils module. Importing at runtime
# instead of at import-time avoids cyclic dependencies if extensions are used
# by tooling before the root package is fully initialised (e.g. during
# editable installs or unit-test collection).

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:  # pragma: no cover – only for static type checkers
    from pathlib import Path


def ensure_project_root_on_path() -> "Path":  # noqa: D401 – keep the historical name
    """Ensure repository root is in sys.path and return it.

    This is a thin alias around utils.path_utils.ensure_project_root to
    provide simple path utilities for existing extensions while enforcing the
    single source of truth policy.
    """
    from utils.path_utils import ensure_project_root as _ensure_root  # local import to avoid cycles
    return _ensure_root()


def setup_extension_paths() -> None:  # noqa: D401 – historical name retained
    """Simple wrapper that calls ensure_project_root_on_path."""
    ensure_project_root_on_path()


def get_extension_path(extension_file: str) -> Path:
    """
    Get the path to an extension directory from an extension file.
    
    Args:
        extension_file: Usually __file__ from the calling extension
        
    Returns:
        Path to the extension directory
        
    Example:
        >>> # From within extensions/heuristics-v0.03/main.py
        >>> extension_path = get_extension_path(__file__)
        >>> # Returns: Path("extensions/heuristics-v0.03")
    """
    return Path(extension_file).parent


def get_dataset_path(
    extension_type: str,
    version: str,
    grid_size: int,
    algorithm: str = "",
    timestamp: str = ""
) -> Path:
    """
    Get standardized dataset path following the grid-size hierarchy.
    
    Args:
        extension_type: Type of extension (heuristics, supervised, etc.)
        version: Version string (e.g., "0.04")
        grid_size: Grid size for the dataset
        algorithm: Algorithm name (optional)
        timestamp: Timestamp string (optional)
        
    Returns:
        Standardized dataset path
        
    Example:
        >>> path = get_dataset_path("heuristics", "0.04", 10, "bfs", "20240101_120000")
        >>> # Returns: logs/extensions/datasets/grid-size-10/heuristics_v0.04_20240101_120000/bfs/
    """
    ensure_project_root_on_path()
    
    # Base dataset directory
    base_path = Path("logs/extensions/datasets")
    
    # Grid-size hierarchy
    grid_path = base_path / f"grid-size-{grid_size}"
    
    # Extension-specific directory
    if timestamp:
        extension_dir = f"{extension_type}_v{version}_{timestamp}"
    else:
        extension_dir = f"{extension_type}_v{version}"
    
    dataset_path = grid_path / extension_dir
    
    # Add algorithm subdirectory if specified
    if algorithm:
        dataset_path = dataset_path / algorithm
    
    return dataset_path


def get_model_path(
    extension_type: str,
    version: str,
    grid_size: int,
    model_name: str = "",
    timestamp: str = ""
) -> Path:
    """
    Get standardized model path following the grid-size hierarchy.
    
    Args:
        extension_type: Type of extension (supervised, reinforcement, etc.)
        version: Version string (e.g., "0.02")
        grid_size: Grid size for the model
        model_name: Model name (optional)
        timestamp: Timestamp string (optional)
        
    Returns:
        Standardized model path
        
    Example:
        >>> path = get_model_path("supervised", "0.02", 10, "mlp", "20240101_120000")
        >>> # Returns: logs/extensions/models/grid-size-10/supervised_v0.02_20240101_120000/mlp/
    """
    ensure_project_root_on_path()
    
    # Base model directory
    base_path = Path("logs/extensions/models")
    
    # Grid-size hierarchy
    grid_path = base_path / f"grid-size-{grid_size}"
    
    # Extension-specific directory
    if timestamp:
        extension_dir = f"{extension_type}_v{version}_{timestamp}"
    else:
        extension_dir = f"{extension_type}_v{version}"
    
    model_path = grid_path / extension_dir
    
    # Add model subdirectory if specified
    if model_name:
        model_path = model_path / model_name
    
    return model_path


def ensure_extension_directories(path: Path) -> Path:
    """
    Ensure extension directories exist, creating them if necessary.
    
    Args:
        path: Path to create directories for
        
    Returns:
        The created path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_path_structure(extension_path: Path) -> bool:
    """
    Validate that an extension path follows the standardized structure.
    
    Args:
        extension_path: Path to validate
        
    Returns:
        True if path structure is valid
        
    Raises:
        ValueError: If path structure is invalid
    """
    # Basic validation - extension should be in extensions/ directory
    if "extensions" not in str(extension_path):
        raise ValueError(f"Extension path should be in extensions/ directory: {extension_path}")
    
    # Check that extension follows naming convention
    extension_name = extension_path.name
    if not any(char.isdigit() for char in extension_name):
        raise ValueError(f"Extension should have version number: {extension_name}")
    
    return True


# ---------------------
# Public re-exports – keep the historical interface intact
# ---------------------

__all__ = [
    "ensure_project_root_on_path",
    "setup_extension_paths",
    "get_extension_path",
    "get_dataset_path", 
    "get_model_path",
    "ensure_extension_directories",
    "validate_path_structure",
]

# Ensure side-effects (cwd + sys.path) are applied as soon as the module is
# imported – this mimics the historical behaviour so that existing extensions
# continue to work without modification.

ensure_project_root_on_path() 