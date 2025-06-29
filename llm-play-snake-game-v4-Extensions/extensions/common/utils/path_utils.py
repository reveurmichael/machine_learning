"""Path utilities for extensions/common/utils/ package.

This file follows the principles from final-decision-10.md:
- All utilities must use simple print logging
- All utilities must be OOP, extensible, and never over-engineered
- Reference: SimpleFactory in factory_utils.py is the canonical factory pattern for all extensions
- See also: agents.md, core.md, config.md, factory-design-pattern.md, extension-evolution-rules.md

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

Design Philosophy:
- Simple, object-oriented utilities that can be inherited and extended
- No tight coupling with ML/DL/RL/LLM-specific concepts
- Simple logging with print() statements
- Enables easy addition of new extensions without friction
- All code examples use print() and create() as canonical patterns.
"""

from __future__ import annotations

# NOTE: The heavy-lifting lives in the root utils module. Importing at runtime
# instead of at import-time avoids cyclic dependencies if extensions are used
# by tooling before the root package is fully initialised (e.g. during
# editable installs or unit-test collection).

from typing import TYPE_CHECKING
from pathlib import Path
import os
import sys

if TYPE_CHECKING:  # pragma: no cover – only for static type checkers
    from pathlib import Path


def ensure_project_root() -> Path:
    """
    Ensure the working directory is the project root.
    Returns:
        Path to the project root.
    """
    root = Path(__file__).parent.parent.parent.parent.resolve()
    os.chdir(root)
    print(f"[PathUtils] Changed working directory to project root: {root}")  # Simple logging
    return root


def setup_extension_paths() -> None:  # noqa: D401 – historical name retained
    """Simple wrapper that calls ensure_project_root_on_path.
    
    Simple logging with print() statements.
    """
    print("[PathUtils] Setting up extension paths...")  # Simple logging
    ensure_project_root()
    print("[PathUtils] Extension paths setup complete")  # Simple logging


def get_extension_path(file: str) -> Path:
    """
    Get the path to the extension directory containing the given file.
    Args:
        file: __file__ from the caller.
    Returns:
        Path to the extension directory.
    """
    path = Path(file).parent.resolve()
    print(f"[PathUtils] Extension path: {path}")  # Simple logging
    return path


def get_dataset_path(extension_type: str, version: str, grid_size: int, algorithm: str, timestamp: str) -> Path:
    """
    Generate standardized dataset path.
    Args:
        extension_type: Type of extension (e.g., 'heuristics').
        version: Version string (e.g., '0.04').
        grid_size: Grid size (int).
        algorithm: Algorithm name.
        timestamp: Timestamp string.
    Returns:
        Path to the dataset directory.
    """
    path = Path(f"logs/extensions/datasets/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}/{algorithm}/")
    print(f"[PathUtils] Dataset path: {path}")  # Simple logging
    return path


def get_model_path(extension_type: str, version: str, grid_size: int, algorithm: str, timestamp: str) -> Path:
    """
    Generate standardized model path.
    Args:
        extension_type: Type of extension (e.g., 'supervised').
        version: Version string (e.g., '0.02').
        grid_size: Grid size (int).
        algorithm: Algorithm name.
        timestamp: Timestamp string.
    Returns:
        Path to the model directory.
    """
    path = Path(f"logs/extensions/models/grid-size-{grid_size}/{extension_type}_v{version}_{timestamp}/{algorithm}/")
    print(f"[PathUtils] Model path: {path}")  # Simple logging
    return path


def ensure_extension_directories(path: Path) -> Path:
    """
    Ensure extension directories exist, creating them if necessary.
    
    Args:
        path: Path to create directories for
        
    Returns:
        The created path
        
    Simple logging with print() statements.
    """
    path.mkdir(parents=True, exist_ok=True)
    print(f"[PathUtils] Extension directories ensured: {path}")  # Simple logging
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
        
    Simple logging with print() statements.
    """
    print(f"[PathUtils] Validating path structure: {extension_path}")  # Simple logging
    
    # Basic validation - extension should be in extensions/ directory
    if "extensions" not in str(extension_path):
        raise ValueError(f"Extension path should be in extensions/ directory: {extension_path}")
    
    # Check that extension follows naming convention
    extension_name = extension_path.name
    if not any(char.isdigit() for char in extension_name):
        raise ValueError(f"Extension should have version number: {extension_name}")
    
    print(f"[PathUtils] Path structure validation passed: {extension_path}")  # Simple logging
    return True


# ---------------------------------------------------------------------------
# Public helper that ensures the repository root is on the import path
# ---------------------------------------------------------------------------

def ensure_project_root_on_path() -> Path:
    """Ensure *both* the current working directory **and** ``sys.path`` point
    to the repository root.

    Many legacy extension packages (e.g. ``heuristics-v0.01`` → ``v0.04``)
    call this helper at import-time.  Historically it used to live in a
    different module so it is re-introduced here to avoid breaking those
    import statements while still honouring the *single source of truth*
    principle.

    Returns
    -------
    Path
        Absolute ``Path`` object of the detected repository root.
    """

    root = ensure_project_root()  # Also prints a log line

    # Add to sys.path *once* – duplicate entries are harmless but we check
    # anyway to keep things tidy.
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        print(f"[PathUtils] Added project root to sys.path: {root}")  # Simple logging

    return root


# ---------------------------------------------------------------------------
# Lightweight OOP façade
# ---------------------------------------------------------------------------
class PathManager:
    """Tiny helper class grouping path-related utilities.

    Rationale:
        * Keeps procedural helpers intact for callers that prefer them.
        * Offers an easy inheritance point for extensions needing tweaks
          (e.g. custom directory naming) without editing common code.
        
    Simple logging: All logging uses simple print() statements.
    """

    # Static helpers are forwarded for clarity – instance methods keep signature.
    # These very thin wrappers intentionally **do not** add new logic.

    def ensure_project_root(self) -> Path:  # noqa: D401
        return ensure_project_root()

    # Dataset / model paths --------------------------------------------------
    def dataset_path(
        self,
        extension_type: str,
        version: str,
        grid_size: int,
        algorithm: str = "",
        timestamp: str = "",
    ) -> Path:
        return get_dataset_path(extension_type, version, grid_size, algorithm, timestamp)

    def model_path(
        self,
        extension_type: str,
        version: str,
        grid_size: int,
        model_name: str = "",
        timestamp: str = "",
    ) -> Path:
        return get_model_path(extension_type, version, grid_size, model_name, timestamp)

    def validate(self, extension_path: Path) -> bool:  # noqa: D401
        return validate_path_structure(extension_path)


# Shared singleton-like instance (stateless)
path_manager = PathManager()

__all__ = [
    # Functional re-exports
    "ensure_project_root_on_path",
    "setup_extension_paths",
    "get_extension_path",
    "get_dataset_path",
    "get_model_path",
    "ensure_extension_directories",
    "validate_path_structure",
    "setup_extension_environment",
    # OOP façade
    "PathManager",
    "path_manager",
]

# Ensure side-effects (cwd + sys.path) are applied as soon as the module is
# imported – this mimics the historical behaviour so that existing extensions
# continue to work without modification.

ensure_project_root()

# ---------------------------------------------------------------------------
# Backward-compatibility shim (import alias)
# ---------------------------------------------------------------------------

def setup_extension_environment():
    """
    Standard extension environment setup following unified-path-management-guide.md
    
    This function ensures proper working directory and returns the paths
    needed for extension operation. It follows SUPREME_RULES from final-decision-10.md
    for simple logging and OOP extensibility.
    
    Returns:
        Tuple of (project_root, extension_path)
    """
    print("[PathUtils] Setting up extension environment")  # Simple logging
    
    # Ensure we're working from project root
    project_root = ensure_project_root()
    
    # Get extension path from caller
    import inspect
    caller_frame = inspect.currentframe().f_back
    caller_file = caller_frame.f_code.co_filename
    extension_path = get_extension_path(caller_file)
    
    # Validate path structure
    validate_path_structure(extension_path)
    
    print("[PathUtils] Extension environment setup complete")  # Simple logging
    return project_root, extension_path 