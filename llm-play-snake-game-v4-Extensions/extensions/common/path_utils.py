"""common path utilities for heuristics extensions

Provides `ensure_project_root_on_path()` and `setup_extension_paths()`
so individual modules don't need to repeat boiler-plate path hacks.
"""

from __future__ import annotations

import sys
import pathlib
from typing import Final


class PathManager:
    """Manages project root path caching to avoid repeated lookups."""

    _repo_root: Final[pathlib.Path | None] = None  # cache

    @classmethod
    def _find_repo_root(cls, start: pathlib.Path | None = None) -> pathlib.Path:
        current = start or pathlib.Path(__file__).resolve()
        for _ in range(10):
            if (current / "config").is_dir():
                return current
            if current.parent == current:
                break
            current = current.parent
        raise RuntimeError(
            "Could not locate repository root containing 'config/' folder"
        )

    @classmethod
    def ensure_project_root_on_path(cls) -> pathlib.Path:
        """Ensure project root is on Python path, with caching."""
        if cls._repo_root is None:
            # Use a different approach to avoid global statement
            cls._repo_root = cls._find_repo_root()
        root_str = str(cls._repo_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return cls._repo_root


def ensure_project_root_on_path() -> pathlib.Path:
    """Ensure project root is on Python path."""
    return PathManager.ensure_project_root_on_path()


def setup_extension_paths() -> None:
    """
    Common path setup for extension modules.
    
    This function handles the boilerplate path setup that was previously
    duplicated across all extension modules:
    
    ```python
    # Add project root to path for imports
    project_root = pathlib.Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    ```
    
    Usage:
        from extensions.common.path_utils import setup_extension_paths
        setup_extension_paths()
    """
    # Ensure project root is on path (handles the common pattern)
    ensure_project_root_on_path()


# Auto-register on import
ensure_project_root_on_path()

__all__ = ["ensure_project_root_on_path", "setup_extension_paths"]
