"""Path utilities for extensions/common/utils/ package.

This file follows the principles from final-decision.md:
- All utilities must use simple print logging
- All utilities must be OOP, extensible, and never over-engineered
- Reference: SimpleFactory in factory_utils.py is the canonical factory pattern for all extensions
- See also: agents.md, core.md, config.md, factory-design-pattern.md, extension-evolution-rules.md

This module provides path management utilities for the extensions package,
serving as a thin façade to the canonical implementation in the root utils
package to maintain the Single Source of Truth principle.

Key Functions:
    get_datasets_root: Get standardized datasets root directory
    get_dataset_path: Get standardized dataset directory paths
    get_model_path: Get standardized model directory paths

Design Philosophy:
- Simple, lightweight utilities that can be inherited and extended
- No tight coupling with ML/DL/RL/LLM-specific concepts
- Simple logging with print() statements
- Enables easy addition of new extensions without friction
- All code examples use print() and create() as canonical patterns.
"""

from __future__ import annotations


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


from typing import TYPE_CHECKING
from utils.print_utils import print_info

if TYPE_CHECKING:  # pragma: no cover – only for static type checkers
    from pathlib import Path


def get_datasets_root() -> Path:
    """
    Get the standardized datasets root directory.
    Returns:
        Path to the datasets root directory.
    """
    path = Path("logs/extensions/datasets")
    print_info(f"[PathUtils] Datasets root: {path}")  # Simple logging
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
    print_info(f"[PathUtils] Dataset path: {path}")  # Simple logging
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
    print_info(f"[PathUtils] Model path: {path}")  # Simple logging
    return path


# ----------------
# Public API
# ----------------

__all__ = [
    "get_datasets_root", 
    "get_dataset_path",
    "get_model_path",
] 
