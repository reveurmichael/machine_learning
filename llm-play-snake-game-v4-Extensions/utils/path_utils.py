"""
Project Setup and Environment Configuration Utilities

This module provides essential setup functions for script entrypoints to ensure
a consistent execution environment across the entire project. It handles:

- Setting the current working directory to the repository root
- Configuring sys.path for absolute imports
- Environment variable setup for headless library execution
- Cross-platform compatibility for different execution contexts

These utilities prevent code duplication across script files and provide
a standardized way to bootstrap the project environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

__all__ = [
    "ensure_project_root",
    "enable_headless_pygame",
    "get_project_root",
]

# Cache the repository root path for efficient repeated access.
# The root is determined by finding the parent directory of the utils folder.
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    
    This is a pure function with no side effects, suitable for situations
    where you only need the path without changing the working directory.
    
    Returns:
        The absolute pathlib.Path to the project root directory.
    """
    return _PROJECT_ROOT


def ensure_project_root() -> Path:
    """
    Ensures the current working directory is the project root and that the
    root directory is in sys.path for absolute imports.

    This function has intentional side effects:
    - Changes the current working directory (os.chdir)
    - Modifies sys.path to enable absolute imports
    - Prints a message if the directory is changed

    Returns:
        The absolute pathlib.Path to the project root directory.
        
    Example:
        >>> from utils.path_utils import ensure_project_root
        >>> root = ensure_project_root()
        >>> print(f"Project running from: {root}")
    """
    current_dir = Path.cwd()
    
    if current_dir != _PROJECT_ROOT:
        print(f"Changing working directory to project root: {_PROJECT_ROOT}")
        os.chdir(_PROJECT_ROOT)
    
    # Ensure the project root is at the beginning of sys.path for import precedence
    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    
    return _PROJECT_ROOT


def enable_headless_pygame() -> None:
    """
    Configures pygame to run in headless mode by setting the SDL_VIDEODRIVER
    environment variable to 'dummy'.

    This is essential for scripts that use pygame for game logic but run in
    environments where creating a display window is not possible or desired,
    such as:
    - Web server environments (Flask, FastAPI)
    - CI/CD pipelines
    - Headless servers
    - Docker containers without X11 forwarding
    
    Example:
        >>> from utils.path_utils import enable_headless_pygame
        >>> enable_headless_pygame()
        >>> import pygame  # Will now run without requiring a display
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    # Also set other common headless environment variables for compatibility
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":99"  # Common virtual display number 