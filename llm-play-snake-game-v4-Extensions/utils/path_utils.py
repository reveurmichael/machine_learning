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

Single Source of Truth: This module provides the canonical implementation
of project root detection and path management for ALL extensions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final, Union

from config.game_constants import LOGS_DIR_NAME, SUMMARY_JSON_FILENAME, GAME_JSON_FILENAME_PATTERN

__all__ = [
    "ensure_project_root",
    "enable_headless_pygame",
    "get_project_root",
    "get_logs_dir_path",
    "create_session_dir_path",
    "ensure_logs_dir",
    "get_default_logs_root",
    "get_summary_json_path",
    "get_game_json_path",
    "get_summary_json_filename",
    "get_game_json_filename",
]

# Cache the repository root path for efficient repeated access.
# The root is determined by finding the parent directory of the utils folder.
# THIS IS VERY IMPORTANT AND SHOULD BE USED ABSOLUTELY BY THE WEB MODE.
#
# Implementation Note: Path(__file__).resolve().parent.parent is CORRECT.
# - __file__ = /path/to/project/utils/path_utils.py
# - .parent = /path/to/project/utils/  
# - .parent.parent = /path/to/project/ (project root)
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    
    This is a pure function with no side effects, suitable for situations
    where you only need the path without changing the working directory.
    
    Project root is identified by the presence of three required directories:
    core/, llm/, and extensions/
    
    Returns:
        The absolute pathlib.Path to the project root directory.
        
    Raises:
        RuntimeError: If project root cannot be found within 10 levels
    """
    # First check if the static _PROJECT_ROOT is valid
    required_dirs = ["core", "llm", "extensions"]
    if all((_PROJECT_ROOT / dir_name).is_dir() for dir_name in required_dirs):
        return _PROJECT_ROOT
    
    # Search upward from current file location
    current = Path(__file__).resolve()
    for _ in range(10):
        if all((current / dir_name).is_dir() for dir_name in required_dirs):
            return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent
    
    raise RuntimeError(
        f"Could not locate project root containing 'core/', 'llm/', and 'extensions/' directories "
        f"within 10 levels from {Path(__file__).resolve()}"
    )


def ensure_project_root() -> Path:
    """
    Ensures the current working directory is the project root and that the
    root directory is in sys.path for absolute imports.

    This function validates the project root by checking for required directories
    (core/, llm/, extensions/) and searches upward if needed.

    Single Source of Truth: This is the ONLY function that should be used
    for project root detection across ALL extensions and scripts.

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
    project_root = get_project_root()
    current_dir = Path.cwd()
    
    if current_dir != project_root:
        print(f"[PathUtils] Changing working directory to project root: {project_root}")
        os.chdir(project_root)
    
    # Ensure the project root is at the beginning of sys.path for import precedence
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    
    return project_root


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


def get_logs_dir_path(base_dir: Union[str, Path, None] = None) -> Path:
    """
    Get the standardized logs directory path.
    
    Args:
        base_dir: Optional base directory (defaults to current working directory)
        
    Returns:
        Path object pointing to the logs directory
        
    Example:
        >>> get_logs_dir_path()
        Path("logs")
        >>> get_logs_dir_path("/project")  
        Path("/project/logs")
    """
    if base_dir is None:
        return Path(LOGS_DIR_NAME)
    return Path(base_dir) / LOGS_DIR_NAME


def create_session_dir_path(model_name: str, timestamp: str, base_dir: Union[str, Path, None] = None) -> Path:
    """
    Create path for session directory using model name and timestamp.
    
    Args:
        model_name: Name/identifier of the model
        timestamp: Timestamp string for the session
        base_dir: Optional base directory (defaults to current working directory)
        
    Returns:
        Path object pointing to the session directory
        
    Example:
        >>> create_session_dir_path("gpt-4", "20240101_120000")
        Path("logs/gpt-4_20240101_120000")
    """
    logs_dir = get_logs_dir_path(base_dir)
    session_name = f"{model_name}_{timestamp}"
    return logs_dir / session_name


def ensure_logs_dir(base_dir: Union[str, Path, None] = None) -> Path:
    """
    Ensure logs directory exists and return its path.
    
    Args:
        base_dir: Optional base directory (defaults to current working directory)
        
    Returns:
        Path object pointing to the created logs directory
        
    Example:
        >>> logs_path = ensure_logs_dir()
        >>> print(f"Logs directory: {logs_path}")
    """
    logs_dir = get_logs_dir_path(base_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_default_logs_root() -> str:
    """
    Get the default logs root directory name.
    
    Returns:
        String name of the logs directory
        
    Example:
        >>> get_default_logs_root()
        "logs"
    """
    return LOGS_DIR_NAME


def get_summary_json_path(base_dir: Union[str, Path, None] = None) -> Path:
    """
    Get path for summary JSON file.
    
    Args:
        base_dir: Optional base directory (defaults to current working directory)
        
    Returns:
        Path object pointing to the summary.json file
        
    Example:
        >>> get_summary_json_path()
        Path("summary.json")
    """
    if base_dir is None:
        return Path(SUMMARY_JSON_FILENAME)
    return Path(base_dir) / SUMMARY_JSON_FILENAME


def get_game_json_path(game_number: int, base_dir: Union[str, Path, None] = None) -> Path:
    """
    Get path for specific game JSON file.
    
    Args:
        game_number: Game number for the filename
        base_dir: Optional base directory (defaults to current working directory)
        
    Returns:
        Path object pointing to the game_N.json file
        
    Example:
        >>> get_game_json_path(1)
        Path("game_1.json")
    """
    filename = GAME_JSON_FILENAME_PATTERN.format(game_number=game_number)
    if base_dir is None:
        return Path(filename)
    return Path(base_dir) / filename


def get_summary_json_filename() -> str:
    """
    Get the standard filename for summary JSON files.
    
    Returns:
        String filename for summary JSON
        
    Example:
        >>> get_summary_json_filename()
        "summary.json"
    """
    return SUMMARY_JSON_FILENAME


def get_game_json_filename(game_number: int) -> str:
    """
    Get the standard filename for game JSON files.
    
    Args:
        game_number: Game number for the filename
        
    Returns:
        String filename for game JSON
        
    Example:
        >>> get_game_json_filename(1)
        "game_1.json"
    """
    return GAME_JSON_FILENAME_PATTERN.format(game_number=game_number) 