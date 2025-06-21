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
#
# This approach is preferred over config folder because:
# 1. Path utilities are infrastructure, not configuration
# 2. Config folder should contain settings, not path detection logic
# 3. This module needs to work before config is loaded
# 4. Keeps path resolution logic centralized in utils
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


# IMPORTANT: this one is very important for scripts in the folder "scripts" and hence for all web modes (human play web mode, replay web mode, main web mode, continue web mode)
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
    Create a standardized session directory path with model name and timestamp.
    
    Args:
        model_name: Name of the model/provider
        timestamp: Timestamp string for the session
        base_dir: Optional base directory (defaults to current working directory)
        
    Returns:
        Path object pointing to the session directory
        
    Example:
        >>> create_session_dir_path("deepseek-r1", "20250118_143022")
        Path("logs/deepseek-r1_20250118_143022")
    """
    logs_dir = get_logs_dir_path(base_dir)
    session_name = f"{model_name}_{timestamp}"
    return logs_dir / session_name


def ensure_logs_dir(base_dir: Union[str, Path, None] = None) -> Path:
    """
    Ensure logs directory exists, creating it if necessary.
    
    Args:
        base_dir: Optional base directory (defaults to current working directory)
        
    Returns:
        Path object pointing to the logs directory (guaranteed to exist)
        
    Example:
        >>> logs_dir = ensure_logs_dir()
        # Directory is created if it doesn't exist
    """
    logs_dir = get_logs_dir_path(base_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_default_logs_root() -> str:
    """
    Get the default logs root directory name as a string.
    
    This function provides backwards compatibility for code that expects 
    a string rather than a Path object.
    
    Returns:
        String name of the logs directory
        
    Example:
        >>> get_default_logs_root()
        "logs"
    """
    return LOGS_DIR_NAME 


def get_summary_json_path(base_dir: Union[str, Path, None] = None) -> Path:
    """
    Get the standardized summary JSON file path.
    
    Args:
        base_dir: Optional base directory (defaults to current working directory)
        
    Returns:
        Path object pointing to the summary JSON file
        
    Example:
        >>> get_summary_json_path()
        Path("logs/summary.json")
        >>> get_summary_json_path("/project")  
        Path("/project/logs/summary.json")
    """
    return get_logs_dir_path(base_dir) / SUMMARY_JSON_FILENAME


def get_game_json_path(game_number: int, base_dir: Union[str, Path, None] = None) -> Path:
    """
    Get the standardized game JSON file path for a specific game number.
    
    Args:
        game_number: The game number
        base_dir: Optional base directory (defaults to current working directory)
        
    Returns:
        Path object pointing to the specific game JSON file
        
    Example:
        >>> get_game_json_path(1)
        Path("logs/game_1.json")
        >>> get_game_json_path(5, "/project")  
        Path("/project/logs/game_5.json")
    """
    filename = GAME_JSON_FILENAME_PATTERN.format(game_number)
    return get_logs_dir_path(base_dir) / filename


def get_summary_json_filename() -> str:
    """
    Get the summary JSON filename as a string.
    
    Returns:
        String name of the summary JSON file
        
    Example:
        >>> get_summary_json_filename()
        "summary.json"
    """
    return SUMMARY_JSON_FILENAME


def get_game_json_filename(game_number: int) -> str:
    """
    Get the game JSON filename as a string for a specific game number.
    
    Args:
        game_number: The game number
        
    Returns:
        String name of the game JSON file
        
    Example:
        >>> get_game_json_filename(1)
        "game_1.json"
    """
    return GAME_JSON_FILENAME_PATTERN.format(game_number) 