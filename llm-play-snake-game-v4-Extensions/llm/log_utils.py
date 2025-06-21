"""LLM logging utilities with single source of truth for file operations.

=== SINGLE SOURCE OF TRUTH FOR LLM FILE OPERATIONS ===
This module provides elegant utility functions for LLM-specific file and directory
operations, following the same pattern as get_prompt_filename().

All LLM directory names and file patterns are centralized here:
- Directory names come from config.game_constants (PROMPTS_DIR_NAME, RESPONSES_DIR_NAME)
- File naming patterns are standardized via get_prompt_filename()
- Directory creation and management via get_llm_directories(), ensure_llm_directories()
- Cleanup operations via cleanup_game_artifacts()

=== DESIGN PHILOSOPHY ===
Like get_prompt_filename(), all functions here follow elegant patterns:
1. **Centralized Constants**: No hardcoded strings
2. **Type Safety**: Union[str, Path] for flexibility
3. **Path Objects**: Modern pathlib.Path usage
4. **Error Handling**: Graceful missing_ok=True operations
5. **Consistent Naming**: Standardized file/directory patterns

=== USAGE EXAMPLES ===
```python
# Get directory paths
prompts_dir, responses_dir = get_llm_directories(log_dir)

# Ensure directories exist
prompts_dir, responses_dir = ensure_llm_directories(log_dir)

# Clean up game artifacts
cleanup_game_artifacts(log_dir, start_game=3)

# Get standardized filename
filename = get_prompt_filename(game_num=1, round_num=2, artefact_type="prompt")
```
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from config.game_constants import PROMPTS_DIR_NAME, RESPONSES_DIR_NAME

__all__ = [
    "clean_prompt_files",
    "save_llm_artefact",
    "get_prompt_filename",
    "get_prompts_dir_path",
    "get_responses_dir_path",
    "get_llm_directories",
    "ensure_llm_directories",
    "cleanup_game_artifacts",
]


def clean_prompt_files(log_dir: Union[str, Path], start_game: int) -> None:
    """
    Legacy function - use cleanup_game_artifacts() instead.
    
    Deletes prompt and response files for a given game number onwards.

    Args:
        log_dir: The root directory for the logging session.
        start_game: The game number from which to clear artifacts.
    """
    cleanup_game_artifacts(log_dir, start_game)


def save_llm_artefact(
    content: str,
    directory: Union[str, Path],
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Saves LLM-related content (like prompts or responses) to a file with metadata.

    Args:
        content: The main text content to save.
        directory: The directory where the file will be saved (e.g., 'prompts').
        filename: The name of the file.
        metadata: An optional dictionary of metadata to prepend to the file.

    Returns:
        The absolute path to the saved file as a string.
    """
    path_dir = Path(directory)
    path_dir.mkdir(parents=True, exist_ok=True)
    file_path = path_dir / filename

    header = ""
    if metadata:
        from datetime import datetime

        if "timestamp" not in (k.lower() for k in metadata):
            metadata["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for key, value in metadata.items():
            header += f"{key}: {value}\n"

        if "prompt" in filename.lower():
            header += "\n\n========== LLM PROMPT ==========\n\n"
        elif "response" in filename.lower():
            header += "\n\n========== LLM RESPONSE ==========\n\n"

    full_content = f"{header}{content}"
    file_path.write_text(full_content, encoding="utf-8")

    return str(file_path.resolve())


def get_prompt_filename(
    game_number: int,
    round_number: int,
    artefact_type: str,
) -> str:
    """
    Generates a standardized filename for an LLM artifact.

    Args:
        game_number: The game number (1-based).
        round_number: The round number within the game (1-based).
        artefact_type: The type of artifact. Valid options are:
                       'prompt', 'raw_response', 'parser_prompt',
                       'parsed_response'.

    Returns:
        A string with the standardized filename (e.g., 'game_1_round_1_prompt.txt').

    Raises:
        ValueError: If an invalid `artefact_type` is provided.
    """
    valid_types = {"prompt", "raw_response", "parser_prompt", "parsed_response"}
    if artefact_type not in valid_types:
        raise ValueError(
            f"Invalid artefact_type '{artefact_type}'. Must be one of: {valid_types}"
        )

    return f"game_{game_number}_round_{round_number}_{artefact_type}.txt"


def get_prompts_dir_path(log_dir: Union[str, Path]) -> Path:
    """
    Get the standardized prompts directory path for a given log directory.
    
    Args:
        log_dir: The base log directory path
        
    Returns:
        Path object pointing to the prompts subdirectory
        
    Example:
        >>> get_prompts_dir_path("/logs/session_123")
        Path("/logs/session_123/prompts")
    """
    return Path(log_dir) / PROMPTS_DIR_NAME


def get_responses_dir_path(log_dir: Union[str, Path]) -> Path:
    """
    Get the standardized responses directory path for a given log directory.
    
    Args:
        log_dir: The base log directory path
        
    Returns:
        Path object pointing to the responses subdirectory
        
    Example:
        >>> get_responses_dir_path("/logs/session_123")
        Path("/logs/session_123/responses")
    """
    return Path(log_dir) / RESPONSES_DIR_NAME


def get_llm_directories(log_dir: Union[str, Path]) -> tuple[Path, Path]:
    """
    Get both prompts and responses directory paths for a given log directory.
    
    Args:
        log_dir: The base log directory path
        
    Returns:
        Tuple of (prompts_dir, responses_dir) Path objects
        
    Example:
        >>> prompts_dir, responses_dir = get_llm_directories("/logs/session_123")
        >>> print(prompts_dir)  # Path("/logs/session_123/prompts")
        >>> print(responses_dir)  # Path("/logs/session_123/responses")
    """
    return get_prompts_dir_path(log_dir), get_responses_dir_path(log_dir)


def ensure_llm_directories(log_dir: Union[str, Path]) -> tuple[Path, Path]:
    """
    Ensure prompts and responses directories exist, creating them if necessary.
    
    Args:
        log_dir: The base log directory path
        
    Returns:
        Tuple of (prompts_dir, responses_dir) Path objects (guaranteed to exist)
        
    Example:
        >>> prompts_dir, responses_dir = ensure_llm_directories("/logs/session_123")
        # Directories are created if they don't exist
    """
    prompts_dir, responses_dir = get_llm_directories(log_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    return prompts_dir, responses_dir


def cleanup_game_artifacts(log_dir: Union[str, Path], start_game: int) -> None:
    """
    Deletes prompt and response files for a given game number onwards.

    This is used when continuing a session to ensure that artifacts from
    previous, partial runs of a game are cleared.

    Args:
        log_dir: The root directory for the logging session.
        start_game: The game number from which to clear artifacts.
    """
    prompts_dir, responses_dir = get_llm_directories(log_dir)
    
    if prompts_dir.exists():
        for f in prompts_dir.glob(f"game_{start_game}_*"):
            f.unlink(missing_ok=True)

    if responses_dir.exists():
        for f in responses_dir.glob(f"game_{start_game}_*"):
            f.unlink(missing_ok=True) 