"""
LLM Logging Utilities

This module provides specialized functions for handling the logging of LLM-related
data, such as prompts, responses, and parser outputs. It ensures that all
Task-0 specific logging is managed in a consistent and organized manner,
decoupled from generic file utilities.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

__all__ = [
    "clean_prompt_files",
    "save_llm_artefact",
    "get_prompt_filename",
]


def clean_prompt_files(log_dir: Union[str, Path], start_game: int) -> None:
    """
    Deletes prompt and response files for a given game number onwards.

    This is used when continuing a session to ensure that artifacts from
    previous, partial runs of a game are cleared.

    Args:
        log_dir: The root directory for the logging session.
        start_game: The game number from which to clear artifacts.
    """
    prompts_dir = Path(log_dir) / "prompts"
    if prompts_dir.exists():
        for f in prompts_dir.glob(f"game_{start_game}_*"):
            f.unlink(missing_ok=True)

    responses_dir = Path(log_dir) / "responses"
    if responses_dir.exists():
        for f in responses_dir.glob(f"game_{start_game}_*"):
            f.unlink(missing_ok=True)


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