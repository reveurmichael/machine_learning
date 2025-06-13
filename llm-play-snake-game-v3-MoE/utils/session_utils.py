"""Utilities to launch replay / main sessions in background subprocesses.
Used by the Streamlit dashboard."""

from __future__ import annotations

import subprocess
import streamlit as st

from utils.file_utils import get_folder_display_name

__all__ = [
    "run_replay",
    "run_web_replay",
    "run_main_web",
    "continue_game",
    "continue_game_web",
]


def run_replay(log_folder: str, game_num: int):
    try:
        cmd = ["python", "replay.py", "--log-dir", log_folder, "--game", str(game_num)]
        subprocess.Popen(cmd)
        st.info(f"Replay started for Game {game_num}. Close the replay window when finished.")
    except Exception as exc:
        st.error(f"Error running replay: {exc}")


def run_web_replay(log_folder: str, game_num: int, host: str, port: int):
    try:
        cmd = [
            "python",
            "replay_web.py",
            "--log-dir",
            log_folder,
            "--game",
            str(game_num),
            "--host",
            host,
            "--port",
            str(port),
        ]
        subprocess.Popen(cmd)
        st.info(f"Web replay started for Game {game_num} at http://{host}:{port}.")
    except Exception as exc:
        st.error(f"Error running web replay: {exc}")


def continue_game(log_folder: str, max_games: int, no_gui: bool):
    try:
        cmd = [
            "python",
            "main.py",
            "--continue-with-game-in-dir",
            log_folder,
            "--max-games",
            str(max_games),
        ]
        if no_gui:
            cmd.append("--no-gui")
        subprocess.Popen(cmd)
        st.info("Continuation started in background.")
    except Exception as exc:
        st.error(f"Error continuing game: {exc}")


def run_main_web(max_games: int, host: str, port: int):
    try:
        cmd = [
            "python",
            "main_web.py",
            "--max-games",
            str(max_games),
            "--host",
            host,
            "--port",
            str(port),
        ]
        subprocess.Popen(cmd)
        st.info(f"Web main session started at http://{host}:{port}.")
    except Exception as exc:
        st.error(f"Error launching web main session: {exc}")


def continue_game_web(log_folder: str, max_games: int, host: str, port: int):
    try:
        cmd = [
            "python",
            "main_web.py",
            "--continue-with-game-in-dir",
            log_folder,
            "--max-games",
            str(max_games),
            "--host",
            host,
            "--port",
            str(port),
        ]
        subprocess.Popen(cmd)
        st.info(
            f"Continuation (web) started for '{get_folder_display_name(log_folder)}' at http://{host}:{port}."
        )
    except Exception as exc:
        st.error(f"Error starting web continuation: {exc}") 