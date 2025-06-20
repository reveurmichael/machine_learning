"""Utilities to launch replay / main sessions in background subprocesses.
Used by the Streamlit dashboard.

This whole module is Task0 specific.
"""

from __future__ import annotations

import subprocess
import streamlit as st
import os

from utils.file_utils import get_folder_display_name
from utils.network_utils import ensure_free_port

__all__ = [
    "run_replay",
    "run_web_replay",
    "run_main_web",
    "continue_game",
    "continue_game_web",
    "run_human_play",
    "run_human_play_web",
]


def run_replay(log_folder: str, game_num: int):
    try:
        cmd = [
            "python",
            os.path.join("scripts", "replay.py"),
            "--log-dir",
            log_folder,
            "--game",
            str(game_num),
        ]
        subprocess.Popen(cmd)
        st.info(f"Replay started for Game {game_num}. Close the replay window when finished.")
    except Exception as exc:
        st.error(f"Error running replay: {exc}")


def run_web_replay(log_folder: str, game_num: int, host: str, port: int):
    try:
        port = ensure_free_port(port)
        cmd = [
            "python",
            os.path.join("scripts", "replay_web.py"),
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
            os.path.join("scripts", "main.py"),
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
    """Launch main_web.py with full CLI options.

    The Streamlit tab already stored all user inputs in session_state.  To
    keep the tab code minimal, we harvest those values here and construct the
    command accordingly.  That way we don't need to touch the UI when new
    arguments are added.
    """
    try:
        port = ensure_free_port(port)

        cmd: list[str] = [
            "python",
            os.path.join("scripts", "main_web.py"),
            "--max-games",
            str(max_games),
            "--host",
            host,
            "--port",
            str(port),
        ]

        # ---------------------
        # Optional CLI parameters harvested from session_state
        # ---------------------
        ss = st.session_state

        _append_arg(cmd, "--provider", ss.get("main_web_provider"))
        _append_arg(cmd, "--model", ss.get("main_web_model"))

        parser_provider = ss.get("main_web_parser_provider")
        parser_model = ss.get("main_web_parser_model")
        if parser_provider and parser_provider != "None":
            _append_arg(cmd, "--parser-provider", parser_provider)
            _append_arg(cmd, "--parser-model", parser_model)

        max_steps = ss.get("main_web_max_steps")
        if max_steps and int(max_steps) > 0:
            _append_arg(cmd, "--max-steps", max_steps)

        sleep_before = ss.get("main_web_sleep")
        if sleep_before and float(sleep_before) > 0:
            _append_arg(cmd, "--sleep-before-launching", sleep_before)

        pause_between_moves = ss.get("main_web_pause_between_moves")
        if pause_between_moves is not None and float(pause_between_moves) >= 0:
            _append_arg(cmd, "--pause-between-moves", pause_between_moves)

        # Back-off after EMPTY sentinel (minutes)
        sleep_after_empty = ss.get("main_web_sleep_after_empty")
        if sleep_after_empty and float(sleep_after_empty) > 0:
            _append_arg(cmd, "--sleep-after-empty-step", sleep_after_empty)

        no_gui = ss.get("main_web_no_gui")
        if no_gui:
            cmd.append("--no-gui")

        # New safety limits – include only when >0
        max_empty = ss.get("main_web_max_empty_moves")
        if max_empty and int(max_empty) > 0:
            _append_arg(cmd, "--max-consecutive-empty-moves-allowed", max_empty)

        max_siw = ss.get("main_web_max_siw")
        if max_siw and int(max_siw) > 0:
            _append_arg(cmd, "--max-consecutive-something-is-wrong-allowed", max_siw)

        max_inv_rev = ss.get("main_web_max_invalid_rev")
        if max_inv_rev and int(max_inv_rev) > 0:
            _append_arg(cmd, "--max-consecutive-invalid-reversals-allowed", max_inv_rev)

        max_no_path = ss.get("main_web_max_no_path")
        if max_no_path and int(max_no_path) > 0:
            _append_arg(cmd, "--max-consecutive-no-path-found-allowed", max_no_path)

        # ---------------------
        subprocess.Popen(cmd)
        st.info(f"Web main session started at http://{host}:{port}.")
    except Exception as exc:
        st.error(f"Error launching web main session: {exc}")


def continue_game_web(
    log_folder: str,
    max_games: int,
    host: str,
    port: int,
    sleep_before: float = 0.0,
    no_gui: bool = False,
):
    try:
        port = ensure_free_port(port)
        cmd = [
            "python",
            os.path.join("scripts", "main_web.py"),
            "--continue-with-game-in-dir",
            log_folder,
            "--max-games",
            str(max_games),
            "--host",
            host,
            "--port",
            str(port),
        ]
        if sleep_before and float(sleep_before) > 0:
            _append_arg(cmd, "--sleep-before-launching", sleep_before)
        if no_gui:
            cmd.append("--no-gui")
        subprocess.Popen(cmd)
        st.info(
            f"Continuation (web) started for '{get_folder_display_name(log_folder)}' at http://{host}:{port}."
        )
    except Exception as exc:
        st.error(f"Error starting web continuation: {exc}")


# ---------------------
# Human Play launchers
# ---------------------


def run_human_play():
    """Launch PyGame human play mode in background."""
    try:
        subprocess.Popen(["python", os.path.join("scripts", "human_play.py")])
        st.info("PyGame Human Play started in a new window.")
    except Exception as exc:
        st.error(f"Error starting human play: {exc}")


def run_human_play_web(host: str, port: int):
    """Launch web human play mode on specified host/port."""
    try:
        port = ensure_free_port(port)
        cmd = [
            "python",
            os.path.join("scripts", "human_play_web.py"),
            "--host",
            host,
            "--port",
            str(port),
        ]
        subprocess.Popen(cmd)
        url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
        st.info(f"Web Human Play started – open {url} in your browser.")
    except Exception as exc:
        st.error(f"Error starting web human play: {exc}")


# Reuse helper to append args consistently
def _append_arg(cmd: list[str], flag: str, value):
    """Append flag/value to cmd if value provided.

    Mirrors the logic used in dashboard.tab_main / tab_continue so that we
    don't duplicate option-building code all over the place.  Boolean flags
    are appended without a value when *value* is truthy; everything else is
    appended as two separate list items.
    """
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)]) 