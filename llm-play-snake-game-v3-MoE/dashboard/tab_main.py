"""
Dashboard – Main Mode tabs (PyGame / Web)

Provides UI widgets to launch *main.py* or *main_web.py* with the full set of
CLI arguments requested by the user.
"""

from __future__ import annotations

import subprocess
import streamlit as st

from utils.network_utils import random_free_port
from utils.session_utils import run_main_web

# Default safety-limit values
from config.constants import (
    MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
    MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
    MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
    MAX_GAMES_ALLOWED,
    MAX_STEPS_ALLOWED,
    AVAILABLE_PROVIDERS,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_PARSER_PROVIDER,
    DEFAULT_PARSER_MODEL,
    PAUSE_BETWEEN_MOVES_SECONDS,
)

# ----------------------------------------
# Helper – build command list from optional args
# ----------------------------------------

def _append_arg(cmd: list[str], flag: str, value):
    """Append *flag value* to *cmd* if *value* is truthy (non-None / non-empty)."""
    if value is None:
        return
    if isinstance(value, bool):
        if value:  # only flag on boolean
            cmd.append(flag)
        return
    # Non-boolean – append flag + value as two separate list items
    cmd.extend([flag, str(value)])


# ----------------------------------------
# Main Mode – PyGame
# ----------------------------------------

def render_main_pygame_tab():
    st.markdown("### Start New Game Session (PyGame)")

    # ----- Provider / model -----
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", AVAILABLE_PROVIDERS, index=AVAILABLE_PROVIDERS.index(DEFAULT_PROVIDER) if DEFAULT_PROVIDER in AVAILABLE_PROVIDERS else 0, key="main_pg_provider")
    with col2:
        model = st.text_input("Model", value=DEFAULT_MODEL, key="main_pg_model")

    # ----- Parser LLM -----
    colp1, colp2 = st.columns(2)
    with colp1:
        parser_provider = st.selectbox("Parser Provider", ["None"] + AVAILABLE_PROVIDERS, index=AVAILABLE_PROVIDERS.index(DEFAULT_PARSER_PROVIDER)+1 if DEFAULT_PARSER_PROVIDER in AVAILABLE_PROVIDERS else 0, key="main_pg_parser_provider")
    with colp2:
        parser_model = st.text_input("Parser Model", value=DEFAULT_PARSER_MODEL, key="main_pg_parser_model")

    # ----- Core limits / timings -----
    col_core1, col_core2 = st.columns(2)
    with col_core1:
        max_games = st.number_input("Maximum Games", 1, 100, MAX_GAMES_ALLOWED, 1, key="main_pg_max_games")
    with col_core2:
        max_steps = st.number_input("Max Steps", 10, 1000, MAX_STEPS_ALLOWED, 10, key="main_pg_max_steps")

    col_time1, col_time2 = st.columns(2)
    with col_time1:
        sleep_before = st.number_input("Sleep Before Launch (minutes)", 0.0, 600.0, 0.0, 0.5, key="main_pg_sleep")
    with col_time2:
        move_pause = st.number_input("Move Pause (seconds)", 0.0, 10.0, PAUSE_BETWEEN_MOVES_SECONDS, 0.1, key="main_pg_move_pause")

    # New safety limits
    col_lim1, col_lim2, col_lim3 = st.columns(3)
    with col_lim1:
        max_empty_moves = st.number_input(
            "Max Consecutive Empty Moves",
            0,
            50,
            MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
            1,
            key="main_pg_max_empty_moves",
        )
    with col_lim2:
        max_siw = st.number_input(
            "Max Consecutive Something-Is-Wrong",
            0,
            50,
            MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
            1,
            key="main_pg_max_siw",
        )
    with col_lim3:
        max_invalid_rev = st.number_input(
            "Max Consecutive Invalid Reversals",
            0,
            50,
            MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
            1,
            key="main_pg_max_invalid_rev",
        )

    no_gui = st.checkbox("Disable GUI", value=False, key="main_pg_no_gui")

    if st.button("Start Main Session (PyGame)", key="start_main_pg"):
        cmd: list[str] = ["python", "main.py", "--max-games", str(max_games)]
        _append_arg(cmd, "--provider", provider.strip() or None)
        _append_arg(cmd, "--model", model.strip() or None)
        _append_arg(cmd, "--parser-provider", None if parser_provider == "None" else parser_provider.strip() or None)
        _append_arg(cmd, "--parser-model", None if parser_provider == "None" else parser_model.strip() or None)
        if max_steps > 0:
            _append_arg(cmd, "--max-steps", max_steps)
        if sleep_before > 0:
            _append_arg(cmd, "--sleep-before-launching", sleep_before)
        if move_pause >= 0:
            _append_arg(cmd, "--move-pause", move_pause)
        if max_empty_moves > 0:
            _append_arg(cmd, "--max-consecutive-empty-moves-allowed", max_empty_moves)
        if max_siw > 0:
            _append_arg(cmd, "--max-consecutive-something-is-wrong-allowed", max_siw)
        if max_invalid_rev > 0:
            _append_arg(cmd, "--max-consecutive-invalid-reversals-allowed", max_invalid_rev)
        if no_gui:
            cmd.append("--no-gui")

        subprocess.Popen(cmd)
        st.success("Main PyGame session launched – check your terminal/window.")


# ----------------------------------------
# Main Mode – Web
# ----------------------------------------

def render_main_web_tab():
    st.markdown("### Start New Game Session (Web)")

    # ----- Provider / model -----
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox(
            "Provider",
            AVAILABLE_PROVIDERS,
            index=AVAILABLE_PROVIDERS.index(DEFAULT_PROVIDER) if DEFAULT_PROVIDER in AVAILABLE_PROVIDERS else 0,
            key="main_web_provider",
        )
    with col2:
        st.text_input("Model", value=DEFAULT_MODEL, key="main_web_model")

    # Parser widgets
    colp1, colp2 = st.columns(2)
    with colp1:
        st.selectbox(
            "Parser Provider",
            ["None"] + AVAILABLE_PROVIDERS,
            index=AVAILABLE_PROVIDERS.index(DEFAULT_PARSER_PROVIDER) + 1 if DEFAULT_PARSER_PROVIDER in AVAILABLE_PROVIDERS else 0,
            key="main_web_parser_provider",
        )
    with colp2:
        st.text_input("Parser Model", value=DEFAULT_PARSER_MODEL, key="main_web_parser_model")

    # Core limits / timings (session_state-only)
    col_core1, col_core2 = st.columns(2)
    with col_core1:
        st.number_input("Maximum Games", 1, 100, MAX_GAMES_ALLOWED, 1, key="main_web_max_games")
    with col_core2:
        st.number_input("Max Steps", 10, 1000, MAX_STEPS_ALLOWED, 10, key="main_web_max_steps")

    col_time1, col_time2 = st.columns(2)
    with col_time1:
        st.number_input("Sleep Before Launch (minutes)", 0.0, 600.0, 0.0, 0.5, key="main_web_sleep")
    with col_time2:
        st.number_input("Move Pause (seconds)", 0.0, 10.0, PAUSE_BETWEEN_MOVES_SECONDS, 0.1, key="main_web_move_pause")

    # Safety limits (web)
    col_lim1, col_lim2, col_lim3 = st.columns(3)
    with col_lim1:
        st.number_input("Max Consecutive Empty Moves", 0, 50, MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED, 1, key="main_web_max_empty_moves")
    with col_lim2:
        st.number_input("Max Consecutive Something-Is-Wrong", 0, 50, MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED, 1, key="main_web_max_siw")
    with col_lim3:
        st.number_input("Max Consecutive Invalid Reversals", 0, 50, MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED, 1, key="main_web_max_invalid_rev")

    # Server details
    colh, colp = st.columns(2)
    with colh:
        host = st.selectbox("Host", ["localhost", "0.0.0.0", "127.0.0.1"], index=0, key="main_web_host")
    with colp:
        default_port = random_free_port(8000, 9000)
        port = st.number_input("Port", 1024, 65535, default_port, key="main_web_port")

    no_gui = st.checkbox("Disable GUI (headless)", value=False, key="main_web_no_gui")

    if st.button("Start Main Session (Web)", key="start_main_web"):
        selected_max_games = int(st.session_state.get("main_web_max_games", MAX_GAMES_ALLOWED))
        run_main_web(selected_max_games, host, port)
        url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
        st.success(f"Web session started – open {url} in your browser.")


