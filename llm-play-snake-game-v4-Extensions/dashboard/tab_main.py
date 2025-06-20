"""Streamlit sub-module: renders *Main* and *Main-Web* tabs.

The functions here only **compose UI components** and build CLI command lists;
they never mutate global state or perform heavy computation.  This keeps the
dashboard responsive and side-effect free outside the explicit *Start* button
callbacks.
"""

from __future__ import annotations

import subprocess
from typing import List
import os

import streamlit as st

from utils.network_utils import random_free_port
from utils.session_utils import run_main_web
from llm.providers import get_available_models
from config.network_constants import HOST_CHOICES

from config.game_constants import (
    MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED,
    MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED,
    MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED,
    MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED,
    MAX_GAMES_ALLOWED,
    MAX_STEPS_ALLOWED,
    list_available_providers,
    PAUSE_BETWEEN_MOVES_SECONDS,
    SLEEP_AFTER_EMPTY_STEP,
)
from config.ui_constants import (
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_PARSER_PROVIDER,
    DEFAULT_PARSER_MODEL,
)

# Lazily obtain provider list to avoid circular-import issues.
AVAILABLE_PROVIDERS = list_available_providers()

# ---------------------
# Helper – build command list from optional args
# ---------------------

def _append_arg(cmd: List[str], flag: str, value) -> None:
    """Append *flag* (and maybe *value*) to *cmd* in place.

    • Booleans append only the flag when *True* (common for `--no-gui`).
    • Any other truthy value appends flag **and** str(value).
    • ``None`` / falsy values are ignored.
    """

    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


# ---------------------
# Main Mode – PyGame
# ---------------------

def render_main_pygame_tab() -> None:
    st.markdown("### Start New Game Session (PyGame)")

    # ----- Primary LLM Provider / Model -----
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox(
            "Provider",
            AVAILABLE_PROVIDERS,
            index=AVAILABLE_PROVIDERS.index(DEFAULT_PROVIDER)
            if DEFAULT_PROVIDER in AVAILABLE_PROVIDERS
            else 0,
            key="main_pg_provider",
        )
    with col2:
        provider_models = get_available_models(provider) or [DEFAULT_MODEL]
        model_default_idx = (
            provider_models.index(DEFAULT_MODEL)
            if DEFAULT_MODEL in provider_models else 0
        )
        model = st.selectbox(
            "Model",
            provider_models,
            index=model_default_idx,
            key="main_pg_model",
        )

    # ----- Secondary LLM Provider / Model -----
    colp1, colp2 = st.columns(2)
    with colp1:
        parser_provider = st.selectbox(
            "Parser Provider",
            ["None"] + AVAILABLE_PROVIDERS,
            index=AVAILABLE_PROVIDERS.index(DEFAULT_PARSER_PROVIDER) + 1
            if DEFAULT_PARSER_PROVIDER in AVAILABLE_PROVIDERS
            else 0,
            key="main_pg_parser_provider",
        )
    with colp2:
        if parser_provider == "None":
            parser_model = st.selectbox(
                "Parser Model",
                ["None"],
                index=0,
                key="main_pg_parser_model",
            )
        else:
            p_models = get_available_models(parser_provider) or [DEFAULT_PARSER_MODEL]
            p_idx = p_models.index(DEFAULT_PARSER_MODEL) if DEFAULT_PARSER_MODEL in p_models else 0
            parser_model = st.selectbox(
                "Parser Model",
                p_models,
                index=p_idx,
                key="main_pg_parser_model",
            )

    # ----- Core limits / timings -----
    col_core1, col_core2 = st.columns(2)
    with col_core1:
        max_games = st.number_input("Maximum Games", 1, 100, MAX_GAMES_ALLOWED, 1, key="main_pg_max_games")
    with col_core2:
        max_steps = st.number_input("Max Steps", 10, 1000, MAX_STEPS_ALLOWED, 10, key="main_pg_max_steps")

    col_time1, col_time2, col_time3 = st.columns(3)
    with col_time1:
        sleep_before = st.number_input("Sleep Before Launch (minutes)", 0.0, 600.0, 0.0, 0.5, key="main_pg_sleep")
    with col_time2:
        pause_between_moves = st.number_input("Move Pause (seconds)", 0.0, 10.0, PAUSE_BETWEEN_MOVES_SECONDS, 0.1, key="main_pg_pause_between_moves")
    with col_time3:
        sleep_after_empty = st.number_input("Sleep After EMPTY (minutes)", 0.0, 1000.0, SLEEP_AFTER_EMPTY_STEP, 0.5, key="main_pg_sleep_after_empty")

    # New safety limits
    col_lim1, col_lim2, col_lim3, col_lim4 = st.columns(4)
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
    with col_lim4:
        max_no_path = st.number_input(
            "Max Consecutive NO_PATH_FOUND",
            0,
            50,
            MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED,
            1,
            key="main_pg_max_no_path",
        )

    no_gui = st.checkbox("Disable GUI", value=False, key="main_pg_no_gui")

    if st.button("Start Main Session (PyGame)", key="start_main_pg"):
        cmd: list[str] = ["python", os.path.join("scripts", "main.py"), "--max-games", str(max_games)]
        _append_arg(cmd, "--provider", provider.strip() or None)
        _append_arg(cmd, "--model", model.strip() or None)
        _append_arg(cmd, "--parser-provider", None if parser_provider == "None" else parser_provider.strip() or None)
        _append_arg(cmd, "--parser-model", None if parser_provider == "None" else parser_model.strip() or None)
        if max_steps > 0:
            _append_arg(cmd, "--max-steps", max_steps)
        if sleep_before > 0:
            _append_arg(cmd, "--sleep-before-launching", sleep_before)
        if pause_between_moves >= 0:
            _append_arg(cmd, "--pause-between-moves", pause_between_moves)
        if sleep_after_empty > 0:
            _append_arg(cmd, "--sleep-after-empty-step", sleep_after_empty)
        if max_empty_moves > 0:
            _append_arg(cmd, "--max-consecutive-empty-moves-allowed", max_empty_moves)
        if max_siw > 0:
            _append_arg(cmd, "--max-consecutive-something-is-wrong-allowed", max_siw)
        if max_invalid_rev > 0:
            _append_arg(cmd, "--max-consecutive-invalid-reversals-allowed", max_invalid_rev)
        if max_no_path > 0:
            _append_arg(cmd, "--max-consecutive-no-path-found-allowed", max_no_path)
        if no_gui:
            cmd.append("--no-gui")

        subprocess.Popen(cmd)
        st.success("Main PyGame session launched – check your terminal/window.")


# ---------------------
# Main Mode – Web
# ---------------------

def render_main_web_tab() -> None:
    st.markdown("### Start New Game Session (Web)")

    # ----- Provider / model -----
    col1, col2 = st.columns(2)
    with col1:
        provider_w = st.selectbox(
            "Provider",
            AVAILABLE_PROVIDERS,
            index=AVAILABLE_PROVIDERS.index(DEFAULT_PROVIDER)
            if DEFAULT_PROVIDER in AVAILABLE_PROVIDERS
            else 0,
            key="main_web_provider",
        )
    with col2:
        provider_w_models = get_available_models(provider_w) or [DEFAULT_MODEL]
        st.selectbox(
            "Model",
            provider_w_models,
            index=provider_w_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in provider_w_models else 0,
            key="main_web_model",
        )

    # Parser widgets
    colp1, colp2 = st.columns(2)
    with colp1:
        parser_provider_w = st.selectbox(
            "Parser Provider",
            ["None"] + AVAILABLE_PROVIDERS,
            index=AVAILABLE_PROVIDERS.index(DEFAULT_PARSER_PROVIDER) + 1
            if DEFAULT_PARSER_PROVIDER in AVAILABLE_PROVIDERS
            else 0,
            key="main_web_parser_provider",
        )
    with colp2:
        if parser_provider_w == "None":
            st.selectbox(
                "Parser Model",
                ["None"],
                index=0,
                key="main_web_parser_model",
            )
        else:
            pp_models = get_available_models(parser_provider_w) or [DEFAULT_PARSER_MODEL]
            st.selectbox(
                "Parser Model",
                pp_models,
                index=pp_models.index(DEFAULT_PARSER_MODEL) if DEFAULT_PARSER_MODEL in pp_models else 0,
                key="main_web_parser_model",
            )

    # Core limits / timings (session_state-only)
    col_core1, col_core2 = st.columns(2)
    with col_core1:
        st.number_input("Maximum Games", 1, 100, MAX_GAMES_ALLOWED, 1, key="main_web_max_games")
    with col_core2:
        st.number_input("Max Steps", 10, 1000, MAX_STEPS_ALLOWED, 10, key="main_web_max_steps")

    col_time1, col_time2, col_time3 = st.columns(3)
    with col_time1:
        sleep_before = st.number_input("Sleep Before Launch (minutes)", 0.0, 600.0, 0.0, 0.5, key="main_web_sleep")
    with col_time2:
        pause_between_moves = st.number_input("Move Pause (seconds)", 0.0, 10.0, PAUSE_BETWEEN_MOVES_SECONDS, 0.1, key="main_web_pause_between_moves")
    with col_time3:
        sleep_after_empty = st.number_input("Sleep After EMPTY (minutes)", 0.0, 1000.0, SLEEP_AFTER_EMPTY_STEP, 0.5, key="main_web_sleep_after_empty")

    # Safety limits (web)
    col_lim1, col_lim2, col_lim3, col_lim4 = st.columns(4)
    with col_lim1:
        st.number_input("Max Consecutive Empty Moves", 0, 50, MAX_CONSECUTIVE_EMPTY_MOVES_ALLOWED, 1, key="main_web_max_empty_moves")
    with col_lim2:
        st.number_input("Max Consecutive Something-Is-Wrong", 0, 50, MAX_CONSECUTIVE_SOMETHING_IS_WRONG_ALLOWED, 1, key="main_web_max_siw")
    with col_lim3:
        st.number_input("Max Consecutive Invalid Reversals", 0, 50, MAX_CONSECUTIVE_INVALID_REVERSALS_ALLOWED, 1, key="main_web_max_invalid_rev")
    with col_lim4:
        st.number_input("Max Consecutive NO_PATH_FOUND", 0, 50, MAX_CONSECUTIVE_NO_PATH_FOUND_ALLOWED, 1, key="main_web_max_no_path")

    # Server details
    colh, colp = st.columns(2)
    with colh:
        host = st.selectbox("Host", HOST_CHOICES, index=0, key="main_web_host")
    with colp:
        default_port = random_free_port(8000, 9000)
        port = st.number_input("Port", 1024, 65535, default_port, key="main_web_port")

    # Headless option – runs with --no-gui so move pauses are skipped, useful for
    # speed-oriented batch runs.  Replay mode ignores this flag.
    no_gui = st.checkbox("Run headless (skip GUI & move pauses)", value=False, key="main_web_no_gui")

    if st.button("Start Main Session (Web)", key="start_main_web"):
        selected_max_games = int(st.session_state.get("main_web_max_games", MAX_GAMES_ALLOWED))
        run_main_web(selected_max_games, host, port)
        url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
        st.success(f"Web session started – open {url} in your browser.")


