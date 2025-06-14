"""
Dashboard – Main Mode tabs (PyGame / Web)

Provides UI widgets to launch *main.py* or *main_web.py* with the full set of
CLI arguments requested by the user.
"""

from __future__ import annotations

import subprocess
import streamlit as st
import importlib
import pathlib

from utils.network_utils import find_free_port, random_free_port
from utils.session_utils import run_main_web

# ---------------------------------------------------------------------------
# Helper – build command list from optional args
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main Mode – PyGame
# ---------------------------------------------------------------------------

def render_main_pygame_tab():
    st.markdown("### Start New Game Session (PyGame)")

    # ----- Basic arguments -----
    max_games = st.number_input("Maximum Games", 1, 100, 2, 1, key="main_pg_max_games")

    # Provider / model
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", AVAILABLE_PROVIDERS, index=AVAILABLE_PROVIDERS.index(DEFAULT_PROVIDER) if DEFAULT_PROVIDER in AVAILABLE_PROVIDERS else 0, key="main_pg_provider")
    with col2:
        model = st.text_input("Model", value=DEFAULT_MODEL, key="main_pg_model")

    # Parser LLM
    colp1, colp2 = st.columns(2)
    with colp1:
        parser_provider = st.selectbox("Parser Provider", ["None"] + AVAILABLE_PROVIDERS, index=AVAILABLE_PROVIDERS.index(DEFAULT_PARSER_PROVIDER)+1 if DEFAULT_PARSER_PROVIDER in AVAILABLE_PROVIDERS else 0, key="main_pg_parser_provider")
    with colp2:
        parser_model = st.text_input("Parser Model", value=DEFAULT_PARSER_MODEL, key="main_pg_parser_model")

    # Limits / misc
    colm1, colm2 = st.columns(2)
    with colm1:
        max_steps = st.number_input("Max Steps", 100, 100_000, 400, 100, key="main_pg_max_steps")
    with colm2:
        sleep_before = st.number_input("Sleep Before Launch (minutes)", 0.0, 60.0, 0.0, 0.5, key="main_pg_sleep")

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
        if no_gui:
            cmd.append("--no-gui")

        subprocess.Popen(cmd)
        st.success("Main PyGame session launched – check your terminal/window.")


# ---------------------------------------------------------------------------
# Main Mode – Web
# ---------------------------------------------------------------------------

def render_main_web_tab():
    st.markdown("### Start New Game Session (Web)")

    # Basic args
    max_games = st.number_input("Maximum Games", 1, 100, 2, 1, key="main_web_max_games")

    # Provider / model
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", AVAILABLE_PROVIDERS, index=AVAILABLE_PROVIDERS.index(DEFAULT_PROVIDER) if DEFAULT_PROVIDER in AVAILABLE_PROVIDERS else 0, key="main_web_provider")
    with col2:
        model = st.text_input("Model", value=DEFAULT_MODEL, key="main_web_model")

    # Parser
    colp1, colp2 = st.columns(2)
    with colp1:
        parser_provider = st.selectbox("Parser Provider", ["None"] + AVAILABLE_PROVIDERS, index=AVAILABLE_PROVIDERS.index(DEFAULT_PARSER_PROVIDER)+1 if DEFAULT_PARSER_PROVIDER in AVAILABLE_PROVIDERS else 0, key="main_web_parser_provider")
    with colp2:
        parser_model = st.text_input("Parser Model", value=DEFAULT_PARSER_MODEL, key="main_web_parser_model")

    # Limits / misc
    colm1, colm2 = st.columns(2)
    with colm1:
        max_steps = st.number_input("Max Steps", 100, 100_000, 400, 100, key="main_web_max_steps")
    with colm2:
        sleep_before = st.number_input("Sleep Before Launch (minutes)", 0.0, 60.0, 0.0, 0.5, key="main_web_sleep")

    # Server details
    colh, colp = st.columns(2)
    with colh:
        host = st.selectbox("Host", ["localhost", "0.0.0.0", "127.0.0.1"], index=0, key="main_web_host")
    with colp:
        default_port = random_free_port(8000, 9000)
        port = st.number_input("Port", 1024, 65535, default_port, key="main_web_port")

    no_gui = st.checkbox("Disable GUI (headless)", value=False, key="main_web_no_gui")

    if st.button("Start Main Session (Web)", key="start_main_web"):
        run_main_web(max_games, host, port)
        url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
        st.success(f"Web session started – open {url} in your browser.")


# ---------------------------------------------------------------------------
# Discover provider modules
# ---------------------------------------------------------------------------

# Discover provider modules once
_PROVIDER_FILES = pathlib.Path("llm/providers").glob("*_provider.py")
AVAILABLE_PROVIDERS = sorted({p.stem.replace("_provider", "") for p in _PROVIDER_FILES if p.stem != "base_provider"})
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "deepseek-r1:14b"
DEFAULT_PARSER_PROVIDER = "ollama"
DEFAULT_PARSER_MODEL = "gemma3:12b-it-qat"
