"""
Dashboard – Continue Mode tabs (PyGame / Web)

Launch continuation sessions with extended CLI arguments.
"""

from __future__ import annotations

import subprocess
import streamlit as st
import json

from config.constants import MAX_GAMES_ALLOWED
from utils.file_utils import (
    get_folder_display_name,
    load_summary_data,
    get_total_games,
)
from utils.network_utils import random_free_port
from utils.session_utils import continue_game_web

# Helper for building cmd (reuse from tab_main)

def _append_arg(cmd: list[str], flag: str, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


def render_continue_pygame_tab(log_folders):
    st.markdown("### Continue Game Session (PyGame)")
    if not log_folders:
        st.warning("No experiment logs found.")
        return

    col_exp, col_info = st.columns(2)

    # Sort experiments alphabetically
    sorted_folders = sorted(log_folders, key=get_folder_display_name)

    with col_exp:
        exp = st.selectbox(
            "Experiment",
            options=sorted_folders,
            format_func=get_folder_display_name,
            key="cont_pg_exp",
            label_visibility="collapsed",
        )

    # Determine number of games already completed (summary.json preferred,
    # fallback to counting game_*.json files)
    total_games_finished: int = 0

    with col_info:
        if exp:
            total_games_finished = get_total_games(exp)
            st.info(f"Games completed so far: {total_games_finished}")
            # Still load full summary if users want to inspect it
            summary_data = load_summary_data(exp)

    # optional expander under full width
    if exp and summary_data:
        with st.expander("Show summary.json"):
            st.code(json.dumps(summary_data, indent=2), language="json")


    col1, col2 = st.columns(2)
    with col1:
        max_games = st.number_input(
            "Max Games",
            min_value=1,
            max_value=1000000000000,
            value=total_games_finished,
            step=1,
            key="cont_pg_max_games",
        )
    with col2:
        sleep_before = st.number_input(
            "Sleep Before Launch (minutes)", 0.0, 600.0, 0.0, 0.5, key="cont_pg_sleep"
        )

    no_gui = st.checkbox("Disable GUI", value=False, key="cont_pg_no_gui")

    if st.button("Start Continuation (PyGame)", key="start_cont_pg"):
        cmd = [
            "python", "main.py", "--continue-with-game-in-dir", exp, "--max-games", str(max_games)
        ]
        if sleep_before > 0:
            _append_arg(cmd, "--sleep-before-launching", sleep_before)
        if no_gui:
            cmd.append("--no-gui")
        subprocess.Popen(cmd)
        st.success("Continuation started in background.")


def render_continue_web_tab(log_folders):
    st.markdown("### Continue Game Session (Web)")
    if not log_folders:
        st.warning("No experiment logs found.")
        return
    
    col_exp_w, col_info_w = st.columns(2)

    # Sort experiments alphabetically
    sorted_folders = sorted(log_folders, key=get_folder_display_name)

    with col_exp_w:
        exp = st.selectbox(
            "Experiment",
            options=sorted_folders,
            format_func=get_folder_display_name,
            key="cont_web_exp",
            label_visibility="collapsed",
        )

    # Determine number of games already completed
    total_games_finished: int = 0

    with col_info_w:
        if exp:
            total_games_finished = get_total_games(exp)
            st.info(f"Games completed so far: {total_games_finished}")
            summary_data = load_summary_data(exp)

    if exp and summary_data:
        with st.expander("Show summary.json"):
            st.code(json.dumps(summary_data, indent=2), language="json")

    # Determine sensible defaults / bounds for Max Games widget.
    default_max_games = total_games_finished or MAX_GAMES_ALLOWED
    upper_limit = max(100, default_max_games)

    col1, col2 = st.columns(2)
    with col1:
        max_games = st.number_input(
            "Max Games",
            min_value=1,
            max_value=upper_limit,
            value=default_max_games,
            step=1,
            key="cont_web_max_games",
        )
    with col2:
        sleep_before = st.number_input(
            "Sleep Before Launch (minutes)", 0.0, 600.0, 0.0, 0.5, key="cont_web_sleep"
        )

    colh, colp = st.columns(2)
    with colh:
        host = st.selectbox("Host", ["localhost", "0.0.0.0", "127.0.0.1"], index=0, key="cont_web_host")
    with colp:
        default_port = random_free_port()
        port = st.number_input("Port", 1024, 65535, default_port, key="cont_web_port")

    no_gui = st.checkbox("Disable GUI", value=False, key="cont_web_no_gui")

    if st.button("Start Continuation (Web)", key="start_cont_web"):
        continue_game_web(exp, max_games, host, port, sleep_before)
        url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
        st.success(f"Continuation (web) started – open {url} to watch.") 
