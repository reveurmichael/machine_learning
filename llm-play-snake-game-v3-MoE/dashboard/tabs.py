"""Individual Streamlit tab renderers for the Snake analytics dashboard."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.overview import (
    display_experiment_overview,
    display_experiment_details,
)

# Import renamed helper modules
from utils.file_utils import (
    get_folder_display_name,
    load_game_data,
)

from utils.session_utils import (
    run_replay,
    run_web_replay,
    run_main_web,
    continue_game,
    continue_game_web,
)

__all__ = [
    "render_overview_tab",
    "render_main_pygame_tab",
    "render_main_web_tab",
    "render_replay_pygame_tab",
    "render_replay_web_tab",
    "render_continue_pygame_tab",
    "render_continue_web_tab",
]

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def render_overview_tab(log_folders):
    st.markdown("### Experiment Overview")
    st.markdown("View statistics and detailed information about all experiments.")
    overview_df = display_experiment_overview(log_folders)
    if overview_df is not None and not overview_df.empty:
        st.markdown("### Experiment Details")
        selected_exp = st.selectbox(
            "Select Experiment",
            options=overview_df["Folder"].tolist(),
            format_func=get_folder_display_name,
            index=0,
        )
        display_experiment_details(selected_exp)

# ---------------------------------------------------------------------------
# Main mode – PyGame
# ---------------------------------------------------------------------------

def render_main_pygame_tab():
    st.markdown("### Start New Game Session (PyGame)")
    max_games = st.number_input("Maximum Games", 1, 100, 10, 1)
    no_gui = st.checkbox("Disable GUI", value=False)
    if st.button("Start Main Session (PyGame)"):
        import subprocess
        cmd = ["python", "main.py", "--max-games", str(max_games)]
        if no_gui:
            cmd.append("--no-gui")
        subprocess.Popen(cmd)
        st.success("Main session started – check your terminal/PyGame window.")

# ---------------------------------------------------------------------------
# Main mode – Web
# ---------------------------------------------------------------------------

def render_main_web_tab():
    st.markdown("### Start New Game Session (Web)")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_games = st.number_input("Max Games", 1, 100, 10, 1, key="main_web_max_games")
    with col2:
        host = st.selectbox("Host", ["localhost", "0.0.0.0", "127.0.0.1"], index=0, key="main_web_host")
    with col3:
        port = st.number_input("Port", 1024, 65535, 8000, key="main_web_port")
    if st.button("Start Main Session (Web)"):
        run_main_web(max_games, host, port)

# ---------------------------------------------------------------------------
# Replay – PyGame
# ---------------------------------------------------------------------------

def render_replay_pygame_tab(log_folders):
    st.markdown("### Replay Game (PyGame)")
    if not log_folders:
        st.warning("No experiment logs found.")
        return
    exp = st.selectbox(
        "Select Experiment", options=log_folders, format_func=get_folder_display_name, key="replay_exp_pg"
    )
    games = load_game_data(exp)
    if not games:
        st.warning("No games found in the selected experiment.")
        return
    game_num = st.selectbox(
        "Select Game", sorted(games), format_func=lambda x: f"Game {x} (Score: {games[x].get('score',0)})"
    )
    if st.button("Start Replay", key="start_replay_pg"):
        run_replay(exp, game_num)

# ---------------------------------------------------------------------------
# Replay – Web
# ---------------------------------------------------------------------------

def render_replay_web_tab(log_folders):
    st.markdown("### Replay Game (Web)")
    if not log_folders:
        st.warning("No experiment logs found.")
        return
    exp = st.selectbox(
        "Select Experiment", options=log_folders, format_func=get_folder_display_name, key="replay_exp_web"
    )
    games = load_game_data(exp)
    if not games:
        st.warning("No games found in the selected experiment.")
        return
    game_num = st.selectbox("Select Game", sorted(games), key="replay_game_web")
    col1, col2 = st.columns(2)
    with col1:
        host = st.selectbox("Host", ["localhost", "0.0.0.0", "127.0.0.1"], index=0, key="replay_web_host")
    with col2:
        port = st.number_input("Port", 1024, 65535, 8000, key="replay_web_port")
    if st.button("Start Web Replay"):
        run_web_replay(exp, game_num, host, port)

# ---------------------------------------------------------------------------
# Continue – PyGame
# ---------------------------------------------------------------------------

def render_continue_pygame_tab(log_folders):
    st.markdown("### Continue Game (PyGame)")
    if not log_folders:
        st.warning("No experiment logs found.")
        return
    exp = st.selectbox(
        "Select Experiment", options=log_folders, format_func=get_folder_display_name, key="cont_exp_pg"
    )
    max_games = st.number_input("Max Games", 1, 100, 10, 1, key="cont_pg_max_games")
    no_gui = st.checkbox("Disable GUI", value=False, key="cont_pg_no_gui")
    if st.button("Start Continuation (PyGame)"):
        continue_game(exp, max_games, no_gui)

# ---------------------------------------------------------------------------
# Continue – Web
# ---------------------------------------------------------------------------

def render_continue_web_tab(log_folders):
    st.markdown("### Continue Session (Web)")
    if not log_folders:
        st.warning("No experiment logs found.")
        return
    exp = st.selectbox(
        "Select Experiment", options=log_folders, format_func=get_folder_display_name, key="cont_exp_web"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        max_games = st.number_input("Max Games", 1, 100, 10, 1, key="cont_web_max_games")
    with col2:
        host = st.selectbox("Host", ["localhost", "0.0.0.0", "127.0.0.1"], index=0, key="cont_web_host")
    with col3:
        port = st.number_input("Port", 1024, 65535, 8000, key="cont_web_port")
    if st.button("Start Continuation (Web)"):
        continue_game_web(exp, max_games, host, port) 