"""Streamlit sub-module: *Replay* tab (PyGame & Web variants).

Provides widgets for selecting an experiment + game number and launching
`replay.py` / `replay_web.py` in a background subprocess.  Only UI/command
composition – no heavy logic.
"""
from __future__ import annotations

import streamlit as st
import json
from typing import Sequence

from core.game_file_manager import FileManager
from utils.session_utils import run_replay, run_web_replay
from utils.network_utils import random_free_port
from config.network_constants import HOST_CHOICES

# Initialize file manager for dashboard operations
_file_manager = FileManager()


def render_replay_pygame_tab(log_folders: Sequence[str]) -> None:
    st.markdown("### Replay Game (PyGame)")
    if not log_folders:
        st.warning("No experiment logs found.")
        return
    col_exp, col_game = st.columns(2)
    # Sort experiments alphabetically
    sorted_folders = sorted(log_folders, key=_file_manager.get_folder_display_name)

    with col_exp:
        exp = st.selectbox(
            "Experiment", options=sorted_folders, format_func=_file_manager.get_folder_display_name, key="replay_exp_pg",
            label_visibility="collapsed"
        )
    games = _file_manager.load_game_data(exp)
    if not games:
        st.warning("No games found in the selected experiment.")
        return
    with col_game:
        game_num = st.selectbox(
            "Game", sorted(games), key="replay_game_pg", 
            format_func=lambda x: f"Game {x} (Score: {games[x].get('score',0)})",
            label_visibility="collapsed"
        )

    # ── Optional expander showing raw game_N.json ────────────────────
    if game_num in games:
        with st.expander(f"Show game_{game_num}.json"):
            st.code(json.dumps(games[game_num], indent=2), language="json")

    if st.button("Start Replay", key="start_replay_pg"):
        run_replay(exp, game_num)


def render_replay_web_tab(log_folders: Sequence[str]) -> None:
    st.markdown("### Replay Game (Web)")
    if not log_folders:
        st.warning("No experiment logs found.")
        return
    col_exp, col_game = st.columns(2)
    # Sort experiments alphabetically (reuse sorted folders)
    sorted_folders = sorted(log_folders, key=_file_manager.get_folder_display_name)

    with col_exp:
        exp = st.selectbox(
            "Experiment", options=sorted_folders, format_func=_file_manager.get_folder_display_name, key="replay_exp_web",
            label_visibility="collapsed"
        )
    games = _file_manager.load_game_data(exp)
    if not games:
        st.warning("No games found in the selected experiment.")
        return
    with col_game:
        game_num = st.selectbox(
            "Game", sorted(games), key="replay_game_web", 
            format_func=lambda x: f"Game {x} (Score: {games[x].get('score',0)})",
            label_visibility="collapsed"
        )

    col1, col2 = st.columns(2)
    with col1:
        host = st.selectbox(
            "Host",
            HOST_CHOICES,
            index=0,
            key="replay_web_host",
        )
    with col2:
        default_port = random_free_port()
        port = st.number_input("Port", 1024, 65535, default_port, key="replay_web_port")

    if game_num in games:
        with st.expander(f"Show game_{game_num}.json"):
            st.code(json.dumps(games[game_num], indent=2), language="json")

    if st.button("Start Web Replay", key="start_replay_web"):
        run_web_replay(exp, game_num, host, port) 