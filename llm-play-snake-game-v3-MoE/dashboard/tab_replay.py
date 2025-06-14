"""
Dashboard – Replay Mode tabs
"""
from __future__ import annotations

import streamlit as st
from utils.file_utils import get_folder_display_name, load_game_data
from utils.session_utils import run_replay, run_web_replay
from utils.network_utils import random_free_port


def render_replay_pygame_tab(log_folders):
    st.markdown("### Replay Game (PyGame)")
    if not log_folders:
        st.warning("No experiment logs found.")
        return
    col_exp, col_game = st.columns(2)
    with col_exp:
        exp = st.selectbox(
            "Experiment", options=log_folders, format_func=get_folder_display_name, key="replay_exp_pg",
            label_visibility="collapsed"
        )
    games = load_game_data(exp)
    if not games:
        st.warning("No games found in the selected experiment.")
        return
    with col_game:
        game_num = st.selectbox(
            "Game", sorted(games), format_func=lambda x: f"Game {x} (Score: {games[x].get('score',0)})",
            label_visibility="collapsed"
        )
    if st.button("Start Replay", key="start_replay_pg"):
        run_replay(exp, game_num)


def render_replay_web_tab(log_folders):
    st.markdown("### Replay Game (Web)")
    if not log_folders:
        st.warning("No experiment logs found.")
        return
    col_exp, col_game = st.columns(2)
    with col_exp:
        exp = st.selectbox(
            "Experiment", options=log_folders, format_func=get_folder_display_name, key="replay_exp_web",
            label_visibility="collapsed"
        )
    games = load_game_data(exp)
    if not games:
        st.warning("No games found in the selected experiment.")
        return
    with col_game:
        game_num = st.selectbox(
            "Game", sorted(games), key="replay_game_web", label_visibility="collapsed",
            format_func=lambda x: f"Game {x} (Score: {games[x].get('score',0)})"
        )
    col1, col2 = st.columns(2)
    with col1:
        host = st.selectbox("Host", ["localhost", "0.0.0.0", "127.0.0.1"], index=0, key="replay_web_host")
    with col2:
        default_port = random_free_port(8000, 9000)
        port = st.number_input("Port", 1024, 65535, default_port, key="replay_web_port")
    if st.button("Start Web Replay"):
        run_web_replay(exp, game_num, host, port) 