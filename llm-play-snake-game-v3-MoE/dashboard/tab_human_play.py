from __future__ import annotations

"""Dashboard – Human Play tabs (PyGame / Web)

Encapsulates the UI for launching human-play modes so *app.py* stays clean.
"""

import streamlit as st

from utils.session_utils import run_human_play, run_human_play_web
from utils.network_utils import random_free_port

__all__ = [
    "render_human_pygame_tab",
    "render_human_web_tab",
]


def render_human_pygame_tab() -> None:
    """Render the *Human Play (PyGame)* tab contents."""
    st.markdown("### 🎮 Play Snake – PyGame Window")
    if st.button("Start PyGame Human Play", key="btn_hp_pygame"):
        run_human_play()


def render_human_web_tab() -> None:
    """Render the *Human Play (Web)* tab contents."""
    st.markdown("### 🌐 Play Snake in Browser")

    col1, col2 = st.columns(2)
    with col1:
        host = st.selectbox(
            "Host",
            ["localhost", "0.0.0.0", "127.0.0.1"],
            index=0,
            key="hp_web_host",
        )
    with col2:
        port = st.number_input(
            "Port", 1024, 65535, random_free_port(), key="hp_web_port"
        )

    if st.button("Start Web Human Play", key="btn_hp_web"):
        run_human_play_web(host, port)
