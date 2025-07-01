from __future__ import annotations

"""Dashboard – Human Play tabs (PyGame / Web)

Encapsulates the UI for launching human-play modes so *app.py* stays clean.
"""

import streamlit as st

from utils.session_utils import run_human_play, run_human_play_web
from config.network_constants import HOST_CHOICES

__all__ = [
    "render_human_pygame_tab",
    "render_human_web_tab",
]


def render_human_pygame_tab() -> None:
    """Render the *Human Play (PyGame)* tab contents."""
    st.markdown("### Play Snake – PyGame Window")
    if st.button("Start PyGame Human Play", key="btn_hp_pygame"):
        run_human_play()


def render_human_web_tab() -> None:
    """Render the *Human Play (Web)* tab contents."""
    st.markdown("### Play Snake in Browser")
    
    st.info("🌐 **Dynamic Port Allocation**: The web application will automatically find an available port.")
    
    host = st.selectbox(
        "Host",
        HOST_CHOICES,
        index=0,
        key="hp_web_host",
        help="Host address for the web server"
    )

    if st.button("Start Web Human Play", key="btn_hp_web"):
        run_human_play_web(host)
