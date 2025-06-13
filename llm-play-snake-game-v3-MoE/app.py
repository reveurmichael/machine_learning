"""
Snake Game Analytics Dashboard

A Streamlit app for analyzing, replaying, and continuing recorded Snake game sessions.
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import subprocess
import dashboard.tabs as tabs
from streamlit.errors import StreamlitAPIException

# Import utilities
from utils.game_stats_utils import filter_experiments, get_experiment_options

# Core file utilities – replaces former *dashboard.helpers* import
from utils.file_utils import find_valid_log_folders

# Set page configuration
st.set_page_config(
    page_title="Snake Game Analytics",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# No custom CSS for better appearance - removed as requested

# Function to display experiment overview
def display_experiment_overview(log_folders):
    """Wrapper – real implementation lives in *dashboard.overview*."""
    from dashboard.overview import display_experiment_overview as _impl
    return _impl(log_folders)

def display_experiment_details(folder_path):
    """Wrapper – real implementation lives in *dashboard.overview*."""
    from dashboard.overview import display_experiment_details as _impl
    return _impl(folder_path)
        
# ---------------------------------------------------------------------------
# Extra helpers for the new tab layout (Main-/Continue-Mode via main_web.py)
# ---------------------------------------------------------------------------

def main():
    """Main application function."""
    # Display header
    st.title("🐍 Snake Game Analytics Dashboard")
    
    # Discover experiment folders (summary.json + games + prompts + responses)
    log_folders = find_valid_log_folders("logs")
    
    # ------------------------------------------------------------------
    # Tabs: Overview ▸ Main (PyGame) ▸ Main (Web) ▸ Replay (PyGame)
    #       ▸ Replay (Web) ▸ Continue (PyGame) ▸ Continue (Web)
    # ------------------------------------------------------------------
    (
        tab_overview,
        tab_main_pg,
        tab_main_web,
        tab_replay_pg,
        tab_replay_web,
        tab_continue_pg,
        tab_continue_web,
    ) = st.tabs([
        "Overview",
        "Main Mode (PyGame)",
        "Main Mode (Web)",
        "Replay Mode (PyGame)",
        "Replay Mode (Web)",
        "Continue Mode (PyGame)",
        "Continue Mode (Web)",
    ])
    
    # --------------------------- Overview ---------------------------
    with tab_overview:
        tabs.render_overview_tab(log_folders)
    
    # ---------------------- Main Mode (PyGame) ---------------------
    with tab_main_pg:
        tabs.render_main_pygame_tab()
    
    # ----------------------- Main Mode (Web) -----------------------
    with tab_main_web:
        tabs.render_main_web_tab()
    
    # --------------------- Replay Mode (PyGame) --------------------
    with tab_replay_pg:
        tabs.render_replay_pygame_tab(log_folders)
    
    # ---------------------- Replay Mode (Web) ----------------------
    with tab_replay_web:
        tabs.render_replay_web_tab(log_folders)
    
    # ------------------- Continue Mode (PyGame) --------------------
    with tab_continue_pg:
        tabs.render_continue_pygame_tab(log_folders)
    
    # -------------------- Continue Mode (Web) ----------------------
    with tab_continue_web:
        tabs.render_continue_web_tab(log_folders)



class App:
    """Entry-point wrapper for the Streamlit dashboard (OOP style)."""

    def __init__(self):
        self.setup_page_config()
        self.main()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def setup_page_config(self):
        """Ensure Streamlit page config is set (no-op if already done)."""
        try:
            st.set_page_config(
                page_title="Snake Game Analytics",
                page_icon="🐍",
                layout="wide",
                initial_sidebar_state="expanded",
            )
        except StreamlitAPIException:
            # set_page_config may only be called once per app; ignore second call
            pass

    def main(self):
        # Delegate to the functional implementation above for now.
        main()

if __name__ == "__main__":
    App()
