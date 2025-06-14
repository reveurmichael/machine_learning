"""
Snake Game Analytics Dashboard

A Streamlit app for analyzing, replaying, and continuing recorded Snake game sessions.
"""

import streamlit as st
from streamlit.errors import StreamlitAPIException
from utils.file_utils import find_valid_log_folders


class App:
    def __init__(self):
        self.setup_page_config()
        self.main()

    def setup_page_config(self):
        try:
            st.set_page_config(
                page_title="Snake Game Analytics",
                page_icon="🐍",
                layout="wide",
                initial_sidebar_state="expanded",
            )
        except StreamlitAPIException:
            pass

    def main(self):
        st.title("🐍 Snake Game Analytics Dashboard")

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
        ) = st.tabs(
            [
                "Overview",
                "Main Mode (PyGame)",
                "Main Mode (Web)",
                "Replay Mode (PyGame)",
                "Replay Mode (Web)",
                "Continue Mode (PyGame)",
                "Continue Mode (Web)",
            ]
        )

        # --------------------------- Overview ---------------------------
        with tab_overview:
            from dashboard.tab_overview import render_overview_tab

            render_overview_tab(log_folders)

        # ---------------------- Main Mode (PyGame) ---------------------
        with tab_main_pg:
            from dashboard.tab_main import render_main_pygame_tab

            render_main_pygame_tab()

        # ----------------------- Main Mode (Web) -----------------------
        with tab_main_web:
            from dashboard.tab_main import render_main_web_tab

            render_main_web_tab()

        # --------------------- Replay Mode (PyGame) --------------------
        with tab_replay_pg:
            from dashboard.tab_replay import render_replay_pygame_tab

            render_replay_pygame_tab(log_folders)

        # ---------------------- Replay Mode (Web) ----------------------
        with tab_replay_web:
            from dashboard.tab_replay import render_replay_web_tab

            render_replay_web_tab(log_folders)

        # ------------------- Continue Mode (PyGame) --------------------
        with tab_continue_pg:
            from dashboard.tab_continue import render_continue_pygame_tab

            render_continue_pygame_tab(log_folders)

        # -------------------- Continue Mode (Web) ----------------------
        with tab_continue_web:
            from dashboard.tab_continue import render_continue_web_tab
            render_continue_web_tab(log_folders)


if __name__ == "__main__":
    App()
