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
            tab_human_pg,
            tab_human_web,
            tab_main_pg,
            tab_main_web,
            tab_replay_pg,
            tab_replay_web,
            tab_continue_pg,
            tab_continue_web,
        ) = st.tabs(
            [
                "Overview",
                "Human Play (PyGame)",
                "Human Play (Web)",
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

        # ---------------------- Human Play (PyGame) -----------------------
        with tab_human_pg:
            from utils.session_utils import run_human_play
            st.markdown("### 🎮 Play Snake – PyGame Window")
            if st.button("Start PyGame Human Play", key="btn_hp_pygame"):
                run_human_play()

        # ----------------------- Human Play (Web) ------------------------
        with tab_human_web:
            from utils.session_utils import run_human_play_web
            from utils.network_utils import find_free_port
            st.markdown("### 🌐 Play Snake in Browser")
            host = st.selectbox("Host", ["localhost", "0.0.0.0", "127.0.0.1"], index=0, key="hp_web_host")
            port = st.number_input("Port", 1024, 65535, find_free_port(8000), key="hp_web_port")
            if st.button("Start Web Human Play", key="btn_hp_web"):
                run_human_play_web(host, port)

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
