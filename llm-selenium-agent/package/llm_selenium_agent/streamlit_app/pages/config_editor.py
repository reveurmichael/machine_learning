import streamlit as st
import yaml
import time
import os
import datetime
from llm_selenium_agent.config import *
from llm_selenium_agent.streamlit_tools import configure_page_navigation
from llm_selenium_agent.logger import logger, log_function_call
import pandas as pd


class ConfigEditor:
    def __init__(self):
        self.config = load_configuration()
        self.setup_page_config()
        configure_page_navigation("config_editor")

        if "chrome_options" not in self.config:
            self.config["chrome_options"] = {
                "ignore-certificate-errors": "true"
            }
            update_configuration(self.config)  

        # Ensure 'selenium' key exists in the config
        if "selenium" not in self.config:
            self.config["selenium"] = {
                "chrome_driver_path": "/path/to/chromedriver",
                "firefox_driver_path": "/path/to/geckodriver",
                "headless": "false"
            }
            update_configuration(self.config)  
        else:
            # Ensure 'headless' key exists within 'selenium'
            if "headless" not in self.config["selenium"]:
                self.config["selenium"]["headless"] = "false"
                update_configuration(self.config)  

        # Ensure 'streamlit' key exists in the config
        if "streamlit" not in self.config:
            self.config["streamlit"] = {
                "address": "127.0.0.1",
                "allow_run_on_save": "true",
                "port": 8501
            }
            update_configuration(self.config)  

        if "sort_by_task_id" not in self.config:
            self.config["sort_by_task_id"] = "true"
            update_configuration(self.config)  

        self.count = 0
        self.main()

    def edit_chrome_options(self):
        """Edit Chrome-specific options."""
        st.markdown(
            "<div class='section-header'>🔧 Chrome Options</div>",
            unsafe_allow_html=True,
        )
        chrome_options = self.config.get("chrome_options", {})

        # 🔒 Ignore Certificate Errors
        ignore_certificate_errors = st.checkbox(
            "🔒 Ignore Certificate Errors",
            value=chrome_options.get("ignore-certificate-errors", False),
            help="Ignore SSL certificate errors in Chrome.",
            key="ignore_certificate_errors",
        )
        self.config["chrome_options"]["ignore-certificate-errors"] = str(
            ignore_certificate_errors
        ).lower()
        update_configuration(self.config)

    def setup_page_config(self):
        """Set up the Streamlit page configuration and custom CSS."""
        st.set_page_config(
            page_title="⚙️ Configuration Editor",
            page_icon="⚙️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Load and inject CSS
        common_css_path = os.path.join(os.path.dirname(__file__), "../css/common.css")
        with open(common_css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        css_path = os.path.join(os.path.dirname(__file__), "../css/config_editor.css")
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def main(self):
        """Main method to display and edit configuration settings."""
        st.markdown(
            '<h1 class="common-title">⚙️ Configuration Editor</h1>',
            unsafe_allow_html=True,
        )

        # Improved informational message about automatic saving
        st.warning(
            "⚠️ All changes on this page are automatically saved to `config.yml` in real-time. Please ensure accuracy before making modifications."
        )

        # Display and edit configuration sections
        self.edit_selenium_settings()
        self.edit_streamlit_settings()
        self.edit_chrome_options()

        # View config.yml Section
        config_path = os.path.join(os.getcwd(), "config.yml")
        if os.path.exists(config_path):
            last_updated = datetime.datetime.fromtimestamp(
                os.path.getmtime(config_path)
            ).strftime("%Y-%m-%d %H:%M:%S")
            with st.expander(
                f"View config.yml (Last Updated: {last_updated})",
                icon="📄",
                expanded=True,
            ):
                st.code(f"Location: {config_path}")
                with open(config_path, "r") as file:
                    config_content = file.read()
                st.code(config_content, language="yaml")
        else:
            st.error(f"❌ `config.yml` not found at {config_path}")

    def edit_selenium_settings(self):
        """Edit Selenium related configurations."""
        st.markdown(
            "<div class='section-header'>🔧 Selenium Settings</div>",
            unsafe_allow_html=True,
        )
        selenium_config = self.config.get("selenium", {})

        col1, col2 = st.columns(2)

        with col1:
            # Headless Mode
            headless = st.checkbox(
                "Headless Mode",
                value=str(selenium_config.get("headless", False)).lower() == "true",
                help="Run browsers in headless mode.",
                key="headless_mode",
            )
            self.config["selenium"]["headless"] = str(headless).lower()
            update_configuration(self.config)
            time.sleep(0.3)

    def edit_streamlit_settings(self):
        """Edit Streamlit server configurations."""
        st.markdown(
            "<div class='section-header'>📡 Streamlit Server Settings</div>",
            unsafe_allow_html=True,
        )
        streamlit_config = self.config.get("streamlit", {})

        col1, col2 = st.columns(2)

        with col1:
            # Streamlit Address
            address = st.selectbox(
                "🌐 Streamlit Address",
                options=["127.0.0.1", "0.0.0.0"],
                index=(
                    0
                    if streamlit_config.get("address", "127.0.0.1") == "127.0.0.1"
                    else 1
                ),
                help="Address where the Streamlit server listens.",
                key="streamlit_address",
            )
            self.config["streamlit"]["address"] = address
            update_configuration(self.config)

        with col2:
            # Streamlit Port
            port = st.number_input(
                "🔢 Streamlit Port",
                min_value=1024,
                max_value=65535,
                value=int(streamlit_config.get("port", 8501)),
                step=1,
                help="Port number for the Streamlit server.",
                key="streamlit_port",
            )
            self.config["streamlit"]["port"] = port
            update_configuration(self.config)

        with col1:
            # Allow Run on Save
            allow_run_on_save = st.checkbox(
                "🔄 Allow Run on Save",
                value=str(streamlit_config.get("allow_run_on_save", True)).lower()
                == "true",
                help="Automatically rerun the app upon saving changes.",
                key="allow_run_on_save",
            )
            self.config["streamlit"]["allow_run_on_save"] = str(
                allow_run_on_save
            ).lower()
            update_configuration(self.config)
            time.sleep(0.3)


if __name__ == "__main__":
    ConfigEditor()
