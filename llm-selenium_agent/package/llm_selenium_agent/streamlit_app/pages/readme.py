import streamlit as st
import os
from llm_selenium_agent.streamlit_tools import configure_page_navigation
import llm_selenium_agent.config


class Readme:
    def __init__(self):
        self.setup_page_config()
        configure_page_navigation("readme")
        st.markdown(
            '<h1 class="common-title">📖 Readme</h1>',
            unsafe_allow_html=True,
        )
        self.display_readme()

    def setup_page_config(self):
        st.set_page_config(
            page_title="📖 Readme",
            page_icon="📖",
            layout="centered",
            initial_sidebar_state="expanded",
        )

        # Load and inject CSS
        common_css_path = os.path.join(os.path.dirname(__file__), "../css/common.css")
        with open(common_css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        readme_css_path = os.path.join(os.path.dirname(__file__), "../css/readme.css")
        with open(readme_css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def display_readme(self):
        dir = os.path.dirname(llm_selenium_agent.config.__file__)
        readme_path = os.path.join(dir, "Readme.md")
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
            # Render the Markdown content directly
            st.markdown(readme_content, unsafe_allow_html=True)
        else:
            st.error("Readme.md file not found.")


if __name__ == "__main__":
    Readme()
