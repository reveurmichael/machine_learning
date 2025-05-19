import streamlit as st
import os
from .base_selenium import BaseSelenium


def configure_page_navigation(current_page):
    """Configure the navigation sidebar for all pages"""
    # Add sidebar title
    st.sidebar.markdown("# 🤖 Selenium WebDriver Setup")

    # Simplified pages configuration - removed log_viewer and screenshots_viewer
    pages_config = {
        "app": {"title": "🏠 Home", "icon": "🏠"},
        "config_editor": {"title": "⚙️ Configurations", "icon": "⚙️"},
        "readme": {"title": "📖 Documentation", "icon": "📖"},
    }

    css_path = os.path.join(os.path.dirname(__file__), "streamlit_app/css/sidebar_navigation.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Create custom navigation menu
    nav_html = '<div class="sidebar-nav">'
    for page_id, page_info in pages_config.items():
        active_class = "active" if page_id == current_page else ""
        nav_html += f'<a href="{page_id}" class="nav-link {active_class}">{page_info["icon"]} {page_info["title"].replace(page_info["icon"] + " ", "")}</a>'
    nav_html += "</div>"

    # Render the navigation
    st.sidebar.markdown(nav_html, unsafe_allow_html=True)
