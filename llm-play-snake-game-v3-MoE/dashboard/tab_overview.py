"""
Dashboard – Overview tab renderer.
"""
from __future__ import annotations

import streamlit as st
from utils.file_utils import get_folder_display_name
from dashboard.overview import (
    display_experiment_overview,
    display_experiment_details,
)

def render_overview_tab(log_folders):
    st.markdown("### Experiment Overview")
    st.markdown("View statistics and detailed information about all experiments.")
    overview_df = display_experiment_overview(log_folders)
    if overview_df is not None and not overview_df.empty:
        st.markdown("### Experiment Details")
        selected_exp = st.selectbox(
            "Select Experiment",
            options=overview_df["Folder"].tolist(),
            format_func=get_folder_display_name,
            index=0,
            key="overview_exp_select",
        )
        display_experiment_details(selected_exp) 