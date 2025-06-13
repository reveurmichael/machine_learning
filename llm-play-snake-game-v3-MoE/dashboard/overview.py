"""Overview & details rendering for the Streamlit Snake dashboard."""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.file_utils import (
    get_folder_display_name,
    load_summary_data,
    load_game_data,
)
from utils.game_stats_utils import filter_experiments, get_experiment_options

__all__ = [
    "display_experiment_overview",
    "display_experiment_details",
]


def display_experiment_overview(log_folders: List[str]):
    """Render the experiment overview table & filters, return filtered DataFrame."""
    if not log_folders:
        st.warning("No experiment logs found.")
        return None

    # Build raw overview DataFrame (largely copied from the original implementation)
    experiments_data: List[Dict[str, Any]] = []
    for folder in log_folders:
        summary_data = load_summary_data(folder)
        if not summary_data:
            continue

        folder_name = get_folder_display_name(folder)
        timestamp = summary_data.get("timestamp", "Unknown")
        config = summary_data.get("configuration", {})

        primary_model = f"{config.get('provider', 'Unknown')}/{config.get('model', 'default')}"
        secondary_model = "None"

        # Providers / models for filtering
        primary_provider = config.get("provider", "Unknown")
        primary_model_name = config.get("model", "default")
        parser_provider = config.get("parser_provider")
        secondary_provider = None
        secondary_model_name = None
        if parser_provider and str(parser_provider).lower() != "none":
            secondary_model = f"{parser_provider}/{config.get('parser_model', 'default')}"
            secondary_provider = parser_provider
            secondary_model_name = config.get("parser_model", "default")

        providers = [primary_provider] + ([secondary_provider] if secondary_provider else [])
        models = [primary_model_name] + ([secondary_model_name] if secondary_model_name else [])

        # Game statistics
        game_stats = summary_data.get("game_statistics", {})
        total_games = game_stats.get("total_games", 0)
        total_score = game_stats.get("total_score", 0)
        total_steps = game_stats.get("total_steps", 0)
        mean_score = total_score / total_games if total_games > 0 else 0
        apples_per_step = total_score / total_steps if total_steps > 0 else 0

        # Step statistics
        step_stats = summary_data.get("step_stats", {})
        valid_steps = step_stats.get("valid_steps", 0)
        empty_steps = step_stats.get("empty_steps", 0)
        something_is_wrong_steps = step_stats.get("something_is_wrong_steps", 0)
        invalid_reversals = step_stats.get("invalid_reversals", 0)

        # JSON parsing statistics
        json_stats = summary_data.get("json_parsing_stats", {})
        total_extractions = json_stats.get("total_extraction_attempts", 0)
        successful_extractions = json_stats.get("successful_extractions", 0)
        json_success_rate = (
            successful_extractions / total_extractions * 100 if total_extractions > 0 else 0
        )

        # Continuation info
        continuation_info = summary_data.get("continuation_info", {})
        is_continuation = continuation_info.get("is_continuation", False)
        continuation_count = continuation_info.get("continuation_count", 0) if is_continuation else 0

        experiments_data.append(
            {
                "Folder": folder,
                "Experiment": folder_name,
                "Timestamp": timestamp,
                "Primary LLM": primary_model,
                "Secondary LLM": secondary_model,
                "Total Games": total_games,
                "Total Score": total_score,
                "Mean Score": mean_score,
                "Apples/Step": apples_per_step,
                "Total Steps": total_steps,
                "Valid Steps": valid_steps,
                "Empty Steps": empty_steps,
                "SOMETHING_IS_WRONG steps": something_is_wrong_steps,
                "Invalid Reversals": invalid_reversals,
                "JSON Success Rate": json_success_rate,
                "providers": providers,
                "models": models,
                "Is Continuation": is_continuation,
                "Continuation Count": continuation_count,
                "primary_provider": primary_provider,
                "primary_model_name": primary_model_name,
                "secondary_provider": secondary_provider,
                "secondary_model_name": secondary_model_name,
            }
        )

    overview_df = pd.DataFrame(experiments_data)

    # Filter widgets (uses utils.game_stats_utils helpers)
    options = get_experiment_options(overview_df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_primary_providers = st.multiselect(
            "Filter by Primary LLM Provider", options=options["primary_providers"], default=[]
        )
    with col2:
        selected_primary_models = st.multiselect(
            "Filter by Primary LLM Model", options=options["primary_models"], default=[]
        )
    with col3:
        selected_secondary_providers = st.multiselect(
            "Filter by Parser LLM Provider", options=options["secondary_providers"], default=[]
        )
    with col4:
        selected_secondary_models = st.multiselect(
            "Filter by Parser LLM Model", options=options["secondary_models"], default=[]
        )

    filtered_df = filter_experiments(
        overview_df,
        selected_primary_providers,
        selected_primary_models,
        selected_secondary_providers,
        selected_secondary_models,
    )

    # Display table (strip helper cols)
    display_df = filtered_df.drop(
        columns=[
            "providers",
            "models",
            "primary_llm",
            "secondary_llm",
            "Folder",
        ],
        errors="ignore",
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    return filtered_df


def display_experiment_details(folder_path: str):
    """Render charts & table for a single experiment."""

    summary_data = load_summary_data(folder_path)
    if not summary_data:
        st.warning("No summary data found.")
        return

    games_data = load_game_data(folder_path)
    if not games_data:
        st.info("No games found in the selected experiment.")
        return

    # ---------- Game score histogram ----------
    st.markdown("## Game Scores Distribution")
    scores = [g.get("score", 0) for g in games_data.values()]
    if scores:
        max_score = max(scores)
        col_hist, col_cfg = st.columns([3, 1])
        with col_cfg:
            bins = st.slider("Bins", 5, 30, min(10, max_score) if max_score else 10)
        with col_hist:
            fig_scores = px.histogram(x=scores, nbins=bins, labels={"x": "Score"})
            st.plotly_chart(fig_scores, use_container_width=True)

    # ---------- LLM response-time histogram ----------
    st.markdown("## Primary LLM Response Time Distribution")
    resp_times: List[float] = []
    for g in games_data.values():
        rounds = g.get("rounds_data", {})
        if isinstance(rounds, dict):
            for rd in rounds.values():
                resp_times.extend(t for t in rd.get("primary_response_times", []) if t > 0)
        if not resp_times:  # fallbacks
            pstats = g.get("prompt_response_stats", {})
            resp_times.extend(
                pstats.get(k, 0)
                for k in (
                    "min_primary_response_time",
                    "avg_primary_response_time",
                    "max_primary_response_time",
                )
                if pstats.get(k, 0) > 0
            )
    if resp_times:
        col_hist, col_cfg = st.columns([3, 1])
        with col_cfg:
            bins_t = st.slider("Bins", 5, 30, 10, key="time_bins")
        with col_hist:
            fig_times = px.histogram(x=resp_times, nbins=bins_t, labels={"x": "Response Time (s)"})
            st.plotly_chart(fig_times, use_container_width=True)
    else:
        st.info("No LLM response time data available for this experiment.")

    # ---------- Game details table ----------
    st.markdown("## Game Details")
    rows: List[Dict[str, Any]] = []
    for num, g in sorted(games_data.items()):
        step_stats = g.get("step_stats", {})
        total_steps = g.get("steps", 0)
        valid_steps = step_stats.get("valid_steps", 0)
        empty_steps = step_stats.get("empty_steps", 0)
        error_steps = step_stats.get("something_is_wrong_steps", 0)

        rows.append(
            {
                "Game": num,
                "Score": g.get("score", 0),
                "Steps": total_steps,
                "Valid %": f"{(valid_steps / total_steps * 100) if total_steps else 0:.1f}%",
                "Empty %": f"{(empty_steps / total_steps * 100) if total_steps else 0:.1f}%",
                "Error %": f"{(error_steps / total_steps * 100) if total_steps else 0:.1f}%",
                "End Reason": g.get("game_end_reason", "Unknown"),
                "Rounds": g.get("round_count", 0),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True) 