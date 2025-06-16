"""
Dashboard – Overview tab renderer.
"""
from __future__ import annotations

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


def render_overview_tab(log_folders):
    st.markdown("### Experiment Overview")
    st.markdown("View statistics and detailed information about all experiments.")
    overview_df = display_experiment_overview(log_folders)
    if overview_df is not None and not overview_df.empty:
        st.markdown("### Experiment Details")
        # Sort experiments alphabetically for easier navigation
        exp_options = sorted(
            overview_df["Folder"].tolist(), key=get_folder_display_name
        )

        selected_exp = st.selectbox(
            "Select Experiment",
            options=exp_options,
            format_func=get_folder_display_name,
            index=0,
            key="overview_exp_select",
        )
        display_experiment_details(selected_exp) 


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

        primary_model = (
            f"{config.get('provider', 'Unknown')}/{config.get('model', 'default')}"
        )
        secondary_model = "None"

        # Providers / models for filtering
        primary_provider = config.get("provider", "Unknown")
        primary_model_name = config.get("model", "default")
        parser_provider = config.get("parser_provider")
        secondary_provider = None
        secondary_model_name = None
        if parser_provider and str(parser_provider).lower() != "none":
            secondary_model = (
                f"{parser_provider}/{config.get('parser_model', 'default')}"
            )
            secondary_provider = parser_provider
            secondary_model_name = config.get("parser_model", "default")

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

        # Time statistics / token usage stats
        time_stats = summary_data.get("time_stats", {})
        total_llm_comm_time = time_stats.get("total_llm_communication_time", 0)

        token_stats = summary_data.get("token_usage_stats", {})
        primary_tokens = token_stats.get("primary_llm", {}).get("total_tokens", 0)
        secondary_tokens = token_stats.get("secondary_llm", {}).get("total_tokens", 0)

        # Average response times direct from summary (primary); secondary left None
        ts_stats = summary_data.get("time_stats", {})
        avg_primary_resp = ts_stats.get("avg_primary_response_time")
        avg_secondary_resp = ts_stats.get("avg_secondary_response_time")

        # Token averages
        avg_primary_tokens = (
            summary_data.get("token_usage_stats", {})
            .get("primary_llm", {})
            .get("avg_completion_tokens")
        )
        avg_secondary_tokens = (
            summary_data.get("token_usage_stats", {})
            .get("secondary_llm", {})
            .get("avg_completion_tokens")
            if secondary_provider
            else None
        )

        # Continuation info
        continuation_info = summary_data.get("continuation_info", {})
        is_continuation = continuation_info.get("is_continuation", False)
        continuation_count = (
            continuation_info.get("continuation_count", 0) if is_continuation else 0
        )

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
                "SWRONG Steps": something_is_wrong_steps,
                "Invalid Reversals": invalid_reversals,
                "Avg Primary Resp (s)": (
                    round(avg_primary_resp, 2) if avg_primary_resp is not None else None
                ),
                "Avg Primary Tokens": int(avg_primary_tokens),
                "Avg Secondary Resp (s)": (
                    round(avg_secondary_resp, 2)
                    if avg_secondary_resp is not None
                    else None
                ),
                "Avg Secondary Tokens": (
                    int(avg_secondary_tokens) if secondary_provider else None
                ),
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
            "Filter by Primary LLM Provider",
            options=options["primary_providers"],
            default=[],
        )
    with col2:
        selected_primary_models = st.multiselect(
            "Filter by Primary LLM Model", options=options["primary_models"], default=[]
        )
    with col3:
        selected_secondary_providers = st.multiselect(
            "Filter by Parser LLM Provider",
            options=options["secondary_providers"],
            default=[],
        )
    with col4:
        selected_secondary_models = st.multiselect(
            "Filter by Parser LLM Model",
            options=options["secondary_models"],
            default=[],
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
            "Folder",
            "primary_provider",
            "primary_model_name",
            "secondary_provider",
            "secondary_model_name",
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
        col_hist, col_cfg = st.columns([3, 1])
        with col_cfg:
            bins = st.slider("Bins", 5, 40, 20, key="score_bins")
        with col_hist:
            fig_scores = px.histogram(
                x=scores,
                nbins=bins,
                labels={"x": "Score"},
                color_discrete_sequence=["seagreen"],
            )
            st.plotly_chart(fig_scores, use_container_width=True)

    # ---------- LLM response-time histogram ----------
    st.markdown("## Primary LLM Response Time Distribution")

    primary_resp_times: List[float] = []
    secondary_resp_times: List[float] = []

    for g in games_data.values():

        # --- Also inspect *prompt_response_stats* for per-game timings ---
        pstats = g.get("prompt_response_stats", {})
        if pstats:
            # Modern log format: full list of per-round timings
            primary_resp_times.extend(
                [t for t in pstats.get("primary_response_times", []) if t > 0]
            )
            secondary_resp_times.extend(
                [t for t in pstats.get("secondary_response_times", []) if t > 0]
            )

    # ---------------- Primary chart ----------------
    if primary_resp_times:
        col_hist, col_cfg = st.columns([3, 1])
        with col_cfg:
            bins_t_primary = st.slider(
                "Bins (Primary)", 5, 40, 20, key="time_bins_primary"
            )
        with col_hist:
            fig_times_primary = px.histogram(
                x=primary_resp_times,
                nbins=bins_t_primary,
                labels={"x": "Primary Response Time (s)"},
            )
            st.plotly_chart(fig_times_primary, use_container_width=True)
    else:
        st.info("No Primary LLM response time data available for this experiment.")

    # ---------------- Secondary chart (optional) ----------------
    if secondary_resp_times:
        st.markdown("## Secondary LLM Response Time Distribution")
        col_hist_s, col_cfg_s = st.columns([3, 1])
        with col_cfg_s:
            bins_t_secondary = st.slider(
                "Bins (Secondary)", 5, 40, 20, key="time_bins_secondary"
            )
        with col_hist_s:
            fig_times_secondary = px.histogram(
                x=secondary_resp_times,
                nbins=bins_t_secondary,
                labels={"x": "Secondary Response Time (s)"},
                color_discrete_sequence=["indianred"],
            )
            st.plotly_chart(fig_times_secondary, use_container_width=True)

    # ---------- Game details table ----------
    st.markdown("## Game Details")
    rows: List[Dict[str, Any]] = []
    for num, g in sorted(games_data.items()):
        step_stats = g.get("step_stats", {})
        total_steps = g.get("steps", 0)

        valid_steps = step_stats.get("valid_steps", 0)
        empty_steps = step_stats.get("empty_steps", 0)
        something_wrong_steps = step_stats.get("something_is_wrong_steps", 0)
        invalid_rev_steps = step_stats.get("invalid_reversals", 0)

        # Use the sum of all recorded categories as denominator to avoid >100% artefacts
        denom = valid_steps + empty_steps + something_wrong_steps + invalid_rev_steps
        denom = denom if denom > 0 else 1  # prevent ZeroDivision

        rows.append(
            {
                "Game": num,
                "Score": g.get("score", 0),
                "Steps (total)": total_steps,
                "Valid": valid_steps,
                "Valid %": f"{valid_steps / denom * 100:.1f}%",
                "EMPTY": empty_steps,
                "EMPTY %": f"{empty_steps / denom * 100:.1f}%",
                "SWRONG": something_wrong_steps,
                "SWRONG %": f"{something_wrong_steps / denom * 100:.1f}%",
                "Invalid Rev": invalid_rev_steps,
                "Invalid Rev %": f"{invalid_rev_steps / denom * 100:.1f}%",
                "End Reason": g.get("game_end_reason", "Unknown"),
                "Rounds": g.get("round_count", 0),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
