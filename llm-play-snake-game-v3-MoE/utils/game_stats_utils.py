"""
Utility module for game statistics and visualization.
Handles game statistics processing and visualization for analysis.
"""

import os
import numpy as np
import json
from datetime import datetime


class NumPyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""

    def default(self, o):
        """Handle NumPy types for JSON serialization.

        Args:
            o: Object to serialize

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def get_experiment_options(stats_df):
    """Return unique provider/model options for Streamlit filter widgets.

    Handles NaN/None values gracefully and avoids cross-type sorting issues.
    """
    def _clean_unique(series):
        if series is None:
            return []
        # Convert to strings, drop NaN/None and duplicates
        cleaned = (
            series.dropna()
            .astype(str)
            .loc[lambda s: s.str.lower() != "none"]
            .unique()
        )
        return sorted(cleaned.tolist())

    primary_providers = _clean_unique(stats_df.get("primary_provider"))
    primary_models = _clean_unique(stats_df.get("primary_model_name"))
    secondary_providers = _clean_unique(stats_df.get("secondary_provider"))
    secondary_models = _clean_unique(stats_df.get("secondary_model_name"))

    # Always include explicit 'None' entry so dashboard filters can target
    # single-LLM experiments even when the raw data contains only NaN.
    if "None" not in secondary_providers:
        secondary_providers.insert(0, "None")

    return {
        "primary_providers": primary_providers,
        "primary_models": primary_models,
        "secondary_providers": secondary_providers,
        "secondary_models": secondary_models,
    }

def filter_experiments(stats_df, selected_primary_providers=None, selected_primary_models=None,
                       selected_secondary_providers=None, selected_secondary_models=None):
    """Filter experiments based on selected criteria.
    
    This function is used by the Streamlit analytics dashboard when run separately.
    
    Args:
        stats_df: DataFrame with experiment statistics
        selected_primary_providers: List of selected primary providers
        selected_primary_models: List of selected primary models
        selected_secondary_providers: List of selected secondary providers
        selected_secondary_models: List of selected secondary models
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = stats_df.copy()

    if selected_primary_providers and len(selected_primary_providers) > 0:
        filtered_df = filtered_df[filtered_df['primary_provider'].isin(selected_primary_providers)]

    if selected_primary_models and len(selected_primary_models) > 0:
        filtered_df = filtered_df[filtered_df['primary_model_name'].isin(selected_primary_models)]

    # --------------- Secondary provider filter ----------------
    if selected_secondary_providers and len(selected_secondary_providers) > 0:
        # Special-case "None": rows where secondary_provider is NaN/None
        providers_no_none = [p for p in selected_secondary_providers if p != "None"]

        mask = False
        if providers_no_none:
            mask = filtered_df["secondary_provider"].isin(providers_no_none)
        if "None" in selected_secondary_providers:
            mask = mask | filtered_df["secondary_provider"].isna()

        filtered_df = filtered_df[mask]

    # --------------- Secondary model filter -------------------
    if selected_secondary_models and len(selected_secondary_models) > 0:
        models_no_none = [m for m in selected_secondary_models if m != "None"]

        mask_m = False
        if models_no_none:
            mask_m = filtered_df["secondary_model_name"].isin(models_no_none)
        if "None" in selected_secondary_models:
            mask_m = mask_m | filtered_df["secondary_model_name"].isna()

        filtered_df = filtered_df[mask_m]

    return filtered_df 


def save_experiment_info_json(args, directory):
    """Save experiment configuration information to a JSON file.

    Args:
        args: Command line arguments
        directory: Directory to save the file in

    Returns:
        Experiment info dictionary
    """
    # Convert args to dict
    args_dict = vars(args)

    # Clean up configuration - set parser info to null if it's 'none'
    config_dict = args_dict.copy()
    if "parser_provider" in config_dict and (
        not config_dict["parser_provider"]
        or config_dict["parser_provider"].lower() == "none"
    ):
        # In single LLM mode, set parser fields to null instead of removing them
        config_dict["parser_provider"] = None
        config_dict["parser_model"] = None

    # Create experiment info
    experiment_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": config_dict,
        "game_statistics": {
            "total_games": 0,
            "total_rounds": 0,
            "total_score": 0,
            "total_steps": 0,
            "scores": [],
            "round_counts": [],
        },
        "time_statistics": {
            "total_llm_communication_time": 0,
            "total_primary_llm_communication_time": 0,
            "total_secondary_llm_communication_time": 0,
        },
        "token_usage_stats": {
            "primary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
            },
            "secondary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
            },
        },
        "step_stats": {
            "empty_steps": 0,
            "something_is_wrong_steps": 0,
            "valid_steps": 0,
            "invalid_reversals": 0,  # Aggregated count across all games
        },
    }

    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Save to file
    file_path = os.path.join(directory, "summary.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(experiment_info, f, indent=2, cls=NumPyJSONEncoder)

    return experiment_info


def save_session_stats(log_dir, **kwargs):
    """Save session statistics to the summary JSON file.

    Args:
        log_dir: Directory containing the summary.json file
        **kwargs: Statistics fields to save
    """
    # Read existing summary file
    summary_path = os.path.join(log_dir, "summary.json")

    if not os.path.exists(summary_path):
        return

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        print(f"Error reading summary.json: {e}")
        return

    def _safe_set(target: dict, key: str, val):
        """Overwrite only with a real, non-zero value."""
        if val:
            target[key] = val

    # Ensure all required sections exist
    if "game_statistics" not in summary:
        summary["game_statistics"] = {
            "total_games": 0,
            "total_rounds": 0,
            "total_score": 0,
            "total_steps": 0,
            "scores": [],
            "round_counts": [],
        }

    if "time_statistics" not in summary:
        summary["time_statistics"] = {
            "total_llm_communication_time": 0,
            "total_primary_llm_communication_time": 0,
            "total_secondary_llm_communication_time": 0,
        }

    if "token_usage_stats" not in summary:
        summary["token_usage_stats"] = {
            "primary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
            },
            "secondary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
            },
        }

    if "step_stats" not in summary:
        summary["step_stats"] = {
            "empty_steps": 0,
            "something_is_wrong_steps": 0,
            "valid_steps": 0,
            "invalid_reversals": 0,  # Aggregated count across all games
        }

    # Apply new statistics values to the appropriate sections
    for key, value in kwargs.items():
        if key == "game_count":
            summary["game_statistics"]["total_games"] = value
        elif key == "total_score":
            summary["game_statistics"]["total_score"] = value
        elif key == "total_steps":
            summary["game_statistics"]["total_steps"] = value
        elif key == "game_scores":
            summary["game_statistics"]["scores"] = value
        elif key == "empty_steps":
            summary["step_stats"][
                "empty_steps"
            ] = value  # Already accumulated in process_game_over
        elif key == "something_is_wrong_steps":
            summary["step_stats"][
                "something_is_wrong_steps"
            ] = value  # Already accumulated in process_game_over
        elif key == "valid_steps":
            summary["step_stats"][
                "valid_steps"
            ] = value  # Already accumulated in process_game_over
        elif key == "invalid_reversals":
            summary["step_stats"]["invalid_reversals"] = value
        elif key == "time_stats":
            # Handle time statistics if provided
            if value and isinstance(value, dict):
                ts = value
                _safe_set(
                    summary["time_statistics"],
                    "total_llm_communication_time",
                    ts.get("llm_communication_time"),
                )
                _safe_set(
                    summary["time_statistics"],
                    "total_primary_llm_communication_time",
                    ts.get("primary_llm_communication_time"),
                )
                _safe_set(
                    summary["time_statistics"],
                    "total_secondary_llm_communication_time",
                    ts.get("secondary_llm_communication_time"),
                )
        elif key == "token_stats":
            # Handle token statistics if provided
            if value and isinstance(value, dict):
                if "primary" in value and isinstance(value["primary"], dict):
                    primary = value["primary"]

                    # Only add token stats if they're not None
                    if (
                        "total_tokens" in primary
                        and primary["total_tokens"] is not None
                    ):
                        summary["token_usage_stats"]["primary_llm"]["total_tokens"] = (
                            primary["total_tokens"]
                        )
                    if (
                        "total_prompt_tokens" in primary
                        and primary["total_prompt_tokens"] is not None
                    ):
                        summary["token_usage_stats"]["primary_llm"][
                            "total_prompt_tokens"
                        ] = primary["total_prompt_tokens"]
                    if (
                        "total_completion_tokens" in primary
                        and primary["total_completion_tokens"] is not None
                    ):
                        summary["token_usage_stats"]["primary_llm"][
                            "total_completion_tokens"
                        ] = primary["total_completion_tokens"]

                if "secondary" in value and isinstance(value["secondary"], dict):
                    secondary = value["secondary"]

                    # Only add token stats if they're not None
                    if (
                        "total_tokens" in secondary
                        and secondary["total_tokens"] is not None
                    ):
                        summary["token_usage_stats"]["secondary_llm"][
                            "total_tokens"
                        ] = secondary["total_tokens"]
                    if (
                        "total_prompt_tokens" in secondary
                        and secondary["total_prompt_tokens"] is not None
                    ):
                        summary["token_usage_stats"]["secondary_llm"][
                            "total_prompt_tokens"
                        ] = secondary["total_prompt_tokens"]
                    if (
                        "total_completion_tokens" in secondary
                        and secondary["total_completion_tokens"] is not None
                    ):
                        summary["token_usage_stats"]["secondary_llm"][
                            "total_completion_tokens"
                        ] = secondary["total_completion_tokens"]
        elif key == "step_stats":
            # Handle step statistics if provided as a complete dictionary
            if value and isinstance(value, dict):
                if "empty_steps" in value:
                    summary["step_stats"]["empty_steps"] = value[
                        "empty_steps"
                    ]  # Already accumulated in process_game_over
                if "something_is_wrong_steps" in value:
                    summary["step_stats"]["something_is_wrong_steps"] = value[
                        "something_is_wrong_steps"
                    ]  # Already accumulated in process_game_over
                if "valid_steps" in value:
                    summary["step_stats"]["valid_steps"] = value[
                        "valid_steps"
                    ]  # Already accumulated in process_game_over
                if "invalid_reversals" in value:
                    summary["step_stats"]["invalid_reversals"] = value[
                        "invalid_reversals"
                    ]  # Already accumulated in process_game_over
        elif key == "round_counts":
            summary["game_statistics"]["round_counts"] = value
        elif key == "total_rounds":
            summary["game_statistics"]["total_rounds"] = value
        else:
            # For any other fields, add them at the top level
            summary[key] = value

    # After merging totals, compute averages – scaled **per round** when available
    total_rounds = summary["game_statistics"].get("total_rounds", 0)
    denom = max(total_rounds, 0.00001)

    # ---------------- Token averages ----------------
    prim = summary["token_usage_stats"].get("primary_llm", {})
    if prim:
        prim["avg_tokens"] = prim.get("total_tokens", 0) / denom
        prim["avg_prompt_tokens"] = prim.get("total_prompt_tokens", 0) / denom
        prim["avg_completion_tokens"] = prim.get("total_completion_tokens", 0) / denom

    sec = summary["token_usage_stats"].get("secondary_llm", {})
    if sec:
        sec_total = sec.get("total_tokens")
        if sec_total is not None:
            sec["avg_tokens"] = sec_total / denom
        sec_prompt = sec.get("total_prompt_tokens")
        if sec_prompt is not None:
            sec["avg_prompt_tokens"] = sec_prompt / denom
        sec_comp = sec.get("total_completion_tokens")
        if sec_comp is not None:
            sec["avg_completion_tokens"] = sec_comp / denom

    # ---------------- Response-time averages ----------------
    ts = summary.get("time_statistics", {})
    if ts:
        if ts.get("total_primary_llm_communication_time") is not None:
            ts["avg_primary_response_time"] = (
                ts.get("total_primary_llm_communication_time", 0) / denom
            )
        if ts.get("total_secondary_llm_communication_time") is not None:
            ts["avg_secondary_response_time"] = (
                ts.get("total_secondary_llm_communication_time", 0) / denom
            )

    # Save the summary file
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, cls=NumPyJSONEncoder)
    except Exception as e:
        print(f"Error writing summary.json: {e}")
