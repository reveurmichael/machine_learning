"""
Utility module for game statistics and visualization.
Handles game statistics processing and visualization for analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def create_game_dataframe(game_data):
    """Create a DataFrame from game data dictionary.
    
    This function is used by the Streamlit analytics dashboard when run separately.
    
    Args:
        game_data: Dictionary containing game data
        
    Returns:
        DataFrame with game information
    """
    game_list = []
    for game_num, data in game_data.items():
        game_list.append({
            'Game #': game_num,
            'Score': data.get('score', 0),
            'Steps': data.get('steps', 0),
            'End Reason': data.get('game_end_reason', 'Unknown'),
            'Has Apple Positions': data.get('has_apple_positions', False),
            'Has Moves': data.get('has_moves', False),
            'Move Count': data.get('move_count', 0)
        })
    
    return pd.DataFrame(game_list)

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
    
    if selected_secondary_providers and len(selected_secondary_providers) > 0:
        # Need to handle None values for secondary provider
        has_provider_mask = filtered_df['secondary_provider'].isin(selected_secondary_providers)
        filtered_df = filtered_df[has_provider_mask]
    
    if selected_secondary_models and len(selected_secondary_models) > 0:
        # Need to handle None values for secondary model
        has_model_mask = filtered_df['secondary_model_name'].isin(selected_secondary_models)
        filtered_df = filtered_df[has_model_mask]
    
    return filtered_df 