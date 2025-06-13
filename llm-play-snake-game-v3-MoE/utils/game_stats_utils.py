"""
Utility module for game statistics and visualization.
Handles game statistics processing and visualization for analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


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
    """Extract experiment options for filtering.
    
    This function is used by the Streamlit analytics dashboard when run separately.
    
    Args:
        stats_df: DataFrame with experiment statistics
        
    Returns:
        Dictionary with options for providers, models, primary and secondary LLMs
    """
    # Use sets to avoid duplicates
    all_providers = set()
    all_models = set()
    
    # Specific provider and model sets
    primary_providers = set()
    primary_models = set()
    secondary_providers = set()
    secondary_models = set()
    
    # Extract separate primary and secondary options
    for _, row in stats_df.iterrows():
        # Add primary provider and model
        if row.get('primary_provider') is not None:
            primary_providers.add(row['primary_provider'])
            all_providers.add(row['primary_provider'])
        
        if row.get('primary_model_name') is not None:
            primary_models.add(row['primary_model_name'])
            all_models.add(row['primary_model_name'])
        
        # Add secondary provider and model if not None
        if row.get('secondary_provider') is not None:
            secondary_providers.add(row['secondary_provider'])
            all_providers.add(row['secondary_provider'])
        
        if row.get('secondary_model_name') is not None:
            secondary_models.add(row['secondary_model_name'])
            all_models.add(row['secondary_model_name'])
    
    # Get unique full LLM strings (these are already strings)
    primary_llms = set(stats_df['primary_llm'].dropna().unique())
    secondary_llms = set(stats_df['secondary_llm'].dropna().unique())
    
    # Remove "None" from secondary providers and models
    if "None" in secondary_providers:
        secondary_providers.remove("None")
    
    if "None" in secondary_models:
        secondary_models.remove("None")
    
    return {
        'providers': sorted(all_providers),
        'models': sorted(all_models),
        'primary_providers': sorted(primary_providers),
        'primary_models': sorted(primary_models),
        'secondary_providers': sorted(secondary_providers),
        'secondary_models': sorted(secondary_models),
        'primary_llms': sorted(primary_llms),
        'secondary_llms': sorted(secondary_llms)
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