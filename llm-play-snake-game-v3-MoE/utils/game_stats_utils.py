"""
Utility module for game statistics and visualization.
Handles game statistics processing and visualization for analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def create_display_dataframe(stats_df):
    """Create a user-friendly display DataFrame for the UI.
    
    This function is used by the Streamlit analytics dashboard when run separately.
    
    Args:
        stats_df: DataFrame with raw game statistics
        
    Returns:
        DataFrame formatted for display
    """
    # Select relevant columns
    display_df = stats_df[['folder', 'total_score', 'total_steps', 'mean_score', 'max_score', 
                          'primary_llm', 'secondary_llm', 'avg_response_time', 
                          'avg_secondary_response_time', 'steps_per_apple', 'json_success_rate']].copy()
    
    # Rename columns for better display
    display_df.columns = ['Folder', 'Total Score', 'Total Steps', 'Mean Score', 'Max Score', 
                          'Primary LLM', 'Secondary LLM', 'Avg Response Time (s)', 
                          'Avg Secondary Response Time (s)', 'Steps Per Apple', 'JSON Success Rate (%)']
    
    # Format display values
    display_df['Folder'] = display_df['Folder'].apply(lambda x: os.path.basename(x))
    display_df['Avg Response Time (s)'] = display_df['Avg Response Time (s)'].apply(lambda x: f"{x:.2f}")
    display_df['Avg Secondary Response Time (s)'] = display_df['Avg Secondary Response Time (s)'].apply(lambda x: f"{x:.2f}")
    display_df['Steps Per Apple'] = display_df['Steps Per Apple'].apply(lambda x: f"{x:.2f}")
    display_df['JSON Success Rate (%)'] = display_df['JSON Success Rate (%)'].apply(lambda x: f"{x:.2f}")
    
    return display_df

def create_game_performance_chart(game_df):
    """Create a matplotlib figure showing game performance.
    
    This function is used by the Streamlit analytics dashboard when run separately.
    
    Args:
        game_df: DataFrame with game data
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(game_df['Game #'], game_df['Score'], color='skyblue')
    ax.set_xlabel('Game Number')
    ax.set_ylabel('Score')
    ax.set_title('Score by Game')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig

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
        Dictionary with options for providers, models, primary LLMs, and secondary LLMs
    """
    # Use sets to avoid duplicates
    all_providers = set()
    all_models = set()
    
    # Add all valid providers and models to sets
    for providers_list in stats_df['providers']:
        for provider in providers_list:
            if provider is not None:
                all_providers.add(provider)
    
    for models_list in stats_df['models']:
        for model in models_list:
            if model is not None:
                all_models.add(model)
    
    # Get unique primary and secondary LLMs (these are already strings, not lists)
    primary_llms = set(stats_df['primary_llm'].dropna().unique())
    secondary_llms = set(stats_df['secondary_llm'].dropna().unique())
    
    return {
        'providers': sorted(all_providers),
        'models': sorted(all_models),
        'primary_llms': sorted(primary_llms),
        'secondary_llms': sorted(secondary_llms)
    }

def filter_experiments(stats_df, selected_providers=None, selected_models=None, 
                       selected_primary_llms=None, selected_secondary_llms=None):
    """Filter experiments based on selected criteria.
    
    This function is used by the Streamlit analytics dashboard when run separately.
    
    Args:
        stats_df: DataFrame with experiment statistics
        selected_providers: List of selected providers
        selected_models: List of selected models
        selected_primary_llms: List of selected primary LLMs
        selected_secondary_llms: List of selected secondary LLMs
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = stats_df.copy()
    
    if selected_providers and len(selected_providers) > 0:
        filtered_df = filtered_df[filtered_df['providers'].apply(lambda x: any(p in x for p in selected_providers))]
    
    if selected_models and len(selected_models) > 0:
        filtered_df = filtered_df[filtered_df['models'].apply(lambda x: any(m in x for m in selected_models))]
    
    if selected_primary_llms and len(selected_primary_llms) > 0:
        filtered_df = filtered_df[filtered_df['primary_llm'].isin(selected_primary_llms)]
    
    if selected_secondary_llms and len(selected_secondary_llms) > 0:
        filtered_df = filtered_df[filtered_df['secondary_llm'].isin(selected_secondary_llms)]
    
    return filtered_df 