"""
Streamlit app for analyzing and replaying recorded Snake game sessions.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from core.snake_game import SnakeGame
from gui.replay_gui import ReplayGUI
from utils.game_stats_utils import (
    create_display_dataframe,
    create_game_performance_chart,
    create_game_dataframe,
    get_experiment_options,
    filter_experiments
)


def main():
    """Main app function."""
    # Set up the page
    st.set_page_config(
        page_title="Snake Game Analysis",
        page_icon="🐍",
        layout="wide"
    )
    
    # Title
    st.title("Snake Game Analysis")
    
    # Load statistics
    stats_df = load_statistics()
    if stats_df is None:
        st.error("No statistics found. Please run some games first.")
        return
        
    # Create display dataframe
    display_df = create_display_dataframe(stats_df)
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Get filter options
    options = get_experiment_options(stats_df)
    
    # Provider filter
    provider = st.sidebar.selectbox(
        "Provider",
        ["All"] + options["providers"]
    )
    
    # Model filter
    model = st.sidebar.selectbox(
        "Model",
        ["All"] + options["models"]
    )
    
    # Primary LLM filter
    primary_llm = st.sidebar.selectbox(
        "Primary LLM",
        ["All"] + options["primary_llms"]
    )
    
    # Secondary LLM filter
    secondary_llm = st.sidebar.selectbox(
        "Secondary LLM",
        ["All"] + options["secondary_llms"]
    )
    
    # Filter the data
    filtered_df = filter_experiments(
        stats_df,
        provider=provider if provider != "All" else None,
        model=model if model != "All" else None,
        primary_llm=primary_llm if primary_llm != "All" else None,
        secondary_llm=secondary_llm if secondary_llm != "All" else None
    )
    
    # Create display dataframe for filtered data
    filtered_display_df = create_display_dataframe(filtered_df)
    
    # Display statistics
    st.header("Statistics")
    st.dataframe(filtered_display_df)
    
    # Create performance chart
    st.header("Performance")
    fig = create_game_performance_chart(filtered_df)
    st.plotly_chart(fig)
    
    # Game replay section
    st.header("Game Replay")
    
    # Get available games
    game_dirs = get_available_games()
    if not game_dirs:
        st.warning("No recorded games found.")
        return
        
    # Game selection
    selected_game = st.selectbox(
        "Select a game to replay",
        game_dirs,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Load game data
    game_data = load_game_data(selected_game)
    if not game_data:
        st.error("Failed to load game data.")
        return
        
    # Create game instance
    game = SnakeGame()
    
    # Create GUI
    gui = ReplayGUI()
    game.set_gui(gui)
    
    # Replay controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Reset"):
            game.reset()
            
    with col2:
        if st.button("Previous Move"):
            if game.steps > 0:
                game.reset()
                for i in range(game.steps - 1):
                    game.move(game_data[i])
                    
    with col3:
        if st.button("Next Move"):
            if game.steps < len(game_data):
                game.move(game_data[game.steps])
                
    # Display game state
    st.image(gui.get_surface(), caption="Game State")


def load_statistics():
    """Load game statistics.
    
    Returns:
        DataFrame containing game statistics
    """
    # Find statistics file
    stats_file = "logs/statistics.json"
    if not os.path.exists(stats_file):
        return None
        
    # Load statistics
    with open(stats_file, "r") as f:
        stats = json.load(f)
        
    # Convert to DataFrame
    return pd.DataFrame(stats)


def get_available_games():
    """Get available recorded games.
    
    Returns:
        List of game directory paths
    """
    # Find game directories
    game_dirs = []
    for root, dirs, files in os.walk("logs"):
        if any(f.startswith("game_") and f.endswith(".json") for f in files):
            game_dirs.append(root)
            
    return sorted(game_dirs)


def load_game_data(game_dir):
    """Load game data from a directory.
    
    Args:
        game_dir: Directory containing game data
        
    Returns:
        List of game moves
    """
    # Find game log file
    game_log = None
    for filename in os.listdir(game_dir):
        if filename.startswith("game_") and filename.endswith(".json"):
            game_log = os.path.join(game_dir, filename)
            break
            
    if not game_log:
        return None
        
    # Load game data
    with open(game_log, "r") as f:
        data = json.load(f)
        return data["moves"]


if __name__ == "__main__":
    main()
