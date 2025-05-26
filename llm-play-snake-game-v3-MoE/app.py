"""
Snake Game Analytics Dashboard

A Streamlit app for analyzing and replaying recorded Snake game sessions.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from utils.file_utils import find_log_folders, extract_game_stats
from replay import ReplaySnakeGame

# Set up page configuration
st.set_page_config(
    page_title="Snake Game Analytics",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_game_details(selected_stats):
    """Display detailed information about a selected game.
    
    Args:
        selected_stats: Statistics for the selected game
    """
    if not selected_stats:
        return
    
    st.subheader("Game Scores")
    
    # If we have game data, create a bar chart of scores
    if 'game_data' in selected_stats and selected_stats['game_data']:
        game_scores = []
        for game_num, data in selected_stats['game_data'].items():
            if 'score' in data:
                game_scores.append((int(game_num), data['score']))
        
        if game_scores:
            game_scores.sort(key=lambda x: x[0])  # Sort by game number
            
            fig = px.bar(
                x=[f"Game {num}" for num, _ in game_scores],
                y=[score for _, score in game_scores],
                title="Score by Game",
                labels={'x': 'Game', 'y': 'Score'},
                color=[score for _, score in game_scores],
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Display model information
    st.subheader("Model Information")
    
    # Create columns for primary and secondary LLM
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Primary LLM")
        st.markdown(f"**Model:** {selected_stats['primary_llm']}")
    
    with col2:
        st.markdown("#### Secondary LLM")
        st.markdown(f"**Model:** {selected_stats['secondary_llm']}")
    
    # Display performance metrics
    st.subheader("Performance Metrics")
    
    # Create metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Score", selected_stats['total_score'])
    
    with col2:
        st.metric("Total Steps", selected_stats['total_steps'])
    
    with col3:
        st.metric("Mean Score", round(selected_stats['mean_score'], 2))
    
    with col4:
        st.metric("Steps per Apple", round(selected_stats['steps_per_apple'], 2))
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Max Score", selected_stats['max_score'])
    
    with col2:
        st.metric("Min Score", selected_stats['min_score'])
    
    with col3:
        st.metric("JSON Success Rate", f"{selected_stats['json_success_rate']:.1f}%")
    
    with col4:
        st.metric("Avg Response Time", f"{selected_stats['avg_response_time']:.2f}s")
    
    # Check if we have per-game data to show
    if 'game_data' in selected_stats and selected_stats['game_data']:
        st.subheader("Games")
        
        # Create a list of games with key metrics
        games_data = []
        for game_num, data in selected_stats['game_data'].items():
            games_data.append({
                "Game": f"Game {game_num}",
                "Score": data.get('score', 0),
                "Steps": data.get('steps', 0),
                "Steps per Apple": data.get('steps_per_apple', 0),
                "JSON Success Rate": f"{data.get('json_success_rate', 0):.1f}%",
                "End Reason": data.get('game_end_reason', 'Unknown')
            })
        
        if games_data:
            # Convert to DataFrame and display as table
            games_df = pd.DataFrame(games_data)
            st.dataframe(games_df, use_container_width=True)

def run_replay(log_folder, game_num, move_pause):
    """Run a replay of a specific game.
    
    Args:
        log_folder: Folder containing the game logs
        game_num: Game number to replay
        move_pause: Pause between moves in seconds
    """
    # Initialize the replay
    replay = ReplaySnakeGame(log_folder, move_pause=move_pause)
    
    # Set the game number
    replay.game_number = game_num
    
    # Run the replay
    replay.run()

def main():
    """Main function to run the Streamlit app."""
    # Set up sidebar
    st.sidebar.title("Snake Game Analytics 🐍")
    st.sidebar.markdown("Analyze and replay Snake game sessions.")
    
    # Main content
    st.title("Snake Game Analytics Dashboard 🐍")
    
    # Find log folders
    with st.spinner("Finding log folders..."):
        log_folders = find_log_folders()
    
    if not log_folders:
        st.warning("No log folders found. Make sure you're running this app from the correct directory.")
        return
    
    # Extract stats from log folders
    with st.spinner("Extracting game statistics..."):
        folder_stats = [extract_game_stats(folder) for folder in log_folders]
    
    # Convert to DataFrame for filtering and display
    stats_df = pd.DataFrame(folder_stats)
    
    # Add date column if missing
    if 'date' not in stats_df:
        stats_df['date'] = None
    
    # Convert date strings to datetime objects for sorting
    stats_df['date'] = pd.to_datetime(stats_df['date'], errors='coerce')
    
    # Sort by date (newest first)
    stats_df.sort_values(by='date', ascending=False, inplace=True, na_position='last')
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Overview", "Game Details", "Replay"])
    
    with tab1:
        # Filter options
        st.header("Filter Options")
        
        # Create filter columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filter by model
            if 'models' in stats_df.columns:
                all_models = []
                for models_list in stats_df['models']:
                    if isinstance(models_list, list):
                        all_models.extend(models_list)
                
                unique_models = sorted(list(set([m for m in all_models if m])))
                
                if unique_models:
                    selected_models = st.multiselect(
                        "Filter by Model",
                        options=unique_models,
                        default=[]
                    )
        
        with col2:
            # Filter by provider
            if 'providers' in stats_df.columns:
                all_providers = []
                for providers_list in stats_df['providers']:
                    if isinstance(providers_list, list):
                        all_providers.extend(providers_list)
                
                unique_providers = sorted(list(set([p for p in all_providers if p])))
                
                if unique_providers:
                    selected_providers = st.multiselect(
                        "Filter by Provider",
                        options=unique_providers,
                        default=[]
                    )
        
        with col3:
            # Filter by date range
            if 'date' in stats_df.columns:
                valid_dates = stats_df['date'].dropna()
                
                if not valid_dates.empty:
                    min_date = valid_dates.min().date()
                    max_date = valid_dates.max().date()
                    
                    date_range = st.date_input(
                        "Filter by Date Range",
                        value=(min_date, max_date)
                    )
        
        # Apply filters
        filtered_df = stats_df.copy()
        
        # Apply model filter
        if 'models' in stats_df.columns and 'selected_models' in locals() and selected_models:
            filtered_df = filtered_df[filtered_df['models'].apply(
                lambda x: any(model in x for model in selected_models) if isinstance(x, list) else False
            )]
        
        # Apply provider filter
        if 'providers' in stats_df.columns and 'selected_providers' in locals() and selected_providers:
            filtered_df = filtered_df[filtered_df['providers'].apply(
                lambda x: any(provider in x for provider in selected_providers) if isinstance(x, list) else False
            )]
        
        # Apply date filter
        if 'date' in stats_df.columns and 'date_range' in locals() and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['date'].dt.date >= start_date) &
                (filtered_df['date'].dt.date <= end_date)
            ]
        
        # Display filtered results
        st.header("Game Statistics")
        
        # Create a display DataFrame with selected columns
        display_df = filtered_df[['folder', 'total_score', 'total_steps', 'mean_score', 'max_score',
                               'steps_per_apple', 'json_success_rate', 'primary_llm', 'date']]
        
        # Rename columns for display
        display_df.columns = ['Folder', 'Total Score', 'Total Steps', 'Mean Score', 'Max Score',
                           'Steps per Apple', 'JSON Success Rate', 'Primary LLM', 'Date']
        
        # Format the folder column to show only the basename
        display_df['Folder'] = display_df['Folder'].apply(lambda x: os.path.basename(x))
        
        # Format the JSON success rate as percentage
        if 'JSON Success Rate' in display_df:
            display_df['JSON Success Rate'] = display_df['JSON Success Rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        
        # Display the table
        st.dataframe(display_df, use_container_width=True)
        
        # Display summary charts
        st.header("Summary Charts")
        
        # Create side-by-side charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Score vs Steps scatter plot
            if 'total_score' in filtered_df.columns and 'total_steps' in filtered_df.columns:
                fig = px.scatter(
                    filtered_df,
                    x='total_steps',
                    y='total_score',
                    color='mean_score',
                    size='total_games',
                    hover_name=filtered_df['folder'].apply(os.path.basename),
                    title="Score vs Steps",
                    labels={'total_steps': 'Total Steps', 'total_score': 'Total Score', 'mean_score': 'Mean Score'},
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Steps per Apple vs JSON Success Rate
            if 'steps_per_apple' in filtered_df.columns and 'json_success_rate' in filtered_df.columns:
                fig = px.scatter(
                    filtered_df,
                    x='json_success_rate',
                    y='steps_per_apple',
                    color='mean_score',
                    size='total_games',
                    hover_name=filtered_df['folder'].apply(os.path.basename),
                    title="Efficiency vs JSON Success Rate",
                    labels={'json_success_rate': 'JSON Success Rate (%)', 'steps_per_apple': 'Steps per Apple', 'mean_score': 'Mean Score'},
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Let user select a folder
        selected_folder_idx = st.selectbox(
            "Select a folder to view details",
            options=range(len(stats_df)),
            format_func=lambda x: f"{stats_df.iloc[x]['folder']} ({stats_df.iloc[x]['date'] if not pd.isna(stats_df.iloc[x]['date']) else 'Unknown date'})",
            key="folder_select"
        )
        
        # Display details for the selected folder
        selected_stats = stats_df.iloc[selected_folder_idx]
        
        st.write(f"**Details for {os.path.basename(selected_stats['folder'])}**")
        
        # Extract game data for the selected folder
        display_game_details(selected_stats)
    
    with tab3:
        # Game replay
        st.header("Game Replay")
        
        # Let user select a folder for replay
        replay_folder_idx = st.selectbox(
            "Select a folder to replay games from",
            options=range(len(stats_df)),
            format_func=lambda x: f"{stats_df.iloc[x]['folder']} ({stats_df.iloc[x]['date'] if not pd.isna(stats_df.iloc[x]['date']) else 'Unknown date'})",
            key="replay_folder_select"
        )
        
        # Get folder details
        replay_stats = stats_df.iloc[replay_folder_idx]
        replay_folder = replay_stats['folder']
        
        # Get available games from the folder
        available_games = []
        if 'game_data' in replay_stats and replay_stats['game_data']:
            available_games = sorted([int(g) for g in replay_stats['game_data'].keys()])
        
        if available_games:
            # Let user select a game
            game_num = st.selectbox(
                "Select a game to replay",
                options=available_games,
                format_func=lambda x: f"Game {x} (Score: {replay_stats['game_data'].get(str(x), {}).get('score', 0)})"
            )
            
            # Let user set replay speed
            move_pause = st.slider(
                "Pause between moves (seconds)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1
            )
            
            # Add a button to start the replay
            if st.button("Start Replay"):
                st.write("Starting replay... (Close the pygame window when done)")
                run_replay(replay_folder, game_num, move_pause)
        else:
            st.warning("No games available for replay in the selected folder")

if __name__ == "__main__":
    main()
