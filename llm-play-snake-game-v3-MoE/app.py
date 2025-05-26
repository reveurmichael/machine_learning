"""
Streamlit application for viewing Snake game statistics and replaying games.
"""

import os
import re
import json
import glob
import time
import pandas as pd
import streamlit as st
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from replay import extract_apple_positions
import numpy as np

def find_log_folders(base_dir='.', max_depth=4):
    """Find all log folders in the given directory and its subdirectories.
    
    Args:
        base_dir: Base directory to start search from
        max_depth: Maximum depth of subdirectories to search
        
    Returns:
        List of paths to log folders containing game data
    """
    log_folders = []
    base_path = Path(base_dir)
    
    for depth in range(max_depth + 1):
        # Create pattern for current depth
        # Use wildcards for each directory level up to the current depth
        if depth == 0:
            # Just look in the base directory
            patterns = [str(base_path / "*")]
        else:
            # Look in subdirectories up to depth
            parts = ['*'] * depth
            patterns = [str(base_path.joinpath(*parts) / "*")]
        
        # Find all potential log folders for current patterns
        for pattern in patterns:
            potential_folders = glob.glob(pattern)
            
            for folder in potential_folders:
                folder_path = Path(folder)
                # Check if folder contains required files and directories
                has_info_json = (folder_path / 'info.json').exists()
                has_summary_files = bool(glob.glob(str(folder_path / 'game*_summary.json')))
                has_prompts_dir = (folder_path / 'prompts').is_dir()
                has_responses_dir = (folder_path / 'responses').is_dir()
                
                # If it has all required components, it's a log folder
                if has_info_json and has_summary_files and has_prompts_dir and has_responses_dir:
                    log_folders.append(folder)
    
    return log_folders

def extract_game_stats(log_folder):
    """Extract game statistics from a log folder.
    
    Args:
        log_folder: Path to the log folder
        
    Returns:
        Dictionary with game statistics
    """
    stats = {
        'folder': log_folder,
        'date': None,
        'total_games': 0,
        'total_score': 0,
        'total_steps': 0,
        'max_score': 0,
        'min_score': 0,
        'mean_score': 0,
        'primary_llm': 'Unknown',
        'secondary_llm': 'Unknown',
        'avg_response_time': 0,
        'avg_secondary_response_time': 0,
        'steps_per_apple': 0,
        'json_success_rate': 0,
        'providers': [],
        'models': [],
        'game_data': {}  # Store per-game data
    }
    
    # Try to extract info from info.json
    info_path = Path(log_folder) / 'info.json'
    if info_path.exists():
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
            
            # Extract experiment information
            stats['date'] = info_data.get('date')
            
            # Extract LLM information
            primary_llm_info = info_data.get('primary_llm', {})
            if primary_llm_info:
                primary_provider = primary_llm_info.get('provider')
                primary_model = primary_llm_info.get('model')
                
                if primary_provider:
                    stats['primary_llm'] = primary_provider
                    stats['providers'].append(primary_provider)
                
                if primary_model:
                    stats['primary_llm'] += f" - {primary_model}"
                    stats['models'].append(primary_model)
            
            secondary_llm_info = info_data.get('secondary_llm', {})
            if secondary_llm_info:
                secondary_provider = secondary_llm_info.get('provider')
                secondary_model = secondary_llm_info.get('model')
                
                if secondary_provider:
                    stats['secondary_llm'] = secondary_provider
                    if secondary_provider not in stats['providers']:
                        stats['providers'].append(secondary_provider)
                
                if secondary_model:
                    stats['secondary_llm'] += f" - {secondary_model}"
                    if secondary_model not in stats['models']:
                        stats['models'].append(secondary_model)
            
            # Extract game statistics
            game_stats = info_data.get('game_statistics', {})
            stats['total_games'] = game_stats.get('total_games', 0)
            stats['total_score'] = game_stats.get('total_score', 0)
            stats['total_steps'] = game_stats.get('total_steps', 0)
            stats['max_score'] = game_stats.get('max_score', 0)
            stats['min_score'] = game_stats.get('min_score', 0)
            stats['mean_score'] = game_stats.get('mean_score', 0.0)
            
            # Extract JSON success rate
            json_stats = info_data.get('json_parsing_stats', {})
            stats['json_success_rate'] = json_stats.get('success_rate', 0.0)
            
        except Exception as e:
            print(f"Error reading info.json: {e}")
    
    # Extract per-game data from JSON summary files
    total_response_time = 0
    total_secondary_response_time = 0
    total_steps_per_apple = 0
    game_count = 0
    
    for i in range(1, 7):  # Assuming max 6 games
        json_summary_file = Path(log_folder) / f"game{i}_summary.json"
        if json_summary_file.exists():
            game_data = extract_game_summary(json_summary_file)
            if game_data:
                stats['game_data'][i] = game_data
                game_count += 1
                
                # Accumulate response times and steps per apple
                if 'avg_primary_response_time' in game_data:
                    total_response_time += game_data['avg_primary_response_time']
                
                if 'avg_secondary_response_time' in game_data:
                    total_secondary_response_time += game_data['avg_secondary_response_time']
                
                if 'steps_per_apple' in game_data:
                    total_steps_per_apple += game_data['steps_per_apple']
    
    # Calculate averages across games
    if game_count > 0:
        stats['avg_response_time'] = total_response_time / game_count
        stats['avg_secondary_response_time'] = total_secondary_response_time / game_count
        stats['steps_per_apple'] = total_steps_per_apple / game_count
    
    # Update total_games if not set from info.json
    if stats['total_games'] == 0:
        stats['total_games'] = len(stats['game_data'])
    
    return stats

def extract_apple_positions(log_dir, game_number):
    """Extract apple positions from a game summary file.
    
    Args:
        log_dir: Path to the log directory
        game_number: Game number to extract apple positions for
        
    Returns:
        List of apple positions as [x, y] arrays
    """
    log_dir_path = Path(log_dir)
    json_summary_file = log_dir_path / f"game{game_number}_summary.json"
    apple_positions = []
    
    if not json_summary_file.exists():
        print(f"No JSON summary file found for game {game_number}")
        return apple_positions
    
    try:
        with open(json_summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Extract apple positions from JSON
        if 'apple_positions' in summary_data and summary_data['apple_positions']:
            for pos in summary_data['apple_positions']:
                apple_positions.append(np.array([pos['x'], pos['y']]))
        
        print(f"Extracted {len(apple_positions)} apple positions from game {game_number} JSON summary")
    
    except Exception as e:
        print(f"Error extracting apple positions from JSON summary: {e}")
    
    return apple_positions

def extract_game_summary(summary_file):
    """Extract game summary from a summary file.
    
    Args:
        summary_file: Path to the game summary file
        
    Returns:
        Dictionary with game summary information
    """
    summary = {
        'file': str(summary_file),
        'score': 0,
        'steps': 0,
        'game_end_reason': None,
        'has_apple_positions': False,
        'has_moves': False,
        'move_count': 0,
        'avg_primary_response_time': 0,
        'avg_secondary_response_time': 0,
        'steps_per_apple': 0,
        'json_success_rate': 0
    }
    
    try:
        # Parse JSON file
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract fields from JSON
        summary['score'] = data.get('score', 0)
        summary['steps'] = data.get('steps', 0)
        summary['game_end_reason'] = data.get('game_end_reason', 'Unknown')
        
        # Check if the file has apple positions
        if 'apple_positions' in data and data['apple_positions']:
            summary['has_apple_positions'] = True
            
        # Check if the file has moves
        if 'moves' in data and data['moves']:
            summary['has_moves'] = True
            summary['move_count'] = len(data['moves'])
        
        # Extract response time metrics if available
        if 'prompt_response_stats' in data:
            prompt_stats = data.get('prompt_response_stats', {})
            summary['avg_primary_response_time'] = prompt_stats.get('avg_primary_response_time', 0)
            summary['avg_secondary_response_time'] = prompt_stats.get('avg_secondary_response_time', 0)
        
        # Extract performance metrics if available
        if 'performance_metrics' in data:
            perf_metrics = data.get('performance_metrics', {})
            summary['steps_per_apple'] = perf_metrics.get('steps_per_apple', 0)
        
        # Extract JSON parsing success rate if available
        if 'json_parsing_stats' in data:
            json_stats = data.get('json_parsing_stats', {})
            summary['json_success_rate'] = json_stats.get('success_rate', 0)
            
    except Exception as e:
        print(f"Error extracting summary from {summary_file}: {e}")
    
    return summary

def run_replay(log_dir, game_number=None, move_pause=1.0):
    """Run the game replay using subprocess.
    
    Args:
        log_dir: Path to the log directory
        game_number: Specific game number to replay (None means all games)
        move_pause: Pause between moves in seconds
    """
    cmd = ["python", "replay.py", "--log-dir", log_dir, "--move-pause", str(move_pause)]
    
    if game_number is not None:
        cmd.extend(["--game", str(game_number)])
        
        # Show information about available moves if a specific game is selected
        try:
            json_summary_file = os.path.join(log_dir, f"game{game_number}_summary.json")
            if os.path.exists(json_summary_file):
                with open(json_summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                move_info = ""
                if 'moves' in summary_data and summary_data['moves']:
                    move_count = len(summary_data['moves'])
                    move_info = f"Found {move_count} stored moves in summary file."
                else:
                    move_info = "No stored moves found in summary file."
                
                st.info(move_info)
        except Exception as e:
            st.warning(f"Error checking for moves in summary file: {e}")
    
    try:
        process = subprocess.Popen(cmd)
        st.write(f"Started replay of {os.path.basename(log_dir)}")
        if game_number is not None:
            st.write(f"Game {game_number}")
        else:
            st.write("All games")
        st.write("Note: The replay window opens in a separate window.")
        
        # Keep the Streamlit app running while the subprocess is active
        while process.poll() is None:
            time.sleep(0.1)
            
        st.write("Replay finished.")
    except Exception as e:
        st.error(f"Error running replay: {e}")

def app():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LLM Snake Game Analyzer",
        page_icon="🐍",
        layout="wide"
    )
    
    st.title("🐍 LLM Snake Game Analyzer")
    
    # Find log folders
    with st.spinner("Finding log folders..."):
        log_folders = find_log_folders()
    
    if not log_folders:
        st.warning("No log folders found. Make sure you're running this app from the correct directory.")
        return
        
    # Extract stats from log folders
    with st.spinner("Extracting game statistics..."):
        folder_stats = [extract_game_stats(folder) for folder in log_folders]
    
    # Create a DataFrame for easier manipulation
    stats_df = pd.DataFrame(folder_stats)
    
    # Sort by date if available
    if 'date' in stats_df.columns and not stats_df['date'].isna().all():
        stats_df = stats_df.sort_values('date', ascending=False)
    
    # Display overall statistics
    st.header("Overall Statistics")
    
    # Create columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Experiments", len(stats_df))
    
    with col2:
        total_games = stats_df['total_games'].sum()
        st.metric("Total Games Played", total_games)
    
    with col3:
        total_score = stats_df['total_score'].sum()
        st.metric("Total Score", total_score)
    
    with col4:
        avg_score = total_score / total_games if total_games > 0 else 0
        st.metric("Overall Avg Score", f"{avg_score:.2f}")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Experiment List", "Game Replay"])
    
    with tab1:
        # Display experiment list
        st.subheader("Experiment List")
        
        # Create filter section
        st.subheader("Filter Experiments")
        col1, col2 = st.columns(2)
        
        # Get unique providers and models from all experiments
        all_providers = []
        all_models = []
        for s in stats_df['providers']:
            all_providers.extend([p for p in s if p not in all_providers])
        for s in stats_df['models']:
            all_models.extend([m for m in s if m not in all_models])
        
        # Provider filter
        with col1:
            selected_providers = st.multiselect(
                "Select Providers:", 
                options=sorted(all_providers),
                default=[]
            )
        
        # Model filter
        with col2:
            selected_models = st.multiselect(
                "Select Models:", 
                options=sorted(all_models),
                default=[]
            )
        
        # Primary and Secondary LLM filters
        col3, col4 = st.columns(2)
        
        # Get unique primary and secondary LLMs
        all_primary_llms = sorted(stats_df['primary_llm'].unique())
        all_secondary_llms = sorted(stats_df['secondary_llm'].unique())
        
        with col3:
            selected_primary_llms = st.multiselect(
                "Select Primary LLMs:", 
                options=all_primary_llms,
                default=[]
            )
        
        with col4:
            selected_secondary_llms = st.multiselect(
                "Select Secondary LLMs:", 
                options=all_secondary_llms,
                default=[]
            )
        
        # Apply filters
        filtered_df = stats_df.copy()
        
        if selected_providers:
            filtered_df = filtered_df[filtered_df['providers'].apply(lambda x: any(p in x for p in selected_providers))]
        
        if selected_models:
            filtered_df = filtered_df[filtered_df['models'].apply(lambda x: any(m in x for m in selected_models))]
        
        if selected_primary_llms:
            filtered_df = filtered_df[filtered_df['primary_llm'].isin(selected_primary_llms)]
        
        if selected_secondary_llms:
            filtered_df = filtered_df[filtered_df['secondary_llm'].isin(selected_secondary_llms)]
        
        # Create a more user-friendly display dataframe
        display_df = filtered_df[['folder', 'total_score', 'total_steps', 'mean_score', 'max_score', 
                                  'primary_llm', 'secondary_llm', 'avg_response_time', 
                                  'avg_secondary_response_time', 'steps_per_apple', 'json_success_rate']].copy()
        
        display_df.columns = ['Folder', 'Total Score', 'Total Steps', 'Mean Score', 'Max Score', 
                              'Primary LLM', 'Secondary LLM', 'Avg Response Time (s)', 
                              'Avg Secondary Response Time (s)', 'Steps Per Apple', 'JSON Success Rate (%)']
        
        display_df['Folder'] = display_df['Folder'].apply(lambda x: os.path.basename(x))
        display_df['Avg Response Time (s)'] = display_df['Avg Response Time (s)'].apply(lambda x: f"{x:.2f}")
        display_df['Avg Secondary Response Time (s)'] = display_df['Avg Secondary Response Time (s)'].apply(lambda x: f"{x:.2f}")
        display_df['Steps Per Apple'] = display_df['Steps Per Apple'].apply(lambda x: f"{x:.2f}")
        display_df['JSON Success Rate (%)'] = display_df['JSON Success Rate (%)'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Select experiment for detailed view
        st.subheader("Experiment Details")
        
        # Let user select a folder
        selected_folder_idx = st.selectbox(
            "Select experiment to view details:",
            range(len(stats_df)),
            format_func=lambda x: f"{stats_df.iloc[x]['folder']} ({stats_df.iloc[x]['date'] if not pd.isna(stats_df.iloc[x]['date']) else 'Unknown date'})",
            key="folder_select"
        )
        
        selected_stats = stats_df.iloc[selected_folder_idx]
        
        # Display selected experiment details
        st.write(f"**Details for {os.path.basename(selected_stats['folder'])}**")
        
        # Extract game data for the selected folder
        game_data = selected_stats['game_data']
        
        if game_data:
            # Create a DataFrame for game data
            game_list = []
            for game_num, data in game_data.items():
                game_list.append({
                    'Game #': game_num,
                    'Score': data.get('score', 0),
                    'Steps': data.get('steps', 0),
                    'End Reason': data.get('game_end_reason', 'Unknown'),
                    'Has Apple Positions': data.get('has_apple_positions', False)
                })
            
            game_df = pd.DataFrame(game_list)
            
            # Show game performance chart
            if game_df.shape[0] > 0:
                st.subheader("Game Performance")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(game_df['Game #'], game_df['Score'], color='skyblue')
                ax.set_xlabel('Game Number')
                ax.set_ylabel('Score')
                ax.set_title('Score by Game')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
            
            # Display game list
            st.subheader("Games")
            
            # Format the display columns
            display_cols = ['Game #', 'Score', 'Steps', 'End Reason']
            if 'Has Apple Positions' in game_df.columns:
                display_cols.append('Has Apple Positions')
                
            st.dataframe(game_df[display_cols], use_container_width=True)
        else:
            st.warning("No game data found for this experiment.")
    
    with tab2:
        # Game replay section
        st.subheader("Game Replay")
        
        # Let user select an experiment for replay
        replay_folder_idx = st.selectbox(
            "Select experiment to replay:",
            range(len(stats_df)),
            format_func=lambda x: f"{stats_df.iloc[x]['folder']} ({stats_df.iloc[x]['date'] if not pd.isna(stats_df.iloc[x]['date']) else 'Unknown date'})",
            key="replay_folder_select"
        )
        
        replay_stats = stats_df.iloc[replay_folder_idx]
        replay_folder = replay_stats['folder']
        
        # Get game data for the selected experiment
        game_data = replay_stats['game_data']
        
        if game_data:
            # Create a list of games with info about apple positions
            game_options = []
            for game_num, data in sorted(game_data.items()):
                has_positions = data.get('has_apple_positions', False)
                has_moves = data.get('has_moves', False)
                move_count = data.get('move_count', 0)
                score = data.get('score', 0)
                steps = data.get('steps', 0)
                
                # Build detailed info string
                position_info = ""
                if has_positions:
                    position_info += f" [✓ apple positions]"
                else:
                    position_info += " [no apple positions]"
                    
                # Add move information
                if has_moves:
                    position_info += f" [✓ {move_count} stored moves]"
                else:
                    position_info += " [no stored moves]"
                    
                option_text = f"Game {game_num} - Score: {score}, Steps: {steps}{position_info}"
                game_options.append((game_num, option_text))
            
            # Add "All Games" option
            game_options.append(("all", "All Games"))
            
            # Let user select a game to replay
            selected_game_option = st.selectbox(
                "Select game to replay:",
                options=[opt[0] for opt in game_options],
                format_func=lambda x: next((opt[1] for opt in game_options if opt[0] == x), str(x)),
                index=len(game_options) - 1  # Default to "All Games"
            )
            
            # Replay configuration
            move_pause = st.slider("Move pause (seconds):", 0.1, 2.0, 1.0, 0.1)
            
            # Replay button
            if st.button("Start Replay"):
                game_num = None if selected_game_option == "all" else selected_game_option
                run_replay(replay_folder, game_num, move_pause)
        else:
            st.warning("No game data found for this experiment.")

if __name__ == "__main__":
    app()
