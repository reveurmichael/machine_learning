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

def find_log_folders(base_dir='.', max_depth=2):
    """Find all log folders in the given directory and its subdirectories.
    
    Args:
        base_dir: Base directory to start search from
        max_depth: Maximum depth of subdirectories to search
        
    Returns:
        List of paths to log folders containing game data
    """
    log_folders = []
    base_path = Path(base_dir)
    
    # Pattern to match log folders (contains info.txt and/or response_*.txt files)
    for depth in range(max_depth + 1):
        # Create pattern for current depth
        # Use wildcards for each directory level up to the current depth
        if depth == 0:
            # Just look in the base directory
            patterns = [str(base_path / "*_*")]  # modelname_timestamp pattern
        else:
            # Look in subdirectories up to depth
            parts = ['*'] * depth
            patterns = [str(base_path.joinpath(*parts) / "*_*")]
        
        # Find all potential log folders for current patterns
        for pattern in patterns:
            potential_folders = glob.glob(pattern)
            
            for folder in potential_folders:
                # Check if folder contains info.txt, response files, or JSON summary files
                if (Path(folder) / 'info.txt').exists() or \
                   glob.glob(str(Path(folder) / 'response_*.txt')) or \
                   glob.glob(str(Path(folder) / 'game*_summary.json')):
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
    
    # Try to extract info from info.txt
    info_path = Path(log_folder) / 'info.txt'
    if info_path.exists():
        with open(info_path, 'r', encoding='utf-8') as f:
            info_content = f.read()
            
            # Extract date
            date_match = re.search(r'Date: ([\d-]+ [\d:]+)', info_content)
            if date_match:
                stats['date'] = date_match.group(1)
                
            # Extract LLM information
            primary_llm_match = re.search(r'PRIMARY LLM Provider: (\w+)', info_content)
            if primary_llm_match:
                primary_provider = primary_llm_match.group(1)
                stats['primary_llm'] = primary_provider
                stats['providers'].append(primary_provider)
                
            primary_model_match = re.search(r'PRIMARY LLM Model: ([^\n]+)', info_content)
            if primary_model_match:
                primary_model = primary_model_match.group(1)
                stats['primary_llm'] += f" - {primary_model}"
                stats['models'].append(primary_model)
                
            secondary_llm_match = re.search(r'SECONDARY LLM Provider: (\w+)', info_content)
            if secondary_llm_match:
                secondary_provider = secondary_llm_match.group(1)
                stats['secondary_llm'] = secondary_provider
                if secondary_provider not in stats['providers']:
                    stats['providers'].append(secondary_provider)
                
                secondary_model_match = re.search(r'SECONDARY LLM Model: ([^\n]+)', info_content)
                if secondary_model_match:
                    secondary_model = secondary_model_match.group(1)
                    stats['secondary_llm'] += f" - {secondary_model}"
                    if secondary_model not in stats['models']:
                        stats['models'].append(secondary_model)
            elif 'SECONDARY LLM: Not used' in info_content:
                stats['secondary_llm'] = 'None'
                
            # Extract game statistics
            total_games_match = re.search(r'Total Games Played: (\d+)', info_content)
            if total_games_match:
                stats['total_games'] = int(total_games_match.group(1))
                
            total_score_match = re.search(r'Total Score: (\d+)', info_content)
            if total_score_match:
                stats['total_score'] = int(total_score_match.group(1))
                
            total_steps_match = re.search(r'Total Steps: (\d+)', info_content)
            if total_steps_match:
                stats['total_steps'] = int(total_steps_match.group(1))
                
            max_score_match = re.search(r'Maximum Score: (\d+)', info_content)
            if max_score_match:
                stats['max_score'] = int(max_score_match.group(1))
                
            min_score_match = re.search(r'Minimum Score: (\d+)', info_content)
            if min_score_match:
                stats['min_score'] = int(min_score_match.group(1))
                
            mean_score_match = re.search(r'Mean Score: ([\d.]+)', info_content)
            if mean_score_match:
                stats['mean_score'] = float(mean_score_match.group(1))
                
            # JSON success rate
            json_success_rate_match = re.search(r'Successful Extractions: \d+ \(([\d.]+)%\)', info_content)
            if json_success_rate_match:
                stats['json_success_rate'] = float(json_success_rate_match.group(1))
    
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
    
    # Check response files if no game data was found from summary files
    if not stats['game_data']:
        # Check if responses directory exists
        responses_dir = Path(log_folder) / "responses"
        if responses_dir.exists() and responses_dir.is_dir():
            response_pattern = str(responses_dir / "response_*.txt")
        else:
            # Fall back to the main directory
            response_pattern = str(Path(log_folder) / "response_*.txt")
            
        response_files = sorted(glob.glob(response_pattern))
        
        for response_file in response_files:
            game_num_match = re.search(r'response_(\d+)\.txt', os.path.basename(response_file))
            if game_num_match:
                game_num = int(game_num_match.group(1))
                stats['game_data'][game_num] = {'file': response_file}
    
    # Update total_games if not set from info.txt
    if stats['total_games'] == 0:
        stats['total_games'] = len(stats['game_data'])
    
    return stats

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
            summary['apple_positions'] = data['apple_positions']
        
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
                score = data.get('score', 0)
                steps = data.get('steps', 0)
                
                # Check if summary file has apple positions
                if has_positions:
                    apple_positions = extract_apple_positions(replay_folder, game_num)
                    position_info = f" [✓ {len(apple_positions)} apple positions]" if apple_positions else ""
                else:
                    position_info = " [no apple positions]"
                    
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
