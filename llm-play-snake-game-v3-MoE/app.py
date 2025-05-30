"""
Snake Game Analytics Dashboard

A Streamlit app for analyzing, replaying, and continuing recorded Snake game sessions.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import subprocess
from datetime import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="Snake Game Analytics",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# No custom CSS for better appearance - removed as requested

# Function to find log folders
def find_log_folders():
    """Find all log folders in the logs directory.
    
    Returns:
        List of log folder paths
    """
    logs_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logs_dir):
        return []
    
    # Get subdirectories in logs folder
    log_folders = []
    for folder in os.listdir(logs_dir):
        folder_path = os.path.join(logs_dir, folder)
        if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "summary.json")):
            log_folders.append(folder_path)
    
    # Sort by folder name (typically contains timestamp)
    log_folders.sort(reverse=True)
    return log_folders

# Function to extract folder name from path
def get_folder_display_name(path):
    """Extract the folder name from a path.
    
    Args:
        path: The folder path
        
    Returns:
        Display name for the folder
    """
    return os.path.basename(path)

# Function to load summary data
def load_summary_data(folder_path):
    """Load summary data from a log folder.
    
    Args:
        folder_path: Path to the log folder
        
    Returns:
        Dictionary with summary data
    """
    summary_path = os.path.join(folder_path, "summary.json")
    try:
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        return summary_data
    except Exception as e:
        st.error(f"Error loading summary data: {e}")
        return None

# Function to load game data
def load_game_data(folder_path):
    """Load all game data from a log folder.
    
    Args:
        folder_path: Path to the log folder
        
    Returns:
        Dictionary of game data keyed by game number
    """
    games = {}
    for file in os.listdir(folder_path):
        if file.startswith("game_") and file.endswith(".json"):
            try:
                with open(os.path.join(folder_path, file), 'r') as f:
                    game_data = json.load(f)
                game_num = int(file.replace("game_", "").replace(".json", ""))
                games[game_num] = game_data
            except Exception:
                pass
    return games

# Function to run replay in a subprocess
def run_replay(log_folder, game_num):
    """Run a replay of a specific game.
    
    Args:
        log_folder: Path to the log folder
        game_num: Game number to replay
    """
    try:
        cmd = ["python", "replay.py", "--log-dir", log_folder, "--game", str(game_num)]
        process = subprocess.Popen(cmd)
        st.info(f"Replay started for Game {game_num}. Close the replay window when finished.")
        return process
    except Exception as e:
        st.error(f"Error running replay: {e}")
        return None

# Function to run continuation in a subprocess
def continue_game(log_folder, max_games, no_gui):
    """Continue a game session.
    
    Args:
        log_folder: Path to the log folder
        max_games: Maximum number of games to play
        no_gui: Whether to disable GUI
    """
    try:
        cmd = ["python", "main.py", "--continue-with-game-in-dir", log_folder, "--max-games", str(max_games)]
        if no_gui:
            cmd.append("--no-gui")
        
        process = subprocess.Popen(cmd)
        st.info(f"Continuation started with max games: {max_games}. The process is running in the background.")
        return process
    except Exception as e:
        st.error(f"Error continuing game: {e}")
        return None

# Function to display experiment overview
def display_experiment_overview(log_folders):
    """Display overview of all experiments.
    
    Args:
        log_folders: List of log folder paths
    """
    if not log_folders:
        st.warning("No experiment logs found.")
        return
    
    # Create a list to store all experiment data
    experiments_data = []
    
    # Process each log folder
    for folder in log_folders:
        try:
            summary_data = load_summary_data(folder)
            if not summary_data:
                continue
            
            # Extract key information
            folder_name = get_folder_display_name(folder)
            timestamp = summary_data.get("timestamp", "Unknown")
            config = summary_data.get("configuration", {})
            
            # LLM information
            primary_model = f"{config.get('provider', 'Unknown')}/{config.get('model', 'default')}"
            secondary_model = f"{config.get('parser_provider', 'None')}/{config.get('parser_model', 'default')}"
            if config.get('parser_provider', '').lower() == 'none':
                secondary_model = "None"
            
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
            error_steps = step_stats.get("error_steps", 0)
            invalid_reversals = step_stats.get("invalid_reversals", 0)
            
            # JSON parsing statistics
            json_stats = summary_data.get("json_parsing_stats", {})
            total_extractions = json_stats.get("total_extraction_attempts", 0)
            successful_extractions = json_stats.get("successful_extractions", 0)
            json_success_rate = (successful_extractions / total_extractions * 100) if total_extractions > 0 else 0
            
            # Continuation information
            continuation_info = summary_data.get("continuation_info", {})
            is_continuation = continuation_info.get("is_continuation", False)
            continuation_count = continuation_info.get("continuation_count", 0) if is_continuation else 0
            
            # Add to experiments data
            experiments_data.append({
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
                "Error Steps": error_steps,
                "Invalid Reversals": invalid_reversals,
                "JSON Success Rate": json_success_rate,
                "Is Continuation": is_continuation,
                "Continuation Count": continuation_count
            })
            
        except Exception as e:
            st.error(f"Error processing {folder}: {e}")
    
    # Create DataFrame from experiments data
    if experiments_data:
        overview_df = pd.DataFrame(experiments_data)
        
        # Display the data
        st.dataframe(
            overview_df,
            column_config={
                "Experiment": st.column_config.TextColumn("Experiment"),
                "Timestamp": st.column_config.TextColumn("Timestamp"),
                "Primary LLM": st.column_config.TextColumn("Primary LLM"),
                "Secondary LLM": st.column_config.TextColumn("Secondary LLM"),
                "Total Games": st.column_config.NumberColumn("Total Games", format="%d"),
                "Total Score": st.column_config.NumberColumn("Total Score", format="%d"),
                "Mean Score": st.column_config.NumberColumn("Mean Score", format="%.2f"),
                "Apples/Step": st.column_config.NumberColumn("Apples/Step", format="%.4f"),
                "Total Steps": st.column_config.NumberColumn("Total Steps", format="%d"),
                "Valid Steps": st.column_config.NumberColumn("Valid Steps", format="%d"),
                "Empty Steps": st.column_config.NumberColumn("Empty Steps", format="%d"),
                "Error Steps": st.column_config.NumberColumn("Error Steps", format="%d"),
                "Invalid Reversals": st.column_config.NumberColumn("Invalid Reversals", format="%d"),
                "JSON Success Rate": st.column_config.NumberColumn("JSON Success Rate", format="%.1f%%"),
                "Is Continuation": st.column_config.CheckboxColumn("Continuation?"),
                "Continuation Count": st.column_config.NumberColumn("# Continuations", format="%d")
            },
            use_container_width=True,
            hide_index=True
        )
        
        return overview_df
    
    return None

def display_experiment_details(folder_path):
    """Display detailed information about an experiment.
    
    Args:
        folder_path: Path to the log folder
    """
    summary_data = load_summary_data(folder_path)
    if not summary_data:
        st.warning("No summary data found.")
        return
    
    # Get game data
    games_data = load_game_data(folder_path)
    
    # Display game scores chart
    if games_data:
        st.markdown("## Game Scores")
        
        # Prepare data for chart
        game_numbers = []
        scores = []
        for game_num, data in sorted(games_data.items()):
            game_numbers.append(f"Game {game_num}")
            scores.append(data.get("score", 0))
        
        # Create bar chart
        fig = px.bar(
            x=game_numbers,
            y=scores,
            labels={'x': 'Game', 'y': 'Score'},
            title="Score by Game",
            color=scores,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Game",
            yaxis_title="Score"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display LLM Time and Token Statistics
        st.markdown("## LLM Performance")
        
        # Get time statistics
        time_stats = summary_data.get("time_statistics", {})
        llm_communication_time = time_stats.get("total_llm_communication_time", 0)
        game_movement_time = time_stats.get("total_game_movement_time", 0)
        waiting_time = time_stats.get("total_waiting_time", 0)
        
        # Create time stats chart
        time_data = {
            "Category": ["LLM Communication", "Game Movement", "Waiting Time"],
            "Time (seconds)": [llm_communication_time, game_movement_time, waiting_time]
        }
        time_df = pd.DataFrame(time_data)
        
        fig_time = px.bar(
            time_df,
            x="Category",
            y="Time (seconds)",
            title="Time Distribution",
            color="Category"
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Get token statistics
        token_stats = summary_data.get("token_usage_stats", {})
        primary_tokens = token_stats.get("primary_llm", {}).get("total_tokens", 0)
        secondary_tokens = token_stats.get("secondary_llm", {}).get("total_tokens", 0)
        
        # Create token stats chart
        token_data = {
            "LLM": ["Primary LLM", "Secondary LLM"],
            "Tokens": [primary_tokens, secondary_tokens]
        }
        token_df = pd.DataFrame(token_data)
        
        fig_tokens = px.bar(
            token_df,
            x="LLM",
            y="Tokens",
            title="Token Usage",
            color="LLM"
        )
        st.plotly_chart(fig_tokens, use_container_width=True)
        
        # Display game details table - kept as requested
        st.markdown("## Game Details")
        
        # Prepare data for table
        games_list = []
        for game_num, data in sorted(games_data.items()):
            step_stats = data.get("step_stats", {})
            total_game_steps = data.get("steps", 0)
            valid_game_steps = step_stats.get("valid_steps", 0)
            empty_game_steps = step_stats.get("empty_steps", 0)
            error_game_steps = step_stats.get("error_steps", 0)
            
            # Calculate step percentages
            valid_pct = (valid_game_steps / total_game_steps * 100) if total_game_steps > 0 else 0
            empty_pct = (empty_game_steps / total_game_steps * 100) if total_game_steps > 0 else 0
            error_pct = (error_game_steps / total_game_steps * 100) if total_game_steps > 0 else 0
            
            games_list.append({
                "Game": game_num,
                "Score": data.get("score", 0),
                "Steps": total_game_steps,
                "Valid Steps": valid_game_steps,
                "Valid %": f"{valid_pct:.1f}%",
                "Empty Steps": empty_game_steps,
                "Empty %": f"{empty_pct:.1f}%",
                "Error Steps": error_game_steps,
                "Error %": f"{error_pct:.1f}%",
                "End Reason": data.get("game_end_reason", "Unknown"),
                "Rounds": data.get("round_count", 0)
            })
        
        # Create DataFrame and display
        games_df = pd.DataFrame(games_list)
        st.dataframe(
            games_df,
            column_config={
                "Game": st.column_config.NumberColumn("Game", format="%d"),
                "Score": st.column_config.NumberColumn("Score", format="%d"),
                "Steps": st.column_config.NumberColumn("Steps", format="%d"),
                "Valid Steps": st.column_config.NumberColumn("Valid Steps", format="%d"),
                "Valid %": st.column_config.TextColumn("Valid %"),
                "Empty Steps": st.column_config.NumberColumn("Empty Steps", format="%d"),
                "Empty %": st.column_config.TextColumn("Empty %"),
                "Error Steps": st.column_config.NumberColumn("Error Steps", format="%d"),
                "Error %": st.column_config.TextColumn("Error %"),
                "End Reason": st.column_config.TextColumn("End Reason"),
                "Rounds": st.column_config.NumberColumn("Rounds", format="%d")
            },
            use_container_width=True,
            hide_index=True
        )

def main():
    """Main application function."""
    # Display header
    st.title("🐍 Snake Game Analytics Dashboard")
    
    # Find log folders
    log_folders = find_log_folders()
    
    # Create tabs - primitive style, no fancy styling
    tab1, tab2, tab3 = st.tabs(["Overview", "Replay Mode", "Continue Mode"])
    
    with tab1:
        # Overview tab
        st.markdown("### Experiment Overview")
        st.markdown("View statistics and detailed information about all Snake game experiments.")
        
        # Display experiment overview
        overview_df = display_experiment_overview(log_folders)
        
        # Allow selecting an experiment for detailed view
        if overview_df is not None and not overview_df.empty:
            st.markdown("### Experiment Details")
            st.markdown("Select an experiment to view detailed information.")
            
            # Create a dropdown to select an experiment
            selected_exp = st.selectbox(
                "Select Experiment",
                options=overview_df["Folder"].tolist(),
                format_func=get_folder_display_name,
                index=0
            )
            
            # Display experiment details
            display_experiment_details(selected_exp)
    
    with tab2:
        # Replay Mode tab
        st.markdown("### Replay Game")
        st.markdown("Select an experiment and game to replay.")
        
        # Select an experiment
        if log_folders:
            replay_exp = st.selectbox(
                "Select Experiment for Replay",
                options=log_folders,
                format_func=get_folder_display_name,
                index=0,
                key="replay_exp"
            )
            
            # Load game data for the selected experiment
            games_data = load_game_data(replay_exp)
            
            if games_data:
                # Select a game to replay
                game_options = sorted(games_data.keys())
                replay_game = st.selectbox(
                    "Select Game to Replay",
                    options=game_options,
                    index=0,
                    format_func=lambda x: f"Game {x} (Score: {games_data[x].get('score', 0)})"
                )
                
                # Display game info
                selected_game = games_data.get(replay_game, {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score", selected_game.get("score", 0))
                with col2:
                    st.metric("Steps", selected_game.get("steps", 0))
                with col3:
                    st.metric("End Reason", selected_game.get("game_end_reason", "Unknown"))
                
                # Add replay button
                if st.button("Start Replay", key="start_replay"):
                    with st.spinner("Starting replay..."):
                        process = run_replay(replay_exp, replay_game)
                        if process:
                            st.success("Replay started! Check the game window.")
            else:
                st.warning("No games found in the selected experiment.")
        else:
            st.warning("No experiment logs found.")
    
    with tab3:
        # Continue Mode tab
        st.markdown("### Continue Game")
        st.markdown("Select an experiment to continue from the last game.")
        
        # Select an experiment
        if log_folders:
            continue_exp = st.selectbox(
                "Select Experiment to Continue",
                options=log_folders,
                format_func=get_folder_display_name,
                index=0,
                key="continue_exp"
            )
            
            # Load summary data
            summary_data = load_summary_data(continue_exp)
            
            if summary_data:
                # Display experiment info
                config = summary_data.get("configuration", {})
                game_stats = summary_data.get("game_statistics", {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Games Played", game_stats.get("total_games", 0))
                with col2:
                    st.metric("Total Score", game_stats.get("total_score", 0))
                with col3:
                    total_games = game_stats.get("total_games", 0)
                    total_score = game_stats.get("total_score", 0)
                    mean_score = total_score / total_games if total_games > 0 else 0
                    st.metric("Mean Score", f"{mean_score:.2f}")
                
                # Configuration for continuation
                st.markdown("### Continuation Settings")
                
                max_games = st.number_input(
                    "Maximum Games to Play",
                    min_value=1,
                    max_value=100,
                    value=int(config.get("max_games", 10)),
                    step=1
                )
                
                no_gui = st.checkbox(
                    "Disable GUI",
                    value=bool(config.get("no_gui", False))
                )
                
                # Add continue button
                if st.button("Start Continuation", key="start_continuation"):
                    with st.spinner("Starting continuation..."):
                        process = continue_game(continue_exp, max_games, no_gui)
                        if process:
                            st.success(f"Continuation started with max games: {max_games}")
                            
                            # Show info about viewing results
                            st.info(
                                "The game is running in the background. " +
                                "Refresh this page later to see updated results in the Overview tab."
                            )
            else:
                st.warning("No summary data found for the selected experiment.")
        else:
            st.warning("No experiment logs found.")

if __name__ == "__main__":
    main()
