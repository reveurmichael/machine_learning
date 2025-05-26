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
    selected_game_dir = st.selectbox(
        "Select a game directory",
        game_dirs,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Find game summary files in the selected directory
    game_summary_files = []
    for filename in os.listdir(selected_game_dir):
        if filename.startswith("game") and filename.endswith("_summary.json"):
            game_summary_files.append(os.path.join(selected_game_dir, filename))
    
    # Sort files by game number
    game_summary_files.sort(key=lambda x: int(os.path.basename(x).split("game")[1].split("_")[0]))
    
    # Create a display name for each file
    game_options = {}
    for file_path in game_summary_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                game_num = int(os.path.basename(file_path).split("game")[1].split("_")[0])
                score = data.get("score", 0)
                steps = data.get("steps", 0)
                end_reason = data.get("game_end_reason", "Unknown")
                game_options[file_path] = f"Game {game_num} - Score: {score}, Steps: {steps}, End: {end_reason}"
        except:
            # If there's an error, just use the filename
            game_options[file_path] = os.path.basename(file_path)
    
    # Game file selection
    if len(game_summary_files) > 1:
        selected_game_file = st.selectbox(
            "Select a game to replay",
            options=list(game_options.keys()),
            format_func=lambda x: game_options[x]
        )
    else:
        selected_game_file = game_summary_files[0] if game_summary_files else None
    
    if not selected_game_file:
        st.error("No game summary files found in the selected directory.")
        return
    
    # Load game data from the selected file
    game_data, game_info = load_game_data_with_info(selected_game_file)
    if not game_data:
        st.error("Failed to load game data.")
        return
    
    # Display game information
    st.subheader("Game Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", game_info.get("score", 0))
    with col2:
        st.metric("Steps", game_info.get("steps", 0))
    with col3:
        st.metric("End Reason", game_info.get("game_end_reason", "Unknown"))
        
    # Show more details in an expander
    with st.expander("Game Details"):
        st.write(f"**Game Time:** {game_info.get('timestamp', 'Unknown')}")
        st.write(f"**Snake Length:** {game_info.get('snake_length', 0)}")
        st.write(f"**Primary LLM:** {game_info.get('primary_provider', 'Unknown')} - {game_info.get('primary_model', 'Unknown')}")
        if game_info.get('parser_provider') != "none":
            st.write(f"**Parser LLM:** {game_info.get('parser_provider', 'Unknown')} - {game_info.get('parser_model', 'Unknown')}")
        else:
            st.write("**Parser LLM:** None (bypassed)")
            
        # Show performance metrics if available
        if "performance_metrics" in game_info:
            st.write("**Performance Metrics:**")
            for key, value in game_info["performance_metrics"].items():
                st.write(f"- {key.replace('_', ' ').title()}: {value:.2f}")
                
        # Show response time metrics with more detail
        if "prompt_response_stats" in game_info:
            st.write("**Response Time Metrics:**")
            prompt_stats = game_info["prompt_response_stats"]
            
            # Primary LLM response times
            st.write("*Primary LLM Response Times:*")
            if "avg_primary_response_time" in prompt_stats:
                st.write(f"- Average: {prompt_stats['avg_primary_response_time']:.3f}s")
            if "min_primary_response_time" in prompt_stats:
                st.write(f"- Minimum: {prompt_stats['min_primary_response_time']:.3f}s")
            if "max_primary_response_time" in prompt_stats:
                st.write(f"- Maximum: {prompt_stats['max_primary_response_time']:.3f}s")
            
            # Secondary LLM response times
            if game_info.get('parser_provider') != "none":
                st.write("*Secondary LLM Response Times:*")
                if "avg_secondary_response_time" in prompt_stats:
                    st.write(f"- Average: {prompt_stats['avg_secondary_response_time']:.3f}s")
                if "min_secondary_response_time" in prompt_stats:
                    st.write(f"- Minimum: {prompt_stats['min_secondary_response_time']:.3f}s")
                if "max_secondary_response_time" in prompt_stats:
                    st.write(f"- Maximum: {prompt_stats['max_secondary_response_time']:.3f}s")
        
        # Show token usage statistics if available
        if "token_stats" in game_info:
            st.write("**Token Usage Statistics:**")
            token_stats = game_info["token_stats"]
            
            # Primary LLM token usage
            if "primary" in token_stats:
                primary_tokens = token_stats["primary"]
                st.write("*Primary LLM Token Usage:*")
                if "total_tokens" in primary_tokens:
                    st.write(f"- Total Tokens: {primary_tokens['total_tokens']}")
                if "total_prompt_tokens" in primary_tokens:
                    st.write(f"- Prompt Tokens: {primary_tokens['total_prompt_tokens']}")
                if "total_completion_tokens" in primary_tokens:
                    st.write(f"- Completion Tokens: {primary_tokens['total_completion_tokens']}")
                if "avg_total_tokens" in primary_tokens:
                    st.write(f"- Average Tokens per Request: {primary_tokens['avg_total_tokens']:.2f}")
            
            # Secondary LLM token usage
            if "secondary" in token_stats and game_info.get('parser_provider') != "none":
                secondary_tokens = token_stats["secondary"]
                st.write("*Secondary LLM Token Usage:*")
                if "total_tokens" in secondary_tokens:
                    st.write(f"- Total Tokens: {secondary_tokens['total_tokens']}")
                if "total_prompt_tokens" in secondary_tokens:
                    st.write(f"- Prompt Tokens: {secondary_tokens['total_prompt_tokens']}")
                if "total_completion_tokens" in secondary_tokens:
                    st.write(f"- Completion Tokens: {secondary_tokens['total_completion_tokens']}")
                if "avg_total_tokens" in secondary_tokens:
                    st.write(f"- Average Tokens per Request: {secondary_tokens['avg_total_tokens']:.2f}")
        
        # Show JSON parsing statistics if available
        if "json_parsing_stats" in game_info:
            st.write("**JSON Parsing Statistics:**")
            json_stats = game_info["json_parsing_stats"]
            if "success_rate" in json_stats:
                st.write(f"- Success Rate: {json_stats['success_rate']:.2f}%")
            if "successful_extractions" in json_stats and "total_extraction_attempts" in json_stats:
                st.write(f"- Successful Extractions: {json_stats['successful_extractions']}/{json_stats['total_extraction_attempts']}")
            
        # Add a section for rounds data overview
        if "rounds_data" in game_info:
            st.write("**Rounds Overview:**")
            rounds = game_info["rounds_data"]
            st.write(f"- Total Rounds: {len(rounds)}")
            
            # Create a table for round summary
            round_data = []
            for round_key, round_info in rounds.items():
                round_num = int(round_key.split('_')[1])
                moves = len(round_info.get("moves", []))
                apple_pos = round_info.get("apple_position", "Unknown")
                round_data.append({
                    "Round": round_num,
                    "Apple Position": str(apple_pos),
                    "Moves": moves
                })
            
            if round_data:
                round_df = pd.DataFrame(round_data)
                st.dataframe(round_df)
    
    # Add a button to launch the external replay
    if st.button("Launch Fullscreen Replay"):
        # Build the command to run the replay.py script
        command = f"python replay.py --game-file {selected_game_file} --move-pause 1.0"
        
        # Use subprocess to run the command
        try:
            import subprocess
            st.info(f"Launching external replay: {command}")
            subprocess.Popen(command, shell=True)
            st.success("Replay launched in a separate window. You can continue using the app.")
        except Exception as e:
            st.error(f"Error launching replay: {str(e)}")
    
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
    # Ensure the game state is drawn to the GUI surface first
    if gui and game_data: # Make sure we have game_data to avoid error if it failed to load
        gui.draw(game, 1, 1, game.steps) # game_number and round_number are placeholders
        pil_image = gui.get_surface_as_image()
        if pil_image:
            st.image(pil_image, caption="Game State")
        else:
            st.warning("Could not render game state.")


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
        if any(f.startswith("game") and f.endswith("_summary.json") for f in files):
            game_dirs.append(root)
            
    return sorted(game_dirs)


def load_game_data(game_dir):
    """Load game data from a directory.
    
    Args:
        game_dir: Directory containing game data
        
    Returns:
        List of game moves
    """
    # Find game summary files
    game_summary_files = []
    for filename in os.listdir(game_dir):
        if filename.startswith("game") and filename.endswith("_summary.json"):
            game_summary_files.append(os.path.join(game_dir, filename))
            
    if not game_summary_files:
        return None
        
    # Use the first game summary file by default
    # Sort files by game number
    game_summary_files.sort(key=lambda x: int(os.path.basename(x).split("game")[1].split("_")[0]))
    game_summary_file = game_summary_files[0]
    
    # Load game data
    with open(game_summary_file, "r") as f:
        data = json.load(f)
        
        # Check if using old format with "moves" array
        if "moves" in data:
            # Check if moves is a list
            if isinstance(data["moves"], list):
                return data["moves"]
            # Check if moves is a dictionary with round keys
            elif isinstance(data["moves"], dict):
                # Convert dictionary of moves by round to flat list of moves
                moves_list = []
                # Sort round keys numerically
                sorted_rounds = sorted(data["moves"].keys(), 
                                      key=lambda x: int(x.split("_")[1]) if x.startswith("round_") else 0)
                for round_key in sorted_rounds:
                    move = data["moves"][round_key]
                    moves_list.append(move)
                return moves_list
        
        # Check if using new format with "rounds_data"
        elif "rounds_data" in data:
            # Extract moves from rounds_data and flatten into a list
            moves_list = []
            # Sort round keys numerically
            sorted_rounds = sorted(data["rounds_data"].keys(),
                                  key=lambda x: int(x.split("_")[1]) if x.startswith("round_") else 0)
            for round_key in sorted_rounds:
                round_data = data["rounds_data"][round_key]
                if "moves" in round_data:
                    moves_list.append(round_data["moves"])
            return moves_list
                
    return None


def load_game_data_with_info(game_file):
    """Load game data and game information from a file.
    
    Args:
        game_file: Path to the game summary file
        
    Returns:
        Tuple containing game data and game information
    """
    try:
        # Load game data and information from the same file
        with open(game_file, "r") as f:
            data = json.load(f)
            
            # Extract moves data
            moves_data = None
            
            # Check if using old format with "moves" array
            if "moves" in data:
                # Check if moves is a list
                if isinstance(data["moves"], list):
                    moves_data = data["moves"]
                # Check if moves is a dictionary with round keys
                elif isinstance(data["moves"], dict):
                    # Convert dictionary of moves by round to flat list of moves
                    moves_list = []
                    # Sort round keys numerically
                    sorted_rounds = sorted(data["moves"].keys(), 
                                         key=lambda x: int(x.split("_")[1]) if x.startswith("round_") else 0)
                    for round_key in sorted_rounds:
                        move = data["moves"][round_key]
                        moves_list.append(move)
                    moves_data = moves_list
            
            # Check if using new format with "rounds_data"
            elif "rounds_data" in data:
                # Extract moves from rounds_data and flatten into a list
                moves_list = []
                # Sort round keys numerically
                sorted_rounds = sorted(data["rounds_data"].keys(),
                                     key=lambda x: int(x.split("_")[1]) if x.startswith("round_") else 0)
                for round_key in sorted_rounds:
                    round_data = data["rounds_data"][round_key]
                    if "moves" in round_data:
                        moves_list.append(round_data["moves"])
                moves_data = moves_list
            
            # Extract game info
            game_info = {
                "score": data.get("score", 0),
                "steps": data.get("steps", 0),
                "game_end_reason": data.get("game_end_reason", "Unknown"),
                "snake_length": data.get("snake_length", 0),
                "timestamp": data.get("timestamp", "Unknown"),
                "primary_model": data.get("primary_model", "Unknown"),
                "primary_provider": data.get("primary_provider", "Unknown"),
                "parser_model": data.get("parser_model", "Unknown"),
                "parser_provider": data.get("parser_provider", "Unknown")
            }
            
            # Include performance metrics if available
            if "performance_metrics" in data:
                game_info["performance_metrics"] = data["performance_metrics"]
                
            # Include prompt response stats if available
            if "prompt_response_stats" in data:
                game_info["prompt_response_stats"] = data["prompt_response_stats"]
                
            # Include JSON parsing stats if available
            if "json_parsing_stats" in data:
                game_info["json_parsing_stats"] = data["json_parsing_stats"]
            
            return moves_data, game_info
    except Exception as e:
        print(f"Error loading game data from {game_file}: {e}")
        return None, None


if __name__ == "__main__":
    main()
