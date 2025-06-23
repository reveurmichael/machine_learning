"""
Heuristics v0.03 - Streamlit Dashboard
=====================================

Evolution from v0.02: Adding comprehensive web interface with game launching and replay capabilities.
Demonstrates natural software progression by building upon the multi-algorithm foundation.

Features:
- Launch any of the 7 heuristic algorithms with configurable parameters
- Replay games with both PyGame and Flask web modes
- Performance analysis and algorithm comparison
- Extensive reuse of Task-0 base classes and utilities
- Overview tab showing experiment statistics
- Comprehensive experiment management

This is the main entry point for users - replacing the CLI-only approach of v0.02
with a modern web-based interface while maintaining all functionality.
"""

import streamlit as st
from pathlib import Path
import os
import sys
import json
import subprocess
from typing import Dict, List
from streamlit.errors import StreamlitAPIException
import pandas as pd
from ..common.path_utils import ensure_project_root_on_path
ensure_project_root_on_path()

# Import base utilities and classes from Task-0
from core.game_file_manager import FileManager
from utils.network_utils import random_free_port, ensure_free_port
from config.network_constants import HOST_CHOICES

# Initialize file manager using Task-0 infrastructure
_file_manager = FileManager()


class HeuristicsApp:
    """
    Streamlit application for Heuristics v0.03.
    
    Demonstrates software evolution by building upon v0.02's multi-algorithm
    foundation with web interface capabilities. Extensively reuses Task-0
    infrastructure while adding heuristic-specific features.
    
    Design Pattern: Facade Pattern
    - Provides simplified interface to complex heuristic algorithm system
    - Coordinates between game launching, replay, and analysis components
    """
    
    def __init__(self):
        """Initialize the Streamlit app with proper configuration."""
        self.setup_page_config()
        self.algorithms = [
            "bfs", "bfs-safe-greedy", "bfs-hamiltonian", 
            "dfs", "astar", "astar-hamiltonian", "hamiltonian"
        ]
        self.algorithm_descriptions = {
            "bfs": "Breadth-First Search - Optimal shortest path",
            "bfs-safe-greedy": "BFS with safety validation - Best performer",
            "bfs-hamiltonian": "BFS with Hamiltonian fallback - Hybrid approach",
            "dfs": "Depth-First Search - Educational/experimental",
            "astar": "A* Algorithm - Optimal with heuristics",
            "astar-hamiltonian": "A* with Hamiltonian fallback - Advanced hybrid",
            "hamiltonian": "Hamiltonian Cycle - Space-filling approach"
        }
        self.main()
    
    def setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        try:
            st.set_page_config(
                page_title="Heuristics v0.03 - Snake Game Dashboard",
                page_icon="🧠",
                layout="wide",
                initial_sidebar_state="expanded",
            )
        except StreamlitAPIException:
            pass
    
    def main(self) -> None:
        """Main application interface."""
        st.title("🧠 Heuristics v0.03 - Snake Game Dashboard")
        st.markdown("**Evolution from v0.02**: Comprehensive web interface with game launching and replay capabilities")
        
        # Create tabs for different functionalities
        tab_overview, tab_launch, tab_replay_pg, tab_replay_web, tab_analysis = st.tabs([
            "📊 Overview",
            "🚀 Launch Games",
            "🎮 Replay (PyGame)", 
            "🌐 Replay (Web)",
            "📈 Performance Analysis"
        ])
        
        # Get available log folders for replay functionality
        log_folders = self._find_heuristic_log_folders()
        
        with tab_overview:
            self._render_overview_tab(log_folders)
        
        with tab_launch:
            self._render_launch_tab()
        
        with tab_replay_pg:
            self._render_replay_pygame_tab(log_folders)
        
        with tab_replay_web:
            self._render_replay_web_tab(log_folders)
        
        with tab_analysis:
            self._render_analysis_tab(log_folders)
    
    def _find_heuristic_log_folders(self) -> List[str]:
        """
        Find all heuristic log folders.
        
        Reuses Task-0 file manager infrastructure to discover log folders
        that match the heuristic naming convention.
        
        Returns:
            List of heuristic log folder paths
        """
        logs_dir = Path("../../logs")  # Relative to heuristics-v0.03 folder
        if not logs_dir.exists():
            return []
        
        heuristic_folders = []
        for folder in logs_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("heuristics-"):
                heuristic_folders.append(str(folder))
        
        return sorted(heuristic_folders)
    
    def _render_overview_tab(self, log_folders: List[str]) -> None:
        """
        Render experiment overview tab.
        
        Shows comprehensive statistics about all heuristic experiments,
        inspired by Task-0's overview functionality.
        """
        st.markdown("### 🧠 Heuristic Experiments Overview")
        st.markdown("Statistics and information about all heuristic algorithm experiments")
        
        if not log_folders:
            st.warning("No heuristic experiments found. Launch some games first!")
            st.info("Use the **🚀 Launch Games** tab to run heuristic algorithms and generate experiment data.")
            return
        
        # Build experiment data
        experiments_data = []
        for folder in log_folders:
            folder_path = Path(folder)
            summary_file = folder_path / "summary.json"
            
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    # Extract key information
                    experiments_data.append({
                        'Experiment': folder_path.name,
                        'Algorithm': summary.get('algorithm', 'Unknown').upper(),
                        'Total Games': summary.get('total_games', 0),
                        'Total Score': summary.get('total_score', 0),
                        'Average Score': round(summary.get('average_score', 0), 2),
                        'Total Rounds': summary.get('total_rounds', 0),
                        'Score per Step': round(summary.get('score_per_step', 0), 3),
                        'Score per Round': round(summary.get('score_per_round', 0), 3),
                        'Best Score': max(summary.get('scores', [0])),
                        'Folder Path': str(folder)
                    })
                except Exception as e:
                    st.error(f"Error loading {folder}: {e}")
        
        if not experiments_data:
            st.warning("No valid experiment data found.")
            return
        
        # Display overview table
        df = pd.DataFrame(experiments_data)
        st.dataframe(
            df.drop(columns=['Folder Path']),  # Hide internal path
            use_container_width=True
        )
        
        # Performance comparison charts
        st.markdown("### 📈 Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Average Score by Algorithm**")
            if len(df) > 0:
                # Aggregate by algorithm for better visualization
                algo_performance = df.groupby('Algorithm')['Average Score'].mean().reset_index()
                st.bar_chart(algo_performance.set_index('Algorithm'))
        
        with col2:
            st.markdown("**Efficiency (Score per Step)**")
            if len(df) > 0:
                algo_efficiency = df.groupby('Algorithm')['Score per Step'].mean().reset_index()
                st.bar_chart(algo_efficiency.set_index('Algorithm'))
        
        # Detailed experiment selection
        st.markdown("### 🔍 Experiment Details")
        selected_exp = st.selectbox(
            "Select Experiment for Details",
            options=df['Experiment'].tolist(),
            key="overview_exp_select"
        )
        
        if selected_exp:
            exp_data = df[df['Experiment'] == selected_exp].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Algorithm", exp_data['Algorithm'])
            with col2:
                st.metric("Total Games", exp_data['Total Games'])
            with col3:
                st.metric("Average Score", exp_data['Average Score'])
            with col4:
                st.metric("Best Score", exp_data['Best Score'])
            
            # Show detailed summary
            with st.expander("Show Detailed Summary"):
                summary_file = Path(exp_data['Folder Path']) / "summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary_data = json.load(f)
                    st.code(json.dumps(summary_data, indent=2), language="json")
    
    def _render_launch_tab(self) -> None:
        """Render the game launching interface."""
        st.markdown("### 🚀 Launch Heuristic Algorithms")
        st.markdown("Configure and launch any of the 7 available heuristic algorithms")
        
        # Algorithm selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_algorithm = st.selectbox(
                "Algorithm",
                self.algorithms,
                format_func=lambda x: f"{x.upper()} - {self.algorithm_descriptions[x]}",
                key="launch_algorithm"
            )
        
        with col2:
            st.info(f"**{selected_algorithm.upper()}**: {self.algorithm_descriptions[selected_algorithm]}")
        
        # Game parameters
        st.markdown("#### Game Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_games = st.number_input(
                "Max Games", 
                min_value=1, 
                max_value=100, 
                value=5,
                help="Number of games to play"
            )
        
        with col2:
            max_steps = st.number_input(
                "Max Steps", 
                min_value=100, 
                max_value=2000, 
                value=800,
                help="Maximum steps per game"
            )
        
        with col3:
            grid_size = st.number_input(
                "Grid Size", 
                min_value=5, 
                max_value=20, 
                value=10,
                help="Size of the game grid"
            )
        
        # Advanced options
        with st.expander("Advanced Options"):
            verbose = st.checkbox("Verbose Output", value=False)
            debug = st.checkbox("Debug Mode", value=False)
        
        # Launch button
        if st.button("🚀 Launch Game Session", type="primary"):
            self._launch_game_session(
                selected_algorithm, max_games, max_steps, 
                grid_size, verbose, debug
            )
    
    def _launch_game_session(
        self, 
        algorithm: str, 
        max_games: int, 
        max_steps: int,
        grid_size: int,
        verbose: bool,
        debug: bool
    ) -> None:
        """
        Launch a game session with the specified parameters.
        
        Uses the scripts/main.py approach following Task-0 patterns,
        maintaining separation between the web interface and game execution.
        """
        try:
            # Build command using scripts folder (v0.03 evolution)
            cmd = [
                sys.executable, 
                os.path.join("scripts", "main.py"),
                "--algorithm", algorithm,
                "--max-games", str(max_games),
                "--max-steps", str(max_steps),
                "--grid-size", str(grid_size)
            ]
            
            if verbose:
                cmd.append("--verbose")
            
            # Set environment variables
            env = os.environ.copy()
            if debug:
                env["HEURISTIC_DEBUG"] = "1"
            
            # Show command being executed
            st.code(" ".join(cmd), language="bash")
            
            # Execute in background following Task-0 session_utils pattern
            with st.spinner(f"Launching {algorithm.upper()} session..."):
                try:
                    # Launch in background like Task-0 session_utils
                    process = subprocess.Popen(
                        cmd,
                        cwd=Path(__file__).parent,
                        env=env
                    )
                    
                    st.success(f"✅ {algorithm.upper()} session launched in background!")
                    st.info(f"🎮 Process ID: {process.pid}")
                    st.info("Check the console output for progress updates.")
                    
                    # Add option to wait for completion
                    if st.button("Wait for Completion", key=f"wait_{algorithm}"):
                        with st.spinner("Waiting for session to complete..."):
                            stdout, stderr = process.communicate()
                            
                            if process.returncode == 0:
                                st.success("Session completed successfully!")
                                if stdout:
                                    st.code(stdout, language="text")
                            else:
                                st.error(f"Session failed with return code {process.returncode}")
                                if stderr:
                                    st.code(stderr, language="text")
                    
                except subprocess.SubprocessError as e:
                    st.error(f"❌ Failed to launch subprocess: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
                        
        except Exception as e:
            st.error(f"❌ Failed to launch session: {e}")
            
    def _run_heuristic_session(self, algorithm: str, max_games: int, host: str, port: int) -> None:
        """
        Run a heuristic session in background, following Task-0 session_utils patterns.
        """
        try:
            port = ensure_free_port(port)
            
            cmd = [
                sys.executable,
                os.path.join("scripts", "main.py"),
                "--algorithm", algorithm,
                "--max-games", str(max_games)
            ]
            
            subprocess.Popen(cmd, cwd=Path(__file__).parent)
            st.info(f"🧠 {algorithm.upper()} session started in background.")
            
        except Exception as e:
            st.error(f"❌ Failed to start heuristic session: {e}")
    
    def _render_replay_pygame_tab(self, log_folders: List[str]) -> None:
        """
        Render PyGame replay interface.
        
        Reuses Task-0 replay infrastructure with heuristic-specific adaptations.
        """
        st.markdown("### 🎮 Replay Games (PyGame)")
        
        if not log_folders:
            st.warning("No heuristic game logs found. Launch some games first!")
            return
        
        # Folder selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_folder = st.selectbox(
                "Experiment Folder",
                log_folders,
                format_func=lambda x: Path(x).name,
                key="replay_pg_folder"
            )
        
        # Load games from selected folder
        games = self._load_games_from_folder(selected_folder)
        
        if not games:
            st.warning("No games found in selected folder.")
            return
        
        with col2:
            selected_game = st.selectbox(
                "Game",
                sorted(games.keys()),
                format_func=lambda x: f"Game {x} (Score: {games[x].get('score', 0)})",
                key="replay_pg_game"
            )
        
        # Show game details
        if selected_game in games:
            game_data = games[selected_game]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", game_data.get('score', 0))
            with col2:
                st.metric("Steps", game_data.get('steps', 0))
            with col3:
                st.metric("Algorithm", game_data.get('algorithm', 'Unknown'))
            
            # Show raw JSON
            with st.expander(f"Show game_{selected_game}.json"):
                st.code(json.dumps(game_data, indent=2), language="json")
            
            # Launch replay options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎮 Task-0 PyGame Replay", type="primary"):
                    self._launch_pygame_replay(selected_folder, selected_game)
            
            with col2:
                if st.button("🧠 Heuristic PyGame Replay", type="secondary"):
                    self._launch_heuristic_pygame_replay(selected_folder, selected_game)
            
            st.info("**Task-0 PyGame Replay**: Uses ROOT replay infrastructure (universal)\n\n**Heuristic PyGame Replay**: Uses heuristic-specific replay with algorithm metrics")
    
    def _render_replay_web_tab(self, log_folders: List[str]) -> None:
        """
        Render web replay interface.
        
        Extends Task-0 web replay infrastructure for heuristic algorithms.
        """
        st.markdown("### 🌐 Replay Games (Web)")
        
        if not log_folders:
            st.warning("No heuristic game logs found. Launch some games first!")
            return
        
        # Folder and game selection (similar to PyGame tab)
        col1, col2 = st.columns(2)
        
        with col1:
            selected_folder = st.selectbox(
                "Experiment Folder",
                log_folders,
                format_func=lambda x: Path(x).name,
                key="replay_web_folder"
            )
        
        games = self._load_games_from_folder(selected_folder)
        
        if not games:
            st.warning("No games found in selected folder.")
            return
        
        with col2:
            selected_game = st.selectbox(
                "Game",
                sorted(games.keys()),
                format_func=lambda x: f"Game {x} (Score: {games[x].get('score', 0)})",
                key="replay_web_game"
            )
        
        # Web server configuration
        st.markdown("#### Web Server Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            host = st.selectbox(
                "Host",
                HOST_CHOICES,
                index=0,
                key="replay_web_host"
            )
        
        with col2:
            default_port = random_free_port()
            port = st.number_input(
                "Port", 
                min_value=1024, 
                max_value=65535, 
                value=default_port,
                key="replay_web_port"
            )
        
        # Show game details
        if selected_game in games:
            game_data = games[selected_game]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", game_data.get('score', 0))
            with col2:
                st.metric("Steps", game_data.get('steps', 0))
            with col3:
                st.metric("Algorithm", game_data.get('algorithm', 'Unknown'))
            
            # Launch web replay options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🌐 Task-0 Web Replay", type="primary"):
                    self._launch_web_replay(selected_folder, selected_game, host, port)
            
            with col2:
                if st.button("🧠 Heuristic Web Replay", type="secondary"):
                    self._launch_heuristic_web_replay(selected_folder, selected_game, host, port)
            
            st.info("**Task-0 Web Replay**: Uses ROOT replay infrastructure (universal)\n\n**Heuristic Web Replay**: Uses heuristic-specific interface with algorithm details")
    
    def _render_analysis_tab(self, log_folders: List[str]) -> None:
        """Render performance analysis interface."""
        st.markdown("### 📊 Performance Analysis")
        
        if not log_folders:
            st.warning("No heuristic game logs found. Launch some games first!")
            return
        
        # Multi-select for comparison
        selected_folders = st.multiselect(
            "Select Experiments to Compare",
            log_folders,
            format_func=lambda x: Path(x).name,
            default=log_folders[:3] if len(log_folders) >= 3 else log_folders
        )
        
        if not selected_folders:
            st.info("Select at least one experiment to analyze.")
            return
        
        # Load and analyze data
        analysis_data = []
        for folder in selected_folders:
            summary_file = Path(folder) / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                        analysis_data.append({
                            'folder': Path(folder).name,
                            'algorithm': summary.get('algorithm', 'Unknown'),
                            'total_games': summary.get('total_games', 0),
                            'total_score': summary.get('total_score', 0),
                            'average_score': summary.get('average_score', 0),
                            'score_per_step': summary.get('score_per_step', 0),
                            'score_per_round': summary.get('score_per_round', 0),
                            'scores': summary.get('scores', [])
                        })
                except Exception as e:
                    st.error(f"Error loading {folder}: {e}")
        
        if analysis_data:
            # Display comparison table
            st.markdown("#### Algorithm Comparison")
            
            import pandas as pd
            df = pd.DataFrame(analysis_data)
            st.dataframe(
                df[['algorithm', 'total_games', 'average_score', 'score_per_step', 'score_per_round']],
                use_container_width=True
            )
            
            # Performance charts
            st.markdown("#### Performance Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Average Score by Algorithm**")
                chart_data = df.set_index('algorithm')['average_score']  # type: ignore
                st.bar_chart(chart_data)
            
            with col2:
                st.markdown("**Efficiency (Score per Step)**")
                chart_data = df.set_index('algorithm')['score_per_step']  # type: ignore
                st.bar_chart(chart_data)
    
    def _load_games_from_folder(self, folder_path: str) -> Dict[int, Dict]:
        """
        Load all games from a heuristic log folder.
        
        Reuses Task-0 file manager patterns for consistent data loading.
        """
        games = {}
        folder = Path(folder_path)
        
        for game_file in folder.glob("game_*.json"):
            try:
                game_number = int(game_file.stem.split('_')[1])
                with open(game_file, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
                    games[game_number] = game_data
            except (ValueError, json.JSONDecodeError) as e:
                st.error(f"Error loading {game_file}: {e}")
        
        return games
    
    def _launch_pygame_replay(self, folder_path: str, game_number: int) -> None:
        """Launch PyGame replay using Task-0 replay infrastructure."""
        try:
            # Use Task-0 replay script with heuristic log folder
            cmd = [
                sys.executable, 
                str(Path(__file__).parent / "scripts" / "replay.py"),
                "--log-dir", folder_path,
                "--game", str(game_number)  # Use --game like Task-0
            ]
            
            st.code(" ".join(cmd), language="bash")
            
            # Launch in background following Task-0 session_utils pattern
            subprocess.Popen(cmd)
            st.success(f"🎮 PyGame replay launched for Game {game_number}")
            st.info("Check your desktop for the PyGame window. Close the replay window when finished.")
            
        except Exception as e:
            st.error(f"❌ Failed to launch PyGame replay: {e}")
    
    def _launch_heuristic_pygame_replay(self, folder_path: str, game_number: int) -> None:
        """Launch heuristic-specific PyGame replay with algorithm metrics."""
        try:
            # Use heuristic-specific replay script
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "scripts" / "replay.py"),
                "--log-dir", folder_path,
                "--game", str(game_number),
                "--verbose"  # Enable heuristic-specific verbose output
            ]
            
            st.code(" ".join(cmd), language="bash")
            
            # Launch in background following Task-0 session_utils pattern
            subprocess.Popen(cmd)
            st.success(f"🧠 Heuristic PyGame replay launched for Game {game_number}")
            st.info("Check your desktop for the PyGame window with algorithm metrics. Close the replay window when finished.")
            
        except Exception as e:
            st.error(f"❌ Failed to launch heuristic PyGame replay: {e}")
    
    def _launch_web_replay(self, folder_path: str, game_number: int, host: str, port: int) -> None:
        """Launch web replay using Task-0 web replay infrastructure."""
        try:
            # Ensure port is available
            port = ensure_free_port(port)
            
            # Use Task-0 web replay script with heuristic log folder
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "scripts" / "replay_web.py"),
                "--log-dir", folder_path,
                "--game", str(game_number),  # Use --game like Task-0
                "--host", host,
                "--port", str(port)
            ]
            
            st.code(" ".join(cmd), language="bash")
            
            # Launch in background following Task-0 session_utils pattern
            subprocess.Popen(cmd)
            
            # Show access information
            st.success(f"🌐 Web replay launched for Game {game_number}")
            st.info(f"Access at: http://{host}:{port}")
            st.markdown(f"[Open Web Replay](http://{host}:{port})")
            
        except Exception as e:
            st.error(f"❌ Failed to launch web replay: {e}")
    
    def _launch_heuristic_web_replay(self, folder_path: str, game_number: int, host: str, port: int) -> None:
        """Launch heuristic-specific web replay using scripts/replay_web.py."""
        try:
            # Ensure port is available
            port = ensure_free_port(port)
            
            # Use heuristic-specific web replay script
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "scripts" / "replay_web.py"),
                "--log-dir", folder_path,
                "--game", str(game_number),
                "--host", host,
                "--port", str(port),
                "--show-metrics",  # Enable heuristic-specific metrics
                "--show-path-info"  # Enable pathfinding information
            ]
            
            st.code(" ".join(cmd), language="bash")
            
            # Launch in background
            subprocess.Popen(cmd)
            
            # Show access information
            st.success(f"🧠 Heuristic Web Replay launched for Game {game_number}")
            st.info(f"Access at: http://{host}:{port}")
            st.markdown(f"[Open Heuristic Replay](http://{host}:{port})")
            
        except Exception as e:
            st.error(f"❌ Failed to launch heuristic web replay: {e}")


if __name__ == "__main__":
    HeuristicsApp()