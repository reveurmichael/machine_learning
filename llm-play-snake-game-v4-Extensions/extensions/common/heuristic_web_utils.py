"""heuristic_web_utils.py - Common web utilities for heuristic extensions

Contains non-essential web helper functions shared across heuristic extensions
(v0.03, v0.04) for Streamlit apps and Flask web interfaces.

The goal is to centralize web infrastructure while keeping algorithm-specific
UI components and interactions in each extension.

Design Philosophy:
- Extract common web patterns from v0.03 and v0.04 
- Standardize Streamlit component creation
- Provide shared Flask route patterns
- Centralize web state management
- Keep algorithm-specific UI logic in extensions
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Callable
import json

__all__ = [
    "create_algorithm_selector",
    "create_parameter_inputs",
    "create_performance_display",
    "format_web_state_response",
    "build_streamlit_tabs",
    "create_replay_controls",
    "format_algorithm_metrics",
]

# ---------------------
# Streamlit component creators
# ---------------------

def create_algorithm_selector(
    available_algorithms: List[str],
    default_algorithm: str = "BFS",
    key_prefix: str = "algo"
) -> str:
    """Create standardized algorithm selector for Streamlit.
    
    Args:
        available_algorithms: List of available algorithm names
        default_algorithm: Default selection
        key_prefix: Unique prefix for widget key
        
    Returns:
        Selected algorithm name
    """
    try:
        import streamlit as st
        
        # Create user-friendly display names
        display_names = {}
        for algo in available_algorithms:
            display_names[algo] = algo.replace('-', ' ').replace('_', ' ').title()
        
        # Create selectbox with display names
        display_options = list(display_names.values())
        default_index = 0
        
        # Find default index
        for i, algo in enumerate(available_algorithms):
            if algo.upper() == default_algorithm.upper():
                default_index = i
                break
        
        selected_display = st.selectbox(
            "Algorithm",
            display_options,
            index=default_index,
            key=f"{key_prefix}_selector"
        )
        
        # Convert back to algorithm name
        for algo, display in display_names.items():
            if display == selected_display:
                return algo
                
        return default_algorithm
        
    except ImportError:
        # Fallback if streamlit not available
        return default_algorithm


def create_parameter_inputs(key_prefix: str = "params") -> Dict[str, Any]:
    """Create standardized parameter input widgets for Streamlit.
    
    Args:
        key_prefix: Unique prefix for widget keys
        
    Returns:
        Dictionary of parameter values
    """
    try:
        import streamlit as st
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_games = st.slider(
                "Max Games", 
                min_value=1, 
                max_value=100, 
                value=10,
                key=f"{key_prefix}_max_games"
            )
            
            grid_size = st.selectbox(
                "Grid Size",
                [8, 10, 12, 16, 20],
                index=1,  # Default to 10
                key=f"{key_prefix}_grid_size"
            )
        
        with col2:
            max_steps = st.slider(
                "Max Steps per Game",
                min_value=100,
                max_value=5000,
                value=1000,
                key=f"{key_prefix}_max_steps"
            )
            
            verbose = st.checkbox(
                "Verbose Output",
                value=False,
                key=f"{key_prefix}_verbose"
            )
        
        return {
            "max_games": max_games,
            "grid_size": grid_size,
            "max_steps": max_steps,
            "verbose": verbose
        }
        
    except ImportError:
        # Fallback if streamlit not available
        return {
            "max_games": 10,
            "grid_size": 10,
            "max_steps": 1000,
            "verbose": False
        }


def create_performance_display(
    performance_data: Dict[str, Any],
    algorithm_name: str = "Unknown"
) -> None:
    """Create standardized performance metrics display for Streamlit.
    
    Args:
        performance_data: Performance metrics dictionary
        algorithm_name: Name of algorithm for display
    """
    try:
        import streamlit as st
        
        st.subheader(f"📊 {algorithm_name} Performance")
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Games Played",
                performance_data.get("total_games", 0)
            )
        
        with col2:
            total_score = performance_data.get("total_score", 0)
            avg_score = performance_data.get("average_score", 0)
            st.metric(
                "Total Score",
                total_score,
                delta=f"Avg: {avg_score:.1f}"
            )
        
        with col3:
            avg_steps = performance_data.get("average_steps", 0)
            st.metric(
                "Avg Steps",
                f"{avg_steps:.0f}"
            )
        
        with col4:
            total_duration = performance_data.get("total_duration", 0)
            st.metric(
                "Total Time",
                f"{total_duration:.1f}s"
            )
        
        # Performance charts if data available
        scores = performance_data.get("scores", [])
        if scores:
            st.subheader("📈 Score History")
            st.line_chart(scores)
            
    except ImportError:
        # Fallback if streamlit not available
        pass


def build_streamlit_tabs(
    algorithms: List[str], 
    tab_content_func: Callable[[str], None]
) -> None:
    """Build standardized algorithm tabs for Streamlit.
    
    Args:
        algorithms: List of algorithm names
        tab_content_func: Function to render content for each algorithm
    """
    try:
        import streamlit as st
        
        # Create user-friendly tab names
        tab_names = []
        for algo in algorithms:
            tab_name = algo.replace('-', ' ').replace('_', ' ').title()
            tab_names.append(tab_name)
        
        # Create tabs
        tabs = st.tabs(tab_names)
        
        # Render content for each tab
        for i, (tab, algorithm) in enumerate(zip(tabs, algorithms)):
            with tab:
                tab_content_func(algorithm)
                
    except ImportError:
        # Fallback if streamlit not available
        pass


def create_replay_controls(key_prefix: str = "replay") -> Dict[str, Any]:
    """Create standardized replay control widgets for Streamlit.
    
    Args:
        key_prefix: Unique prefix for widget keys
        
    Returns:
        Dictionary of control states
    """
    try:
        import streamlit as st
        
        st.subheader("🎮 Replay Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            play_pygame = st.button(
                "🎮 PyGame Replay",
                key=f"{key_prefix}_pygame",
                help="Launch desktop replay with PyGame"
            )
        
        with col2:
            play_web = st.button(
                "🌐 Web Replay",
                key=f"{key_prefix}_web",
                help="Launch web-based replay interface"
            )
        
        with col3:
            show_stats = st.checkbox(
                "📊 Show Statistics",
                value=True,
                key=f"{key_prefix}_stats"
            )
        
        # Replay options
        with st.expander("⚙️ Replay Options"):
            replay_speed = st.slider(
                "Replay Speed",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key=f"{key_prefix}_speed"
            )
            
            auto_advance = st.checkbox(
                "Auto Advance",
                value=True,
                key=f"{key_prefix}_auto"
            )
        
        return {
            "play_pygame": play_pygame,
            "play_web": play_web,
            "show_stats": show_stats,
            "replay_speed": replay_speed,
            "auto_advance": auto_advance
        }
        
    except ImportError:
        # Fallback if streamlit not available
        return {
            "play_pygame": False,
            "play_web": False,
            "show_stats": True,
            "replay_speed": 1.0,
            "auto_advance": True
        }


# ---------------------
# Flask/Web response formatting
# ---------------------

def format_web_state_response(
    game_state: Dict[str, Any],
    algorithm_info: Dict[str, str],
    performance_metrics: Optional[Dict[str, Any]] = None,
    extra_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Format standardized web state response for Flask APIs.
    
    Args:
        game_state: Basic game state data
        algorithm_info: Algorithm name and display info
        performance_metrics: Optional performance data
        extra_data: Optional additional data
        
    Returns:
        Standardized web response dictionary
    """
    response = {
        "game_state": game_state,
        "algorithm_info": algorithm_info,
        "timestamp": str(round(time.time() * 1000))  # JavaScript timestamp
    }
    
    if performance_metrics:
        response["performance_metrics"] = performance_metrics
    
    if extra_data:
        response.update(extra_data)
    
    return response


def format_algorithm_metrics(
    algorithm_name: str,
    performance_data: Dict[str, Any],
    include_efficiency: bool = True
) -> Dict[str, Any]:
    """Format algorithm performance metrics for web display.
    
    Args:
        algorithm_name: Name of algorithm
        performance_data: Raw performance data
        include_efficiency: Whether to include efficiency calculations
        
    Returns:
        Formatted metrics dictionary
    """
    formatted = {
        "algorithm": algorithm_name,
        "basic_stats": {
            "games_played": performance_data.get("total_games", 0),
            "total_score": performance_data.get("total_score", 0),
            "average_score": round(performance_data.get("average_score", 0), 2),
            "max_score": performance_data.get("max_score", 0),
            "min_score": performance_data.get("min_score", 0)
        }
    }
    
    if include_efficiency and "average_steps" in performance_data:
        formatted["efficiency_stats"] = {
            "average_steps": round(performance_data.get("average_steps", 0), 1),
            "average_rounds": round(performance_data.get("average_rounds", 0), 1),
            "average_duration": round(performance_data.get("average_duration", 0), 2),
            "score_per_step": round(
                performance_data.get("total_score", 0) / 
                max(sum(performance_data.get("steps", [1])), 1), 4
            )
        }
    
    return formatted


# ---------------------
# Algorithm comparison utilities
# ---------------------

def create_algorithm_comparison_table(
    algorithm_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Create comparison table data for multiple algorithms.
    
    Args:
        algorithm_results: Dictionary mapping algorithm names to their results
        
    Returns:
        Comparison table data structure
    """
    try:
        import streamlit as st
        import pandas as pd
        
        # Prepare data for comparison
        comparison_data = []
        for algo_name, results in algorithm_results.items():
            perf = results.get("performance_summary", {})
            row = {
                "Algorithm": algo_name.replace('-', ' ').title(),
                "Avg Score": round(perf.get("average_score", 0), 2),
                "Max Score": perf.get("max_score", 0),
                "Avg Steps": round(perf.get("average_steps", 0), 1),
                "Success Rate": f"{(perf.get('total_games', 0) / max(perf.get('total_games', 1), 1)) * 100:.1f}%",
                "Efficiency": round(perf.get("average_score", 0) / max(perf.get("average_steps", 1), 1) * 100, 2)
            }
            comparison_data.append(row)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.subheader("🏆 Algorithm Comparison")
            st.dataframe(df, use_container_width=True)
            
            # Highlight best performer in each category
            st.caption("📊 Higher efficiency = better score per step ratio")
        
        return {"comparison_data": comparison_data}
        
    except ImportError:
        # Fallback if streamlit/pandas not available
        return {"comparison_data": list(algorithm_results.keys())}


# ---------------------
# Progress tracking utilities  
# ---------------------

import time

def create_progress_tracker(total_items: int, description: str = "Processing"):
    """Create progress tracker for long-running operations.
    
    Args:
        total_items: Total number of items to process
        description: Description of the operation
        
    Returns:
        Progress tracker object
    """
    try:
        import streamlit as st
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        class ProgressTracker:
            def __init__(self):
                self.current = 0
                self.total = total_items
                self.start_time = time.time()
            
            def update(self, increment: int = 1, status: str = ""):
                self.current = min(self.current + increment, self.total)
                progress = self.current / self.total
                progress_bar.progress(progress)
                
                elapsed = time.time() - self.start_time
                if self.current > 0:
                    eta = elapsed * (self.total - self.current) / self.current
                    status_text.text(f"{description}: {self.current}/{self.total} - ETA: {eta:.1f}s {status}")
                else:
                    status_text.text(f"{description}: {self.current}/{self.total} {status}")
            
            def complete(self):
                progress_bar.progress(1.0)
                elapsed = time.time() - self.start_time
                status_text.text(f"{description}: Completed in {elapsed:.1f}s")
        
        return ProgressTracker()
        
    except ImportError:
        # Fallback progress tracker
        class SimpleProgressTracker:
            def __init__(self):
                self.current = 0
                self.total = total_items
            
            def update(self, increment: int = 1, status: str = ""):
                self.current += increment
                print(f"{description}: {self.current}/{self.total} {status}")
            
            def complete(self):
                print(f"{description}: Completed")
        
        return SimpleProgressTracker() 