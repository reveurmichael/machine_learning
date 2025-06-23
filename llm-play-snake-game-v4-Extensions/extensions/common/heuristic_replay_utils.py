"""heuristic_replay_utils.py - Common replay utilities for heuristic extensions

Contains non-essential replay helper functions shared across heuristic extensions
(v0.03, v0.04) to eliminate duplication in replay engines and GUI components.

The goal is to centralize replay infrastructure while keeping algorithm-specific
visualization and analysis in each extension.

Design Philosophy:
- Extract common replay patterns from v0.03 and v0.04
- Standardize algorithm display names and descriptions
- Provide shared performance metrics calculation
- Centralize replay state management
- Keep algorithm-specific insights in extensions
"""

from __future__ import annotations

import json
from typing import Dict, Any, Optional, List
from pathlib import Path

__all__ = [
    "ALGORITHM_DISPLAY_NAMES",
    "ALGORITHM_DESCRIPTIONS", 
    "get_algorithm_display_name",
    "get_algorithm_description",
    "extract_heuristic_replay_data",
    "calculate_heuristic_performance_metrics",
    "format_algorithm_insights",
    "build_heuristic_state_dict",
]

# ---------------------
# Algorithm metadata
# ---------------------

ALGORITHM_DISPLAY_NAMES = {
    "bfs": "Breadth-First Search (BFS)",
    "bfs-safe-greedy": "BFS + Safety Heuristics", 
    "bfs-hamiltonian": "BFS + Hamiltonian Concepts",
    "dfs": "Depth-First Search (DFS)",
    "astar": "A* Pathfinding",
    "astar-hamiltonian": "A* + Hamiltonian Optimization",
    "hamiltonian": "Hamiltonian Path Algorithm"
}

ALGORITHM_DESCRIPTIONS = {
    "bfs": "Guarantees shortest path to apple using breadth-first exploration",
    "bfs-safe-greedy": "Enhanced BFS with safety checks and greedy optimization", 
    "bfs-hamiltonian": "BFS enhanced with Hamiltonian path concepts for better space utilization",
    "dfs": "Educational depth-first exploration algorithm",
    "astar": "Optimal pathfinding using heuristic distance estimation",
    "astar-hamiltonian": "A* pathfinding enhanced with Hamiltonian optimization",
    "hamiltonian": "Space-filling algorithm that covers the entire board systematically"
}

def get_algorithm_display_name(algorithm: str) -> str:
    """Get user-friendly display name for algorithm."""
    return ALGORITHM_DISPLAY_NAMES.get(algorithm.lower(), algorithm.title())

def get_algorithm_description(algorithm: str) -> str:
    """Get description of algorithm behavior."""
    return ALGORITHM_DESCRIPTIONS.get(algorithm.lower(), "Heuristic pathfinding algorithm")


# ---------------------
# Replay data extraction
# ---------------------

def extract_heuristic_replay_data(game_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract heuristic-specific data from game JSON.
    
    Standardizes the data extraction pattern used in replay engines
    across v0.03 and v0.04.
    
    Args:
        game_data: Loaded game JSON data
        
    Returns:
        Dictionary with extracted heuristic replay data
    """
    algorithm_name = game_data.get('algorithm', 'Unknown')
    
    # Extract basic performance metrics
    performance_metrics = {
        'score': game_data.get('score', 0),
        'steps': game_data.get('steps', 0),
        'round_count': game_data.get('round_count', 0),
        'snake_length': len(game_data.get('snake_positions', [])),
        'duration_seconds': game_data.get('duration_seconds', 0.0)
    }
    
    # Calculate derived metrics
    performance_metrics.update({
        'score_per_step': performance_metrics['score'] / max(performance_metrics['steps'], 1),
        'score_per_round': performance_metrics['score'] / max(performance_metrics['round_count'], 1)
    })
    
    # Extract detailed history information
    detailed_history = game_data.get('detailed_history', {})
    pathfinding_info = {
        'total_moves': len(detailed_history.get('moves', [])),
        'apple_positions': detailed_history.get('apple_positions', []),
        'rounds_data': detailed_history.get('rounds_data', {})
    }
    
    # Extract heuristic-specific statistics if available  
    heuristic_stats = game_data.get('heuristic_stats', {})
    
    return {
        'algorithm_name': algorithm_name,
        'algorithm_display_name': get_algorithm_display_name(algorithm_name),
        'algorithm_description': get_algorithm_description(algorithm_name),
        'performance_metrics': performance_metrics,
        'pathfinding_info': pathfinding_info,
        'heuristic_stats': heuristic_stats,
        'game_end_reason': game_data.get('game_end_reason', 'UNKNOWN')
    }


def calculate_heuristic_performance_metrics(replay_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate performance metrics for heuristic replay display.
    
    Args:
        replay_data: Extracted replay data from extract_heuristic_replay_data
        
    Returns:
        Dictionary with calculated performance metrics
    """
    perf = replay_data['performance_metrics']
    
    # Basic efficiency metrics
    efficiency_metrics = {
        'moves_per_apple': perf['steps'] / max(perf['score'], 1),
        'rounds_per_apple': perf['round_count'] / max(perf['score'], 1),
        'time_per_move': perf['duration_seconds'] / max(perf['steps'], 1),
        'time_per_apple': perf['duration_seconds'] / max(perf['score'], 1)
    }
    
    # Algorithm performance rating
    score_per_step = perf['score_per_step']
    if score_per_step > 0.1:
        performance_rating = "Excellent"
    elif score_per_step > 0.05:
        performance_rating = "Good" 
    elif score_per_step > 0.02:
        performance_rating = "Fair"
    else:
        performance_rating = "Poor"
    
    return {
        **efficiency_metrics,
        'performance_rating': performance_rating,
        'total_efficiency': score_per_step * 100  # Percentage
    }


# ---------------------
# Algorithm insights
# ---------------------

def format_algorithm_insights(algorithm: str) -> List[str]:
    """Get algorithm-specific insights for display.
    
    These insights help users understand the characteristics and trade-offs
    of different heuristic algorithms.
    
    Args:
        algorithm: Algorithm name
        
    Returns:
        List of insight strings
    """
    insights_map = {
        "bfs": [
            "🎯 Optimal pathfinding algorithm - shortest paths guaranteed",
            "⚡ Fast execution with predictable performance",
            "🧠 Educational value: demonstrates breadth-first exploration"
        ],
        "astar": [
            "🎯 Optimal pathfinding algorithm - shortest paths guaranteed", 
            "🔍 Uses heuristic distance estimation for efficiency",
            "⚖️ Balances optimality with computational efficiency"
        ],
        "bfs-safe-greedy": [
            "🛡️ Safety-enhanced BFS - balanced performance and safety",
            "🎯 Combines shortest-path guarantees with safety heuristics",
            "💡 Good for learning about safety-first game strategies"
        ],
        "bfs-hamiltonian": [
            "🔀 Hybrid algorithm - combines pathfinding with space-filling",
            "🎯 BFS pathfinding enhanced with Hamiltonian concepts",
            "🌀 Better space utilization than pure BFS"
        ],
        "astar-hamiltonian": [
            "🔀 Hybrid algorithm - combines A* with space-filling",
            "⚡ Optimal pathfinding with enhanced space management", 
            "🎯 Best of both worlds: efficiency and coverage"
        ],
        "hamiltonian": [
            "🔄 Space-filling algorithm - covers entire board systematically",
            "♾️ Theoretical guarantee of infinite survival",
            "📚 Educational: demonstrates Hamiltonian cycle concepts"
        ],
        "dfs": [
            "🔍 Depth-first exploration - educational algorithm",
            "📚 Demonstrates recursive pathfinding concepts",
            "⚠️ Not optimal - may find longer paths"
        ]
    }
    
    return insights_map.get(algorithm.lower(), [
        "🤖 Heuristic pathfinding algorithm",
        "🎯 Focuses on reaching the apple efficiently"
    ])


# ---------------------
# State building utilities
# ---------------------

def build_heuristic_state_dict(
    snake_positions: List[List[int]],
    apple_position: List[int], 
    performance_metrics: Dict[str, Any],
    algorithm_info: Dict[str, str],
    replay_progress: Optional[Dict[str, Any]] = None,
    pathfinding_info: Optional[Dict[str, Any]] = None,
    grid_size: int = 10
) -> Dict[str, Any]:
    """Build standardized state dictionary for heuristic replay/web interfaces.
    
    This centralizes the state building pattern used across replay engines
    and web interfaces, ensuring consistency in data format.
    
    Args:
        snake_positions: Current snake body positions
        apple_position: Current apple position
        performance_metrics: Performance metrics dictionary
        algorithm_info: Algorithm name and display info
        replay_progress: Optional replay progress information
        pathfinding_info: Optional pathfinding statistics
        grid_size: Board grid size
        
    Returns:
        Standardized state dictionary
    """
    # Base state following Task-0 patterns
    base_state = {
        'snake_positions': snake_positions,
        'apple_position': apple_position,
        'score': performance_metrics.get('score', 0),
        'steps': performance_metrics.get('steps', 0),
        'grid_size': grid_size,
        'game_active': True  # Will be overridden by caller if needed
    }
    
    # Add heuristic-specific extensions
    heuristic_extensions = {
        'algorithm_name': algorithm_info.get('algorithm_name', 'Unknown'),
        'algorithm_display_name': algorithm_info.get('algorithm_display_name', 'Unknown'),
        'performance_metrics': performance_metrics
    }
    
    # Add optional components
    if replay_progress:
        heuristic_extensions['replay_progress'] = replay_progress
        
    if pathfinding_info:
        heuristic_extensions['pathfinding_info'] = pathfinding_info
    
    # Merge all components
    base_state.update(heuristic_extensions)
    
    return base_state


# ---------------------
# Replay navigation utilities
# ---------------------

def validate_replay_navigation(
    current_move: int,
    total_moves: int, 
    navigation_command: str,
    jump_position: Optional[int] = None
) -> tuple[bool, int, str]:
    """Validate and calculate new position for replay navigation.
    
    Args:
        current_move: Current move index
        total_moves: Total number of moves
        navigation_command: Navigation command ('next', 'prev', 'jump', etc.)
        jump_position: Target position for jump commands
        
    Returns:
        Tuple of (is_valid, new_position, error_message)
    """
    if navigation_command == "next":
        if current_move < total_moves - 1:
            return True, current_move + 1, ""
        else:
            return False, current_move, "Already at last move"
            
    elif navigation_command == "prev":
        if current_move > 0:
            return True, current_move - 1, ""
        else:
            return False, current_move, "Already at first move"
            
    elif navigation_command == "jump":
        if jump_position is None:
            return False, current_move, "Jump position not specified"
        if 0 <= jump_position < total_moves:
            return True, jump_position, ""
        else:
            return False, current_move, f"Invalid jump position: {jump_position}"
            
    elif navigation_command == "first":
        return True, 0, ""
        
    elif navigation_command == "last":
        return True, max(0, total_moves - 1), ""
        
    else:
        return False, current_move, f"Unknown navigation command: {navigation_command}" 