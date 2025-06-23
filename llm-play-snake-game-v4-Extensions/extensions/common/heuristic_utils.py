"""heuristic_utils.py - Common utilities for heuristic Snake extensions

Contains non-essential helper functions shared across all heuristic extensions
(v0.01, v0.02, v0.03, v0.04) to eliminate code duplication while keeping the
core heuristic algorithms visible in each extension.

The goal is to centralize infrastructure code (logging, performance tracking,
console output) so that each extension's code stays focused on its conceptual
heuristic ideas.

Design Philosophy:
- Extract common session management patterns
- Standardize logging and directory structure
- Provide shared performance tracking utilities
- Centralize console output formatting
- Keep algorithm-specific logic in extensions
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from colorama import Fore
import json

__all__ = [
    "HeuristicSessionConfig",
    "HeuristicLogger", 
    "HeuristicPerformanceTracker",
    "setup_heuristic_logging",
    "format_heuristic_console_output",
    "save_heuristic_session_summary",
    "validate_algorithm_name",
]


# ---------------------
# Configuration and constants
# ---------------------

class HeuristicSessionConfig:
    """Configuration container for heuristic sessions."""
    
    def __init__(self, args):
        self.algorithm_name: str = getattr(args, "algorithm", "BFS")
        self.verbose: bool = getattr(args, "verbose", False)
        self.max_games: int = getattr(args, "max_games", 10)
        self.max_steps: int = getattr(args, "max_steps", 1000)
        self.session_start_time: datetime = datetime.now()
        
    def get_log_directory_name(self) -> str:
        """Generate standardized log directory name."""
        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        algo_name = self.algorithm_name.lower().replace('-', '_')
        return f"heuristics-{algo_name}_{timestamp}"


# ---------------------
# Logging utilities
# ---------------------

def setup_heuristic_logging(config: HeuristicSessionConfig) -> str:
    """Setup standardized logging directory for heuristic extensions.
    
    CRITICAL: All heuristic extensions write their outputs under:
        ROOT/logs/extensions/<experiment_folder>/
    
    This keeps extension logs separate from Task-0 logs while maintaining
    a single top-level logs/ folder for backup and analytics.
    
    Args:
        config: Session configuration with algorithm and timing info
        
    Returns:
        Path to created log directory
    """
    # Use common constants from extensions.common
    from extensions.common.config import EXTENSIONS_LOGS_DIR
    
    experiment_folder = config.get_log_directory_name()
    log_dir = os.path.join(EXTENSIONS_LOGS_DIR, experiment_folder)
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir


class HeuristicLogger:
    """Standardized logging for heuristic extensions."""
    
    def __init__(self, config: HeuristicSessionConfig):
        self.config = config
        self.verbose = config.verbose
        
    def log_session_start(self):
        """Log session startup information."""
        print(Fore.GREEN + "🚀 Starting heuristics session...")
        print(Fore.CYAN + f"📊 Target games: {self.config.max_games}")
        print(Fore.CYAN + f"🧠 Algorithm: {self.config.algorithm_name}")
        
    def log_session_complete(self, game_count: int, total_score: int):
        """Log session completion statistics."""
        print(Fore.GREEN + "✅ Heuristics session completed!")
        print(Fore.CYAN + f"📊 Games played: {game_count}")
        print(Fore.CYAN + f"🏆 Total score: {total_score}")
        if game_count > 0:
            avg_score = total_score / game_count
            print(Fore.CYAN + f"📈 Average score: {avg_score:.1f}")
    
    def log_game_start(self, game_count: int):
        """Log game start."""
        if self.verbose:
            print(Fore.BLUE + f"\n🎮 Starting Game {game_count} with {self.config.algorithm_name}")
        else:
            print(Fore.BLUE + f"\n🎮 Game {game_count}")
    
    def log_apple_eaten(self, score: int):
        """Log apple eaten event."""
        if self.verbose:
            print(Fore.GREEN + f"🍎 Apple eaten! Score: {score}")
        else:
            print(Fore.GREEN + f"🍎 Score: {score}")
    
    def log_pathfinding_failure(self, consecutive_failures: int, algorithm: str):
        """Log pathfinding failure."""
        if self.verbose:
            print(Fore.YELLOW + f"⚠️  No path found (attempt {consecutive_failures})")
        
        if consecutive_failures >= 5:
            print(Fore.RED + f"❌ Too many consecutive pathfinding failures with {algorithm}")
    
    def log_agent_created(self, agent_class_name: str):
        """Log agent creation in verbose mode."""
        if self.verbose:
            print(Fore.CYAN + f"🏭 Created {agent_class_name} for {self.config.algorithm_name}")
    
    def log_initialization_complete(self, agent_info: str, log_dir: str):
        """Log initialization completion."""
        print(Fore.GREEN + f"🤖 Heuristics initialized with {self.config.algorithm_name} algorithm")
        if self.verbose and agent_info:
            print(Fore.CYAN + f"🔍 Agent: {agent_info}")
        print(Fore.CYAN + f"📂 Logs: {log_dir}")


# ---------------------
# Performance tracking
# ---------------------

class HeuristicPerformanceTracker:
    """Track performance metrics across heuristic sessions."""
    
    def __init__(self):
        self.game_scores: List[int] = []
        self.game_steps: List[int] = []
        self.game_rounds: List[int] = []
        self.game_durations: List[float] = []
        
    def record_game(self, score: int, steps: int, rounds: int, duration: float):
        """Record metrics for a completed game."""
        self.game_scores.append(score)
        self.game_steps.append(steps) 
        self.game_rounds.append(rounds)
        self.game_durations.append(duration)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all recorded games."""
        if not self.game_scores:
            return {}
            
        return {
            "total_games": len(self.game_scores),
            "total_score": sum(self.game_scores),
            "average_score": sum(self.game_scores) / len(self.game_scores),
            "max_score": max(self.game_scores),
            "min_score": min(self.game_scores),
            "average_steps": sum(self.game_steps) / len(self.game_steps),
            "average_rounds": sum(self.game_rounds) / len(self.game_rounds),
            "average_duration": sum(self.game_durations) / len(self.game_durations),
            "total_duration": sum(self.game_durations),
            "scores": self.game_scores,
            "steps": self.game_steps,
            "rounds": self.game_rounds,
            "durations": self.game_durations
        }


# ---------------------
# Console output formatting
# ---------------------

def format_heuristic_console_output(algorithm: str, verbose: bool = False) -> Dict[str, str]:
    """Get standardized console output strings for heuristic algorithms.
    
    Args:
        algorithm: Algorithm name
        verbose: Whether to include verbose messages
        
    Returns:
        Dictionary of formatted output strings
    """
    return {
        "session_start": f"🚀 Starting heuristics session with {algorithm}...",
        "algorithm_info": f"🧠 Algorithm: {algorithm}",
        "pathfinding_round": f"{algorithm} pathfinding",
        "agent_factory": f"🏭 Creating agent for {algorithm}",
        "session_complete": f"✅ Heuristics session with {algorithm} completed!",
        "pathfinding_failure": f"❌ Too many consecutive pathfinding failures with {algorithm}"
    }


# ---------------------
# Session summary utilities
# ---------------------

def save_heuristic_session_summary(
    log_dir: str,
    config: HeuristicSessionConfig,
    performance: HeuristicPerformanceTracker,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save standardized session summary for heuristic extensions.
    
    Args:
        log_dir: Directory to save summary
        config: Session configuration
        performance: Performance tracking data
        extra_metadata: Additional metadata to include
    """
    session_end_time = datetime.now()
    stats = performance.get_summary_stats()
    
    summary = {
        "session_info": {
            "algorithm": config.algorithm_name,
            "start_time": config.session_start_time.isoformat(),
            "end_time": session_end_time.isoformat(),
            "duration_seconds": (session_end_time - config.session_start_time).total_seconds(),
            "target_games": config.max_games,
            "max_steps_per_game": config.max_steps
        },
        "performance_summary": stats,
        "extension_type": "heuristics",
        "timestamp": session_end_time.isoformat()
    }
    
    # Add extra metadata if provided
    if extra_metadata:
        summary["extra_metadata"] = extra_metadata
    
    # Save to file
    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ---------------------
# Validation utilities
# ---------------------

def validate_algorithm_name(algorithm: str, available_algorithms: List[str]) -> str:
    """Validate and normalize algorithm name.
    
    Args:
        algorithm: Algorithm name to validate
        available_algorithms: List of available algorithm names
        
    Returns:
        Normalized algorithm name
        
    Raises:
        ValueError: If algorithm is not available
    """
    algorithm_upper = algorithm.upper()
    available_upper = [alg.upper() for alg in available_algorithms]
    
    if algorithm_upper not in available_upper:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available_algorithms}")
    
    # Return the properly cased version from available list
    index = available_upper.index(algorithm_upper)
    return available_algorithms[index]


# ---------------------
# Game execution helpers  
# ---------------------

def execute_heuristic_game_loop(
    game_logic,
    agent,
    config: HeuristicSessionConfig,
    logger: HeuristicLogger,
    max_consecutive_failures: int = 5
) -> Dict[str, Any]:
    """Execute a single heuristic game with standardized error handling.
    
    This centralizes the common game execution pattern used across all
    heuristic extensions while keeping algorithm-specific logic in the agents.
    
    Args:
        game_logic: Game logic instance
        agent: Heuristic agent instance
        config: Session configuration
        logger: Logger instance
        max_consecutive_failures: Max consecutive pathfinding failures allowed
        
    Returns:
        Dictionary with game results
    """
    consecutive_no_path_found = 0
    game_active = True
    
    # Game execution timing
    start_time = time.time()
    
    while game_active and game_logic.game_state.steps < config.max_steps:
        # Get next move from agent
        planned_move = game_logic.get_next_planned_move()
        
        # Handle pathfinding failures
        if planned_move == "NO_PATH_FOUND":
            consecutive_no_path_found += 1
            logger.log_pathfinding_failure(consecutive_no_path_found, config.algorithm_name)
            
            if consecutive_no_path_found >= max_consecutive_failures:
                game_active = False
                break
        else:
            consecutive_no_path_found = 0
            
            # Execute move
            game_continues, apple_eaten = game_logic.make_move(planned_move)
            
            # Log apple eaten
            if apple_eaten:
                logger.log_apple_eaten(game_logic.game_state.score)
            
            if not game_continues:
                game_active = False
    
    duration = time.time() - start_time
    
    return {
        "score": game_logic.game_state.score,
        "steps": game_logic.game_state.steps,
        "duration": duration,
        "game_end_reason": getattr(game_logic.game_state, 'game_end_reason', 'UNKNOWN'),
        "consecutive_failures": consecutive_no_path_found,
        "success": consecutive_no_path_found < max_consecutive_failures
    } 