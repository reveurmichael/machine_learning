"""Heuristic Game Manager for Task-1.

This module implements HeuristicGameManager which extends BaseGameManager
to provide session management specifically for heuristic algorithms.

The manager handles:
- Agent selection and initialization
- Logging to logs/heuristics/ directory
- Session statistics for algorithm performance
- Round tracking for planning cycles
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

from core.game_manager import BaseGameManager
from core.game_logic import BaseGameLogic
from extensions.heuristics.game_logic import HeuristicGameLogic
from extensions.heuristics.config import (
    HEURISTIC_LOG_DIR,
    DEFAULT_ALGORITHM,
    validate_algorithm,
)
from extensions.heuristics.agents import (
    BFSAgent,
    AStarAgent,
    HamiltonianAgent,
    LongestPathAgent,
    WallHuggerAgent,
)

if TYPE_CHECKING:
    import argparse
    from core.game_agents import SnakeAgent


class HeuristicGameManager(BaseGameManager):
    """Game manager for heuristic algorithms (Task-1).
    
    Extends BaseGameManager with heuristic-specific functionality:
    - Agent selection and configuration
    - Algorithm-specific logging and statistics
    - Heuristic session management
    - Performance tracking for algorithm comparison
    
    This manager works with any heuristic agent that implements SnakeAgent.
    """

    # Use generic game logic (no LLM dependencies)
    GAME_LOGIC_CLS = HeuristicGameLogic

    def __init__(self, args: "argparse.Namespace") -> None:
        """Initialize heuristic game manager.
        
        Args:
            args: Command line arguments containing algorithm selection
        """
        super().__init__(args)
        
        # Heuristic-specific attributes
        self.algorithm_name = getattr(args, "algorithm", DEFAULT_ALGORITHM)
        self.agent: Optional[SnakeAgent] = None
        
        # Algorithm performance tracking
        self.algorithm_stats = {
            "total_search_time": 0.0,
            "total_searches": 0,
            "successful_searches": 0,
            "average_search_time": 0.0,
            "success_rate": 0.0,
            "total_path_length": 0,
            "average_path_length": 0.0,
        }
        
        # Session metadata
        self.session_start_time = 0.0
        self.total_session_time = 0.0
        self.time_stats = {}  # Add time_stats for compatibility with base game loop
        self.token_stats = {}  # Add token_stats for compatibility with base game loop
        
    def initialize(self) -> None:
        """Initialize heuristic-specific components."""
        import time
        self.session_start_time = time.time()
        
        # Validate and create the selected algorithm
        if not validate_algorithm(self.algorithm_name):
            available_algs = ', '.join(['BFS', 'A_STAR', 'HAMILTONIAN', 'LONGEST_PATH', 'WALL_HUGGER'])
            raise ValueError(f"Unknown algorithm '{self.algorithm_name}'. Available: {available_algs}")
        
        # Create the agent instance
        self.agent = self._create_agent(self.algorithm_name)
        
        # Setup logging directory
        self.setup_logging(HEURISTIC_LOG_DIR, "heuristics")
        
        # Setup the game with generic logic
        self.setup_game()
        
        print(f"🧮 Initialized {self.algorithm_name} agent for heuristic session")
        print(f"📁 Logging to: {self.log_dir}")
        
    def _create_agent(self, algorithm_name: str) -> SnakeAgent:
        """Create an agent instance based on algorithm name.
        
        Args:
            algorithm_name: Name of the algorithm to create
            
        Returns:
            Agent instance implementing SnakeAgent protocol
            
        Raises:
            ValueError: If algorithm name is not recognized
        """
        agent_map = {
            "BFS": BFSAgent,
            "A_STAR": AStarAgent,
            "HAMILTONIAN": HamiltonianAgent,
            "LONGEST_PATH": LongestPathAgent,
            "WALL_HUGGER": WallHuggerAgent,
        }
        
        if algorithm_name not in agent_map:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
        return agent_map[algorithm_name]()
    
    def run(self) -> None:
        """Execute the heuristic session using the heuristic game loop."""
        from extensions.heuristics.loop import HeuristicGameLoop
        
        print(f"🚀 Starting heuristic session with {self.algorithm_name}")
        print(f"🎮 Games to play: {self.args.max_games}")
        print(f"🎯 Grid size: {getattr(self.args, 'grid_size', 10)}")
        print(f"👁️  GUI mode: {'enabled' if self.use_gui else 'disabled'}")
        
        # Use the heuristic-specific game loop
        heuristic_loop = HeuristicGameLoop(self)
        heuristic_loop.run()
        
        # Calculate final session statistics
        import time
        self.total_session_time = time.time() - self.session_start_time
        
        # Report final results
        self.report_final_statistics()
    
    def setup_logging(self, base_dir: str, task_name: str = "heuristics") -> None:
        """Setup logging directory for heuristic session.
        
        Args:
            base_dir: Base logging directory
            task_name: Name of the task (for directory naming)
        """
        import time
        from pathlib import Path
        
        # Create timestamped session directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = f"{self.algorithm_name.lower()}_{timestamp}"
        self.log_dir = os.path.join(base_dir, session_dir)
        
        # Ensure directory exists
        log_path = Path(self.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        (log_path / "games").mkdir(exist_ok=True)
        (log_path / "stats").mkdir(exist_ok=True)
        
        print(f"📂 Created logging directory: {self.log_dir}")
    
    def save_session_summary(self) -> None:
        """Save session summary with heuristic-specific statistics."""
        if not self.log_dir:
            return
            
        import json
        import time
        from pathlib import Path
        
        # Update algorithm statistics from agent
        if self.agent and hasattr(self.agent, 'get_search_stats'):
            agent_stats = self.agent.get_search_stats()
            self.algorithm_stats.update(agent_stats)
        
        # Calculate session-level statistics
        session_stats = {
            "session_info": {
                "algorithm": self.algorithm_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_session_time": self.total_session_time,
                "grid_size": getattr(self.args, 'grid_size', 10),
                "use_gui": self.use_gui,
                "args": vars(self.args),
            },
            "game_results": {
                "total_games": self.game_count,
                "total_score": self.total_score,
                "total_steps": self.total_steps,
                "total_rounds": self.total_rounds,
                "game_scores": self.game_scores,
                "round_counts": self.round_counts,
                "average_score": self.total_score / max(1, self.game_count),
                "average_steps": self.total_steps / max(1, self.game_count),
                "average_rounds": self.total_rounds / max(1, self.game_count),
            },
            "error_tracking": {
                "valid_steps": self.valid_steps,
                "invalid_reversals": self.invalid_reversals,
                "no_path_found_steps": self.no_path_found_steps,
                "error_rate": self.invalid_reversals / max(1, self.total_steps),
                "path_failure_rate": self.no_path_found_steps / max(1, self.total_steps),
            },
            "algorithm_performance": self.algorithm_stats,
        }
        
        # Save summary to JSON file
        summary_path = Path(self.log_dir) / "session_summary.json"
        summary_json = json.dumps(session_stats, indent=2, default=str)
        summary_path.write_text(summary_json, encoding="utf-8")
        
        print(f"💾 Saved session summary to: {summary_path}")
    
    def report_final_statistics(self) -> None:
        """Report final session statistics to console."""
        print("\n" + "="*60)
        print(f"🧮 HEURISTIC SESSION COMPLETE: {self.algorithm_name}")
        print("="*60)
        
        print(f"\n📊 GAME RESULTS:")
        print(f"   Total Games: {self.game_count}")
        print(f"   Total Score: {self.total_score}")
        print(f"   Average Score: {self.total_score / max(1, self.game_count):.2f}")
        print(f"   Total Steps: {self.total_steps}")
        print(f"   Average Steps per Game: {self.total_steps / max(1, self.game_count):.2f}")
        
        print(f"\n🔍 ALGORITHM PERFORMANCE:")
        if hasattr(self.agent, 'get_search_stats'):
            stats = self.agent.get_search_stats()
            print(f"   Algorithm: {stats.get('algorithm', self.algorithm_name)}")
            print(f"   Success Rate: {stats.get('success_rate', 0.0)*100:.1f}%")
            print(f"   Average Search Time: {stats.get('average_search_time', 0.0)*1000:.2f}ms")
            
            # Algorithm-specific statistics
            if 'last_nodes_explored' in stats:
                print(f"   Nodes Explored: {stats['last_nodes_explored']}")
            if 'shortcuts_taken' in stats:
                print(f"   Shortcuts Taken: {stats['shortcuts_taken']}")
            if 'cycle_length' in stats:
                print(f"   Cycle Length: {stats['cycle_length']}")
        
        print(f"\n⚠️  ERROR TRACKING:")
        print(f"   Invalid Reversals: {self.invalid_reversals}")
        print(f"   No Path Found: {self.no_path_found_steps}")
        print(f"   Error Rate: {self.invalid_reversals / max(1, self.total_steps)*100:.2f}%")
        
        print(f"\n⏱️  TIMING:")
        print(f"   Total Session Time: {self.total_session_time:.2f}s")
        print(f"   Average Time per Game: {self.total_session_time / max(1, self.game_count):.2f}s")
        
        # Save summary to file
        self.save_session_summary()
        
        print(f"\n📁 Session data saved to: {self.log_dir}")
        print("="*60) 
    def setup_game(self) -> None:
        """Create game logic and heuristic-specific GUI interface."""
        # Use the specified game logic class (BaseGameLogic by default)
        self.game = self.GAME_LOGIC_CLS(use_gui=self.use_gui)

        # Attach heuristic-specific GUI if visual mode is requested
        if self.use_gui:
            from extensions.heuristics.gui_heuristics import HeuristicGUI
            gui = HeuristicGUI(algorithm=self.algorithm_name)
            self.game.set_gui(gui)
