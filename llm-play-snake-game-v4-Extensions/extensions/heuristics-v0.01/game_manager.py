"""
Heuristic Game Manager - Session management for heuristic algorithms
==================================================================

This module extends BaseGameManager to provide session management
specifically for heuristic algorithms while maintaining compatibility
with the base game infrastructure.

Design Philosophy:
- Extends BaseGameManager (inherits all generic session management)
- Uses HeuristicGameLogic for game mechanics
- No LLM dependencies (no token stats, no continuation mode)
- Generates same log format as Task-0 for compatibility
"""

from __future__ import annotations
import argparse
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

from colorama import Fore

from core.game_manager import BaseGameManager
from core.game_stats_manager import BaseGameStatsManager
import json
from game_logic import HeuristicGameLogic
from bfs_agent import BFSAgent


class HeuristicGameManager(BaseGameManager):
    """
    Session manager for heuristic algorithms.
    
    Extends BaseGameManager with heuristic-specific functionality while
    maintaining the same session management patterns and log output format.
    
    Design Patterns:
    - Template Method: Inherits base session management structure
    - Factory Pattern: Uses HeuristicGameLogic for game logic
    - Strategy Pattern: Pluggable heuristic algorithms
    """
    
    # Use heuristic-specific game logic
    GAME_LOGIC_CLS = HeuristicGameLogic
    
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize heuristic game manager.
        
        Args:
            args: Command line arguments (compatible with Task-0 format)
        """
        super().__init__(args)
        
        # Heuristic-specific attributes
        self.algorithm_name: str = getattr(args, 'algorithm', 'v0.01-BFS')
        self.agent: Optional[BFSAgent] = None
        
        # Setup stats manager for log output
        self.stats_manager: Optional[BaseGameStatsManager] = None
        self.log_dir: Optional[str] = None
        
        # Session tracking
        self.session_start_time = datetime.now()
        
    def initialize(self) -> None:
        """Initialize the heuristic game manager."""
        # Setup logging directory
        self._setup_logging()
        
        # Create and configure heuristic agent
        self._setup_agent()
        
        # Initialize base components
        self.setup_game()
        
        # Configure game with agent
        if isinstance(self.game, HeuristicGameLogic) and self.agent:
            self.game.set_agent(self.agent)
        
        print(Fore.GREEN + f"🤖 Heuristic Game Manager initialized with {self.algorithm_name} algorithm")
        print(Fore.CYAN + f"📂 Logs will be saved to: {self.log_dir}")
    
    def _setup_logging(self) -> None:
        """Setup logging directory and stats manager."""
        # Create log directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/heuristics-v0.01-{self.algorithm_name.lower()}_{timestamp}"
        
        # Create directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize stats manager (singleton pattern)
        self.stats_manager = BaseGameStatsManager(self.log_dir)
    
    def _setup_agent(self) -> None:
        """Setup the heuristic agent."""
        if self.algorithm_name.upper() == "BFS":
            self.agent = BFSAgent()
        else:
            # Default to BFS for now
            print(Fore.YELLOW + f"⚠️  Algorithm '{self.algorithm_name}' not implemented, using BFS")
            self.agent = BFSAgent()
    
    def run(self) -> None:
        """
        Run the heuristic game session.
        
        This method follows the same pattern as Task-0 GameManager.run()
        but uses heuristic planning instead of LLM planning.
        """
        try:
            print(Fore.GREEN + f"🚀 Starting heuristic game session...")
            print(Fore.CYAN + f"📊 Target games: {self.args.max_games}")
            print(Fore.CYAN + f"🧠 Algorithm: {self.algorithm_name}")
            
            # Main game loop
            while self.game_count < self.args.max_games and self.running:
                self._run_single_game()
                
            # Save session summary
            self._save_session_summary()
            
            print(Fore.GREEN + f"✅ Heuristic session completed!")
            print(Fore.CYAN + f"📊 Games played: {self.game_count}")
            print(Fore.CYAN + f"🏆 Total score: {self.total_score}")
            
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n⚠️  Session interrupted by user")
            self._save_session_summary()
        except Exception as e:
            print(Fore.RED + f"❌ Session error: {e}")
            raise
    
    def _run_single_game(self) -> None:
        """
        Run a single game with heuristic planning.
        
        Follows the same pattern as Task-0 but uses heuristic agents
        instead of LLM responses.
        """
        self.game_count += 1
        self.round_count = 1
        
        print(Fore.BLUE + f"\n🎮 Starting Game {self.game_count}")
        
        # Reset game state
        self.setup_game()
        if isinstance(self.game, HeuristicGameLogic) and self.agent:
            self.game.set_agent(self.agent)
        
        # Reset game manager state for new game
        self.game_active = True
        self.need_new_plan = True
        self.consecutive_no_path_found = 0
        self.no_path_found_steps = 0
        
        game_start_time = time.time()
        
        # Game loop
        while self.game_active and self.game.game_state.steps < self.args.max_steps:
            # Start new round for planning
            self.start_new_round("Heuristic pathfinding")
            
            # Get next move from heuristic agent
            planned_move = self.game.get_next_planned_move()
            
            # Execute move
            if planned_move == "NO_PATH_FOUND":
                self.consecutive_no_path_found += 1
                self.no_path_found_steps += 1
                
                if self.consecutive_no_path_found >= 5:  # Same limit as Task-0
                    print(Fore.RED + f"❌ Too many consecutive pathfinding failures")
                    self.game_active = False
                    break
            else:
                self.consecutive_no_path_found = 0
                
                # Make the move
                game_continues, apple_eaten = self.game.make_move(planned_move)
                
                if not game_continues:
                    self.game_active = False
                    
                if apple_eaten:
                    print(Fore.GREEN + f"🍎 Apple eaten! Score: {self.game.game_state.score}")
        
        # Record game completion
        game_duration = time.time() - game_start_time
        self._finalize_game(game_duration)
    
    def _finalize_game(self, game_duration: float) -> None:
        """Finalize game and save results."""
        # Update session stats
        self.total_score += self.game.game_state.score
        self.total_steps += self.game.game_state.steps
        self.total_rounds += self.round_count
        
        self.game_scores.append(self.game.game_state.score)
        self.round_counts.append(self.round_count)
        
        # Create game data for logging
        game_data = self._create_game_log_data(game_duration)
        
        # Save game log
        game_filename = f"game_{self.game_count}.json"
        game_filepath = os.path.join(self.log_dir, game_filename)
        with open(game_filepath, 'w') as f:
            json.dump(game_data, f, indent=2)
        
        print(Fore.CYAN + f"📊 Game {self.game_count} completed:")
        print(Fore.CYAN + f"   Score: {self.game.game_state.score}")
        print(Fore.CYAN + f"   Steps: {self.game.game_state.steps}")
        print(Fore.CYAN + f"   Rounds: {self.round_count}")
        print(Fore.CYAN + f"   Duration: {game_duration:.2f}s")
    
    def _create_game_log_data(self, game_duration: float) -> Dict[str, Any]:
        """
        Create game log data compatible with Task-0 format.
        
        This maintains the same JSON structure as Task-0 logs but with
        heuristic-specific data instead of LLM data.
        """
        game_state = self.game.game_state
        
        return {
            "score": game_state.score,
            "steps": game_state.steps,
            "snake_length": len(self.game.snake_positions),
            "game_over": not self.game_active,
            "game_end_reason": game_state.game_end_reason or "UNKNOWN",
            "round_count": self.round_count,
            
            # Heuristic info (replaces llm_info)
            "heuristic_info": {
                "algorithm": self.algorithm_name,
                "agent_type": type(self.agent).__name__ if self.agent else "None"
            },
            
            # Time stats (simplified, no LLM communication time)
            "time_stats": {
                "start_time": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration_seconds": game_duration,
                "heuristic_computation_time": getattr(game_state, 'total_search_time', 0.0)
            },
            
            # Heuristic performance stats (replaces token_stats)
            "pathfinding_stats": game_state.get_heuristic_stats() if hasattr(game_state, 'get_heuristic_stats') else {},
            
            # Step stats (reuses base step counting)
            "step_stats": {
                "valid_steps": game_state.steps - self.no_path_found_steps,
                "no_path_found_steps": self.no_path_found_steps,
                "invalid_reversals": self.invalid_reversals
            },
            
            # Metadata
            "metadata": {
                "timestamp": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "game_number": self.game_count,
                "round_count": self.round_count
            },
            
            # Detailed history (simplified for v0.01)
            "detailed_history": {
                "apple_positions": [],  # Simplified for v0.01
                "moves": [],            # Simplified for v0.01
                "rounds_data": {}       # Simplified for v0.01
            }
        }
    
    def _save_session_summary(self) -> None:
        """Save session summary in Task-0 compatible format."""
        summary_data = {
            "timestamp": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "algorithm": self.algorithm_name,
                "max_games": self.args.max_games,
                "max_steps": getattr(self.args, 'max_steps', 800),
                "no_gui": True  # Heuristics are always headless
            },
            "game_statistics": {
                "total_games": self.game_count,
                "total_rounds": self.total_rounds,
                "total_score": self.total_score,
                "total_steps": self.total_steps,
                "scores": self.game_scores,
                "round_counts": self.round_counts
            },
            "heuristic_statistics": {
                "algorithm_name": self.algorithm_name,
                "average_score": self.total_score / max(self.game_count, 1),
                "average_steps": self.total_steps / max(self.game_count, 1),
                "average_rounds": self.total_rounds / max(self.game_count, 1)
            },
            "step_stats": {
                "no_path_found_steps": self.no_path_found_steps,
                "valid_steps": self.total_steps - self.no_path_found_steps,
                "invalid_reversals": self.invalid_reversals
            }
        }
        
        summary_filepath = os.path.join(self.log_dir, "summary.json")
        with open(summary_filepath, 'w') as f:
            json.dump(summary_data, f, indent=2) 