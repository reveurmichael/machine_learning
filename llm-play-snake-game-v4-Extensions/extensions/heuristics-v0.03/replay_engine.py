#!/usr/bin/env python3
"""
Heuristic Replay Engine
======================

Extension of Task-0's BaseReplayEngine for heuristic algorithms.
Demonstrates extensive reuse of base infrastructure while adding
heuristic-specific replay capabilities.

This engine can replay games from any of the 7 heuristic algorithms:
- BFS, BFS-Safe-Greedy, BFS-Hamiltonian
- DFS, A*, A*-Hamiltonian, Hamiltonian
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path

# Add root directory to Python path for accessing base classes
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

# Import Task-0 base replay infrastructure
from replay.replay_engine import BaseReplayEngine
from replay.replay_utils import load_game_json, parse_game_data
from config.ui_constants import TIME_DELAY


class HeuristicReplayEngine(BaseReplayEngine):
    """
    Replay engine specifically designed for heuristic algorithms.
    
    Extends Task-0's BaseReplayEngine with heuristic-specific features:
    - Algorithm-aware replay display
    - Pathfinding visualization
    - Performance metrics during replay
    - Support for all 7 heuristic algorithms
    
    Design Pattern: Adapter Pattern
    - Adapts Task-0 replay infrastructure for heuristic algorithm needs
    - Maintains compatibility with existing replay scripts and GUI
    
    Extensive Reuse:
    - Inherits all game mechanics from BaseReplayEngine
    - Reuses Task-0 JSON parsing and file management
    - Leverages existing GUI and web replay infrastructure
    """
    
    def __init__(
        self,
        log_dir: str,
        pause_between_moves: float = 1.0,
        auto_advance: bool = False,
        use_gui: bool = True,
    ) -> None:
        """Initialize heuristic replay engine."""
        super().__init__(log_dir, pause_between_moves, auto_advance, use_gui)
        
        # Heuristic-specific replay state
        self.algorithm_name: str = "Unknown"
        self.pathfinding_info: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        
        # Algorithm display names for better UX
        self.algorithm_display_names = {
            "bfs": "Breadth-First Search",
            "bfs-safe-greedy": "BFS with Safety Validation",
            "bfs-hamiltonian": "BFS with Hamiltonian Fallback",
            "dfs": "Depth-First Search",
            "astar": "A* Algorithm",
            "astar-hamiltonian": "A* with Hamiltonian Fallback",
            "hamiltonian": "Hamiltonian Cycle"
        }
    
    def load_game_data(self, game_number: int) -> Optional[Dict[str, Any]]:
        """
        Load heuristic game data from JSON file.
        
        Extends base loading with heuristic-specific data extraction:
        - Algorithm identification
        - Performance metrics
        - Pathfinding statistics
        
        Args:
            game_number: Game number to load
            
        Returns:
            Game data dictionary or None if loading fails
        """
        try:
            # Use Task-0 JSON loading infrastructure
            game_data = load_game_json(self.log_dir, game_number)
            
            if not game_data:
                print(f"❌ Could not load game {game_number} from {self.log_dir}")
                return None
            
            # Extract heuristic-specific information
            self.algorithm_name = game_data.get('algorithm', 'Unknown')
            
            # Parse performance metrics
            self.performance_metrics = {
                'score': game_data.get('score', 0),
                'steps': game_data.get('steps', 0),
                'round_count': game_data.get('round_count', 0),
                'score_per_step': game_data.get('score', 0) / max(game_data.get('steps', 1), 1),
                'score_per_round': game_data.get('score', 0) / max(game_data.get('round_count', 1), 1)
            }
            
            # Extract pathfinding information if available
            detailed_history = game_data.get('detailed_history', {})
            self.pathfinding_info = {
                'total_moves': len(detailed_history.get('moves', [])),
                'apple_positions': detailed_history.get('apple_positions', []),
                'rounds_data': detailed_history.get('rounds_data', {})
            }
            
            # Use base class parsing for common game data
            parsed_data = parse_game_data(game_data)
            
            if parsed_data:
                # Update replay state with parsed data
                self.apple_positions = parsed_data.get('apple_positions', [])
                self.moves = parsed_data.get('moves', [])
                self.game_stats = parsed_data.get('game_stats', {})
                
                # Reset replay indices
                self.apple_index = 0
                self.move_index = 0
                self.moves_made = []
                
                print(f"✅ Loaded {self.algorithm_display_names.get(self.algorithm_name, self.algorithm_name)} game {game_number}")
                print(f"📊 Score: {self.performance_metrics['score']}, Steps: {self.performance_metrics['steps']}")
                print(f"⚡ Efficiency: {self.performance_metrics['score_per_step']:.3f} score/step")
                
                return game_data
            else:
                print(f"❌ Failed to parse game data for game {game_number}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading heuristic game {game_number}: {e}")
            return None
    
    def update(self) -> None:
        """
        Update replay state with heuristic-specific enhancements.
        
        Extends base update with:
        - Algorithm-specific status display
        - Pathfinding progress tracking
        - Performance metrics updates
        """
        if not self.running:
            return
        
        # Check if it's time for the next move
        current_time = time.time()
        if current_time - self.last_move_time < self.pause_between_moves:
            return
        
        # Execute the next move if available
        if self.move_index < len(self.moves):
            move = self.moves[self.move_index]
            
            # Display algorithm-specific information
            algorithm_display = self.algorithm_display_names.get(self.algorithm_name, self.algorithm_name)
            print(f"🧠 {algorithm_display} | Move {self.move_index + 1}/{len(self.moves)}: {move}")
            
            # Execute the move using base class functionality
            game_continues = self.execute_replay_move(move)
            
            if not game_continues:
                self.running = False
                self._display_final_statistics()
                return
            
            self.move_index += 1
            self.moves_made.append(move)
            self.last_move_time = current_time
            
            # Update GUI if available
            if self.gui:
                self.gui.update_display()
        else:
            # Replay completed
            self.running = False
            self._display_final_statistics()
    
    def _display_final_statistics(self) -> None:
        """Display comprehensive final statistics for heuristic replay."""
        print(f"\n🎯 Heuristic Replay Completed!")
        print(f"🧠 Algorithm: {self.algorithm_display_names.get(self.algorithm_name, self.algorithm_name)}")
        print(f"🏆 Final Score: {self.performance_metrics['score']}")
        print(f"👣 Total Steps: {self.performance_metrics['steps']}")
        print(f"🔄 Total Rounds: {self.performance_metrics['round_count']}")
        print(f"⚡ Score per Step: {self.performance_metrics['score_per_step']:.3f}")
        print(f"🎯 Score per Round: {self.performance_metrics['score_per_round']:.3f}")
        
        # Algorithm-specific insights
        if self.algorithm_name in ["bfs", "astar"]:
            print(f"🎯 Optimal pathfinding algorithm - shortest paths guaranteed")
        elif self.algorithm_name == "bfs-safe-greedy":
            print(f"🛡️ Safety-enhanced BFS - balanced performance and safety")
        elif self.algorithm_name == "hamiltonian":
            print(f"🔄 Space-filling algorithm - covers entire board systematically")
        elif "hamiltonian" in self.algorithm_name:
            print(f"🔀 Hybrid algorithm - combines pathfinding with space-filling")
        elif self.algorithm_name == "dfs":
            print(f"🔍 Depth-first exploration - educational algorithm")
    
    def handle_events(self) -> None:
        """
        Handle replay events with heuristic-specific controls.
        
        Extends base event handling with:
        - Algorithm information display
        - Performance metrics on demand
        - Pathfinding visualization controls
        """
        # Use base class event handling for common functionality
        super().handle_events()
        
        # Add heuristic-specific event handling if needed
        # This can be extended for algorithm-specific visualizations
    
    def run(self) -> None:
        """
        Main replay loop with heuristic-specific enhancements.
        
        Provides rich console output and algorithm-specific information
        during replay execution.
        """
        print(f"\n🚀 Starting Heuristic Replay")
        print(f"📁 Log Directory: {self.log_dir}")
        print(f"🧠 Algorithm: {self.algorithm_display_names.get(self.algorithm_name, self.algorithm_name)}")
        print(f"⏱️  Pause Between Moves: {self.pause_between_moves}s")
        print(f"🎮 Auto Advance: {self.auto_advance}")
        print(f"🖥️  GUI Enabled: {self.use_gui}")
        
        # Load initial game
        if not self.load_game_data(self.game_number):
            print(f"❌ Failed to load initial game {self.game_number}")
            return
        
        # Initialize game state
        self.reset()
        
        # Main replay loop
        try:
            while self.running:
                self.update()
                self.handle_events()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Replay interrupted by user")
        except Exception as e:
            print(f"\n❌ Replay error: {e}")
        finally:
            print(f"🏁 Heuristic replay session ended")
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get comprehensive algorithm information for external use.
        
        Returns:
            Dictionary containing algorithm details and performance metrics
        """
        return {
            'algorithm_name': self.algorithm_name,
            'algorithm_display_name': self.algorithm_display_names.get(self.algorithm_name, self.algorithm_name),
            'performance_metrics': self.performance_metrics,
            'pathfinding_info': self.pathfinding_info,
            'replay_state': {
                'current_move': self.move_index,
                'total_moves': len(self.moves),
                'moves_made': len(self.moves_made),
                'game_active': self.running
            }
        }


def create_heuristic_replay_engine(
    log_dir: str,
    pause_between_moves: float = 1.0,
    auto_advance: bool = False,
    use_gui: bool = True
) -> HeuristicReplayEngine:
    """
    Factory function for creating heuristic replay engines.
    
    Design Pattern: Factory Pattern
    - Provides consistent interface for creating replay engines
    - Encapsulates configuration and initialization logic
    
    Args:
        log_dir: Directory containing heuristic game logs
        pause_between_moves: Delay between replay moves in seconds
        auto_advance: Whether to automatically advance through moves
        use_gui: Whether to enable GUI display
        
    Returns:
        Configured HeuristicReplayEngine instance
    """
    return HeuristicReplayEngine(
        log_dir=log_dir,
        pause_between_moves=pause_between_moves,
        auto_advance=auto_advance,
        use_gui=use_gui
    )