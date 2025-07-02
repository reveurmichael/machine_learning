"""
Replay Web Game Application - Simple Flask Interface
===================================================

Simple Flask web interface for replaying Snake game logs.
Uses existing replay infrastructure for consistent behavior.

Design Philosophy:
- KISS: Simple, direct implementation
- DRY: Reuses existing replay infrastructure
- No Over-Preparation: Only implements what's needed now
- Extensible: Easy for Tasks 1-5 to copy and modify
"""

from typing import Dict, Any
import os
import threading
import time

from web.base_app import GameFlaskApp
from config.game_constants import PAUSE_BETWEEN_MOVES_SECONDS
from replay.replay_engine import ReplayEngine
from utils.web_utils import to_list, build_color_map, translate_end_reason


class ReplayWebApp(GameFlaskApp):
    """
    Simple Flask web app for game replay.
    
    Uses existing ReplayEngine for consistent replay behavior.
    Provides web interface for viewing game replays.
    """
    
    def __init__(self, log_dir: str, game_number: int = 1, port: int = None):
        """Initialize replay web app."""
        super().__init__("Snake Game Replay", port)
        self.log_dir = log_dir
        self.game_number = game_number
        
        # Default pause between moves comes from central game constants
        self.move_pause: float = PAUSE_BETWEEN_MOVES_SECONDS
        
        # Use existing ReplayEngine, passing default pause value
        self.replay_engine = ReplayEngine(
            log_dir=log_dir,
            pause_between_moves=self.move_pause,
            use_gui=False,
        )
        self.replay_data = None
        # Sequence of raw move strings parsed from *game_N.json*
        self.steps: list[str] = []
        self.current_step = 0
        
        # --- Auto-play (background loop) state --------------------------------
        # Whether the background replay loop should advance the step index.
        # Starts in a paused state – the front-end can switch to "play".
        self.paused: bool = True
        
        # Load replay data and start background loop
        self.load_replay_data()
        
        # Start background loop **after** data are available so thread doesn't
        # immediately exit. The daemon flag ensures the thread exits with the
        # main interpreter (no explicit join necessary).
        threading.Thread(target=self._replay_loop, daemon=True).start()
        
        print(f"[ReplayWebApp] Loaded game {game_number} from {log_dir}")
    
    def load_replay_data(self) -> bool:
        """Load replay data from log directory."""
        try:
            # Use ReplayEngine to load data (engine already knows log_dir)
            self.replay_data = self.replay_engine.load_game_data(self.game_number)

            if not self.replay_data:
                print(f"[ReplayWebApp] No replay data found for game {self.game_number}")
                return False

            # Always use moves from the ReplayEngine (single-source of truth)
            self.steps = self.replay_engine.moves
            
            # Reset both indices to start from the beginning
            self.current_step = 0
            self.replay_engine.move_index = 0

            print(f"[ReplayWebApp] Loaded {len(self.steps)} moves")
            return True
            
        except Exception as e:
            print(f"[ReplayWebApp] Error loading replay data: {e}")
            self.replay_data = None
            self.steps = []
            return False
    
    def get_template_name(self) -> str:
        """Get template for replay interface."""
        return 'replay.html'
    
    def get_template_data(self) -> Dict[str, Any]:
        """Get template data for replay interface."""
        return {
            'name': self.name,
            'mode': 'replay',
            'log_dir': self.log_dir,
            'game_number': self.game_number,
            'has_data': self.replay_data is not None
        }
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current replay state."""
        if not self.replay_data:
            return {
                'mode': 'replay',
                'error': 'No replay data loaded',
                'log_dir': self.log_dir,
                'game_number': self.game_number
            }
        
        steps = self.steps
        
        # Since JSON schema is always consistent, we always have moves
        if not steps:
            return {
                'mode': 'replay',
                'error': 'No moves available',
                'current_step': self.current_step,
                'total_steps': 0
            }

        # Allow *exactly* len(steps) to represent "finished" state so progress
        # can show N/N. Only clamp when we somehow overshoot.
        if self.current_step > len(steps):
            self.current_step = len(steps)
        
        # Get current state from the ReplayEngine (which computes state from moves)
        engine_state = self.replay_engine._build_state_base()
        
        # Convert numpy arrays to lists for JSON serialization
        engine_state['snake_positions'] = to_list(engine_state.get('snake_positions', []))
        engine_state['apple_position'] = to_list(engine_state.get('apple_position', [0, 0]))
        
        # Get end reason from the replay data (not from individual steps)
        from utils.web_utils import translate_end_reason
        raw_reason = self.replay_data.get('game_end_reason') or self.replay_data.get('end_reason')
        end_reason = translate_end_reason(raw_reason) if raw_reason else None
        
        return {
            'mode': 'replay',
            'log_dir': self.log_dir,
            'game_number': self.game_number,
            'current_step': self.current_step,
            'total_steps': len(steps),
            'move_index': self.current_step,
            'total_moves': len(steps),
            'paused': self.paused,
            'move_pause': self.move_pause,
            'pause_between_moves': self.move_pause,
            'snake_positions': engine_state.get('snake_positions', []),
            'apple_position': engine_state.get('apple_position', [0, 0]),
            'score': engine_state.get('score', 0),
            'colors': build_color_map(as_list=True),
            'game_end_reason': end_reason
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle replay controls."""
        if not self.replay_data:
            return {'status': 'error', 'message': 'No replay data loaded'}
        
        # Future-proof API: single source – "action" key only (no legacy fallback)
        action = data.get('action', '')
        steps = self.steps
        
        if action == 'play':
            self.paused = False
            # Sync with underlying engine when using move-based playback
            self.replay_engine.paused = False
            return {'status': 'ok', 'message': 'Playing'}
        elif action == 'pause':
            self.paused = True
            self.replay_engine.paused = True
            return {'status': 'ok', 'message': 'Paused'}
        elif action == 'speed_up':
            # Decrease pause duration but keep a sensible lower bound
            self.move_pause = max(0.05, self.move_pause - 0.1)
            self.replay_engine.pause_between_moves = self.move_pause
            return {'status': 'ok', 'move_pause': self.move_pause}
        elif action == 'speed_down':
            self.move_pause += 0.1
            self.replay_engine.pause_between_moves = self.move_pause
            return {'status': 'ok', 'move_pause': self.move_pause}
        elif action == 'next_game':
            # Load next game if available
            self.game_number += 1
            if self.load_replay_data():
                self.current_step = 0
                return {'status': 'ok', 'game_number': self.game_number}
            else:
                # Roll back if load failed
                self.game_number -= 1
                return {'status': 'error', 'message': 'No next game'}
        elif action == 'prev_game':
            if self.game_number > 1:
                self.game_number -= 1
                self.load_replay_data()
                self.current_step = 0
                return {'status': 'ok', 'game_number': self.game_number}
            return {'status': 'error', 'message': 'Already at first game'}
        elif action == 'restart_game':
            self.load_replay_data()
            self.current_step = 0
            return {'status': 'ok', 'game_number': self.game_number}
        
        return {'status': 'error', 'message': 'Invalid action'}
    
    def handle_move(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move (not applicable for replay)."""
        return {'status': 'error', 'message': 'Moves not applicable in replay mode'}
    
    def handle_reset(self) -> Dict[str, Any]:
        """Reset replay to beginning."""
        self.current_step = 0
        # Keep the paused state – caller may explicitly choose to play.
        return {'status': 'ok', 'message': 'Replay reset to beginning', 'step': 0}

    # ---------------------------------------------------------------------
    # Background loop utilities
    # ---------------------------------------------------------------------

    def _replay_loop(self) -> None:
        """Continuously advance `current_step` when *not* paused.

        This background thread emulates the behaviour of the pre-MVC design
        where the game progressed automatically on the server-side. The loop
        sleeps for ``self.move_pause`` seconds between each advancement to
        avoid hogging CPU resources.
        """
        while True:
            # Guard: sleep when paused
            if self.paused:
                time.sleep(self.move_pause)
                continue

            # Check if we've reached the end of moves
            if self.current_step >= len(self.steps):
                # Stay at the last valid step
                time.sleep(self.move_pause)
                continue

            # Manually advance the step and apply the move
            if self.current_step < len(self.steps):
                # Get the move for this step
                move = self.steps[self.current_step]
                
                # Apply the move to the ReplayEngine
                self.replay_engine.execute_replay_move(move)
                
                # Advance to next step
                self.current_step += 1
                
                # Update the ReplayEngine's move_index to match
                self.replay_engine.move_index = self.current_step

            # Sleep based on current move_pause
            time.sleep(self.move_pause)

