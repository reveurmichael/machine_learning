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
from replay.replay_engine import ReplayEngine
from utils.web_utils import to_list, build_color_map


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
        
        # Use existing ReplayEngine with log_dir parameter
        self.replay_engine = ReplayEngine(log_dir=log_dir, use_gui=False)
        self.replay_data = None
        # List of per-step dictionaries representing the replay timeline
        self.steps = []  # type: list[dict[str, Any]]
        # Indicates whether self.steps contains full snapshot dicts (True) or raw move strings (False).
        self._steps_are_snapshots: bool = False
        self.current_step = 0
        
        # --- Auto-play (background loop) state --------------------------------
        # Whether the background replay loop should advance the step index.
        # Starts in a paused state – the front-end can switch to "play".
        self.paused: bool = True
        # Sleep time (seconds) between automatic step advances.
        self.move_pause: float = 1.0
        
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

            # Normalise steps list – fall back to ReplayEngine.moves if the JSON
            # contains no per-step snapshots (legacy logs).
            steps_field = []
            if self.replay_data:
                steps_field = self.replay_data.get('steps', [])
                if not isinstance(steps_field, list):
                    steps_field = []

            if steps_field and isinstance(steps_field[0], dict):
                # JSON already contains per-step snapshots – prefer these.
                self.steps = steps_field
                self._steps_are_snapshots = True
            else:
                # Fall back to raw moves sequence – we'll drive the original
                # ReplayEngine to compute dynamic states.
                self.steps = self.replay_engine.moves
                self._steps_are_snapshots = False

            print(f"[ReplayWebApp] Loaded {len(self.steps)} steps")
        except Exception as e:
            print(f"[ReplayWebApp] Error loading replay data: {e}")
            self.replay_data = None
            self.steps = []
            return False

        return bool(self.replay_data)
    
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
            # Attempt to return live engine state instead of erroring out so UI can still display.
            if self.replay_engine.moves:
                engine_state = self.replay_engine._build_state_base()
                engine_state['snake_positions'] = to_list(engine_state.get('snake_positions', []))
                engine_state['apple_position'] = to_list(engine_state.get('apple_position', [0, 0]))
                engine_state.update({
                    'mode': 'replay',
                    'log_dir': self.log_dir,
                    'game_number': self.game_number,
                    'current_step': self.replay_engine.move_index,
                    'total_steps': len(self.replay_engine.moves),
                    'colors': build_color_map(as_list=True),
                })
                return engine_state
            else:
                return {
                    'mode': 'replay',
                    'error': 'No replay data loaded',
                    'log_dir': self.log_dir,
                    'game_number': self.game_number
                }
        
        steps = self.steps
        
        # Improved fallback: only error if both steps and moves are empty
        if not steps or not isinstance(steps[0], dict):
            if self.replay_engine.moves:
                engine_state = self.replay_engine._build_state_base()
                # Convert NumPy arrays to lists for JSON serialisation
                engine_state['snake_positions'] = to_list(engine_state.get('snake_positions', []))
                engine_state['apple_position'] = to_list(engine_state.get('apple_position', [0, 0]))
                # Enrich with replay-specific metadata expected by the front-end
                engine_state.update({
                    'mode': 'replay',
                    'log_dir': self.log_dir,
                    'game_number': self.game_number,
                    'current_step': self.replay_engine.move_index,
                    'total_steps': len(self.replay_engine.moves),
                    'colors': build_color_map(as_list=True),
                })
                return engine_state
            else:
                return {
                    'mode': 'replay',
                    'error': 'No steps available',
                    'current_step': self.current_step,
                    'total_steps': 0
                }

        if self.current_step >= len(steps):
            return {
                'mode': 'replay',
                'error': 'No steps available',
                'current_step': self.current_step,
                'total_steps': len(steps)
            }
        
        current_data = steps[self.current_step]
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
            'snake_positions': to_list(current_data.get('snake_positions', [])),
            'apple_position': to_list(current_data.get('apple_position', [0, 0])),
            'score': current_data.get('score', 0),
            'colors': build_color_map(as_list=True)
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
            return {'status': 'ok', 'move_pause': self.move_pause}
        elif action == 'speed_down':
            self.move_pause += 0.1
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
            # Guard: do nothing if paused or we have reached the last frame.
            if self.paused:
                time.sleep(self.move_pause)
                continue

            if self._steps_are_snapshots:
                if self.current_step < max(len(self.steps) - 1, 0):
                    self.current_step += 1
            else:
                # Drive original update loop which mutates replay_engine state
                self.replay_engine.update()
            # Sleep regardless (even when paused) to relinquish the GIL.
            time.sleep(self.move_pause)

