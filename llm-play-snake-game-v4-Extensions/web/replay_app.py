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
        
        # Use existing ReplayEngine
        self.replay_engine = ReplayEngine()
        self.replay_data = None
        self.current_step = 0
        
        # Load replay data
        self.load_replay_data()
        
        print(f"[ReplayWebApp] Loaded game {game_number} from {log_dir}")
    
    def load_replay_data(self):
        """Load replay data from log directory."""
        try:
            # Use ReplayEngine to load data
            self.replay_data = self.replay_engine.load_game_data(self.log_dir, self.game_number)
            if self.replay_data:
                print(f"[ReplayWebApp] Loaded {len(self.replay_data.get('steps', []))} steps")
            else:
                print(f"[ReplayWebApp] No data found for game {self.game_number}")
        except Exception as e:
            print(f"[ReplayWebApp] Error loading replay data: {e}")
            self.replay_data = None
    
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
        
        steps = self.replay_data.get('steps', [])
        if not steps or self.current_step >= len(steps):
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
            'snake_positions': to_list(current_data.get('snake_positions', [])),
            'apple_position': to_list(current_data.get('apple_position', [0, 0])),
            'score': current_data.get('score', 0),
            'colors': build_color_map(as_list=True)
        }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle replay controls."""
        if not self.replay_data:
            return {'status': 'error', 'message': 'No replay data loaded'}
        
        action = data.get('action', '')
        steps = self.replay_data.get('steps', [])
        
        if action == 'next' and self.current_step < len(steps) - 1:
            self.current_step += 1
            return {'status': 'ok', 'step': self.current_step}
        elif action == 'previous' and self.current_step > 0:
            self.current_step -= 1
            return {'status': 'ok', 'step': self.current_step}
        elif action == 'reset':
            self.current_step = 0
            return {'status': 'ok', 'step': self.current_step}
        elif action == 'goto':
            step = data.get('step', 0)
            if 0 <= step < len(steps):
                self.current_step = step
                return {'status': 'ok', 'step': self.current_step}
            else:
                return {'status': 'error', 'message': 'Invalid step number'}
        
        return {'status': 'error', 'message': 'Invalid action or step'}
    
    def handle_move(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move (not applicable for replay)."""
        return {'status': 'error', 'message': 'Moves not applicable in replay mode'}
    
    def handle_reset(self) -> Dict[str, Any]:
        """Reset replay to beginning."""
        self.current_step = 0
        return {'status': 'ok', 'message': 'Replay reset to beginning', 'step': 0}

