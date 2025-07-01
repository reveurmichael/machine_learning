"""
Replay Web Game Application with Enhanced Naming
===============================================

Web application for replaying recorded Snake game sessions.
Provides Flask interface for game replay with ReplayEngine integration.
Uses enhanced naming and inherits from universal BaseReplayApp.

Design Philosophy:
- SSOT: Uses existing ReplayEngine from replay module for consistent behavior
- Educational: Shows replay web interface patterns with enhanced naming
- Layered Architecture: Inherits from BaseReplayApp for universal replay functionality
- Enhanced Naming: Clear, explicit naming that indicates web + game + replay domain
- Extension Pattern: Template for future replay web interfaces

Educational Value:
- Shows proper integration of web interface with replay functionality
- Demonstrates ReplayEngine usage for web-based replay with enhanced clarity
- Provides template for extension replay web applications with layered inheritance

Extension Pattern:
Extensions can copy this enhanced pattern and customize for their specific replay
needs while maintaining consistent web interface patterns and universal base functionality.

Reference: web/base_app.py for layered web infrastructure with BaseReplayApp
"""

from typing import Dict, Any
import os

# Import Task-0 replay components for proper integration
from replay.replay_engine import ReplayEngine

# Import base application class from layered web infrastructure
from web.base_app import BaseReplayApp

# Import utilities following SSOT principles
from utils.web_utils import build_color_map
from utils.print_utils import create_logger

# Create logger for this module
print_log = create_logger("ReplayWebGameApp")


class ReplayWebGameApp(BaseReplayApp):
    """
    Replay web game application with enhanced naming.
    
    Uses the existing ReplayEngine from replay module for consistent
    replay behavior across CLI and web interfaces. Enhanced naming
    clearly indicates this is a web-based game application for replay functionality.
    Inherits from BaseReplayApp for universal replay infrastructure.
    
    Design Pattern: Adapter Pattern (Web + Game + Replay Integration)
    Purpose: Adapts ReplayEngine interface for web API consumption with clear naming
    Educational Value: Shows proper integration of replay functionality with layered inheritance
    Extension Pattern: Copy this enhanced pattern for extension replay web interfaces
    
    Enhanced Features:
    - Clear naming: ReplayWebGameApp indicates web + game + replay focus
    - Layered inheritance: Inherits from BaseReplayApp for universal replay infrastructure
    - ReplayEngine integration: Same behavior as scripts/replay.py CLI interface
    - Educational clarity: Enhanced naming for better understanding
    """
    
    def __init__(self, log_dir: str, game_number: int = 1, **config):
        """Initialize replay web game app with ReplayEngine integration.
        
        Args:
            log_dir: Directory containing game logs
            game_number: Game number to replay (default: 1)
            **config: Additional configuration options
            
        Raises:
            ValueError: If log directory doesn't exist
            
        Educational Value: Shows proper validation and error handling with enhanced naming
        Extension Pattern: Extensions can copy this validation pattern
        """
        super().__init__("Snake Game Web Replay", log_dir, **config)
        self.game_number = game_number
        
        # Validate log directory exists
        if not os.path.isdir(log_dir):
            raise ValueError(f"Log directory does not exist: {log_dir}")
        
        print_log(f"Replay web game mode: {os.path.basename(log_dir)}, game {game_number}")
        print_log("ReplayEngine will be integrated - same behavior as scripts/replay.py")
    
    def setup_replay_infrastructure(self) -> None:
        """Setup replay infrastructure with ReplayEngine integration.
        
        Educational Value: Shows universal replay infrastructure setup with ReplayEngine
        Extension Pattern: Extensions can override for specialized replay setup
        """
        super().setup_replay_infrastructure()
        
        # Initialize ReplayEngine - same as scripts/replay.py
        self.replay_engine = ReplayEngine(
            log_dir=self.log_dir,
            pause_between_moves=self.config.get('pause_between_moves', 1.0),
            auto_advance=self.config.get('auto_advance', False)
        )
        
        # Set initial game number
        self.replay_engine.game_number = self.game_number
        
        print_log("ReplayEngine integrated - same behavior as scripts/replay.py CLI interface")
    
    def get_app_data(self) -> Dict[str, Any]:
        """Get application data for template rendering.
        
        Returns:
            Dictionary with replay data for HTML template
            
        Educational Value: Shows replay-specific template data preparation with enhanced naming
        Extension Pattern: Extensions can copy this for their replay template data
        """
        return {
            'name': self.name,
            'mode': 'replay_web',
            'log_dir': os.path.basename(self.log_dir),
            'game_number': self.game_number,
            'status': 'ready',
            'type': 'replay_web_game_app'
        }
    
    def get_replay_state(self) -> Dict[str, Any]:
        """Get replay game state for API consumption.
        
        Returns:
            Dictionary with current replay state
            
        Educational Value: Shows replay-specific state representation with enhanced structure
        Extension Pattern: Extensions can copy this state format for their replay apps
        """
        try:
            # Get current state from ReplayEngine
            current_state = self.replay_engine.get_current_state()
            
            if current_state:
                return {
                    'mode': 'replay_web',
                    'app_type': 'replay_web_game_app',
                    'log_dir': os.path.basename(self.log_dir),
                    'game_number': self.replay_engine.game_number,
                    'current_step': self.replay_engine.current_step,
                    'total_steps': len(self.replay_engine.moves) if hasattr(self.replay_engine, 'moves') else 0,
                    'snake_positions': current_state.get('snake_positions', []),
                    'apple_position': current_state.get('apple_position', [0, 0]),
                    'score': current_state.get('score', 0),
                    'steps': current_state.get('steps', 0),
                    'running': not self.replay_engine.paused,
                    'game_active': not current_state.get('game_over', False),
                    'game_over': current_state.get('game_over', False),
                    'paused': self.replay_engine.paused,
                    'status': 'replaying',
                    'colors': build_color_map(as_list=True)
                }
            else:
                # No current state available
                return {
                    'mode': 'replay_web',
                    'app_type': 'replay_web_game_app',
                    'log_dir': os.path.basename(self.log_dir),
                    'game_number': self.game_number,
                    'snake_positions': [],
                    'apple_position': [0, 0],
                    'score': 0,
                    'steps': 0,
                    'running': False,
                    'game_active': False,
                    'game_over': False,
                    'paused': True,
                    'status': 'loading',
                    'colors': build_color_map(as_list=True)
                }
        except Exception as e:
            print_log(f"Error getting replay state: {e}")
            return {
                'mode': 'replay_web',
                'app_type': 'replay_web_game_app',
                'status': 'error',
                'message': str(e),
                'colors': build_color_map(as_list=True)
            }
    
    def handle_replay_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle replay web game controls.
        
        Args:
            data: Control data from client
            
        Returns:
            Response dictionary with operation result
            
        Educational Value: Shows replay-specific control handling with enhanced context
        Extension Pattern: Extensions can copy this for their replay control logic
        """
        action = data.get('action', '')
        
        try:
            if action == 'play':
                print_log("Replay web game play")
                self.replay_engine.paused = False
                return {
                    'action': 'play', 
                    'status': 'playing', 
                    'paused': False,
                    'app_type': 'replay_web_game_app'
                }
                
            elif action == 'pause':
                print_log("Replay web game pause")
                self.replay_engine.paused = True
                return {
                    'action': 'pause', 
                    'status': 'paused', 
                    'paused': True,
                    'app_type': 'replay_web_game_app'
                }
                
            elif action == 'reset':
                print_log("Replay web game reset")
                self.replay_engine.reset_current_game()
                return {
                    'action': 'reset', 
                    'status': 'reset',
                    'app_type': 'replay_web_game_app'
                }
                
            elif action == 'next_game':
                print_log("Replay web game next game")
                self.replay_engine.next_game()
                return {
                    'action': 'next_game', 
                    'status': 'changed_game',
                    'game_number': self.replay_engine.game_number,
                    'app_type': 'replay_web_game_app'
                }
                
            elif action == 'prev_game':
                print_log("Replay web game previous game")
                self.replay_engine.previous_game()
                return {
                    'action': 'prev_game', 
                    'status': 'changed_game',
                    'game_number': self.replay_engine.game_number,
                    'app_type': 'replay_web_game_app'
                }
                
            elif action == 'speed_up':
                print_log("Replay web game speed up")
                # Implementation would adjust replay speed
                return {
                    'action': 'speed_up', 
                    'status': 'speed_changed',
                    'app_type': 'replay_web_game_app'
                }
                
            elif action == 'speed_down':
                print_log("Replay web game speed down")
                # Implementation would adjust replay speed
                return {
                    'action': 'speed_down', 
                    'status': 'speed_changed',
                    'app_type': 'replay_web_game_app'
                }
                
            else:
                return {
                    'status': 'error', 
                    'app_type': 'replay_web_game_app',
                    'message': f'Unknown action: {action}. Supported: play, pause, reset, next_game, prev_game, speed_up, speed_down.'
                }
                
        except Exception as e:
            print_log(f"Error handling replay control: {e}")
            return {
                'status': 'error', 
                'app_type': 'replay_web_game_app',
                'message': str(e)
            }
    
    def get_replay_info(self) -> Dict[str, Any]:
        """Get replay information and metadata.
        
        Returns:
            Dictionary with replay information
            
        Educational Value: Shows replay metadata patterns with enhanced structure
        Extension Pattern: Extensions can override for additional replay info
        """
        base_info = super().get_replay_info()
        base_info.update({
            'app_type': 'replay_web_game_app',
            'game_number': self.game_number,
            'replay_engine_available': hasattr(self, 'replay_engine'),
            'log_dir_basename': os.path.basename(self.log_dir)
        })
        return base_info
    
    def get_template_name(self) -> str:
        """Get template name for replay web game interface.
        
        Returns:
            Template filename for replay web game interface
            
        Educational Value: Shows replay template selection patterns with enhanced naming
        Extension Pattern: Extensions can override for custom replay templates
        """
        return 'replay.html'

