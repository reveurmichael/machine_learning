"""
Main Web Game Application - Full GameManager Integration
=======================================================

Full Flask web interface for LLM-controlled Snake game with complete GameManager integration.
Provides real-time LLM functionality with web interface, mirroring all features of main.py.

Design Philosophy:
- Full Feature Parity: Mirrors all main.py capabilities
- Real-time Integration: Background GameManager with live state updates
- No Over-Preparation: Only implements what's needed for web interface
- Extensible: Easy for Tasks 1-5 to copy and modify

Educational Value:
- Shows proper Flask + GameManager integration
- Demonstrates real-time LLM game state synchronization
- Provides template for extension web interfaces

Integration Architecture:
- Background Thread: GameManager runs in daemon thread
- State Translation: build_state_dict() converts GameManager state to JSON
- Real-time Updates: 100ms polling for live game visualization
- Full CLI Support: All main.py arguments supported
"""

import threading
from typing import Dict, Any, Optional

from core.game_manager import GameManager
from core.game_controller import GameControllerAdapter
from llm.agent_llm import SnakeAgent
from web.base_app import GameFlaskApp
from utils.web_utils import build_state_dict
from core.game_file_manager import FileManager


class MainWebApp(GameFlaskApp):
    """
    Main Web Application with Full GameManager Integration.
    
    Provides complete LLM functionality with real-time web interface,
    mirroring all features of main.py mode.
    
    Design Pattern: MVC Pattern (Model-View-Controller)
    Purpose: Separate concerns between game logic, web interface, and request handling
    Educational Value: Demonstrates clean MVC implementation with Flask
    """
    
    def __init__(self, provider: str = "hunyuan", model: str = "hunyuan-turbos-latest", 
                 grid_size: int = 10, max_games: int = 1, port: int = None,
                 continue_from_folder: Optional[str] = None, no_gui: bool = True,
                 game_args: Optional[Any] = None):
        """Initialize main web app with full GameManager integration."""
        super().__init__("Main Snake Game", port)
        
        # Controller state
        self.provider = provider
        self.model = model
        self.grid_size = grid_size
        self.max_games = max_games
        self.continue_from_folder = continue_from_folder
        self.no_gui = no_gui
        self.game_args = game_args
        
        # Model components (GameManager + SnakeAgent)
        self.game_manager = None
        self.controller = None
        self.game_thread = None
        self.is_running = False
        
        # Initialize Model layer
        self._setup_model()
        
        print(f"[MainWebApp] Full GameManager integration: {provider}/{model}")
        print(f"[MainWebApp] Max games: {max_games}")
        if continue_from_folder:
            print(f"[MainWebApp] Continuing from: {continue_from_folder}")
    
    def _setup_model(self) -> None:
        """Setup Model layer (GameManager + SnakeAgent) with full integration."""
        try:
            # Use game_args if provided (full CLI support)
            if self.game_args:
                args = self.game_args
            else:
                # Create argument namespace for GameManager
                from argparse import Namespace
                args = Namespace(
                    provider=self.provider,
                    model=self.model,
                    parser_provider="none",
                    parser_model=None,
                    max_games=self.max_games,
                    pause_between_moves=1.0,  # Web mode uses 1 second pause
                    sleep_before_launching=0.0,
                    max_steps=800,
                    max_consecutive_empty_moves_allowed=5,
                    max_consecutive_something_is_wrong_allowed=5,
                    max_consecutive_invalid_reversals_allowed=10,
                    max_consecutive_no_path_found_allowed=5,
                    sleep_after_empty_step=0.0,
                    no_gui=self.no_gui,
                    log_dir=None,
                    continue_with_game_in_dir=self.continue_from_folder
                )
            
            # Initialize Model: GameManager
            if self.continue_from_folder:
                self.game_manager = GameManager.continue_from_directory(args)
                
                # Handle continuation mode configuration
                self._load_continuation_config()
            else:
                self.game_manager = GameManager(args)
            
            # Initialize Model: SnakeAgent
            self.game_manager.agent = SnakeAgent(
                self.game_manager,
                provider=self.provider,
                model=self.model
            )
            
            # Initialize Model: GameControllerAdapter (for state access)
            self.controller = GameControllerAdapter(self.game_manager)
            
            # Start GameManager in background thread
            self._start_game_manager_thread()
            
            print("[MainWebApp] Model layer initialized successfully")
            
        except Exception as e:
            print(f"[MainWebApp] Error setting up Model layer: {e}")
            raise
    
    def _load_continuation_config(self) -> None:
        """Load configuration from continuation directory."""
        if not self.continue_from_folder:
            return
            
        try:
            summary = FileManager().load_summary_data(self.continue_from_folder)
            if summary:
                original_cfg = summary.get("configuration", {})
                
                # Update GameManager args with original configuration
                for k in (
                    "provider", "model", "parser_provider", "parser_model",
                    "move_pause", "max_steps", "max_consecutive_empty_moves_allowed",
                    "max_consecutive_something_is_wrong_allowed",
                    "max_consecutive_invalid_reversals_allowed",
                    "max_consecutive_no_path_found_allowed",
                    "sleep_after_empty_step", "no_gui",
                ):
                    if k in original_cfg:
                        setattr(self.game_manager.args, k, original_cfg[k])
                        
                print("[MainWebApp] Loaded continuation config from summary.json")
        except Exception as e:
            print(f"[MainWebApp] Warning: Could not load continuation config: {e}")
    
    def _start_game_manager_thread(self) -> None:
        """Start GameManager in background thread."""
        if self.game_manager:
            self.game_thread = threading.Thread(
                target=self._game_manager_worker,
                args=(self.game_manager,),
                daemon=True
            )
            self.game_thread.start()
            print("[MainWebApp] GameManager thread started")
    
    def _game_manager_worker(self, gm: GameManager) -> None:
        """Background worker for running the game."""
        try:
            if self.continue_from_folder:
                # Continue from existing session
                # The GameManager instance `gm` is already configured for continuation
                # by GameManager.continue_from_directory in _setup_model.
                # We just need to run it.
                gm.run()
            else:
                # Start new game session
                gm.run()
        except Exception as e:
            print(f"[MainWebApp] GameManager thread crashed: {e}")
    
    # =============================================================================
    # Controller Methods (Request Handling)
    # =============================================================================
    
    def get_template_name(self) -> str:
        """Get View template name."""
        return 'main.html'
    
    def get_template_data(self) -> Dict[str, Any]:
        """Get View template data."""
        return {
            'name': self.name,
            'mode': 'main',
            'provider': self.provider,
            'model': self.model,
            'grid_size': self.grid_size,
            'max_games': self.max_games
        }
    
    def get_game_state(self) -> Dict[str, Any]:
        """Controller: Transform Model state for View consumption."""
        try:
            # If GameManager or current game is not yet initialised, return placeholder state
            if not self.game_manager or self.game_manager.game is None:
                return {
                    'mode': 'main',
                    'status': 'initialising',
                    'provider': self.provider,
                    'model': self.model
                }

            game = self.game_manager.game

            # Translate end-reason for readability
            from utils.web_utils import translate_end_reason, build_state_dict

            reason_code = getattr(game.game_state, 'game_end_reason', None)
            end_reason_readable = translate_end_reason(reason_code)

            extra_state = {
                'mode': 'main',
                'provider': self.provider,
                'model': self.model,
                'game_number': self.game_manager.game_count + 1,
                'round_count': self.game_manager.round_count,
                'running': self.game_manager.running,
                'game_active': self.game_manager.game_active,
                'planned_moves': getattr(game, 'planned_moves', []),
                'llm_response': getattr(game, 'processed_response', ''),
                'move_pause': self.game_manager.get_pause_between_moves(),
                'game_end_reason': end_reason_readable,
                'session_finished': not self.game_manager.game_active,
            }

            return build_state_dict(
                snake_positions=game.snake_positions,
                apple_position=game.apple_position,
                score=game.score,
                steps=game.steps,
                grid_size=self.grid_size,
                extra=extra_state,
            )

        except Exception as e:
            print(f"[MainWebApp] Error getting game state: {e}")
            return {
                'mode': 'main',
                'error': f'Error getting game state: {str(e)}',
                'provider': self.provider,
                'model': self.model
            }
    
    def handle_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Controller: Handle control commands from View."""
        action = data.get('action', '')
        
        if action == 'start':
            return self._start_game()
        elif action == 'stop':
            return self._stop_game()
        elif action == 'reset':
            return self._reset_game()
        else:
            return {'status': 'error', 'message': 'Unknown action'}
    
    def handle_move(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Controller: Handle move (not applicable for main mode)."""
        return {
            'status': 'error', 
            'message': 'Moves not applicable in main mode - game is controlled by AI'
        }
    
    def handle_reset(self) -> Dict[str, Any]:
        """Controller: Reset the game session."""
        return self._reset_game()
    
    # =============================================================================
    # Game Control Methods
    # =============================================================================
    
    def _start_game(self) -> Dict[str, Any]:
        """Start the game."""
        if self.game_manager:
            self.game_manager.running = True
            return {'status': 'ok', 'message': 'Game started'}
        return {'status': 'error', 'message': 'GameManager not available'}
    
    def _stop_game(self) -> Dict[str, Any]:
        """Stop the game."""
        if self.game_manager:
            self.game_manager.running = False
            return {'status': 'ok', 'message': 'Game stopped'}
        return {'status': 'error', 'message': 'GameManager not available'}
    
    def _reset_game(self) -> Dict[str, Any]:
        """Reset the game session."""
        if self.game_manager and self.game_manager.game:
            # Reset the current game
            self.game_manager.game.reset()
            return {'status': 'ok', 'message': 'Game reset'}
        return {'status': 'error', 'message': 'GameManager not available'}

