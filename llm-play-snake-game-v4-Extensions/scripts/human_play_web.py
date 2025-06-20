"""
Snake Game - Human Player Web Interface

A Flask-based web application that provides an interactive Snake game experience
for human players through a browser interface. This module offers:

- Full web-based Snake gameplay with keyboard controls
- Real-time game state synchronization via REST API
- Responsive game board rendering
- Score tracking and game statistics
- Comprehensive input validation and error handling
- Cross-platform browser compatibility

Technical Architecture:
- Flask web server for HTTP interface and static file serving
- WebGameController extends core GameController for web compatibility
- Real-time state updates through AJAX polling
- RESTful API for game control and move input
- Thread-safe game state management

This module provides a browser-based alternative to the pygame desktop version,
making the Snake game accessible through any modern web browser without
requiring local pygame installation.
"""

from __future__ import annotations

import sys
import pathlib
from typing import Any, Dict, Optional, Union

# Bootstrap repository root for consistent imports
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.path_utils import ensure_project_root

# Ensure consistent working directory
ensure_project_root()

# Standard library imports
import argparse
import logging
import threading
import time
from contextlib import contextmanager

# Third-party imports
from flask import Flask, render_template, request, jsonify
from flask.typing import ResponseReturnValue

# Project imports
from config.web_constants import FLASK_STATIC_FOLDER, FLASK_TEMPLATE_FOLDER
from core.game_controller import GameController
from utils.network_utils import find_free_port
from utils.web_utils import build_state_dict, translate_end_reason, create_health_check_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress Flask's verbose request logging
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Flask application setup
app = Flask(__name__, static_folder=FLASK_STATIC_FOLDER, template_folder=FLASK_TEMPLATE_FOLDER)

# Global state management
_game_controller: Optional["WebGameController"] = None
_heartbeat_thread: Optional[threading.Thread] = None
_shutdown_event = threading.Event()

# Game configuration constants
DEFAULT_GRID_SIZE = 10
VALID_DIRECTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}
HEARTBEAT_INTERVAL = 0.1  # 100ms heartbeat for responsiveness


class WebGameController(GameController):
    """
    Game controller optimized for web-based human interaction.
    
    Extends the core GameController with web-specific functionality including
    JSON serialization, enhanced error handling, thread-safe state access,
    and comprehensive game statistics tracking.
    
    Features:
    - Thread-safe game state management
    - JSON-serializable state representation
    - Enhanced input validation for web contexts
    - Real-time performance monitoring
    - Graceful error recovery and reporting
    """

    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE) -> None:
        """
        Initialize the web game controller.
        
        Args:
            grid_size: Size of the game grid (grid_size x grid_size).
        """
        # Initialize parent without GUI since we use web interface
        super().__init__(grid_size=grid_size, use_gui=False)
        
        # Web-specific state management
        self._state_lock = threading.RLock()
        self._last_move_time = time.time()
        self._error_count = 0
        self._max_errors = 10
        
        # Game state tracking
        self.grid_size = grid_size
        self.game_over = False
        self.game_end_reason: Optional[str] = None
        self._move_count = 0
        self._game_start_time = time.time()
        
        # Performance statistics
        self._performance_stats = {
            "moves_per_second": 0.0,
            "avg_response_time": 0.0,
            "total_moves": 0,
            "game_duration": 0.0,
        }

    @contextmanager
    def _thread_safe_access(self):
        """Context manager for thread-safe game state access."""
        with self._state_lock:
            yield

    def get_current_state(self) -> Dict[str, Any]:
        """
        Build comprehensive state dictionary for web clients.
        
        Returns a JSON-serializable dictionary containing all necessary
        game state information including position data, scores, game status,
        and performance metrics.
        
        Returns:
            Dictionary with complete game state for web rendering.
            
        Raises:
            RuntimeError: If game state is corrupted or inaccessible.
        """
        try:
            with self._thread_safe_access():
                # Build base state using utility function
                state = build_state_dict(
                    snake_positions=self.snake_positions,
                    apple_position=self.apple_position,
                    score=self.score,
                    steps=self.steps,
                    grid_size=self.grid_size,
                    extra={
                        # Game status
                        "game_over": self.game_over,
                        "game_end_reason": translate_end_reason(self.game_end_reason),
                        "game_active": not self.game_over,
                        
                        # Session statistics
                        "move_count": self._move_count,
                        "game_duration": time.time() - self._game_start_time,
                        
                        # System status
                        "timestamp": time.time(),
                        "error_count": self._error_count,
                        "performance": self._performance_stats.copy(),
                        
                        # Input validation
                        "valid_directions": list(VALID_DIRECTIONS),
                        "last_move_time": self._last_move_time,
                    }
                )
                
                # Update performance tracking
                self._update_performance_stats()
                
                return state
                
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error building game state: {e}")
            
            # Return safe fallback state
            return {
                "error": f"State access failed: {str(e)}",
                "error_count": self._error_count,
                "timestamp": time.time(),
                "grid_size": self.grid_size,
                "game_over": True,
                "score": getattr(self, "score", 0),
                "steps": getattr(self, "steps", 0),
            }

    def make_move(self, direction_key: str) -> tuple[bool, bool]:
        """
        Execute a game move with enhanced validation and error handling.
        
        Args:
            direction_key: Movement direction ("UP", "DOWN", "LEFT", "RIGHT").
            
        Returns:
            Tuple of (game_still_active, apple_eaten).
            
        Raises:
            ValueError: If direction is invalid.
        """
        try:
            with self._thread_safe_access():
                # Validate direction
                if direction_key not in VALID_DIRECTIONS:
                    raise ValueError(f"Invalid direction: {direction_key}")
                
                # Prevent moves on already finished games
                if self.game_over:
                    logger.warning(f"Move attempted on finished game: {direction_key}")
                    return False, False
                
                # Execute move through parent class
                game_active, apple_eaten = super().make_move(direction_key)
                
                # Update game state tracking
                self._move_count += 1
                self._last_move_time = time.time()
                
                # Check for game termination
                if not game_active:
                    self.game_over = True
                    collision_type = getattr(self, 'last_collision_type', None)
                    
                    if collision_type == "WALL":
                        self.game_end_reason = "WALL"
                    elif collision_type == "SELF":
                        self.game_end_reason = "SELF"
                    else:
                        self.game_end_reason = "UNKNOWN"
                    
                    logger.info(f"Game ended: {self.game_end_reason}, Score: {self.score}")
                
                return game_active, apple_eaten
                
        except Exception as e:
            self._error_count += 1
            logger.error(f"Move execution failed: {e}")
            self.game_over = True
            self.game_end_reason = "ERROR"
            return False, False

    def reset(self) -> None:
        """
        Reset the game to initial state with comprehensive cleanup.
        """
        try:
            with self._thread_safe_access():
                # Reset parent game state
                super().reset()
                
                # Reset web-specific state
                self.game_over = False
                self.game_end_reason = None
                self._move_count = 0
                self._game_start_time = time.time()
                self._last_move_time = time.time()
                
                # Reset performance statistics
                self._performance_stats = {
                    "moves_per_second": 0.0,
                    "avg_response_time": 0.0,
                    "total_moves": 0,
                    "game_duration": 0.0,
                }
                
                logger.info("Game reset successfully")
                
        except Exception as e:
            self._error_count += 1
            logger.error(f"Game reset failed: {e}")

    def _update_performance_stats(self) -> None:
        """Update internal performance tracking metrics."""
        try:
            current_time = time.time()
            game_duration = current_time - self._game_start_time
            
            # Update basic statistics
            self._performance_stats["total_moves"] = self._move_count
            self._performance_stats["game_duration"] = game_duration
            
            # Calculate moves per second
            if game_duration > 0:
                self._performance_stats["moves_per_second"] = self._move_count / game_duration
            
            # Simple moving average for response time
            time_since_last = current_time - self._last_move_time
            if time_since_last > 0:
                response_time = time_since_last * 1000  # Convert to milliseconds
                current_avg = self._performance_stats["avg_response_time"]
                self._performance_stats["avg_response_time"] = (
                    (current_avg * 0.9) + (response_time * 0.1)
                )
            
        except Exception as e:
            logger.warning(f"Performance stats update failed: {e}")

    def handle_error(self, error: Exception) -> None:
        """
        Enhanced error handling for web contexts.
        
        Args:
            error: Exception that occurred during game execution.
        """
        self._error_count += 1
        logger.error(f"Game controller error ({self._error_count}): {error}")
        
        if self._error_count >= self._max_errors:
            logger.critical("Maximum error count reached, marking game as over")
            self.game_over = True
            self.game_end_reason = "SYSTEM_ERROR"


def _heartbeat_loop() -> None:
    """
    Background heartbeat to keep the application responsive.
    
    This function maintains a minimal background thread to ensure the
    Python interpreter remains active and responsive to web requests,
    particularly important for pygame integration.
    """
    try:
        logger.info("Starting heartbeat thread")
        
        while not _shutdown_event.is_set():
            time.sleep(HEARTBEAT_INTERVAL)
        
    except Exception as e:
        logger.error(f"Heartbeat thread error: {e}")
    finally:
        logger.info("Heartbeat thread finished")


# -------------------
# Flask Routes
# -------------------

@app.route("/")
def index() -> str:
    """Serve the main human play interface."""
    return render_template("human_play.html")


@app.route("/api/state")
def api_get_state() -> ResponseReturnValue:
    """
    API endpoint to retrieve current game state.
    
    Returns:
        JSON response with current game state or error information.
    """
    global _game_controller
    
    if _game_controller is None:
        return jsonify({
            "error": "Game controller not initialized",
            "timestamp": time.time(),
            "status": "not_ready"
        })
    
    return jsonify(_game_controller.get_current_state())


@app.route("/api/move", methods=["POST"])
def api_move() -> ResponseReturnValue:
    """
    API endpoint for player move input.
    
    Expected JSON payload:
    {
        "direction": "UP|DOWN|LEFT|RIGHT"
    }
    
    Returns:
        JSON response with move result and updated game state.
    """
    global _game_controller
    
    if _game_controller is None:
        return jsonify({
            "status": "error",
            "message": "Game controller not available",
            "timestamp": time.time()
        })
    
    try:
        # Parse and validate request
        data = request.get_json(silent=True) or {}
        direction = data.get("direction", "").upper()
        
        if not direction:
            return jsonify({
                "status": "error",
                "message": "Direction parameter required",
                "valid_directions": list(VALID_DIRECTIONS),
                "timestamp": time.time()
            })
        
        if direction not in VALID_DIRECTIONS:
            return jsonify({
                "status": "error",
                "message": f"Invalid direction: {direction}",
                "valid_directions": list(VALID_DIRECTIONS),
                "timestamp": time.time()
            })
        
        # Execute move
        game_active, apple_eaten = _game_controller.make_move(direction)
        
        # Build response with comprehensive game state
        response_data = {
            "status": "success",
            "message": f"Move {direction} executed",
            "game_active": game_active,
            "apple_eaten": apple_eaten,
            "score": _game_controller.score,
            "steps": _game_controller.steps,
            "game_over": _game_controller.game_over,
            "timestamp": time.time()
        }
        
        # Add game end reason if applicable
        if _game_controller.game_over and _game_controller.game_end_reason:
            response_data["game_end_reason"] = translate_end_reason(
                _game_controller.game_end_reason
            )
        
        return jsonify(response_data)
        
    except ValueError as e:
        logger.warning(f"Invalid move request: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "valid_directions": list(VALID_DIRECTIONS),
            "timestamp": time.time()
        })
    
    except Exception as e:
        logger.error(f"Move API error: {e}")
        if _game_controller:
            _game_controller.handle_error(e)
        return jsonify({
            "status": "error",
            "message": f"Move execution failed: {str(e)}",
            "timestamp": time.time()
        })


@app.route("/api/reset", methods=["POST"])
def api_reset() -> ResponseReturnValue:
    """
    API endpoint to reset the game to initial state.
    
    Returns:
        JSON response with reset confirmation and new game state.
    """
    global _game_controller
    
    if _game_controller is None:
        return jsonify({
            "status": "error",
            "message": "Game controller not available",
            "timestamp": time.time()
        })
    
    try:
        _game_controller.reset()
        logger.info("Game reset via web API")
        
        return jsonify({
            "status": "success",
            "message": "Game reset successfully",
            "score": _game_controller.score,
            "steps": _game_controller.steps,
            "game_over": _game_controller.game_over,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Reset API error: {e}")
        if _game_controller:
            _game_controller.handle_error(e)
        return jsonify({
            "status": "error",
            "message": f"Game reset failed: {str(e)}",
            "timestamp": time.time()
        })


@app.route("/api/stats")
def api_stats() -> ResponseReturnValue:
    """
    API endpoint to retrieve detailed game statistics.
    
    Returns:
        JSON response with comprehensive game statistics and performance metrics.
    """
    global _game_controller
    
    if _game_controller is None:
        return jsonify({
            "error": "Game controller not available",
            "timestamp": time.time()
        })
    
    try:
        with _game_controller._thread_safe_access():
            stats = {
                # Basic game statistics
                "current_score": _game_controller.score,
                "current_steps": _game_controller.steps,
                "move_count": _game_controller._move_count,
                "game_active": not _game_controller.game_over,
                
                # Timing information
                "game_duration": time.time() - _game_controller._game_start_time,
                "last_move_time": _game_controller._last_move_time,
                
                # Performance metrics
                "performance": _game_controller._performance_stats.copy(),
                
                # System information
                "error_count": _game_controller._error_count,
                "grid_size": _game_controller.grid_size,
                "timestamp": time.time(),
            }
            
            return jsonify(stats)
            
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            "error": f"Statistics retrieval failed: {str(e)}",
            "timestamp": time.time()
        })


@app.route("/api/health")
def api_health() -> ResponseReturnValue:
    """
    Health check endpoint for monitoring and diagnostics.
    
    Returns:
        JSON response with standardized system health status and performance metrics.
    """
    global _game_controller, _heartbeat_thread
    
    # Use centralized health check with component mapping
    components = {
        "web_server": app,  # Flask app is always operational if we reach this point
        "game_controller": _game_controller,
        "heartbeat_thread": _heartbeat_thread,
    }
    
    health_response = create_health_check_response(components, error_threshold=5)
    
    # Add game-specific performance metrics if available
    if _game_controller and hasattr(_game_controller, "_performance_stats"):
        try:
            health_response["performance"] = _game_controller._performance_stats.copy()
        except Exception as e:
            logger.warning(f"Failed to get game controller stats: {e}")
    
    return jsonify(health_response)


# -------------------
# Argument Parser and Configuration
# -------------------

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create modern, future-proof argument parser for web human play mode.
    
    Designed for clarity, usability, and extensibility without legacy constraints.
    
    Returns:
        Configured argument parser with intuitive, well-organized options.
    """
    parser = argparse.ArgumentParser(
        description="Web Interface for Human Snake Game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎮 Examples:
  %(prog)s
  %(prog)s --size 15 --port 8080
  %(prog)s --size 20 --host 0.0.0.0 --debug
  %(prog)s --size 12 --theme dark

🎯 Game Controls (in browser):
  Arrow Keys/WASD - Move snake
  R               - Reset game
  Space           - Pause/Resume

🎨 Customization:
  Grid sizes from 8x8 to 25x25 supported
  Responsive design adapts to screen size
        """
    )
    
    # -------------------
    # Game Configuration
    # -------------------
    game_group = parser.add_argument_group("🐍 Game Settings")
    game_group.add_argument(
        "--size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        choices=range(8, 26),
        help=f"Game grid size (8-25, default: {DEFAULT_GRID_SIZE})"
    )
    game_group.add_argument(
        "--theme",
        type=str,
        choices=["classic", "modern", "dark", "neon"],
        default="modern",
        help="Visual theme for the game (default: %(default)s)"
    )
    game_group.add_argument(
        "--show-stats",
        action="store_true",
        default=True,
        help="Display real-time game statistics (default: enabled)"
    )
    
    # -------------------
    # Web Server Configuration
    # -------------------
    web_group = parser.add_argument_group("🌐 Web Server")
    web_group.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host IP address for web server (default: %(default)s)"
    )
    web_group.add_argument(
        "--port",
        type=int,
        default=find_free_port(8000),
        help="Port number for web server (default: auto-selected)"
    )
    web_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode (development only)"
    )
    web_group.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open browser when server starts"
    )
    
    return parser


def _convert_to_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Convert modern argument structure to internal format expected by game components.
    
    Args:
        args: Modern argument namespace
        
    Returns:
        Converted namespace for internal use
    """
    # Map modern names to internal names
    args.grid_size = args.size
    
    return args


# -------------------
# Main Entry Point
# -------------------

def main() -> None:
    """
    Main entry point for the web-based human Snake game.
    
    Initializes the Flask web server, sets up the game controller,
    and serves the interactive web interface for human gameplay.
    """
    global _game_controller, _heartbeat_thread
    
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Convert modern arguments to internal format
        args = _convert_to_legacy_args(args)
        
        logger.info("Initializing web-based Snake game for human players")
        
        # Initialize game controller
        _game_controller = WebGameController(grid_size=args.grid_size)
        
        # Start background heartbeat thread
        _heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            daemon=True,
            name="HeartbeatThread"
        )
        _heartbeat_thread.start()
        
        # Start web server
        host = args.host
        port = args.port
        
        print(f"\n🐍 Snake Game - Human Player Web Interface")
        print(f"🔗 URL: http://{host}:{port}")
        print(f"🎮 Grid Size: {args.grid_size}x{args.grid_size}")
        print(f"📊 API endpoints: /api/state, /api/move, /api/reset, /api/stats, /api/health")
        print(f"⌨️  Press Ctrl+C to stop")
        print()
        print("🎯 How to Play:")
        print("   🡸🡹🡺🡻 Use arrow keys or WASD to move")
        print("   🔄 Press R to reset the game")
        print("   🎯 Eat apples to grow and increase your score")
        print("   ⚠️  Avoid walls and your own body")
        print()
        
        # Auto-open browser if requested
        if args.open_browser:
            import webbrowser
            def open_browser():
                time.sleep(1.5)  # Give server time to start
                webbrowser.open(f"http://{host}:{port}")
            threading.Thread(target=open_browser, daemon=True).start()
        
        # Run Flask application
        app.run(
            host=host,
            port=port,
            debug=args.debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down web game interface...")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        _shutdown_event.set()
        
        if _heartbeat_thread and _heartbeat_thread.is_alive():
            logger.info("Waiting for heartbeat thread to finish...")
            _heartbeat_thread.join(timeout=1.0)
        
        logger.info("Web game interface shutdown complete")


if __name__ == "__main__":
    main() 
