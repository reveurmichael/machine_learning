"""
Main Web Script - Full GameManager Integration
============================================

Script for launching LLM-controlled Snake game with full GameManager integration.
This provides complete LLM functionality with real-time web interface.

Design Philosophy:
- Full Feature Parity: Mirrors all main.py capabilities
- Real-time Integration: Background GameManager with live state updates
- No Over-Preparation: Only implements what's needed for web interface
- Extensible: Template for Tasks 1-5 LLM implementations

Educational Value:
- Shows proper Flask + GameManager integration
- Demonstrates real-time LLM game state synchronization
- Provides template for extension web interfaces

Usage:
    python scripts/main_web.py                          # Default LLM settings
    python scripts/main_web.py --provider deepseek      # Different LLM provider  
    python scripts/main_web.py --model gpt-4 --port 8080  # Custom model and port
    python scripts/main_web.py --max-games 3            # Play multiple games
    python scripts/main_web.py --continue-with-game-in-dir logs/...  # Continue session
"""

import sys
import argparse
import threading
import time
import os
from pathlib import Path

# Ensure project root in sys.path early
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import shared web utility for template/static dirs
from utils.web_utils import get_web_dirs

TEMPLATE_DIR, STATIC_DIR = get_web_dirs()

# Project root already in sys.path

# Import main.py components for full feature parity
from scripts.main import get_parser, parse_arguments
from core.game_manager import GameManager
from llm.agent_llm import SnakeAgent
from utils.web_utils import translate_end_reason, build_color_map, build_state_dict
from utils.network_utils import find_free_port
from utils.validation_utils import validate_port
from config.ui_constants import GRID_SIZE

# Flask imports
from flask import Flask, render_template, jsonify, request

# Global state for cross-request access
manager = None
manager_thread = None
import threading

# Thread safety for shared state access
_state_lock = threading.Lock()


def parse_web_arguments():
    """Parse command line arguments for web mode."""
    parser = argparse.ArgumentParser(description="Snake Game - Full LLM Web Interface")
    
    # Web-specific arguments
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number (default: auto-detect)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)"
    )
    
    # Parse web-specific args first
    web_args, remaining_argv = parser.parse_known_args()
    
    # Now parse all the main.py arguments
    sys.argv_backup = sys.argv.copy()
    sys.argv = [sys.argv[0]] + remaining_argv
    
    try:
        game_args = parse_arguments()
    finally:
        sys.argv = sys.argv_backup
    
    return web_args, game_args


def manager_thread_fn(gm: GameManager, args):
    """Background worker for running the game."""
    try:
        print(f"[MainWebFull] Starting GameManager thread")
        
        # Handle continuation mode
        cont_dir = getattr(args, "continue_with_game_in_dir", None)
        if cont_dir:
            try:
                # Validate continuation directory
                if not os.path.isdir(cont_dir):
                    print(f"[MainWebFull] Error: Continuation directory does not exist: {cont_dir}")
                    return
                
                # Determine next game number for continuation
                from utils.file_utils import get_next_game_number
                next_game = get_next_game_number(cont_dir)
                print(f"[MainWebFull] Continuing from game {next_game} in {cont_dir}")
                
                # Load existing game session
                gm.continue_from_session(cont_dir, next_game)
            except Exception as e:
                print(f"[MainWebFull] Error in continuation mode: {e}")
                # Fall back to new session
                print(f"[MainWebFull] Falling back to new session")
                gm.run()
        else:
            # Start new game session
            print(f"[MainWebFull] Starting new game session")
            gm.run()
            
    except Exception as e:
        print(f"[MainWebFull] GameManager thread crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[MainWebFull] GameManager thread finished")


def build_game_state_dict(gm: GameManager):
    """Convert GameManager state to JSON-serializable dict."""
    game = gm.game
    
    # Get basic state using the utility function
    reason_code = getattr(game.game_state, "game_end_reason", None)
    end_reason_readable = translate_end_reason(reason_code)
    
    # Build extra state for LLM mode
    extra_state = {
        'game_number': gm.game_count + 1,
        'round_count': gm.round_count,
        'running': gm.running,
        'game_active': gm.game_active,
        'planned_moves': getattr(game, 'planned_moves', []),
        'llm_response': getattr(game, 'processed_response', ''),
        'move_pause': gm.get_pause_between_moves(),
        'game_end_reason': end_reason_readable,
    }
    
    # Use the utility function for basic state
    return build_state_dict(
        snake_positions=game.snake_positions,
        apple_position=game.apple_position,
        score=game.score,
        steps=game.steps,
        grid_size=GRID_SIZE,
        extra=extra_state
    )


def main():
    """Main entry point for full LLM web interface."""
    global manager, manager_thread
    
    try:
        print("[MainWebFull] Starting Snake Game - Full LLM Web Interface")
        
        # Parse arguments (web + game)
        web_args, game_args = parse_web_arguments()
        
        # Validate port (simple like human_play_web.py and replay_web.py)
        port = validate_port(web_args.port) if web_args.port else None
        host = web_args.host
        
        # Use the same port allocation logic as other scripts
        if port is None:
            from utils.network_utils import random_free_port
            port = random_free_port()

        # Print startup banner
        print("=" * 60)
        print(f"[MainWebFull] Host: {host}")
        print(f"[MainWebFull] Port: {port}")
        print(f"[MainWebFull] LLM Provider: {getattr(game_args, 'provider', None)}/{getattr(game_args, 'model', None)}")
        print(f"[MainWebFull] Max Games: {getattr(game_args, 'max_games', None)}")
        if getattr(game_args, 'continue_with_game_in_dir', None):
            print(f"[MainWebFull] Continuing from: {game_args.continue_with_game_in_dir}")
        if host == '0.0.0.0':
            print('[MainWebFull] WARNING: Host is 0.0.0.0 (public). The server will be accessible from any device on the network!')
        print("=" * 60)
        
        # Set up pygame dummy driver for headless mode
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        
        # Create Flask app
        app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATE_DIR))
        
        # Store the actual port that will be used (for URL display)
        actual_port = port
        
        # Enable GUI mode for web interface to get the 3-second preview pause
        # This allows users to see the LLM's plan before the snake starts moving
        # The pause occurs in core/game_loop.py when use_gui=True and a new plan is received
        game_args.use_gui = True
        
        # -------------------------------
        # Step 1 – Create GameManager with full CLI support
        # -------------------------------
        print(f"[MainWebFull] Creating GameManager with full integration")
        manager = GameManager(game_args)
        
        # Set up LLM agent for Task-0
        manager.agent = SnakeAgent(
            manager, 
            provider=game_args.provider, 
            model=game_args.model
        )
        
        # -------------------------------
        # Step 2 – Handle continuation mode configuration
        # -------------------------------
        if game_args.continue_with_game_in_dir:
            try:
                import json
                summary_path = os.path.join(game_args.continue_with_game_in_dir, "summary.json")
                if not os.path.exists(summary_path):
                    print(f"[MainWebFull] Warning: No summary.json found in {game_args.continue_with_game_in_dir}")
                else:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    original_cfg = summary.get("configuration", {})
                    
                    # Validate configuration structure
                    if not isinstance(original_cfg, dict):
                        print(f"[MainWebFull] Warning: Invalid configuration format in {summary_path}")
                    else:
                        # Restore configuration with validation
                        restored_keys = []
                        for k in (
                            "provider", "model", "parser_provider", "parser_model",
                            "move_pause", "max_steps", "max_consecutive_empty_moves_allowed",
                            "max_consecutive_something_is_wrong_allowed",
                            "max_consecutive_invalid_reversals_allowed",
                            "max_consecutive_no_path_found_allowed",
                            "sleep_after_empty_step", "no_gui",
                        ):
                            if k in original_cfg:
                                try:
                                    setattr(manager.args, k, original_cfg[k])
                                    restored_keys.append(k)
                                except (TypeError, ValueError) as e:
                                    print(f"[MainWebFull] Warning: Could not restore {k}: {e}")
                        
                        print(f"[MainWebFull] Restored {len(restored_keys)} config keys from {summary_path}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[MainWebFull] Error loading continuation config: {e}")
            except Exception as e:
                print(f"[MainWebFull] Unexpected error in continuation config: {e}")
        
        # -------------------------------
        # Step 3 – Start GameManager in background thread
        # -------------------------------
        manager_thread = threading.Thread(
            target=manager_thread_fn, 
            args=(manager, game_args), 
            daemon=True
        )
        manager_thread.start()
        
        # -------------------------------
        # Step 4 – Define Flask routes
        # -------------------------------
        @app.route('/')
        def index():
            """Serve the main HTML page."""
            return render_template('main.html')
        
        @app.route('/api/state')
        def api_state():
            """Return current game state for frontend polling."""
            with _state_lock:
                if manager is None or manager.game is None:
                    return jsonify({'error': 'game not started'})
                return jsonify(build_game_state_dict(manager))
        
        @app.route('/api/control', methods=['POST'])
        def api_control():
            """Handle control commands (pause/play)."""
            with _state_lock:
                if manager is None:
                    return jsonify({'status': 'error', 'msg': 'no manager'})
                
                cmd = request.json.get('command') if request.is_json else None
                if cmd == 'pause':
                    manager.running = False  # stops outer while loops – this effectively pauses
                    return jsonify({'status': 'paused'})
                if cmd == 'play':
                    # Not trivial to resume once gm.run() has returned, so just acknowledge
                    return jsonify({'status': 'unpause-not-supported'})
                
                return jsonify({'status': 'error', 'msg': 'unknown command'})
        
        # -------------------------------
        # Step 5 – Start the Flask app (blocking)
        # -------------------------------
        print("[MainWebFull] Starting server...")
        print("[MainWebFull] Use the web interface to watch the game in real-time")
        print("[MainWebFull] Note: 3-second preview pause enabled for better UX")
        print("[MainWebFull] Press Ctrl+C to stop")

        try:
            # Start the Flask app and capture the actual port used
            app.run(host=host, port=port, threaded=True, use_reloader=False)
        except OSError as e:
            print(f"[MainWebFull] Error: Could not bind to {host}:{port}: {e}")
            # Try to find a free port and suggest it
            try:
                new_port = find_free_port(port + 1 if port else 8000)
                print(f"[MainWebFull] Suggestion: Try --port {new_port}")
            except:
                pass
            if manager:
                manager.running = False
            if manager_thread and manager_thread.is_alive():
                manager_thread.join(timeout=5)
            return 1
        
    except KeyboardInterrupt:
        print("\n[MainWebFull] Shutting down gracefully...")
        if manager:
            manager.running = False
        if manager_thread and manager_thread.is_alive():
            manager_thread.join(timeout=5)
        print("[MainWebFull] Shutdown complete")
    except Exception as e:
        print(f"[MainWebFull] Error: {e}")
        if manager:
            manager.running = False
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 