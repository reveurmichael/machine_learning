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

# Import MainWebApp from web/main_app.py
from web.main_app import MainWebApp


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


def main():
    """Main entry point for full LLM web interface."""
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
        
        # -------------------------------
        # Create and launch MainWebApp (clean reuse of web/main_app.py)
        # -------------------------------
        web_app = MainWebApp(
            provider=game_args.provider,
            model=game_args.model,
            grid_size=GRID_SIZE,
            max_games=game_args.max_games,
            port=port,
            continue_from_folder=game_args.continue_with_game_in_dir,
            no_gui=game_args.no_gui,
            game_args=game_args,
        )

        # Start the web application (blocking call)
        print("[MainWebFull] Starting server…")
        web_app.run(host=host, port=port)

        return 0
        
    except KeyboardInterrupt:
        print("\n[MainWebFull] Shutting down gracefully...")
        print("[MainWebFull] Shutdown complete")
    except Exception as e:
        print(f"[MainWebFull] Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 