"""
Replay Web Script - Simple Flask Interface
==========================================

Simple script to launch Snake game replay web interface.
Uses simplified web architecture following KISS principles.

Design Philosophy:
- KISS: Simple, direct script without over-engineering
- DRY: Minimal code duplication
- No Over-Preparation: Only what's needed for replay
- Extensible: Easy for Tasks 1-5 to copy and modify

Usage:
    python scripts/replay_web.py logs/session_dir              # Basic replay
    python scripts/replay_web.py logs/session_dir --game 3     # Specific game
    python scripts/replay_web.py logs/session_dir --port 8080  # Specific port
"""

import sys
import argparse
import os
from pathlib import Path

# Ensure project root is on sys.path before importing project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.web_utils import get_web_dirs  # Ensures project root on sys.path

# Ensure project root and prepare template/static dirs (even if not directly used here)
TEMPLATE_DIR, STATIC_DIR = get_web_dirs()

from web import create_replay_web_app
from utils.validation_utils import validate_port


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Snake Game - Replay Web Interface")
    
    parser.add_argument(
        "log_dir",
        type=str,
        help="Directory containing game logs"
    )
    
    parser.add_argument(
        "--game",
        type=int,
        default=1,
        help="Game number to replay (default: 1)"
    )
    
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
    
    return parser.parse_args()


def main():
    """Main entry point for replay web interface."""
    try:
        print("[ReplayWeb] Starting Snake Game - Replay Web Interface")
        
        # Parse arguments
        args = parse_arguments()
        
        # Validate log directory
        if not os.path.isdir(args.log_dir):
            print(f"[ReplayWeb] Error: Directory not found: {args.log_dir}")
            return 1
        
        # Validate port
        port = validate_port(args.port) if args.port else None
        
        # Create and run app
        app = create_replay_web_app(
            log_dir=args.log_dir,
            game_number=args.game,
            port=port
        )
        
        print(f"[ReplayWeb] Server starting on http://{args.host}:{app.port}")
        print(f"[ReplayWeb] Replaying game {args.game} from {args.log_dir}")
        print("[ReplayWeb] Press Ctrl+C to stop")
        
        app.run(host=args.host)
        
    except KeyboardInterrupt:
        print("\n[ReplayWeb] Server stopped by user")
    except Exception as e:
        print(f"[ReplayWeb] Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
