"""
Snake Game - Replay Web Interface Script (Enhanced Architecture)
==============================================================

Flask-based web application for game replay using the enhanced layered web
infrastructure. This script demonstrates Task-0's modern replay architecture
and serves as the canonical template for all future extension replay interfaces.

Enhanced Features:
- Layered Web Infrastructure: BaseWebApp → BaseReplayApp → ReplayWebGameApp
- Enhanced Naming: Clear, explicit naming throughout the replay application stack
- Universal Factory Utilities: Uses factory_utils from ROOT/utils/ following SSOT
- Dynamic Port Allocation: Network utilities with conflict resolution
- Replay Engine Integration: Same ReplayEngine as CLI scripts for consistency
- Future-Proof Design: Template for Task 1-5 extension replay interfaces

Design Patterns (Enhanced):
    - Template Method Pattern: Layered replay application lifecycle
    - Factory Pattern: Universal factory utilities with canonical create() method
    - Adapter Pattern: ReplayEngine integration with web interface
    - Facade Pattern: Simplified replay application launcher interface

Educational Goals:
    - Demonstrate enhanced replay architecture for Task-0
    - Show layered inheritance patterns for replay functionality
    - Illustrate ReplayEngine integration with web infrastructure
    - Provide canonical template for extension replay interfaces

Extension Template for Future Tasks:
    Task-1 (Heuristics): Copy this structure, replace with HeuristicReplayWebGameApp
    Task-2 (Supervised): Copy this structure, replace with SupervisedReplayWebGameApp
    Task-3 (RL): Copy this structure, replace with RLReplayWebGameApp
    Task-4 (LLM Fine-tuning): Copy this structure, replace with LLMFinetuningReplayWebGameApp
    Task-5 (Distillation): Copy this structure, replace with DistillationReplayWebGameApp

Reference: docs/extensions-guideline/mvc-flask-factory.md for comprehensive patterns
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Ensure project root for consistent imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import enhanced web infrastructure with layered architecture
from web import ReplayWebGameApp, ReplayWebAppLauncher
from web.factories import create_replay_web_game_app

# Import universal utilities following SSOT principles
from utils.validation_utils import validate_replay_web_arguments
from utils.print_utils import create_logger

# Enhanced logging with consistent naming
print_log = create_logger("ReplayWebScript")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for replay web game interface.
    
    Educational Value: Shows consistent argument handling with enhanced naming
    Extension Template: Copy this exact pattern for all extension replay scripts
    
    Returns:
        Configured argument parser with replay-specific options
    """
    parser = argparse.ArgumentParser(
        description="Snake Game - Replay Web Game Interface (Enhanced Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:
  python scripts/replay_web.py logs/session_20250101          # Basic enhanced replay
  python scripts/replay_web.py logs/session_20250101 --game 5 # Start from game 5 with validation
  python scripts/replay_web.py logs/session_20250101 --port 8080 # Specific port with conflict detection
  python scripts/replay_web.py logs/session_20250101 --debug  # Debug mode with enhanced logging

Extension Template Pattern:
  Future extensions should copy this replay script structure:
  1. Import enhanced replay infrastructure: {ExtensionType}ReplayWebGameApp
  2. Use universal factory functions: create_{extension}_replay_web_game_app()
  3. Apply enhanced naming conventions throughout
  4. Leverage universal validation utilities
  5. Maintain same elegant replay launcher pattern

Replay Architecture Benefits:
  - Layered inheritance: BaseWebApp → BaseReplayApp → ReplayWebGameApp
  - Enhanced naming: Clear domain indication (Replay + Web + Game + App)
  - ReplayEngine integration: Same engine as CLI scripts for consistency
  - Educational clarity: Self-documenting enhanced replay architecture

Replay Features:
  - Step-by-step game playback with enhanced controls
  - Navigation controls (play/pause/seek) with improved UX
  - Game state inspection with detailed information
  - Performance analysis with enhanced metrics
        """
    )
    
    parser.add_argument(
        "log_dir",
        type=str,
        help="Directory containing game logs to replay"
    )
    
    parser.add_argument(
        "--game",
        type=int,
        default=1,
        help="Game number to start replay from (default: 1)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for the web server (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number with conflict detection (default: auto-detect free port)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode with enhanced logging"
    )
    
    return parser


def create_replay_web_application(args) -> ReplayWebGameApp:
    """
    Create replay web game application using enhanced factory function.
    
    Args:
        args: Validated command line arguments
        
    Returns:
        Configured replay web game application
        
    Educational Value: Shows enhanced factory pattern usage for replay applications
    Extension Template: Copy this creation pattern for all extension replay applications
    """
    print_log("Creating replay web game application using enhanced factory...")
    
    # Use enhanced factory function with universal utilities
    app = create_replay_web_game_app(
        log_dir=args.log_dir,
        game_number=args.game,
        port=args.port
    )
    
    print_log(f"Replay web game app created: {app.name} on port {app.port}")
    return app


def display_application_info(app: ReplayWebGameApp, host: str) -> None:
    """
    Display enhanced application information with clear formatting.
    
    Args:
        app: Replay web game application
        host: Server host address
        
    Educational Value: Shows enhanced naming and information display for replay
    Extension Template: Copy this display pattern for all extension replay applications
    """
    print_log("🎯 Enhanced Replay Application Information:")
    print_log(f"   Application: {app.name}")
    print_log(f"   Architecture: Enhanced Layered Replay Infrastructure")
    print_log(f"   Type: Replay Web Game App")
    print_log(f"   Log Directory: {app.log_dir}")
    print_log(f"   Starting Game: {app.game_number}")
    print_log(f"   Port: {app.port} (with conflict detection)")
    print_log(f"   URL: http://{host}:{app.port}")
    print_log("")
    
    print_log("🎮 Enhanced Replay Controls:")
    print_log("   Play/Pause: Control replay playback")
    print_log("   Previous/Next: Navigate between games")
    print_log("   Step Forward/Back: Frame-by-frame navigation")
    print_log("   Speed Control: Adjust playback speed")
    print_log("   Reset: Restart current game replay")
    print_log("   Ctrl+C: Stop server")
    print_log("")
    
    print_log("🏗️ Replay Architecture Layers:")
    print_log("   BaseWebApp → BaseReplayApp → ReplayWebGameApp")
    print_log("   ReplayEngine integration (same as CLI scripts)")
    print_log("   Universal utilities from ROOT/utils/ (SSOT compliance)")
    print_log("   Enhanced naming for educational clarity")
    print_log("")


def main() -> int:
    """
    Main entry point for replay web game interface script.
    
    Educational Value: Shows elegant replay application launcher lifecycle
    Extension Template: Copy this exact main() pattern for all extension replay scripts
    
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    try:
        print_log("🎬 Starting Snake Game - Replay Web Interface (Enhanced)")
        print_log("📊 Architecture: Enhanced Layered Replay Infrastructure")
        print_log("🎯 Mode: Game Replay with Enhanced Naming")
        print_log("")
        
        # Parse and validate command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        validate_replay_web_arguments(args)
        
        # Create enhanced replay web application
        app = create_replay_web_application(args)
        
        # Display enhanced application information
        display_application_info(app, args.host)
        
        print_log("🚀 Extension Template Information:")
        print_log("   This script demonstrates enhanced replay architecture")
        print_log("   Copy this structure for all extension replay interfaces")
        print_log("   Replace ReplayWebGameApp with {Extension}ReplayWebGameApp")
        print_log("   Maintain enhanced naming and layered replay architecture")
        print_log("")
        
        # Start the enhanced replay web application
        print_log("✅ Starting enhanced replay web server...")
        app.run(host=args.host, port=app.port, debug=args.debug)
        
        return 0
        
    except KeyboardInterrupt:
        print_log("🛑 Replay server stopped by user")
        return 0
    except Exception as e:
        print_log(f"❌ Failed to start replay web interface: {e}")
        print_log("   Check enhanced error handling and replay validation")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
