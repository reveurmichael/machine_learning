"""
Snake Game - Human Web Interface Script (Enhanced Architecture)
=============================================================

Flask-based web application for human-controlled Snake gameplay using the enhanced
layered web infrastructure. This script demonstrates Task-0's modern web architecture
and serves as the canonical template for all future extension web interfaces.

Enhanced Features:
- Layered Web Infrastructure: BaseWebApp → SimpleFlaskApp → HumanWebGameApp
- Enhanced Naming: Clear, explicit naming throughout the application stack
- Universal Factory Utilities: Uses factory_utils from ROOT/utils/ following SSOT
- Dynamic Port Allocation: Network utilities with conflict resolution
- KISS Principles: Simple, elegant error handling and clean code structure
- Future-Proof Design: Template for Task 1-5 extension web interfaces

Design Patterns (Enhanced):
    - Template Method Pattern: Layered Flask application lifecycle
    - Factory Pattern: Universal factory utilities with canonical create() method
    - Strategy Pattern: Pluggable game modes with enhanced naming
    - Facade Pattern: Simplified web application launcher interface

Educational Goals:
    - Demonstrate enhanced web architecture for Task-0
    - Show layered inheritance patterns for educational value
    - Illustrate enhanced naming conventions for clarity
    - Provide canonical template for extension web interfaces

Extension Template for Future Tasks:
    Task-1 (Heuristics): Copy this structure, replace with HeuristicWebGameApp
    Task-2 (Supervised): Copy this structure, replace with SupervisedWebGameApp
    Task-3 (RL): Copy this structure, replace with RLWebGameApp
    Task-4 (LLM Fine-tuning): Copy this structure, replace with LLMFinetuningWebGameApp
    Task-5 (Distillation): Copy this structure, replace with DistillationWebGameApp

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
from web import HumanWebGameApp, HumanWebAppLauncher
from web.factories import create_human_web_game_app

# Import universal utilities following SSOT principles
from utils.validation_utils import validate_human_web_arguments
from utils.print_utils import create_logger
from config.ui_constants import GRID_SIZE as DEFAULT_GRID_SIZE

# Enhanced logging with consistent naming
print_log = create_logger("HumanWebScript")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for human web game interface.
    
    Educational Value: Shows consistent argument handling with enhanced naming
    Extension Template: Copy this exact pattern for all extension web scripts
    
    Returns:
        Configured argument parser with human-specific options
    """
    parser = argparse.ArgumentParser(
        description="Snake Game - Human Web Game Interface (Enhanced Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:
  python scripts/human_play_web.py                      # Default enhanced settings
  python scripts/human_play_web.py --grid-size 15       # Larger grid with validation
  python scripts/human_play_web.py --port 8080          # Specific port with conflict detection
  python scripts/human_play_web.py --debug              # Debug mode with enhanced logging

Extension Template Pattern:
  Future extensions should copy this script structure:
  1. Import enhanced web infrastructure: {ExtensionType}WebGameApp
  2. Use universal factory functions: create_{extension}_web_game_app()
  3. Apply enhanced naming conventions throughout
  4. Leverage universal validation utilities
  5. Maintain same elegant launcher pattern

Web Architecture Benefits:
  - Layered inheritance: BaseWebApp → SimpleFlaskApp → HumanWebGameApp
  - Enhanced naming: Clear domain indication (Human + Web + Game + App)
  - Universal utilities: Validation, logging, factory patterns from SSOT
  - Educational clarity: Self-documenting enhanced architecture
        """
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Size of the game grid with validation (default: {DEFAULT_GRID_SIZE})"
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


def create_human_web_application(args) -> HumanWebGameApp:
    """
    Create human web game application using enhanced factory function.
    
    Args:
        args: Validated command line arguments
        
    Returns:
        Configured human web game application
        
    Educational Value: Shows enhanced factory pattern usage with universal utilities
    Extension Template: Copy this creation pattern for all extension applications
    """
    print_log("Creating human web game application using enhanced factory...")
    
    # Use enhanced factory function with universal utilities
    app = create_human_web_game_app(
        grid_size=args.grid_size,
        port=args.port
    )
    
    print_log(f"Human web game app created: {app.name} on port {app.port}")
    return app


def display_application_info(app: HumanWebGameApp, host: str) -> None:
    """
    Display enhanced application information with clear formatting.
    
    Args:
        app: Human web game application
        host: Server host address
        
    Educational Value: Shows enhanced naming and information display patterns
    Extension Template: Copy this display pattern for all extension applications
    """
    print_log("🎯 Enhanced Application Information:")
    print_log(f"   Application: {app.name}")
    print_log(f"   Architecture: Enhanced Layered Web Infrastructure")
    print_log(f"   Type: Human Web Game App")
    print_log(f"   Grid Size: {app.grid_size}x{app.grid_size}")
    print_log(f"   Port: {app.port} (with conflict detection)")
    print_log(f"   URL: http://{host}:{app.port}")
    print_log("")
    
    print_log("🎮 Enhanced Web Controls:")
    print_log("   Arrow Keys: Move snake")
    print_log("   R: Reset game")  
    print_log("   Space: Pause/Resume")
    print_log("   Ctrl+C: Stop server")
    print_log("")
    
    print_log("🏗️ Architecture Layers:")
    print_log("   BaseWebApp → SimpleFlaskApp → HumanWebGameApp")
    print_log("   Universal utilities from ROOT/utils/ (SSOT compliance)")
    print_log("   Enhanced naming for educational clarity")
    print_log("")


def main() -> int:
    """
    Main entry point for human web game interface script.
    
    Educational Value: Shows elegant application launcher lifecycle with enhanced architecture
    Extension Template: Copy this exact main() pattern for all extension web scripts
    
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    try:
        print_log("🐍 Starting Snake Game - Human Web Interface (Enhanced)")
        print_log("📊 Architecture: Enhanced Layered Web Infrastructure")
        print_log("🎯 Mode: Human Player with Enhanced Naming")
        print_log("")
        
        # Parse and validate command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        validate_human_web_arguments(args)
        
        # Create enhanced human web application
        app = create_human_web_application(args)
        
        # Display enhanced application information
        display_application_info(app, args.host)
        
        print_log("🚀 Extension Template Information:")
        print_log("   This script demonstrates enhanced web architecture")
        print_log("   Copy this structure for all extension web interfaces")
        print_log("   Replace HumanWebGameApp with {Extension}WebGameApp")
        print_log("   Maintain enhanced naming and layered architecture")
        print_log("")
        
        # Start the enhanced web application
        print_log("✅ Starting enhanced web server...")
        app.run(host=args.host, port=app.port, debug=args.debug)
        
        return 0
        
    except KeyboardInterrupt:
        print_log("🛑 Server stopped by user")
        return 0
    except Exception as e:
        print_log(f"❌ Failed to start human web interface: {e}")
        print_log("   Check enhanced error handling and universal validation")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
