"""
Web Application Entry Points with Enhanced Naming
================================================

Application entry point classes for command-line usage of web interfaces.
Follows patterns from scripts/main.py and scripts/replay.py for consistency.
Uses enhanced naming for maximum clarity and consistency.

Design Philosophy:
- Consistency: Same patterns as CLI application entry points with enhanced naming
- OOP: Object-oriented application classes with clear lifecycle methods
- Argument Integration: Proper integration with argument parsing from universal utilities
- Enhanced Naming: Clear, explicit naming that indicates web app launcher purpose
- Educational: Shows application entry point patterns with enhanced clarity

Educational Value:
- Shows OOP application design for web interfaces with enhanced naming
- Demonstrates argument parsing integration with universal validation utilities
- Provides template for extension application entry points with clear inheritance

Extension Pattern:
Extensions can copy these enhanced application classes and modify for their specific
needs while maintaining consistent argument handling and setup patterns.

Reference: utils/validation_utils.py for universal validation utilities
"""

import argparse
from typing import Optional, Any

# Import enhanced factory functions following enhanced naming
from web.factories import create_human_web_game_app, create_llm_web_game_app, create_replay_web_game_app

# Import utilities following SSOT principles
from utils.validation_utils import validate_grid_size, validate_port
from utils.print_utils import create_logger

# Import configuration constants
from config.web_constants import DEFAULT_HOST, DEFAULT_PORT_RANGE_START, DEFAULT_PORT_RANGE_END

# Create logger for this module  
print_log = create_logger("WebAppLaunchers")


# Argument parsers (inlined for KISS principle)
# -----------------------------------------------

def parse_human_arguments() -> Any:
    """Parse command line arguments for human web interface.
    
    Educational Value: Shows standardized argument parsing with enhanced naming
    Extension Pattern: Extensions can copy this parsing approach
    """
    parser = argparse.ArgumentParser(
        description="Snake Game - Human Web Game Interface"
    )
    parser.add_argument("--grid-size", type=int, default=10, help="Size of the game grid (default: 10)")
    parser.add_argument("--port", type=int, default=None, help="Port number (default: auto-detect)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    return parser.parse_args()


def parse_llm_arguments() -> Any:
    """Parse command line arguments for LLM web interface.
    
    Educational Value: Shows LLM-specific argument parsing with enhanced naming
    Extension Pattern: Extensions can copy this for their LLM interfaces
    """
    parser = argparse.ArgumentParser(
        description="Snake Game - LLM Web Game Interface"
    )
    parser.add_argument("--provider", type=str, default="hunyuan", help="LLM provider (default: hunyuan)")
    parser.add_argument("--model", type=str, default="hunyuan-turbos-latest", help="LLM model")
    parser.add_argument("--grid-size", type=int, default=10, help="Size of the game grid (default: 10)")
    parser.add_argument("--port", type=int, default=None, help="Port number (default: auto-detect)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    return parser.parse_args()


def parse_replay_arguments() -> Any:
    """Parse command line arguments for replay web interface.
    
    Educational Value: Shows replay-specific argument parsing with enhanced naming
    Extension Pattern: Extensions can copy this for their replay interfaces
    """
    parser = argparse.ArgumentParser(
        description="Snake Game - Replay Web Game Interface"
    )
    parser.add_argument("log_dir", type=str, help="Directory containing game logs")
    parser.add_argument("--game", type=int, default=1, help="Game number to replay (default: 1)")
    parser.add_argument("--port", type=int, default=None, help="Port number (default: auto-detect)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    return parser.parse_args()


# Application Launcher Classes with Enhanced Naming
# -------------------------------------------------


class HumanWebAppLauncher:
    """
    Human web application launcher with enhanced naming.
    
    Following the pattern from scripts/main.py MainApplication class,
    this provides an OOP wrapper for the human web application.
    Enhanced naming clearly indicates this is a launcher for human web apps.
    
    Design Pattern: Facade Pattern (Web App Launcher)
    Purpose: Provides simplified interface to human web application subsystem
    Educational Value: Shows application lifecycle management with enhanced clarity
    Extension Pattern: Extensions can copy this enhanced pattern for their applications
    """
    
    def __init__(self, args=None):
        """Initialize human web application launcher.
        
        Args:
            args: Parsed command line arguments (optional, will parse if None)
            
        Educational Value: Shows proper application initialization with enhanced naming
        Extension Pattern: Extensions can copy this initialization approach
        """
        self.args = args or parse_human_arguments()
        self.app = None
        
        print_log("Initialized HumanWebAppLauncher")
    
    def validate_arguments(self) -> None:
        """Validate command line arguments using universal validation utilities.
        
        Educational Value: Shows argument validation patterns with universal utilities
        Extension Pattern: Extensions can copy this validation approach
        """
        # Validate grid size using universal validation
        self.args.grid_size = validate_grid_size(self.args.grid_size)
        
        # Validate port using universal validation
        self.args.port = validate_port(self.args.port)
        
        print_log(f"Human web app arguments validated: grid_size={self.args.grid_size}, port={self.args.port}")
    
    def setup_application(self) -> None:
        """Set up the human web application.
        
        Educational Value: Shows application setup patterns with enhanced factory functions
        Extension Pattern: Extensions can copy this setup approach
        """
        self.validate_arguments()
        
        # Create human web game app using enhanced factory function
        self.app = create_human_web_game_app(
            grid_size=self.args.grid_size,
            port=self.args.port
        )
        
        print_log("Human web application setup complete")
    
    def run_application(self) -> None:
        """Run the complete human web application.
        
        Educational Value: Shows application execution patterns with enhanced context
        Extension Pattern: Extensions can copy this execution approach
        """
        try:
            self.setup_application()
            
            print_log(f"Starting human web game application on {self.args.host}:{self.app.port}")
            self.app.run(
                host=self.args.host,
                port=self.args.port
            )
            
        except KeyboardInterrupt:
            print_log("Human web application interrupted by user")
        except Exception as e:
            print_log(f"Error in human web application: {e}")
            raise


class LLMWebAppLauncher:
    """
    LLM web application launcher with enhanced naming.
    
    Following the same pattern as HumanWebAppLauncher for consistency.
    Enhanced naming clearly indicates this is a launcher for LLM web apps.
    
    Design Pattern: Facade Pattern (LLM Web App Launcher)
    Purpose: Provides simplified interface to LLM web application subsystem
    Educational Value: Shows LLM application lifecycle management with enhanced clarity
    Extension Pattern: Extensions can copy this enhanced pattern for their LLM applications
    """
    
    def __init__(self, args=None):
        """Initialize LLM web application launcher.
        
        Args:
            args: Parsed command line arguments (optional, will parse if None)
            
        Educational Value: Shows LLM application initialization with enhanced naming
        Extension Pattern: Extensions can copy this for their LLM apps
        """
        self.args = args or parse_llm_arguments()
        self.app = None
        
        print_log("Initialized LLMWebAppLauncher")
    
    def validate_arguments(self) -> None:
        """Validate command line arguments using universal validation utilities.
        
        Educational Value: Shows LLM-specific validation patterns with universal utilities
        Extension Pattern: Extensions can copy this validation approach
        """
        # Validate grid size using universal validation
        self.args.grid_size = validate_grid_size(self.args.grid_size)
        
        # Validate port using universal validation
        self.args.port = validate_port(self.args.port)
        
        # Validate provider and model (basic validation)
        if not self.args.provider:
            raise ValueError("Provider is required for LLM web application")
        if not self.args.model:
            raise ValueError("Model is required for LLM web application")
        
        print_log(f"LLM web app arguments validated: {self.args.provider}/{self.args.model}")
    
    def setup_application(self) -> None:
        """Set up the LLM web application.
        
        Educational Value: Shows LLM application setup patterns with enhanced factory functions
        Extension Pattern: Extensions can copy this LLM setup approach
        """
        self.validate_arguments()
        
        # Create LLM web game app using enhanced factory function
        self.app = create_llm_web_game_app(
            provider=self.args.provider,
            model=self.args.model,
            grid_size=self.args.grid_size,
            port=self.args.port
        )
        
        print_log("LLM web application setup complete")
    
    def run_application(self) -> None:
        """Run the complete LLM web application.
        
        Educational Value: Shows LLM application execution patterns with enhanced context
        Extension Pattern: Extensions can copy this execution approach
        """
        try:
            self.setup_application()
            
            print_log(f"Starting LLM web game application on {self.args.host}:{self.app.port}")
            self.app.run(
                host=self.args.host,
                port=self.args.port
            )
            
        except KeyboardInterrupt:
            print_log("LLM web application interrupted by user")
        except Exception as e:
            print_log(f"Error in LLM web application: {e}")
            raise


class ReplayWebAppLauncher:
    """
    Replay web application launcher with enhanced naming.
    
    Following the same pattern as other web application launchers for consistency.
    Enhanced naming clearly indicates this is a launcher for replay web apps.
    
    Design Pattern: Facade Pattern (Replay Web App Launcher)
    Purpose: Provides simplified interface to replay web application subsystem
    Educational Value: Shows replay application lifecycle management with enhanced clarity
    Extension Pattern: Extensions can copy this enhanced pattern for their replay applications
    """
    
    def __init__(self, args=None):
        """Initialize replay web application launcher.
        
        Args:
            args: Parsed command line arguments (optional, will parse if None)
            
        Educational Value: Shows replay application initialization with enhanced naming
        Extension Pattern: Extensions can copy this for their replay apps
        """
        self.args = args or parse_replay_arguments()
        self.app = None
        
        print_log("Initialized ReplayWebAppLauncher")
    
    def validate_arguments(self) -> None:
        """Validate command line arguments using universal validation utilities.
        
        Educational Value: Shows replay-specific validation patterns with universal utilities
        Extension Pattern: Extensions can copy this validation approach
        """
        # Validate log directory (required)
        if not self.args.log_dir:
            raise ValueError("Log directory is required for replay web application")
        
        # Validate port using universal validation
        self.args.port = validate_port(self.args.port)
        
        # Validate game number
        if self.args.game < 1:
            raise ValueError(f"Game number must be >= 1, got: {self.args.game}")
        
        print_log(f"Replay web app arguments validated: {self.args.log_dir}, game {self.args.game}")
    
    def setup_application(self) -> None:
        """Set up the replay web application.
        
        Educational Value: Shows replay application setup patterns with enhanced factory functions
        Extension Pattern: Extensions can copy this replay setup approach
        """
        self.validate_arguments()
        
        # Create replay web game app using enhanced factory function
        self.app = create_replay_web_game_app(
            log_dir=self.args.log_dir,
            game_number=self.args.game,
            port=self.args.port
        )
        
        print_log("Replay web application setup complete")
    
    def run_application(self) -> None:
        """Run the complete replay web application.
        
        Educational Value: Shows replay application execution patterns with enhanced context
        Extension Pattern: Extensions can copy this execution approach
        """
        try:
            self.setup_application()
            
            print_log(f"Starting replay web game application on {self.args.host}:{self.app.port}")
            self.app.run(
                host=self.args.host,
                port=self.args.port
            )
            
        except KeyboardInterrupt:
            print_log("Replay web application interrupted by user")
        except Exception as e:
            print_log(f"Error in replay web application: {e}")
            raise


# =============================================================================
# Convenience Functions for Direct Usage
# =============================================================================

def run_human_web() -> None:
    """Run human web application directly.
    
    Convenience function for direct execution without class instantiation.
    
    Educational Value: Shows function-based application entry point with enhanced naming
    Extension Pattern: Extensions can provide similar convenience functions
    """
    launcher = HumanWebAppLauncher()
    launcher.run_application()


def run_llm_web() -> None:
    """Run LLM web application directly.
    
    Convenience function for direct execution without class instantiation.
    
    Educational Value: Shows function-based application entry point with enhanced naming
    Extension Pattern: Extensions can provide similar convenience functions
    """
    launcher = LLMWebAppLauncher()
    launcher.run_application()


def run_replay_web() -> None:
    """Run replay web application directly.
    
    Convenience function for direct execution without class instantiation.
    
    Educational Value: Shows function-based application entry point with enhanced naming
    Extension Pattern: Extensions can provide similar convenience functions
    """
    launcher = ReplayWebAppLauncher()
    launcher.run_application()


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Maintain backward compatibility while encouraging enhanced naming
HumanWebApplication = HumanWebAppLauncher  # Backward compatibility alias
LLMWebApplication = LLMWebAppLauncher  # Backward compatibility alias
ReplayWebApplication = ReplayWebAppLauncher  # Backward compatibility alias

print_log("Web application launchers initialized with enhanced naming and backward compatibility") 