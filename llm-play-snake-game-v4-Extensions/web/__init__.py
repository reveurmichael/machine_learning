"""
Snake Game Web Interface Package with Enhanced Naming
====================================================

Modular web interface for Snake Game AI with support for human, LLM, and replay modes.
Implements layered web infrastructure with enhanced naming for maximum clarity.
Uses universal factory utilities following SSOT principles.

Design Philosophy:
- Layered Architecture: BaseWebApp → SimpleFlaskApp → Specific Apps
- Enhanced Naming: Clear, explicit naming for better understanding
- Universal Utilities: Uses factory_utils from ROOT/utils/ following SSOT
- KISS Principle: Simple imports and usage patterns with clear hierarchy
- Backward Compatibility: Maintains existing import patterns with aliases
- Educational Value: Clear structure for learning and extension

Package Structure (Enhanced):
- base_app: Layered web infrastructure (BaseWebApp, SimpleFlaskApp, BaseReplayApp)
- factories: Enhanced factory classes with canonical create() methods
- human_app: HumanWebGameApp (enhanced naming)
- llm_app: LLMWebGameApp (enhanced naming)
- replay_app: ReplayWebGameApp (enhanced naming)
- applications: Enhanced application launchers (HumanWebAppLauncher, etc.)

Educational Value:
- Shows proper Python package organization with enhanced naming
- Demonstrates layered architecture and inheritance patterns
- Provides template for extension package structure
- Uses universal factory utilities following SUPREME_RULES

Extension Pattern:
Extensions can organize their web interfaces using the same layered structure
while importing specific components they need for customization from the universal base.

Reference: utils/factory_utils.py for universal factory utilities
"""

# Import layered web infrastructure components
from web.base_app import BaseWebApp, SimpleFlaskApp, BaseReplayApp

# Import enhanced factory classes and functions
from web.factories import (
    GameWebAppFactory, 
    create_human_web_game_app, 
    create_llm_web_game_app, 
    create_replay_web_game_app
)

# Import enhanced web applications
from web.human_app import HumanWebGameApp
from web.llm_app import LLMWebGameApp
from web.replay_app import ReplayWebGameApp

# Import enhanced application launchers
from web.applications import (
    HumanWebAppLauncher,
    LLMWebAppLauncher, 
    ReplayWebAppLauncher,
    run_human_web,
    run_llm_web,
    run_replay_web,
    parse_human_arguments,
    parse_llm_arguments,
    parse_replay_arguments
)

# Import universal utilities following SSOT principles
from utils.factory_utils import WebAppFactory, SimpleFactory
from utils.web_utils import (
    build_color_map, 
    to_list, 
    build_error_response, 
    build_success_response
)
from utils.validation_utils import validate_grid_size, validate_port
from utils.print_utils import create_logger

# Define what gets imported with "from web import *" (Enhanced)
__all__ = [
    # Layered web infrastructure
    'BaseWebApp',
    'SimpleFlaskApp', 
    'BaseReplayApp',
    
    # Enhanced factory pattern (canonical create() methods per SUPREME_RULES)
    'GameWebAppFactory',
    'WebAppFactory',  # Universal factory from utils/
    'SimpleFactory',  # Universal factory from utils/
    'create_human_web_game_app',
    'create_llm_web_game_app',
    'create_replay_web_game_app',
    
    # Enhanced web applications
    'HumanWebGameApp',
    'LLMWebGameApp',
    'ReplayWebGameApp',
    
    # Enhanced application launchers
    'HumanWebAppLauncher',
    'LLMWebAppLauncher',
    'ReplayWebAppLauncher',
    
    # Argument parsing (following scripts/ patterns)
    'parse_human_arguments',
    'parse_llm_arguments',
    'parse_replay_arguments',
    
    # Convenience functions
    'run_human_web',
    'run_llm_web',
    'run_replay_web',
    
    # Universal utilities
    'build_color_map',
    'to_list',
    'validate_grid_size',
    'validate_port',
    'build_error_response',
    'build_success_response',
    'create_logger',
]

# Backward compatibility aliases for existing code
# (Maintains compatibility while encouraging enhanced naming)

# Legacy factory functions (backward compatibility)
create_human_app = create_human_web_game_app
create_llm_app = create_llm_web_game_app
create_replay_app = create_replay_web_game_app

# Legacy application classes (backward compatibility)
HumanGameApp = HumanWebGameApp
LLMGameApp = LLMWebGameApp  
ReplayGameApp = ReplayWebGameApp

# Legacy application launchers (backward compatibility)
HumanWebApplication = HumanWebAppLauncher
LLMWebApplication = LLMWebAppLauncher
ReplayWebApplication = ReplayWebAppLauncher

# Package metadata
__version__ = '2.0.0'  # Incremented for enhanced naming and layered architecture
__author__ = 'Snake Game AI Project'
__description__ = 'Layered web interface for Snake Game AI with enhanced naming'

# Enhanced quick start examples for documentation
__examples__ = """
Enhanced Quick Start Examples:

1. Human Web Game Interface (Enhanced):
   >>> from web import create_human_web_game_app
   >>> app = create_human_web_game_app(grid_size=15)
   >>> app.run()

2. LLM Web Game Interface (Enhanced):
   >>> from web import create_llm_web_game_app
   >>> app = create_llm_web_game_app(provider='deepseek', model='deepseek-chat')
   >>> app.run()

3. Replay Web Game Interface (Enhanced):
   >>> from web import create_replay_web_game_app
   >>> app = create_replay_web_game_app('logs/session_20250101_120000')
   >>> app.run()

4. Enhanced Application Launchers:
   >>> from web import HumanWebAppLauncher
   >>> launcher = HumanWebAppLauncher()
   >>> launcher.run_application()

5. Universal Factory Pattern (SUPREME_RULES compliant):
   >>> from web import GameWebAppFactory
   >>> app = GameWebAppFactory.create('human', grid_size=12)
   >>> app.run()

6. Layered Web Infrastructure:
   >>> from web import BaseWebApp, SimpleFlaskApp
   >>> class MyApp(SimpleFlaskApp):
   ...     def __init__(self):
   ...         super().__init__("My Custom App")

Backward Compatibility Examples:
   >>> from web import create_human_app  # Legacy alias
   >>> app = create_human_app(grid_size=10)
   >>> app.run()

Extension Pattern with Layered Architecture:
Extensions can inherit from the layered infrastructure:
   >>> from web.base_app import BaseReplayApp
   >>> from utils.factory_utils import SimpleFactory
   >>> from utils.print_utils import create_logger
   >>> 
   >>> class HeuristicReplayApp(BaseReplayApp):
   ...     def __init__(self, log_dir, algorithm, **config):
   ...         super().__init__("Heuristic Replay", log_dir, **config)
   ...         self.algorithm = algorithm
   ...         log = create_logger("HeuristicReplay")
   ...         log(f"Heuristic replay app initialized: {algorithm}")
   ...
   ...     def setup_replay_infrastructure(self):
   ...         super().setup_replay_infrastructure()
   ...         # Add heuristic-specific replay setup
   ...         log(f"Setting up {self.algorithm} replay infrastructure")
"""

# Migration guidance for existing code
__migration_guide__ = """
Migration Guide from Original Structure:

OLD (Original):                    NEW (Enhanced):
-------------------------------------------------------------------
from web import create_human_app   → from web import create_human_web_game_app
from web import HumanGameApp       → from web import HumanWebGameApp
from web import HumanWebApplication → from web import HumanWebAppLauncher
from web import WebAppFactory      → from web import GameWebAppFactory

Backward Compatibility:
All old imports still work due to aliases, but new code should use enhanced naming.

Benefits of Enhanced Naming:
- Clear indication of web + game domain
- Better consistency across the project
- Educational clarity for learning
- Template for extension development
"""

print_log = create_logger("WebPackage")
print_log("Snake Game web package initialized with enhanced naming and layered architecture") 