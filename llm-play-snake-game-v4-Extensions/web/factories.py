"""
Game Web Application Factories with Enhanced Naming
==================================================

Factory classes and functions for creating Snake Game web applications.
Implements Factory Pattern with canonical create() methods per SUPREME_RULES.
Uses enhanced naming for maximum clarity and consistency.

Design Philosophy:
- Factory Pattern: Centralized creation of web applications with clear naming
- SUPREME_RULES: Uses canonical create() method names from universal factory_utils
- Enhanced Naming: Clear, explicit names that indicate game domain and purpose
- Educational: Clear factory pattern implementation with universal utilities
- Extensibility: Easy to add new application types with consistent patterns

Educational Value:
- Shows Factory Pattern with canonical create() methods from universal utilities
- Demonstrates parameter forwarding in factory functions with enhanced naming
- Provides template for extension factory patterns with clear inheritance

Extension Pattern:
Extensions can add their application types to the factory registry
and provide their own convenience functions following the same patterns.

Reference: utils/factory_utils.py for universal factory utilities
"""

from typing import Dict, Any

# Import universal factory utilities following SSOT principles
from utils.factory_utils import WebAppFactory
from utils.print_utils import create_logger

# Create logger for this module
print_log = create_logger("GameWebFactory")


class GameWebAppFactory(WebAppFactory):
    """
    Enhanced factory for creating Snake Game web applications.
    
    Design Pattern: Factory Pattern (Enhanced Implementation with Game Focus)
    Purpose: Create game web applications using canonical create() method with clear naming
    Educational Value: Shows factory pattern specialized for game web applications
    Extension Pattern: Extensions can copy this enhanced factory pattern
    
    Enhanced Features:
    - Game-specific naming for maximum clarity
    - Inherits canonical create() method from universal WebAppFactory
    - Enhanced error messages with game context
    - Clear inheritance from universal factory utilities
    
    IMPORTANT: Uses canonical create() method name as mandated by SUPREME_RULES
    """
    
    # Enhanced registry with clearer naming
    _registry = {
        "HUMAN": "HumanWebGameApp",
        "LLM": "LLMWebGameApp", 
        "REPLAY": "ReplayWebGameApp",
    }
    
    @classmethod
    def create(cls, app_type: str, **kwargs) -> Any:  # CANONICAL create() method
        """Create game web application using canonical create() method.
        
        Following SUPREME_RULES from final-decision-10.md and universal factory_utils.py,
        all factories must use the canonical create() method name for consistency.
        
        Args:
            app_type: Type of game application to create ('human', 'llm', 'replay')
            **kwargs: Configuration parameters for the application
            
        Returns:
            Configured game web application instance
            
        Raises:
            ValueError: If app_type is not supported
            
        Educational Value: Shows enhanced factory method with game-specific context
        """
        app_class_name = cls._registry.get(app_type.upper())
        if not app_class_name:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown game app type: {app_type}. Available game apps: {available}")
        
        print_log(f"Creating game web app: {app_type}")  # Simple logging per SUPREME_RULES
        
        # Import classes here to avoid circular imports
        if app_class_name == "HumanWebGameApp":
            from web.human_app import HumanWebGameApp
            return HumanWebGameApp(**kwargs)
        elif app_class_name == "LLMWebGameApp":
            from web.llm_app import LLMWebGameApp
            return LLMWebGameApp(**kwargs)
        elif app_class_name == "ReplayWebGameApp":
            from web.replay_app import ReplayWebGameApp
            return ReplayWebGameApp(**kwargs)
        else:
            raise ValueError(f"Unknown game app class: {app_class_name}")
    
    @classmethod
    def register(cls, app_type: str, app_class_name: str) -> None:
        """Register a new game application type.
        
        Args:
            app_type: Type identifier for the game application
            app_class_name: Name of the game application class
            
        Educational Value: Shows how to extend factory registries with game focus
        Extension Pattern: Extensions can register their game app types
        """
        cls._registry[app_type.upper()] = app_class_name
        print_log(f"Registered game web app type: {app_type} -> {app_class_name}")
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available game application types.
        
        Returns:
            List of available game application type names
            
        Educational Value: Shows factory introspection patterns with game context
        """
        return list(cls._registry.keys())


# =============================================================================
# Enhanced Convenience Factory Functions 
# =============================================================================

def create_human_web_game_app(grid_size: int = 10, **config):
    """Create human web game app using enhanced factory pattern.
    
    Args:
        grid_size: Size of the game grid (default: 10)
        **config: Additional configuration options
        
    Returns:
        Configured HumanWebGameApp instance
        
    Design Pattern: Factory Function (Enhanced Convenience Wrapper)
    Purpose: Provides simple function interface to GameWebAppFactory.create()
    Educational Value: Shows how to provide multiple interfaces to factories with clear naming
    Extension Pattern: Extensions can copy this enhanced pattern for their app creation
    
    Example:
        >>> app = create_human_web_game_app(grid_size=15)
        >>> app.run()
    """
    print_log(f"Creating human web game app with grid_size={grid_size}")
    return GameWebAppFactory.create("human", grid_size=grid_size, **config)


def create_llm_web_game_app(provider: str = "hunyuan", model: str = "hunyuan-turbos-latest",
                           grid_size: int = 10, **config):
    """Create LLM web game app using enhanced factory pattern.
    
    Args:
        provider: LLM provider name (default: 'hunyuan')
        model: LLM model name (default: 'hunyuan-turbos-latest')
        grid_size: Size of the game grid (default: 10)
        **config: Additional configuration options
        
    Returns:
        Configured LLMWebGameApp instance
        
    Design Pattern: Factory Function (Enhanced Convenience Wrapper)
    Purpose: Provides simple function interface to GameWebAppFactory.create()
    Educational Value: Shows parameter forwarding in factory functions with enhanced naming
    Extension Pattern: Extensions can copy this enhanced pattern for their app creation
    
    Note: For full LLM functionality with GameManager integration,
    use scripts/main_web.py which provides complete LLM session management.
    This function provides a lightweight demo interface.
    
    Example:
        >>> app = create_llm_web_game_app(provider='deepseek', model='deepseek-chat')
        >>> app.run()
    """
    print_log(f"Creating LLM web game app with provider={provider}, model={model}")
    return GameWebAppFactory.create("llm", provider=provider, model=model, 
                                   grid_size=grid_size, **config)


def create_replay_web_game_app(log_dir: str, game_number: int = 1, **config):
    """Create replay web game app using enhanced factory pattern.
    
    Args:
        log_dir: Directory containing game logs
        game_number: Game number to replay (default: 1)
        **config: Additional configuration options
        
    Returns:
        Configured ReplayWebGameApp instance
        
    Design Pattern: Factory Function (Enhanced Convenience Wrapper)
    Purpose: Provides simple function interface to GameWebAppFactory.create()
    Educational Value: Shows required vs optional parameter handling with enhanced naming
    Extension Pattern: Extensions can copy this enhanced pattern for their app creation
    
    Example:
        >>> app = create_replay_web_game_app('logs/session_20250101_120000', game_number=3)
        >>> app.run()
    """
    print_log(f"Creating replay web game app with log_dir={log_dir}, game_number={game_number}")
    return GameWebAppFactory.create("replay", log_dir=log_dir, game_number=game_number, **config)

