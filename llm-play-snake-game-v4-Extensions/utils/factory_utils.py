"""
Universal Factory Utilities for Snake Game AI Project
====================================================

This module provides canonical factory patterns used throughout the entire project,
following SUPREME_RULES from final-decision-10.md. These utilities are used by:
- Core game components (Task-0)
- Web applications (all tasks)  
- All extensions (Tasks 1-5)

Design Philosophy:
- Universal: One factory pattern used everywhere
- SSOT: Single Source of Truth for factory implementation
- Educational: Clear examples of canonical factory patterns
- Canonical Method: All factories use create() method name
- Simple Logging: Uses print() statements only

Reference: docs/extensions-guideline/final-decision-10.md
"""

from typing import Dict, Type, Any, List, Optional, Union


class SimpleFactory:
    """
    Universal simple factory class following final-decision-10.md SUPREME_RULES.
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Provides universal factory pattern for all project components
    Educational Value: Demonstrates simple, extensible factory implementation
    Usage: Used by core, web, and all extensions for consistent object creation
    
    Key Features:
    - Canonical create() method name (SUPREME_RULES compliance)
    - Simple dictionary-based registry
    - Clear error messages with available options
    - Simple logging using print() statements
    - Easy extension and customization
    
    Example:
        >>> factory = SimpleFactory()
        >>> factory.register("agent", MyAgent)
        >>> agent = factory.create("agent", param1="value1")
    """
    
    def __init__(self, name: str = "SimpleFactory"):
        """Initialize factory with optional name for logging."""
        self.name = name
        self._registry: Dict[str, Type] = {}
        print(f"[{self.name}] Factory initialized")  # Simple logging - SUPREME_RULES
    
    def register(self, name: str, cls: Type) -> None:
        """Register a class with the factory.
        
        Args:
            name: Name identifier for the class
            cls: Class to register
        """
        self._registry[name.upper()] = cls
        print(f"[{self.name}] Registered: {name} -> {cls.__name__}")  # Simple logging
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create instance by name - CANONICAL METHOD NAME (SUPREME_RULES).
        
        All factory methods must be named create(), never create_agent(), 
        create_model(), etc. This ensures consistency across the entire project.
        
        Args:
            name: Name of the registered class
            *args: Positional arguments for class constructor
            **kwargs: Keyword arguments for class constructor
            
        Returns:
            Instance of the requested class
            
        Raises:
            ValueError: If name is not found in registry
        """
        cls = self._registry.get(name.upper())
        if not cls:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown type: {name}. Available: {available}")
        
        print(f"[{self.name}] Creating: {name}")  # Simple logging
        return cls(*args, **kwargs)
    
    def list_available(self) -> List[str]:
        """List all available registered types."""
        return list(self._registry.keys())
    
    def get_class(self, name: str) -> Type:
        """Get class by name without instantiation."""
        cls = self._registry.get(name.upper())
        if not cls:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown type: {name}. Available: {available}")
        return cls
    
    def is_registered(self, name: str) -> bool:
        """Check if a type is registered."""
        return name.upper() in self._registry
    
    def unregister(self, name: str) -> bool:
        """Unregister a type from the factory."""
        if name.upper() in self._registry:
            del self._registry[name.upper()]
            print(f"[{self.name}] Unregistered: {name}")  # Simple logging
            return True
        return False


class WebAppFactory:
    """
    Factory for creating web applications with canonical create() methods.
    
    Design Pattern: Factory Pattern (Canonical Implementation for Web Apps)
    Purpose: Create web applications using canonical create() method
    Educational Value: Shows factory pattern specialized for web applications
    Usage: Used by ROOT/web/ and extension web modules
    
    IMPORTANT: Uses canonical create() method name as mandated by SUPREME_RULES
    """
    
    _registry = {
        "HUMAN": "HumanWebApp",
        "MAIN": "MainWebApp", 
        "REPLAY": "ReplayWebApp",
    }
    
    @classmethod
    def create(cls, app_type: str, **kwargs) -> Any:  # CANONICAL create() method
        """Create web application using canonical create() method.
        
        Following SUPREME_RULES from final-decision-10.md, all factories must use
        the canonical create() method name for consistency across the project.
        
        Args:
            app_type: Type of application to create ('human', 'main', 'replay')
            **kwargs: Configuration parameters for the application
            
        Returns:
            Configured web application instance
            
        Raises:
            ValueError: If app_type is not supported
        """
        app_class_name = cls._registry.get(app_type.upper())
        if not app_class_name:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown app type: {app_type}. Available: {available}")
        
        print(f"[WebAppFactory] Creating web app: {app_type}")  # Simple logging
        
        # Import classes here to avoid circular imports
        if app_class_name == "HumanWebApp":
            from web.human_app import HumanWebApp
            return HumanWebApp(**kwargs)
        elif app_class_name == "MainWebApp":
            from web.main_app import MainWebApp
            return MainWebApp(**kwargs)
        elif app_class_name == "ReplayWebApp":
            from web.replay_app import ReplayWebApp
            return ReplayWebApp(**kwargs)
        else:
            raise ValueError(f"Unknown app class: {app_class_name}")
    
    @classmethod
    def register(cls, app_type: str, app_class_name: str) -> None:
        """Register a new application type."""
        cls._registry[app_type.upper()] = app_class_name
        print(f"[WebAppFactory] Registered web app type: {app_type} -> {app_class_name}")
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available application types."""
        return list(cls._registry.keys())


# =============================================================================
# Specialized Factory Classes for Universal Use
# =============================================================================

class AgentFactory:
    """
    Universal agent factory for all tasks and extensions.
    
    Design Pattern: Factory Pattern (Canonical Implementation for Agents)  
    Purpose: Create game agents using canonical create() method
    Educational Value: Shows agent factory patterns used across all extensions
    Usage: Used by Task-0 and all extensions for consistent agent creation
    """
    
    def __init__(self, name: str = "AgentFactory"):
        self.name = name
        self._registry: Dict[str, Type] = {}
        print(f"[{self.name}] Agent factory initialized")  # Simple logging
    
    def register(self, agent_type: str, agent_class: Type) -> None:
        """Register an agent class."""
        self._registry[agent_type.upper()] = agent_class
        print(f"[{self.name}] Registered agent: {agent_type} -> {agent_class.__name__}")
    
    def create(self, agent_type: str, **kwargs) -> Any:  # CANONICAL create() method
        """Create agent instance using canonical create() method."""
        agent_class = self._registry.get(agent_type.upper())
        if not agent_class:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")
        
        print(f"[{self.name}] Creating agent: {agent_type}")
        return agent_class(**kwargs)
    
    def list_available(self) -> List[str]:
        """List all available agent types."""
        return list(self._registry.keys())


class GameAppFactory:
    """
    Universal game app factory for all tasks and extensions.
    
    Design Pattern: Factory Pattern (Canonical Implementation for Game Apps)
    Purpose: Create game applications using canonical create() method
    Educational Value: Shows game app factory patterns used across all extensions
    Usage: Used by Task-0 and all extensions for consistent game app creation
    """
    
    def __init__(self, name: str = "GameAppFactory"):
        self.name = name
        self._registry: Dict[str, Type] = {}
        print(f"[{self.name}] Game app factory initialized")  # Simple logging
    
    def register(self, app_type: str, app_class: Type) -> None:
        """Register a game app class."""
        self._registry[app_type.upper()] = app_class
        print(f"[{self.name}] Registered game app: {app_type} -> {app_class.__name__}")
    
    def create(self, app_type: str, **kwargs) -> Any:  # CANONICAL create() method
        """Create game app instance using canonical create() method."""
        app_class = self._registry.get(app_type.upper())
        if not app_class:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown game app type: {app_type}. Available: {available}")
        
        print(f"[{self.name}] Creating game app: {app_type}")
        return app_class(**kwargs)
    
    def list_available(self) -> List[str]:
        """List all available game app types."""
        return list(self._registry.keys())


# =============================================================================
# Factory Creation Functions (Simple Factory Pattern)
# =============================================================================

def create_simple_factory(name: str = "SimpleFactory") -> SimpleFactory:
    """Create a simple factory instance."""
    return SimpleFactory(name)


def create_agent_factory(name: str = "AgentFactory") -> AgentFactory:
    """Create an agent factory instance."""
    return AgentFactory(name)


def create_game_app_factory(name: str = "GameAppFactory") -> GameAppFactory:
    """Create a game app factory instance."""
    return GameAppFactory(name)


def create_web_app_factory() -> type[WebAppFactory]:
    """Create a web app factory class."""
    return WebAppFactory


# =============================================================================
# Convenience Functions for Web App Creation
# =============================================================================

def create_human_web_app(grid_size: int = 10, port: Optional[int] = None) -> Any:
    """Create human web application."""
    return WebAppFactory.create("human", grid_size=grid_size, port=port)


def create_main_web_app(grid_size: int = 10, port: Optional[int] = None) -> Any:
    """Create main web application."""
    return WebAppFactory.create("main", grid_size=grid_size, port=port)


def create_replay_web_app(log_dir: str = "", game_number: int = 1, port: Optional[int] = None) -> Any:
    """Create replay web application."""
    return WebAppFactory.create("replay", log_dir=log_dir, game_number=game_number, port=port)


# =============================================================================
# Validation and Utility Functions
# =============================================================================

def validate_factory_registry(factory: SimpleFactory, required_types: List[str]) -> bool:
    """Validate that factory has all required types registered."""
    available = factory.list_available()
    missing = [t for t in required_types if t.upper() not in available]
    
    if missing:
        print(f"[FactoryValidation] Missing required types: {missing}")
        print(f"[FactoryValidation] Available types: {available}")
        return False
    
    print(f"[FactoryValidation] All required types registered: {required_types}")
    return True


# =============================================================================
# Example Usage and Testing
# =============================================================================

def example_usage():
    """Example usage of factory patterns."""
    
    # Example agent class
    class DemoAgent:
        def __init__(self, name: str):
            self.name = name
            print(f"Created demo agent: {name}")
    
    # Create and use simple factory
    factory = create_simple_factory("DemoFactory")
    factory.register("agent", DemoAgent)
    
    # Create agent instance
    agent = factory.create("agent", name="test_agent")
    print(f"Agent name: {agent.name}")
    
    # Create web app factory
    web_factory = create_web_app_factory()
    print(f"Available web app types: {web_factory.get_available_types()}")


if __name__ == "__main__":
    example_usage() 