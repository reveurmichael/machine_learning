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
        "LLM": "LLMWebApp", 
        "REPLAY": "ReplayWebApp",
    }
    
    @classmethod
    def create(cls, app_type: str, **kwargs) -> Any:  # CANONICAL create() method
        """Create web application using canonical create() method.
        
        Following SUPREME_RULES from final-decision-10.md, all factories must use
        the canonical create() method name for consistency across the project.
        
        Args:
            app_type: Type of application to create ('human', 'llm', 'replay')
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
        elif app_class_name == "LLMWebApp":
            from web.llm_app import LLMWebApp
            return LLMWebApp(**kwargs)
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
        """Create agent using canonical create() method (SUPREME_RULES compliance)."""
        agent_class = self._registry.get(agent_type.upper())
        if not agent_class:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")
        
        print(f"[{self.name}] Creating agent: {agent_type}")  # Simple logging
        return agent_class(**kwargs)
    
    def list_available(self) -> List[str]:
        """List all available agent types."""
        return list(self._registry.keys())


class GameAppFactory:
    """
    Universal game application factory.
    
    Design Pattern: Factory Pattern (Canonical Implementation for Game Apps)
    Purpose: Create game applications (CLI, web, GUI) using canonical create()
    Educational Value: Shows universal factory pattern for different app types
    Usage: Used across all tasks for consistent application creation
    """
    
    def __init__(self, name: str = "GameAppFactory"):
        self.name = name
        self._registry: Dict[str, Type] = {}
        print(f"[{self.name}] Game app factory initialized")  # Simple logging
    
    def register(self, app_type: str, app_class: Type) -> None:
        """Register an application class."""
        self._registry[app_type.upper()] = app_class
        print(f"[{self.name}] Registered app: {app_type} -> {app_class.__name__}")
    
    def create(self, app_type: str, **kwargs) -> Any:  # CANONICAL create() method
        """Create application using canonical create() method (SUPREME_RULES compliance)."""
        app_class = self._registry.get(app_type.upper())
        if not app_class:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown app type: {app_type}. Available: {available}")
        
        print(f"[{self.name}] Creating app: {app_type}")  # Simple logging
        return app_class(**kwargs)
    
    def list_available(self) -> List[str]:
        """List all available application types."""
        return list(self._registry.keys())


# =============================================================================
# Convenience Functions Following SUPREME_RULES
# =============================================================================

def create_simple_factory(name: str = "SimpleFactory") -> SimpleFactory:
    """Create a simple factory instance - canonical function."""
    print(f"[FactoryUtils] Creating simple factory: {name}")  # Simple logging
    return SimpleFactory(name)


def create_agent_factory(name: str = "AgentFactory") -> AgentFactory:
    """Create an agent factory instance - canonical function."""
    print(f"[FactoryUtils] Creating agent factory: {name}")  # Simple logging
    return AgentFactory(name)


def create_game_app_factory(name: str = "GameAppFactory") -> GameAppFactory:
    """Create a game application factory instance - canonical function."""
    print(f"[FactoryUtils] Creating game app factory: {name}")  # Simple logging
    return GameAppFactory(name)


def create_web_app_factory() -> type[WebAppFactory]:
    """Create a web application factory class - canonical function."""
    print("[FactoryUtils] Creating web app factory")  # Simple logging
    return WebAppFactory


# Simple web app creation functions following KISS principles
def create_human_web_app(grid_size: int = 10, port: Optional[int] = None) -> Any:
    """Create human web application using factory pattern."""
    return WebAppFactory.create("human", grid_size=grid_size, port=port)


def create_llm_web_app(grid_size: int = 10, port: Optional[int] = None) -> Any:
    """Create LLM web application using factory pattern."""
    return WebAppFactory.create("llm", grid_size=grid_size, port=port)


def create_replay_web_app(session_path: str = "", game_number: int = 1, port: Optional[int] = None) -> Any:
    """Create replay web application using factory pattern."""
    return WebAppFactory.create("replay", session_path=session_path, game_number=game_number, port=port)


def validate_factory_registry(factory: SimpleFactory, required_types: List[str]) -> bool:
    """Validate that factory has required types registered."""
    available = factory.list_available()
    missing = [t for t in required_types if t.upper() not in available]
    
    if missing:
        print(f"[FactoryUtils] WARNING: Missing types: {missing}")  # Simple logging
        return False
    
    print("[FactoryUtils] Factory validation passed")  # Simple logging
    return True


# =============================================================================
# Educational Examples and Usage Patterns
# =============================================================================

def example_usage():
    """
    Educational examples showing how to use factory utilities.
    
    This function demonstrates the canonical patterns used throughout
    the Snake Game AI project for consistent object creation.
    """
    print("[FactoryUtils] Running educational examples...")  # Simple logging
    
    # Example 1: Simple Factory Usage
    class DemoAgent:
        def __init__(self, name: str):
            self.name = name
            print(f"[DemoAgent] Created: {name}")
    
    # Create and use simple factory
    factory = create_simple_factory("DemoFactory")
    factory.register("demo", DemoAgent)
    agent = factory.create("demo", name="TestAgent")  # CANONICAL create() method
    print(f"[Example] Agent name: {agent.name}")
    
    # Example 2: Agent Factory Usage
    agent_factory = create_agent_factory()
    agent_factory.register("demo", DemoAgent)
    demo_agent = agent_factory.create("demo", name="AgentDemo")  # CANONICAL create()
    
    # Example 3: Factory Validation
    is_valid = validate_factory_registry(factory, ["demo"])
    print(f"[Example] Factory validation result: {is_valid}")
    
    print("[FactoryUtils] Educational examples completed")  # Simple logging


if __name__ == "__main__":
    # Run educational examples when script is executed directly
    example_usage() 