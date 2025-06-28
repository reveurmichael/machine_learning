"""
Factory utilities for Snake Game AI extensions.

This module follows the principles from final-decision-10.md:
- Canonical factory method is create() 
- OOP extensibility is prioritized
- Logging is always simple (print())
- No ML/DL/RL/LLM-specific coupling

Reference: docs/extensions-guideline/final-decision-10.md
"""

from typing import Dict, Type, Any, List
from abc import ABC, abstractmethod

class SimpleFactory:
    """
    Simple factory class following final-decision-10.md principles
    
    Design Pattern: Factory Pattern
    - Simple dictionary-based registry
    - Clear create() method interface (CANONICAL)
    - Easy extension for new types
    - No over-engineering or complex inheritance
    
    Educational Value:
    Shows how factory patterns can be implemented simply
    without complex inheritance hierarchies or over-engineering.
    
    Simple logging: All logging uses simple print() statements.
    """
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
        print(f"[{self.__class__.__name__}] Factory initialized")  # Simple logging
    
    def register(self, name: str, cls: Type) -> None:
        """Register a class with the factory"""
        self._registry[name.upper()] = cls
        print(f"[{self.__class__.__name__}] Registered: {name} -> {cls.__name__}")  # Simple logging
    
    def create(self, name: str, **kwargs) -> Any:
        """
        Create instance by name - CANONICAL METHOD NAME
        
        All factory methods must be named create(),
        never create_agent(), create_model(), etc.
        """
        cls = self._registry.get(name.upper())
        if not cls:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown type: {name}. Available: {available}")
        
        print(f"[{self.__class__.__name__}] Creating: {name}")  # Simple logging
        return cls(**kwargs)
    
    def list_available(self) -> List[str]:
        """List all available types"""
        return list(self._registry.keys())
    
    def get_class(self, name: str) -> Type:
        """Get class by name without instantiation"""
        cls = self._registry.get(name.upper())
        if not cls:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown type: {name}. Available: {available}")
        return cls

# Simple utility functions
def create_simple_factory() -> SimpleFactory:
    """Create a simple factory instance"""
    print(f"[FactoryUtils] Creating simple factory")  # Simple logging
    return SimpleFactory()

def validate_factory_registry(factory: SimpleFactory, required_types: List[str]) -> bool:
    """Validate that factory has required types registered"""
    available = factory.list_available()
    missing = [t for t in required_types if t.upper() not in available]
    
    if missing:
        print(f"[FactoryUtils] WARNING: Missing types: {missing}")  # Simple logging
        return False
    
    print(f"[FactoryUtils] Factory validation passed")  # Simple logging
    return True

# Example usage for documentation and educational value
if __name__ == "__main__":
    class Dummy:
        def __init__(self, x):
            self.x = x
    
    # Use SimpleFactory as referenced in documentation
    f = SimpleFactory()
    f.register("dummy", Dummy)
    d = f.create("dummy", x=42)  # CANONICAL create() method
    print(f"Created: {d.__class__.__name__}, x={d.x}")
    print(f"Available: {f.list_available()}") 