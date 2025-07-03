import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

"""
Simple Factory Pattern Utilities for Snake Game AI Extensions

This module provides a generic, educational SimpleFactory implementation
following SUPREME_RULES from final-decision-10.md. It demonstrates the
canonical factory pattern with the standard create() method.

Design Philosophy:
- Simple, dictionary-based factory pattern
- Canonical create() method (not create_agent() or other variants)
- Educational value for demonstrating design patterns
- No over-engineering - KISS principle

Reference: docs/extensions-guideline/final-decision-10.md
"""

from typing import Any, Dict, Type
from utils.print_utils import print_info


class SimpleFactory:
    """
    Generic factory implementation following SUPREME_RULES from final-decision-10.md
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create objects using canonical create() method
    Educational Value: Shows how SUPREME_RULES apply consistently across extensions
    """
    
    def __init__(self):
        """Initialize empty factory registry."""
        self._registry: Dict[str, Type] = {}
        print_info("[SimpleFactory] Initialized factory registry")
    
    def register(self, name: str, cls: Type) -> None:
        """Register a class with the factory."""
        self._registry[name.upper()] = cls
        print_info(f"[SimpleFactory] Registered: {name} -> {cls.__name__}")
    
    def create(self, name: str, **kwargs) -> Any:  # CANONICAL create() method per SUPREME_RULES
        """Create instance using canonical create() method following SUPREME_RULES from final-decision-10.md"""
        name = name.upper()
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown item: {name}. Available: {available}")
        
        cls = self._registry[name]
        print_info(f"[SimpleFactory] Creating instance: {name} ({cls.__name__})")
        return cls(**kwargs)
    
    def list_available(self) -> list[str]:
        """List all available items in the factory."""
        return list(self._registry.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if an item is registered."""
        return name.upper() in self._registry 