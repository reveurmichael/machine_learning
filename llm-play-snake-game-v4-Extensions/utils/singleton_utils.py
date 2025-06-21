"""
Singleton utilities for implementing the Singleton pattern with abstract base classes.

This module provides the SingletonABCMeta metaclass that combines the Singleton pattern
with abstract base class functionality, resolving metaclass conflicts while ensuring
thread safety.

This whole module is generic and can be used by any task (Tasks 0-5).
"""

from __future__ import annotations

import threading
from abc import ABCMeta
from typing import Any, Dict

__all__ = [
    "SingletonABCMeta",
]


class SingletonABCMeta(ABCMeta):
    """
    Thread-safe Singleton metaclass that combines ABC and Singleton patterns.
    
    This metaclass implements the Singleton pattern using double-checked locking
    while also supporting abstract base class functionality. This resolves the
    metaclass conflict between ABC and Singleton patterns.
    
    Design Pattern: **Singleton Pattern + Abstract Base Class**
    Purpose: Ensure only one instance exists per class while maintaining
    abstract base class functionality for inheritance.
    
    Benefits:
    - Thread safety through double-checked locking
    - Memory efficiency (single instance per class)
    - Abstract base class functionality
    - Centralized instance control
    - Prevents metaclass conflicts
    
    Usage:
        class MyClass(ABC, metaclass=SingletonABCMeta):
            def __init__(self):
                if not hasattr(self, '_initialized'):
                    self._initialized = True
                    # Initialization code here
            
            @abstractmethod
            def my_method(self):
                pass
    
    Thread Safety:
        Uses double-checked locking pattern to minimize synchronization overhead
        while ensuring thread safety during instance creation.
    """
    
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """
        Thread-safe singleton instance creation with double-checked locking.
        
        The double-checked locking pattern ensures thread safety while
        minimizing the performance overhead of synchronization by only
        acquiring the lock when necessary.
        
        Args:
            *args: Positional arguments for class constructor
            **kwargs: Keyword arguments for class constructor
            
        Returns:
            The singleton instance of the class
        """
        # First check (without locking for performance)
        if cls not in cls._instances:
            # Acquire lock for thread safety
            with cls._lock:
                # Second check (with lock to prevent race conditions)
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        
        return cls._instances[cls]
    
    @classmethod
    def clear_instances(mcs) -> None:
        """
        Clear all singleton instances (useful for testing).
        
        This method allows clearing the singleton instances, which is
        particularly useful in testing scenarios where you need fresh
        instances between test cases.
        
        Warning:
            Use with caution in production code as it breaks the singleton
            contract. Primarily intended for testing purposes.
        """
        with mcs._lock:
            mcs._instances.clear()
    
    @classmethod
    def get_instance_count(mcs) -> int:
        """
        Get the number of singleton instances currently managed.
        
        Returns:
            Number of singleton instances in the registry
        """
        return len(mcs._instances) 