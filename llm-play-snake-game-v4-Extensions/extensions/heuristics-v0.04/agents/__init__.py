"""
Heuristics Agents Package - Factory Pattern Implementation
--------------------

This package provides a factory pattern for creating heuristic agents.
It demonstrates software evolution through inheritance and encapsulation.

Available Algorithms:
1. BFS - Basic breadth-first search
2. BFS-SAFE-GREEDY - Enhanced BFS with safety validation (inherits from BFS)

Design Patterns:
- Factory Pattern: create_agent() function for instantiation
- Registry Pattern: ALGORITHM_REGISTRY for algorithm mapping
- Inheritance: Progressive enhancement through class hierarchy
- Strategy Pattern: Interchangeable algorithms
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


# Ensure project root is set and properly configured
from utils.path_utils import ensure_project_root
ensure_project_root()

from typing import Dict, Type, Optional, List, Any

# Import all agent classes
from .agent_bfs import BFSAgent
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent

# Algorithm registry mapping names to classes
ALGORITHM_REGISTRY: Dict[str, Type] = {
    "BFS": BFSAgent,
    "BFS-SAFE-GREEDY": BFSSafeGreedyAgent,
}

# After ALGORITHM_REGISTRY definition
DEFAULT_ALGORITHM: str = "BFS"

def create_agent(algorithm_name: str) -> Any:
    """
    Factory function to create an agent instance.
    
    Args:
        algorithm_name: Name of the algorithm (case-insensitive)
        
    Returns:
        Agent instance
        
    Raises:
        ValueError: If algorithm name is not recognized
    """
    algorithm_name = algorithm_name.upper()
    
    if algorithm_name not in ALGORITHM_REGISTRY:
        available = ", ".join(ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'. "
            f"Available algorithms: {available}"
        )
    
    agent_class = ALGORITHM_REGISTRY[algorithm_name]
    return agent_class()

def get_available_algorithms() -> List[str]:
    """
    Get list of available algorithm names.
    
    Returns:
        List of algorithm names
    """
    return list(ALGORITHM_REGISTRY.keys())

def get_algorithm_info(algorithm_name: str) -> Dict[str, Any]:
    """
    Get information about a specific algorithm.
    
    Args:
        algorithm_name: Name of the algorithm
        
    Returns:
        Dictionary with algorithm information
        
    Raises:
        ValueError: If algorithm name is not recognized
    """
    algorithm_name = algorithm_name.upper()
    
    if algorithm_name not in ALGORITHM_REGISTRY:
        available = ", ".join(ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'. "
            f"Available algorithms: {available}"
        )
    
    agent_class = ALGORITHM_REGISTRY[algorithm_name]
    agent_instance = agent_class()
    
    return {
        "name": getattr(agent_instance, "name", algorithm_name),
        "description": getattr(agent_instance, "description", "No description available"),
        "algorithm_name": getattr(agent_instance, "algorithm_name", algorithm_name),
        "complexity": _get_algorithm_complexity(algorithm_name),
        "category": _get_algorithm_category(algorithm_name),
    }

def _get_algorithm_complexity(algorithm_name: str) -> str:
    """Get time complexity information for educational purposes."""
    complexities = {
        "BFS": "O(V + E) - Optimal for shortest path",
        "BFS-SAFE-GREEDY": "O(V + E) - BFS + safety validation",
    }
    return complexities.get(algorithm_name, "Unknown complexity")

def _get_algorithm_category(algorithm_name: str) -> str:
    """Get educational category for the algorithm."""
    categories = {
        "BFS": "Basic Search",
        "BFS-SAFE-GREEDY": "Enhanced Search",
    }
    return categories.get(algorithm_name, "Unknown category")

# Public API
__all__ = [
    # Agent classes
    "BFSAgent",
    "BFSSafeGreedyAgent", 

    # Factory functions
    "create_agent",
    "get_available_algorithms",
    "get_algorithm_info",
    
    # Registry
    "ALGORITHM_REGISTRY",
    "DEFAULT_ALGORITHM",
] 
