"""
Heuristics Agents Package - Canonical Factory Pattern Implementation
----------------

This package provides a canonical factory pattern for creating heuristic agents.
It demonstrates software evolution through inheritance and encapsulation.

Available Algorithms:
1. BFS - Basic breadth-first search
2. BFS-SAFE-GREEDY - Enhanced BFS with safety validation (inherits from BFS)

Design Patterns:
- Factory Pattern: Canonical create() method for instantiation (SUPREME_RULES)
- Strategy Pattern: Interchangeable algorithms
- Inheritance: Progressive enhancement through class hierarchy

Reference: docs/extensions-guideline/factory-design-pattern.md
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# Ensure project root is set and properly configured
try:
    from utils.path_utils import ensure_project_root
    ensure_project_root()
except ImportError:
    # Fallback if utils module is not available
    pass

from typing import Dict, Type, Optional, List, Any

# Import canonical factory utilities
try:
    from utils.factory_utils import SimpleFactory
except ImportError:
    # Fallback for when running from extension directory
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from utils.factory_utils import SimpleFactory

# Import all agent classes
from .agent_bfs import BFSAgent
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent

# Canonical factory instance following SUPREME_RULES
_agent_factory = SimpleFactory("HeuristicAgentFactory")

# Register agents with canonical factory
_agent_factory.register("BFS", BFSAgent)
_agent_factory.register("BFS-SAFE-GREEDY", BFSSafeGreedyAgent)

# Default algorithm
DEFAULT_ALGORITHM: str = "BFS"

def create(algorithm_name: str, **kwargs) -> Any:
    """
    Canonical factory method to create an agent instance.
    
    Following SUPREME_RULES from final-decision-10.md, all factories must use
    the canonical create() method name for consistency across the project.
    
    Args:
        algorithm_name: Name of the algorithm (case-insensitive)
        **kwargs: Additional arguments for agent initialization
        
    Returns:
        Agent instance
        
    Raises:
        ValueError: If algorithm name is not recognized
    """
    return _agent_factory.create(algorithm_name, **kwargs)

def get_available_algorithms() -> List[str]:
    """
    Get list of available algorithm names.
    
    Returns:
        List of algorithm names
    """
    return _agent_factory.list_available()


# Public API
__all__ = [
    # Agent classes
    "BFSAgent",
    "BFSSafeGreedyAgent",

    # Canonical factory method
    "create",

    # Default
    "DEFAULT_ALGORITHM",
] 
