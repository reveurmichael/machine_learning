"""
Heuristics Agents Package - Canonical Factory Pattern Implementation
----------------

This package provides a canonical factory pattern for creating heuristic agents.
It demonstrates software evolution through inheritance and encapsulation.

Available Algorithms (Token Variants Only):
1. BFS-512 - Token-limited BFS with concise explanations (~512 tokens)
2. BFS-1024 - Token-limited BFS with moderate explanations (~1024 tokens)
3. BFS-2048 - Token-limited BFS with detailed explanations (~2048 tokens)
4. BFS-4096 - Token-limited BFS with full explanations (~4096 tokens)
5. BFS-SAFE-GREEDY-4096 - Token-limited BFS-SAFE-GREEDY with full explanations (~4096 tokens)

Note: Base classes (BFS, BFS-SAFE-GREEDY) are blueprint templates and not registered
for dataset generation. Only token variants are used for actual JSONL generation.

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

# Import all agent classes (including base classes for inheritance)
from .agent_bfs import BFSAgent
from .agent_bfs_safe_greedy import BFSSafeGreedyAgent
from .agent_bfs_tokens_512 import BFS512TokenAgent
from .agent_bfs_tokens_1024 import BFS1024TokenAgent
from .agent_bfs_tokens_2048 import BFS2048TokenAgent
from .agent_bfs_tokens_4096 import BFS4096TokenAgent
from .agent_bfs_safe_greedy_tokens_4096 import BFSSafeGreedy4096TokenAgent

# Canonical factory instance following SUPREME_RULES
_agent_factory = SimpleFactory("HeuristicAgentFactory")

# Register agents with canonical factory
_agent_factory.register("BFS", BFSAgent)
_agent_factory.register("BFS-SAFE-GREEDY", BFSSafeGreedyAgent)
_agent_factory.register("BFS-512", BFS512TokenAgent)
_agent_factory.register("BFS-1024", BFS1024TokenAgent)
_agent_factory.register("BFS-2048", BFS2048TokenAgent)
_agent_factory.register("BFS-4096", BFS4096TokenAgent)
_agent_factory.register("BFS-SAFE-GREEDY-4096", BFSSafeGreedy4096TokenAgent)

# Default algorithm (use 512 token variant as default)
DEFAULT_ALGORITHM: str = "BFS-512"

def create(algorithm_name: str, **kwargs) -> Any:
    """
    Canonical factory method to create an agent instance.
    
    Following SUPREME_RULES from final-decision.md, all factories must use
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
        List of algorithm names (token variants only)
    """
    return _agent_factory.list_available()


# Public API
__all__ = [
    # Agent classes (including base classes for inheritance)
    "BFSAgent",
    "BFSSafeGreedyAgent",
    "BFS512TokenAgent",
    "BFS1024TokenAgent", 
    "BFS2048TokenAgent",
    "BFS4096TokenAgent",
    "BFSSafeGreedy4096TokenAgent",

    # Canonical factory method
    "create",

    # Utility functions
    "get_available_algorithms",

    # Default
    "DEFAULT_ALGORITHM",
] 
