from __future__ import annotations

from utils.path_utils import ensure_project_root
ensure_project_root()

"""
Heuristics v0.04 - Advanced Heuristic Agents with Streamlit Interface
--------------------

This extension demonstrates software evolution from v0.03 to v0.04,
adding a comprehensive Streamlit interface and enhanced replay capabilities.

Key Features:
- Streamlit dashboard with multiple tabs (overview, launch, replay, analysis)
- PyGame and web-based replay systems
- Factory pattern for 7 different heuristic algorithms
- Inheritance hierarchy showing software evolution
- Reuse of Task-0 base classes and utilities

Evolution from v0.03:
- Added Streamlit app.py as primary interface
- Enhanced replay capabilities (PyGame + Flask web)
- Improved agent factory with better error handling
- Added comprehensive documentation and educational content

Design Patterns:
- Factory Pattern: Agent creation through factory functions
- Inheritance: Progressive enhancement through class hierarchy
- MVC Pattern: Separation of game logic, data, and presentation
- Singleton Pattern: File manager and other shared resources
- Strategy Pattern: Interchangeable pathfinding algorithms

Available Algorithms:
1. BFS - Basic breadth-first search
2. BFS-SAFE-GREEDY - Enhanced BFS with safety validation
3. BFS-HAMILTONIAN - BFS with Hamiltonian cycle fallback
4. DFS - Depth-first search (educational)
5. ASTAR - A* pathfinding with Manhattan heuristic
6. ASTAR-HAMILTONIAN - A* with Hamiltonian fallback
7. HAMILTONIAN - Pure Hamiltonian cycle

Inheritance Hierarchy:
- BFSAgent (base)
  └── BFSSafeGreedyAgent (adds safety)
      └── BFSHamiltonianAgent (adds Hamiltonian)
- AStarAgent (base)
  └── AStarHamiltonianAgent (adds Hamiltonian)
- DFSAgent (standalone)
- HamiltonianAgent (standalone)
"""

# Import factory functions from agents package
from .agents import (
    create_agent,
    get_available_algorithms,
    get_algorithm_info,
    ALGORITHM_REGISTRY
)

# Import main components
from . import game_manager
from . import game_logic

# Version information
__version__ = "0.04"
__author__ = "Heuristics Extension Team"

# Public API
__all__ = [
    # Factory functions
    "create_agent",
    "get_available_algorithms", 
    "get_algorithm_info",
    "ALGORITHM_REGISTRY",
    
    # Main modules
    "game_manager",
    "game_logic",
    
    # Version info
    "__version__",
    "__author__",
] 