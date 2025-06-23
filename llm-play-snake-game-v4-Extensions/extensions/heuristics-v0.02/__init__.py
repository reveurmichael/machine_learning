"""
Heuristics v0.02 - Multi-Algorithm Snake Agents
--------------------

Evolution from v0.01: This version demonstrates natural software progression
by expanding from a single BFS algorithm to a comprehensive suite of 7
heuristic algorithms, while maintaining the same base class architecture.

Key improvements in v0.02:
- Multiple algorithm support (BFS, DFS, A*, Hamiltonian, and hybrids)
- Factory pattern for algorithm selection
- Enhanced safety validation
- Verbose mode for debugging
- Simplified logging (no Task-0 replay compatibility)
- Organized agents into proper package structure

Available Algorithms:
1. BFS - Pure breadth-first search (from v0.01)
2. BFS-SAFE-GREEDY - Enhanced BFS with safety checks (inherits from BFS)
3. BFS-HAMILTONIAN - BFS with Hamiltonian fallback (inherits from BFS-Safe-Greedy)
4. DFS - Depth-first search (educational comparison)
5. ASTAR - A* pathfinding with Manhattan heuristic
6. ASTAR-HAMILTONIAN - A* with Hamiltonian fallback (inherits from A*)
7. HAMILTONIAN - Pure Hamiltonian cycle (guaranteed safety)

Design Philosophy:
- Extends BaseGameManager, BaseGameLogic, and BaseGameData
- No GUI dependencies (headless by default)
- Generates the same log format as Task-0 (game_N.json, summary.json)
- Uses SnakeAgent protocol for clean integration
- Demonstrates inheritance patterns for software evolution

This extension demonstrates how future tasks can leverage the base classes
while implementing their own specific algorithms.
"""

from ..common.path_utils import ensure_project_root_on_path
ensure_project_root_on_path()

# Import from agents package (factory pattern)
from .agents import (
    BFSAgent,
    BFSSafeGreedyAgent,
    BFSHamiltonianAgent,
    DFSAgent,
    AStarAgent,
    AStarHamiltonianAgent,
    HamiltonianAgent,
    create_agent,
    get_available_algorithms,
    get_algorithm_info,
)

# Import core components
from .game_manager import HeuristicGameManager
from .game_logic import HeuristicGameLogic

__all__ = [
    # Agents (7 algorithms)
    "BFSAgent",
    "BFSSafeGreedyAgent",
    "BFSHamiltonianAgent", 
    "DFSAgent",
    "AStarAgent",
    "AStarHamiltonianAgent",
    "HamiltonianAgent",
    
    # Factory functions
    "create_agent",
    "get_available_algorithms", 
    "get_algorithm_info",
    
    # Core components
    "HeuristicGameManager",
    "HeuristicGameLogic",
] 