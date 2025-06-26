from __future__ import annotations

from ..common.path_utils import ensure_project_root_on_path
ensure_project_root_on_path()

"""Heuristics v0.03 – CORE DATA GENERATORS
--------------------

Heuristic algorithms are *foundational* to the entire second-citizen
research pipeline:

1.  They produce **high-quality trajectories** that supervise the first wave
    of ML models (Task-2 – supervised learning).
2.  Those same trajectories, once converted to language-rich JSONL in
    *heuristics-v0.04*, become the **seed corpus for LLM fine-tuning**
    (Task-4).

Therefore *every* later task depends transitively on the correctness and
variety of this package.  Treat modifications with the same care you would
apply to core engine changes – tests first, benchmarks second.
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
__version__ = "0.03"
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