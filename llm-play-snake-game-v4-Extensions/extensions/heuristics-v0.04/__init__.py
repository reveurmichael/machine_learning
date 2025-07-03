import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from __future__ import annotations

from utils.path_utils import ensure_project_root
ensure_project_root()

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
] 