import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from __future__ import annotations

from utils.path_utils import ensure_project_root
ensure_project_root()

# Import canonical factory functions from agents package
from .agents import (
    create,
    get_available_algorithms,
    get_algorithm_info,
    DEFAULT_ALGORITHM
)

# Import main components
from . import game_manager
from . import game_logic

# Version information
__version__ = "0.04"
__author__ = "Heuristics Extension Team"

# Public API
__all__ = [
    # Canonical factory functions
    "create",
    "get_available_algorithms", 
    "get_algorithm_info",
    "DEFAULT_ALGORITHM",
    
    # Main modules
    "game_manager",
    "game_logic",
] 