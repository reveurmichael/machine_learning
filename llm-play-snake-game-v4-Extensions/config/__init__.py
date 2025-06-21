"""Config package – provides constants and prompt templates.

This file re-exports all names from submodules so existing imports like
`from config import COLORS, GRID_SIZE` continue to work unchanged.
"""

# Configuration package exports
# 
# This module exports both base configuration classes (for Tasks 0-5)
# and task-specific configuration classes following the BaseClassBlabla philosophy.

# Universal constants (used by ALL tasks 0-5)
from .game_constants import *  # noqa: F401,F403
from .ui_constants import *  # noqa: F401,F403
from .network_constants import *  # noqa: F401,F403
from .web_constants import *  # noqa: F401,F403

# Task-0 specific constants (LLM-specific)
from .llm_constants import *  # noqa: F401,F403
from .prompt_templates import *  # noqa: F401,F403
