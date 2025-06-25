"""Config package – provides constants and prompt templates.

This file re-exports all names from submodules so existing imports like
`from config import COLORS, GRID_SIZE` continue to work unchanged.
"""
from .game_constants import *  # noqa: F401,F403
from .llm_constants import *  # noqa: F401,F403
from .prompt_templates import *  # noqa: F401,F403
from .ui_constants import *  # noqa: F401,F403
from .network_constants import *  # noqa: F401,F403
