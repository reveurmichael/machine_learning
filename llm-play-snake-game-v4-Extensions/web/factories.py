"""
Simple Web Application Factory Functions
=======================================

Simple factory functions for creating Snake Game web applications.
Follows KISS principle: simple functions instead of complex classes.

Design Philosophy:
- KISS: Simple functions instead of complex factory classes
- DRY: Minimal duplication, maximum clarity
- No Over-Preparation: Only what's needed for Task-0
- Extensible: Easy for Tasks 1-5 to add their own functions
"""

from typing import Optional

from web.human_app import HumanWebApp
from web.llm_app import LLMWebApp
from web.replay_app import ReplayWebApp


def create_human_web_app(grid_size: int = 10, port: Optional[int] = None) -> HumanWebApp:
    """Create human web application."""
    return HumanWebApp(grid_size=grid_size, port=port)


def create_llm_web_app(provider: str = "hunyuan", model: str = "hunyuan-turbos-latest",
                      grid_size: int = 10, port: Optional[int] = None) -> LLMWebApp:
    """Create LLM web application."""
    return LLMWebApp(provider=provider, model=model, grid_size=grid_size, port=port)


def create_replay_web_app(log_dir: str, game_number: int = 1, 
                         port: Optional[int] = None) -> ReplayWebApp:
    """Create replay web application."""
    return ReplayWebApp(log_dir=log_dir, game_number=game_number, port=port)

