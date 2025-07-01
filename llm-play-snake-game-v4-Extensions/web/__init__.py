"""
Simple Web Module for Snake Game AI
==================================

Simple web interface components for Task-0 and extensions.
Follows KISS principle: exports only what's needed.

Design Philosophy:
- KISS: Simple, direct exports
- No Over-Preparation: Only current essentials
- Extensible: Easy for Tasks 1-5 to import and extend

Available Components:
- Base Apps: FlaskGameApp, GameFlaskApp
- Specific Apps: HumanWebApp, LLMWebApp, ReplayWebApp  
- Factory Functions: create_human_web_app, create_llm_web_app, create_replay_web_app
"""

# Base application classes
from web.base_app import FlaskGameApp, GameFlaskApp

# Specific application implementations
from web.human_app import HumanWebApp
from web.llm_app import LLMWebApp
from web.replay_app import ReplayWebApp

# Factory functions from centralized factory utilities
from utils.factory_utils import (
    WebAppFactory,
    create_human_web_app,
    create_llm_web_app,
    create_replay_web_app
)

# Export simple interface
__all__ = [
    # Base classes for extensions to inherit
    'FlaskGameApp',
    'GameFlaskApp', 
    
    # Task-0 specific apps
    'HumanWebApp',
    'LLMWebApp',
    'ReplayWebApp',
    
    # Simple factory functions
    'WebAppFactory',
    'create_human_web_app',
    'create_llm_web_app',
    'create_replay_web_app'
] 