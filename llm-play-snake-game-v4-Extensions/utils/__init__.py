"""
Lightweight utilities package initializer.

This package intentionally avoids importing submodules at import time to prevent
heavy transitive dependencies (LLM clients, network providers) from being pulled
in during headless extension runs. Always import the specific submodule you need,
e.g. `from utils.print_utils import print_info`.
"""

__all__ = []
