from __future__ import annotations

"""Agents package – Reinforcement v0.03

This directory intentionally **proxies** to the v0.02 implementation.  The
source files remain byte-for-byte identical to prevent version drift, fulfilling
the guideline that *agents folders are copied unchanged across version bumps*.

If you need to inspect or modify an algorithm, open
`extensions/reinforcement-v0.02/agents/`.
"""

from importlib import import_module as _im
import sys as _sys

_real_pkg = "extensions.reinforcement-v0.02.agents"
_mod = _im(_real_pkg)
_sys.modules[__name__] = _mod

# Re-export everything --------------------
from extensions.reinforcement-v0.02.agents import *  # type: ignore

__all__ = _mod.__all__  # type: ignore[attr-defined] 