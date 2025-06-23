from __future__ import annotations

"""Reinforcement-Learning v0.03 – Dashboard & Replay Upgrade

This version builds on v0.02 (multi-algorithm head-less training) and adds:

• Streamlit dashboard for live training curves.  
• Optional PyGame & web replays (see `scripts/`).  
• Same *agents* folder copied verbatim – absolutely **no behavioural drift**.

It publicly re-exports :class:`RLConfig` and :pyfunc:`create_rl_agent` from
v0.02 so that external code keeps working.
"""

from importlib import import_module as _im

# Re-export config & factory from v0.02 ---------------------------------------
_RL02 = _im("extensions.reinforcement-v0.02")
RLConfig = _RL02.RLConfig  # type: ignore[attr-defined]
create_rl_agent = _RL02.create_rl_agent  # type: ignore[attr-defined]

__all__ = [
    "RLConfig",
    "create_rl_agent",
] 