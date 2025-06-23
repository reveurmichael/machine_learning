from __future__ import annotations

"""RL Game Manager – v0.02

Coordinates training sessions for *multiple* RL algorithms (DQN, PPO, A3C,
SAC).  It extends :class:`core.game_manager.BaseGameManager` via the Template
Method pattern but leaves the main game loop untouched – we only plug custom
behaviour into the *initialisation* and *episode* hooks.

This file deliberately focusses on **architecture** rather than performance.
It demonstrates how second-citizen tasks can reuse Task-0 infrastructure while
remaining completely isolated in the ``extensions/`` namespace.
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import argparse

from core.game_manager import BaseGameManager

from .game_logic import RLGameLogic
from . import create_rl_agent


class RLGameManager(BaseGameManager):  # noqa: D101 – docstring above
    GAME_LOGIC_CLS = RLGameLogic

    # ------------------------------------------------------------------
    # Initialisation ----------------------------------------------------
    # ------------------------------------------------------------------

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.algorithm = getattr(args, "algorithm", "DQN").upper()
        self.episodes = getattr(args, "episodes", 1000)
        self.output_dir = Path(
            getattr(args, "output_dir", "logs/extensions/models")
        ).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Log directory --------------------------------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path("logs") / "extensions" / f"reinforcement-{self.algorithm.lower()}_{ts}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RL-v0.02] Logging to {self.log_dir.relative_to(Path.cwd())}")

        # Create RL agent via factory
        self.agent = create_rl_agent(
            self.algorithm,
            state_size=4,  # from RLGameLogic._obs_size
            action_size=4,  # up/down/left/right
        )
        self.agent.add_observer(self._on_agent_event)

    # ------------------------------------------------------------------
    # Observer callback --------------------------------------------------
    # ------------------------------------------------------------------

    def _on_agent_event(self, event_type: str, data: Dict[str, Any]) -> None:  # noqa: D401
        if event_type == "episode_complete":
            ep_id = data["id"]
            reward = data["reward"]
            print(f"Episode {ep_id} finished – reward = {reward:.2f}")

    # ------------------------------------------------------------------
    # Main run loop (Template override) ---------------------------------
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: D401
        """Training entry-point – minimal loop to showcase integration."""
        env: RLGameLogic = self.game  # type: ignore[assignment]
        for ep in range(1, self.episodes + 1):
            state = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.learn()
                state = next_state
                total_reward += reward
            # Notify agent/observers
            if hasattr(self.agent, "end_episode"):
                self.agent.end_episode(ep, total_reward)

        print("\n[RL-v0.02] Training complete 🚀") 