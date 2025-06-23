from __future__ import annotations

"""RL Game Logic – v0.02

A **very lightweight** snake environment that adheres to the gym-like API that
RL agents expect: ``reset()`` returns an observation, ``step(action)`` returns
``obs, reward, done, info``.

For full-fledged physics (self-collision, apple spawning, etc.) consult the
v0.01 implementation; here we keep only the essentials so the codebase remains
readable in a single glance.
"""

from typing import Dict, Tuple, Any

import numpy as np

from core.game_logic import BaseGameLogic
from config.game_constants import VALID_MOVES

from .game_data import RLGameData


class RLGameLogic(BaseGameLogic):  # noqa: D101 – docstring above
    GAME_DATA_CLS = RLGameData

    def __init__(self, grid_size: int = 10, use_gui: bool | None = False):
        super().__init__(grid_size=grid_size, use_gui=bool(use_gui))
        # Observation encoder (very naive: flattened one-hot of head + apple)
        self._obs_size = 4  # head_x, head_y, apple_x, apple_y

    # ------------------------------------------------------------------
    # Public API expected by RL agents
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:  # noqa: D401
        """Reset environment and return initial observation."""
        self.snake_positions = np.array([[self.grid_size // 2, self.grid_size // 2]])
        self.current_direction = None
        self.apple_position = self._generate_apple()
        self.game_state.reset_episode()
        return self._encode_state()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:  # noqa: D401
        """Perform one environment step.

        Parameters
        ----------
        action_idx
            Integer in ``[0, 3]`` mapping to ``VALID_MOVES``: ``UP, DOWN, LEFT, RIGHT``.
        """
        direction = VALID_MOVES[action_idx]
        self._move_snake(direction)

        # Reward: +1 apple, -1 death, small living penalty
        reward = -0.01
        done = False
        if np.array_equal(self.head_position, self.apple_position):
            reward = 1.0
            self._grow_snake()
            self.apple_position = self._generate_apple()
        elif self._is_collision(self.head_position):
            reward = -1.0
            done = True

        self.game_state.record_step(reward, score=len(self.snake_positions) - 1)
        if done:
            self.game_state.end_episode()
        return self._encode_state(), reward, done, {}

    # ------------------------------------------------------------------
    # Internal helpers (very bare-bones) --------------------------------
    # ------------------------------------------------------------------

    def _generate_apple(self):  # noqa: D401
        return np.random.randint(0, self.grid_size, size=2)

    def _move_snake(self, direction: str):
        dir_map = {
            "UP": np.array([0, 1]),
            "DOWN": np.array([0, -1]),
            "LEFT": np.array([-1, 0]),
            "RIGHT": np.array([1, 0]),
        }
        delta = dir_map[direction]
        new_head = self.snake_positions[-1] + delta
        self.snake_positions = np.vstack([self.snake_positions[1:], new_head])
        self.head_position = new_head

    def _grow_snake(self):  # noqa: D401
        self.snake_positions = np.vstack([self.snake_positions, self.snake_positions[-1]])

    def _is_collision(self, pos: np.ndarray) -> bool:  # noqa: D401
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        # Self collision (ignore last tail cell because it moves forward)
        return any((pos == seg).all() for seg in self.snake_positions[:-1])

    def _encode_state(self) -> np.ndarray:  # noqa: D401
        # Normalise coordinates to [0,1]
        hx, hy = self.snake_positions[-1] / self.grid_size
        ax, ay = self.apple_position / self.grid_size
        return np.array([hx, hy, ax, ay], dtype=np.float32) 