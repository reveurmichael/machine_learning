from __future__ import annotations

from collections import deque
from typing import Deque

import streamlit as st


class MetricsBuffer:  # noqa: D101 – simple helper
    """Fixed-length record buffer feeding Streamlit line chart."""

    def __init__(self, maxlen: int = 200):
        self.rewards: Deque[float] = deque(maxlen=maxlen)
        self.episodes: Deque[int] = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    def add_record(self, episode: int, reward: float) -> None:  # noqa: D401
        self.episodes.append(episode)
        self.rewards.append(reward)
        self._render()

    # ------------------------------------------------------------------
    def _render(self) -> None:  # noqa: D401 – internal
        with st.container():
            st.line_chart(
                {
                    "episode": list(self.episodes),
                    "reward": list(self.rewards),
                },
                x="episode",
                y="reward",
            ) 