from __future__ import annotations

"""Minimal stub for LSTMAgent to satisfy imports in supervised learning v0.02.
This stub allows the rest of the training pipeline to run even though a full
implementation is not yet available. Calling train/predict will raise
NotImplementedError.
"""

from typing import Dict, Any
import numpy as np

from core.game_agents import BaseAgent  # type: ignore


class BaseMLAgent(BaseAgent):
    """Minimal base class placeholder for ML Agents."""

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore
        raise NotImplementedError


class LSTMAgent(BaseMLAgent):
    """Stub LSTM agent (not implemented)."""

    def __init__(self, *args, **kwargs):
        self.description = "Stub LSTM agent (not implemented)"

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("LSTMAgent is a stub and cannot be trained yet.")

    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore
        raise NotImplementedError("LSTMAgent is a stub and cannot predict yet.") 