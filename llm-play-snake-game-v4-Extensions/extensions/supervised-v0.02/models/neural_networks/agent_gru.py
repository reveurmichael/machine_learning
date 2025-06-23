from __future__ import annotations

"""Minimal stub for GRUAgent to satisfy imports in supervised learning v0.02."""
from typing import Dict, Any
import numpy as np
from core.game_agents import BaseAgent  # type: ignore

class BaseMLAgent(BaseAgent):
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore
        raise NotImplementedError

class GRUAgent(BaseMLAgent):
    def __init__(self, *args, **kwargs):
        self.description = "Stub GRU agent (not implemented)"
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("GRUAgent is a stub and cannot be trained yet.")
    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore
        raise NotImplementedError("GRUAgent is a stub and cannot predict yet.") 