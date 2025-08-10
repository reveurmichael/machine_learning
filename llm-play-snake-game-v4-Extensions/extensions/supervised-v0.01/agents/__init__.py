from __future__ import annotations

from typing import Any, List

from utils.factory_utils import SimpleFactory

from .mlp_agent import MLPAgent
from .cnn_agent import CNNAgent
from .rnn_agent import RNNAgent
from .lstm_agent import LSTMAgent
from .lightgbm_agent import LightGBMAgent
from .xgboost_agent import XGBoostAgent

_factory = SimpleFactory("SupervisedAgentFactory")
# Supervised ML family
_factory.register("MLP", MLPAgent)
_factory.register("CNN", CNNAgent)
_factory.register("RNN", RNNAgent)
_factory.register("LSTM", LSTMAgent)
_factory.register("LIGHTGBM", LightGBMAgent)
_factory.register("XGBOOST", XGBoostAgent)

DEFAULT_ALGORITHM = "MLP"

def create(algorithm_name: str, **kwargs) -> Any:
    return _factory.create(algorithm_name, **kwargs)

def get_available_algorithms() -> List[str]:
    return _factory.list_available()

__all__ = [
    "create",
    "get_available_algorithms",
    "DEFAULT_ALGORITHM",
    "MLPAgent",
    "CNNAgent",
    "RNNAgent",
    "LSTMAgent",
    "LightGBMAgent",
    "XGBoostAgent",
]