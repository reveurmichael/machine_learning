from __future__ import annotations

from typing import Any, List

from utils.factory_utils import SimpleFactory

from .agent_mlp import MLPAgent
from .agent_cnn import CNNAgent
from .agent_rnn import RNNAgent
from .agent_lstm import LSTMAgent
from .agent_lightgbm import LightGBMAgent
from .agent_xgboost import XGBoostAgent

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