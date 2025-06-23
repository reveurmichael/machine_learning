"""
XGBoostAgent for Supervised Learning v0.02
--------------------

Implements an XGBoost-based agent for tabular data.
Inherits from BaseAgent and provides train/predict interface.
"""
from core.game_agents import SnakeAgent
import xgboost as xgb
import json
from pathlib import Path
from extensions.common.model_utils import get_model_directory

class XGBoostAgent(SnakeAgent):
    """
    XGBoost agent for tabular feature data.
    """
    def __init__(self, grid_size: int = 10, **kwargs):
        self.grid_size = grid_size
        self.model = xgb.XGBClassifier(**kwargs)
        self.is_trained = False
    def predict(self, X):
        return self.model.predict(X)
    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
    def save_model(self, model_name: str):
        """
        Save the XGBoost model in JSON format with metadata.
        """
        model_dir = get_model_directory('xgboost', self.grid_size)
        model_path = Path(model_dir) / f"{model_name}.json"
        self.model.get_booster().save_model(str(model_path))
        metadata = {
            'framework': 'XGBoost',
            'grid_size': self.grid_size,
            'model_class': self.__class__.__name__,
            'timestamp': str(Path(model_path).stat().st_mtime),
        }
        with open(Path(model_dir) / f"{model_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model and metadata saved to {model_dir}")
    def load_model(self, model_name: str):
        """
        Load the XGBoost model from JSON format.
        """
        model_dir = get_model_directory('xgboost', self.grid_size)
        model_path = Path(model_dir) / f"{model_name}.json"
        self.model.load_model(str(model_path))
        self.is_trained = True 