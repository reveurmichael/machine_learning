from core.game_agents import BaseAgent
import lightgbm as lgb
import json
from pathlib import Path
from extensions.common.model_utils import get_model_directory

class LightGBMAgent(BaseAgent):
    """
    LightGBM agent for tabular feature data.
    """
    def __init__(self, grid_size: int = 10, **kwargs):
        self.grid_size = grid_size
        self.model = lgb.LGBMClassifier(**kwargs)
        self.is_trained = False
    def predict(self, X):
        return self.model.predict(X)
    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
    def save_model(self, model_name: str):
        """
        Save the LightGBM model in text format with metadata.
        """
        model_dir = get_model_directory('lightgbm', self.grid_size)
        model_path = Path(model_dir) / f"{model_name}.txt"
        self.model.booster_.save_model(str(model_path))
        metadata = {
            'framework': 'LightGBM',
            'grid_size': self.grid_size,
            'model_class': self.__class__.__name__,
            'timestamp': str(Path(model_path).stat().st_mtime),
        }
        with open(Path(model_dir) / f"{model_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model and metadata saved to {model_dir}")
    def load_model(self, model_name: str):
        """
        Load the LightGBM model from text format.
        """
        model_dir = get_model_directory('lightgbm', self.grid_size)
        model_path = Path(model_dir) / f"{model_name}.txt"
        self.model = lgb.Booster(model_file=str(model_path))
        self.is_trained = True 