"""pipeline.py - Multi-Framework Heuristics → Supervised Learning Pipeline v0.02

Evolution from v0.01: Production-ready pipeline supporting multiple heuristic algorithms
and multiple ML frameworks with comprehensive configuration management.

Key Features:
- Multi-algorithm dataset generation (BFS, A*, Hamiltonian, etc.)
- Multi-framework model training (PyTorch, XGBoost, LightGBM, scikit-learn)
- Advanced configuration management with templates
- Comprehensive evaluation and model comparison
- Progress monitoring and logging
- Checkpoint support and model persistence

Design Patterns:
- Template Method Pattern: Base pipeline with customizable steps
- Strategy Pattern: Different model training strategies
- Factory Pattern: Model and dataset creation
- Observer Pattern: Progress monitoring and logging
- Command Pattern: Pipeline step execution

Usage Examples:
    # Multi-algorithm, multi-model training
    python pipeline.py --algorithms BFS ASTAR HAMILTONIAN --models MLP XGBOOST LIGHTGBM
    
    # Specific configuration with custom parameters
    python pipeline.py --config production.json --output-dir results/experiment1
    
    # Quick development run
    python pipeline.py --dev-mode --algorithms BFS --models MLP --games 100
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fix Python path for extensions
import sys
import os
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure we're working from project root
if os.getcwd() != str(project_root):
    os.chdir(str(project_root))

# Import common utilities
from extensions.common.dataset_directory_manager import DatasetDirectoryManager
from extensions.common.model_utils import save_model_standardized
from extensions.common.training_logging_utils import TrainingLogger
from extensions.common.path_utils import setup_extension_paths

# Import v0.02 components (lazy loading)
from . import check_pytorch_availability, check_xgboost_availability, check_lightgbm_availability, check_sklearn_availability

setup_extension_paths()

__all__ = [
    "MultiFrameworkConfig",
    "ModelStrategy",
    "PyTorchStrategy",
    "XGBoostStrategy", 
    "LightGBMStrategy",
    "SklearnStrategy",
    "MultiFrameworkPipeline",
    "PipelineResults",
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultiFrameworkConfig:
    """Configuration for multi-framework pipeline.
    
    Design Pattern: Configuration Object
    - Encapsulates all pipeline settings
    - Supports serialization and validation
    - Provides default values and templates
    """
    
    # Dataset generation parameters
    algorithms: List[str] = field(default_factory=lambda: ["BFS", "ASTAR"])
    games_per_algorithm: int = 1000
    grid_size: int = 10
    
    # Model training parameters
    models: List[str] = field(default_factory=lambda: ["MLP", "XGBOOST"])
    test_size: float = 0.2
    random_state: int = 42
    
    # Training hyperparameters
    pytorch_epochs: int = 100
    pytorch_batch_size: int = 32
    pytorch_learning_rate: float = 0.001
    pytorch_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    
    lightgbm_n_estimators: int = 100
    lightgbm_max_depth: int = 6
    lightgbm_learning_rate: float = 0.1
    
    sklearn_hidden_layer_sizes: Tuple[int, ...] = field(default_factory=lambda: (128, 64))
    sklearn_max_iter: int = 200
    sklearn_learning_rate_init: float = 0.001
    
    # Pipeline settings
    output_dir: str = "output/heuristics-supervised-v0.02"
    experiment_name: Optional[str] = None
    dev_mode: bool = False
    enable_checkpoints: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.dev_mode:
            # Reduce parameters for development
            self.games_per_algorithm = min(self.games_per_algorithm, 100)
            self.pytorch_epochs = min(self.pytorch_epochs, 10)
            self.xgboost_n_estimators = min(self.xgboost_n_estimators, 10)
            self.lightgbm_n_estimators = min(self.lightgbm_n_estimators, 10)
            self.sklearn_max_iter = min(self.sklearn_max_iter, 20)
        
        if self.experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            algorithms_str = "_".join(self.algorithms[:2])  # First 2 algorithms
            models_str = "_".join(self.models[:2])  # First 2 models
            self.experiment_name = f"{algorithms_str}_{models_str}_{timestamp}"
    
    def save(self, config_path: Union[str, Path]):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, config_path: Union[str, Path]) -> 'MultiFrameworkConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def development_config(cls) -> 'MultiFrameworkConfig':
        """Create development configuration with small parameters."""
        return cls(
            algorithms=["BFS"],
            games_per_algorithm=50,
            models=["MLP"],
            pytorch_epochs=5,
            dev_mode=True
        )
    
    @classmethod
    def production_config(cls) -> 'MultiFrameworkConfig':
        """Create production configuration with optimized parameters."""
        return cls(
            algorithms=["BFS", "ASTAR", "HAMILTONIAN"],
            games_per_algorithm=5000,
            models=["MLP", "XGBOOST", "LIGHTGBM"],
            pytorch_epochs=200,
            xgboost_n_estimators=500,
            lightgbm_n_estimators=500,
            dev_mode=False
        )
    
    @classmethod
    def research_config(cls) -> 'MultiFrameworkConfig':
        """Create research configuration with all algorithms and models."""
        return cls(
            algorithms=["BFS", "ASTAR", "HAMILTONIAN", "DFS"],
            games_per_algorithm=2000,
            models=["MLP", "CNN", "LSTM", "XGBOOST", "LIGHTGBM", "RANDOMFOREST"],
            pytorch_epochs=150,
            dev_mode=False
        )


@dataclass
class PipelineResults:
    """Results from pipeline execution.
    
    Contains all metrics, model paths, and execution statistics.
    """
    
    config: MultiFrameworkConfig
    dataset_stats: Dict[str, Any]
    model_results: Dict[str, Dict[str, Any]]
    execution_time: float
    output_dir: Path
    experiment_name: str
    
    def save(self, results_path: Union[str, Path]):
        """Save results to JSON file."""
        results_dict = {
            "config": self.config.__dict__,
            "dataset_stats": self.dataset_stats,
            "model_results": self.model_results,
            "execution_time": self.execution_time,
            "output_dir": str(self.output_dir),
            "experiment_name": self.experiment_name,
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """Get the best performing model."""
        best_model = None
        best_score = 0.0
        
        for model_name, results in self.model_results.items():
            score = results.get("validation_accuracy", 0.0)
            if score > best_score:
                best_score = score
                best_model = (model_name, results)
        
        return best_model
    
    def print_summary(self):
        """Print a formatted summary of results."""
        print("\n" + "="*60)
        print(f"🎯 Pipeline Results Summary - {self.experiment_name}")
        print("="*60)
        
        print("\n📊 Dataset Statistics:")
        print(f"  Algorithms: {', '.join(self.config.algorithms)}")
        print(f"  Total samples: {self.dataset_stats.get('total_samples', 'N/A')}")
        print(f"  Features: {self.dataset_stats.get('num_features', 'N/A')}")
        
        print("\n🏆 Model Performance:")
        for model_name, results in self.model_results.items():
            accuracy = results.get("validation_accuracy", 0.0)
            training_time = results.get("training_time", 0.0)
            print(f"  {model_name}: {accuracy:.3f} accuracy ({training_time:.1f}s)")
        
        best_model, best_results = self.get_best_model()
        if best_model:
            print(f"\n🥇 Best Model: {best_model[0]} ({best_results['validation_accuracy']:.3f} accuracy)")
        
        print(f"\n⏱️  Total Execution Time: {self.execution_time:.1f} seconds")
        print(f"📁 Output Directory: {self.output_dir}")


class ModelStrategy(ABC):
    """Abstract base class for model training strategies.
    
    Design Pattern: Strategy Pattern
    - Encapsulates different model training algorithms
    - Provides consistent interface across frameworks
    - Allows runtime strategy selection
    """
    
    def __init__(self, name: str, framework: str):
        self.name = name
        self.framework = framework
        self.logger = TrainingLogger(f"strategy_{name.lower()}")
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the required framework is available."""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              config: MultiFrameworkConfig) -> Dict[str, Any]:
        """Train the model and return results."""
        pass
    
    @abstractmethod
    def save_model(self, model: Any, model_path: Path, 
                   config: MultiFrameworkConfig, results: Dict[str, Any]):
        """Save the trained model."""
        pass


class PyTorchStrategy(ModelStrategy):
    """PyTorch neural network training strategy."""
    
    def __init__(self, model_type: str = "MLP"):
        super().__init__(model_type, "pytorch")
        self.model_type = model_type
    
    def is_available(self) -> bool:
        """Check if PyTorch is available."""
        return check_pytorch_availability()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              config: MultiFrameworkConfig) -> Dict[str, Any]:
        """Train PyTorch model."""
        if not self.is_available():
            raise RuntimeError("PyTorch is not available")
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        self.logger.info(f"Training {self.model_type} with PyTorch")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config.pytorch_batch_size, shuffle=True)
        
        # Create model
        if self.model_type == "MLP":
            model = self._create_mlp_model(X_train.shape[1], config.pytorch_hidden_sizes)
        elif self.model_type == "CNN":
            model = self._create_cnn_model(X_train.shape[1])
        else:
            raise ValueError(f"Unsupported PyTorch model type: {self.model_type}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.pytorch_learning_rate)
        
        # Training loop
        start_time = time.time()
        train_losses = []
        val_accuracies = []
        
        for epoch in range(config.pytorch_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
            
            train_losses.append(epoch_loss / len(train_loader))
            val_accuracies.append(val_accuracy)
            
            if (epoch + 1) % 20 == 0:
                self.logger.info(f"Epoch {epoch+1}/{config.pytorch_epochs}: "
                               f"Loss={train_losses[-1]:.4f}, Val Acc={val_accuracy:.4f}")
        
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            train_predictions = torch.argmax(train_outputs, dim=1)
            train_accuracy = (train_predictions == y_train_tensor).float().mean().item()
            
            val_outputs = model(X_val_tensor)
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
        
        return {
            "model": model,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time": training_time,
            "train_losses": train_losses,
            "val_accuracies": val_accuracies,
            "framework": "pytorch",
            "model_type": self.model_type,
        }
    
    def _create_mlp_model(self, input_size: int, hidden_sizes: List[int]) -> 'torch.nn.Module':
        """Create MLP model."""
        import torch.nn as nn
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 4))  # 4 output classes (UP, DOWN, LEFT, RIGHT)
        
        return nn.Sequential(*layers)
    
    def _create_cnn_model(self, input_size: int) -> 'torch.nn.Module':
        """Create CNN model (simplified for tabular data)."""
        import torch.nn as nn
        
        # Simple CNN-like architecture for tabular data
        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    
    def save_model(self, model: Any, model_path: Path, 
                   config: MultiFrameworkConfig, results: Dict[str, Any]):
        """Save PyTorch model."""
        import torch
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
            'config': config.__dict__,
            'results': results,
        }, model_path)


class XGBoostStrategy(ModelStrategy):
    """XGBoost training strategy."""
    
    def __init__(self):
        super().__init__("XGBOOST", "xgboost")
    
    def is_available(self) -> bool:
        """Check if XGBoost is available."""
        return check_xgboost_availability()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              config: MultiFrameworkConfig) -> Dict[str, Any]:
        """Train XGBoost model."""
        if not self.is_available():
            raise RuntimeError("XGBoost is not available")
        
        import xgboost as xgb
        
        self.logger.info("Training XGBoost model")
        
        start_time = time.time()
        
        # Create XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=config.xgboost_n_estimators,
            max_depth=config.xgboost_max_depth,
            learning_rate=config.xgboost_learning_rate,
            random_state=config.random_state,
            eval_metric='mlogloss'
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        
        return {
            "model": model,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time": training_time,
            "feature_importance": model.feature_importances_.tolist(),
            "framework": "xgboost",
            "model_type": "XGBOOST",
        }
    
    def save_model(self, model: Any, model_path: Path, 
                   config: MultiFrameworkConfig, results: Dict[str, Any]):
        """Save XGBoost model."""
        save_model_standardized(
            model=model,
            framework="xgboost",
            grid_size=config.grid_size,
            model_name=f"xgboost_{config.experiment_name}",
            model_class="XGBClassifier",
            input_size=results.get("input_size", 0),
            output_size=4,
            training_params={
                "n_estimators": config.xgboost_n_estimators,
                "max_depth": config.xgboost_max_depth,
                "learning_rate": config.xgboost_learning_rate,
                "val_acc": results["validation_accuracy"],
                "training_time": results["training_time"],
            },
        )


class LightGBMStrategy(ModelStrategy):
    """LightGBM training strategy."""
    
    def __init__(self):
        super().__init__("LIGHTGBM", "lightgbm")
    
    def is_available(self) -> bool:
        """Check if LightGBM is available."""
        return check_lightgbm_availability()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              config: MultiFrameworkConfig) -> Dict[str, Any]:
        """Train LightGBM model."""
        if not self.is_available():
            raise RuntimeError("LightGBM is not available")
        
        import lightgbm as lgb
        
        self.logger.info("Training LightGBM model")
        
        start_time = time.time()
        
        # Create LightGBM model
        model = lgb.LGBMClassifier(
            n_estimators=config.lightgbm_n_estimators,
            max_depth=config.lightgbm_max_depth,
            learning_rate=config.lightgbm_learning_rate,
            random_state=config.random_state,
            verbosity=-1
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        
        return {
            "model": model,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time": training_time,
            "feature_importance": model.feature_importances_.tolist(),
            "framework": "lightgbm",
            "model_type": "LIGHTGBM",
        }
    
    def save_model(self, model: Any, model_path: Path, 
                   config: MultiFrameworkConfig, results: Dict[str, Any]):
        """Save LightGBM model."""
        save_model_standardized(
            model=model,
            framework="lightgbm",
            grid_size=config.grid_size,
            model_name=f"lightgbm_{config.experiment_name}",
            model_class="LGBMClassifier",
            input_size=results.get("input_size", 0),
            output_size=4,
            training_params={
                "n_estimators": config.lightgbm_n_estimators,
                "max_depth": config.lightgbm_max_depth,
                "learning_rate": config.lightgbm_learning_rate,
                "val_acc": results["validation_accuracy"],
                "training_time": results["training_time"],
            },
        )


class SklearnStrategy(ModelStrategy):
    """Scikit-learn training strategy."""
    
    def __init__(self, model_type: str = "MLP"):
        super().__init__(model_type, "sklearn")
        self.model_type = model_type
    
    def is_available(self) -> bool:
        """Check if scikit-learn is available."""
        return check_sklearn_availability()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              config: MultiFrameworkConfig) -> Dict[str, Any]:
        """Train scikit-learn model."""
        if not self.is_available():
            raise RuntimeError("scikit-learn is not available")
        
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        self.logger.info(f"Training {self.model_type} with scikit-learn")
        
        start_time = time.time()
        
        # Create model based on type
        if self.model_type == "MLP":
            model = MLPClassifier(
                hidden_layer_sizes=config.sklearn_hidden_layer_sizes,
                max_iter=config.sklearn_max_iter,
                learning_rate_init=config.sklearn_learning_rate_init,
                random_state=config.random_state
            )
        elif self.model_type == "RANDOMFOREST":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=config.random_state
            )
        else:
            raise ValueError(f"Unsupported scikit-learn model type: {self.model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Evaluate
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        
        results = {
            "model": model,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time": training_time,
            "framework": "sklearn",
            "model_type": self.model_type,
        }
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            results["feature_importance"] = model.feature_importances_.tolist()
        
        return results
    
    def save_model(self, model: Any, model_path: Path, 
                   config: MultiFrameworkConfig, results: Dict[str, Any]):
        """Save scikit-learn model."""
        save_model_standardized(
            model=model,
            framework="sklearn",
            grid_size=config.grid_size,
            model_name=f"sklearn_{self.model_type.lower()}_{config.experiment_name}",
            model_class=type(model).__name__,
            input_size=results.get("input_size", 0),
            output_size=4,
            training_params={
                "model_type": self.model_type,
                "val_acc": results["validation_accuracy"],
                "training_time": results["training_time"],
            },
        )


class MultiFrameworkPipeline:
    """Main pipeline for multi-framework heuristics to supervised learning.
    
    Design Pattern: Template Method
    - Defines pipeline structure with customizable steps
    - Orchestrates dataset generation, model training, and evaluation
    - Provides progress monitoring and result management
    """
    
    def __init__(self, config: MultiFrameworkConfig):
        self.config = config
        self.logger = TrainingLogger("multi_framework_pipeline")
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_strategies(self) -> Dict[str, ModelStrategy]:
        """Initialize all available model strategies."""
        strategies = {}
        
        # PyTorch strategies
        if "MLP" in self.config.models:
            strategies["MLP"] = PyTorchStrategy("MLP")
        if "CNN" in self.config.models:
            strategies["CNN"] = PyTorchStrategy("CNN")
        
        # XGBoost strategy
        if "XGBOOST" in self.config.models:
            strategies["XGBOOST"] = XGBoostStrategy()
        
        # LightGBM strategy
        if "LIGHTGBM" in self.config.models:
            strategies["LIGHTGBM"] = LightGBMStrategy()
        
        # Scikit-learn strategies
        if "SKLEARN_MLP" in self.config.models:
            strategies["SKLEARN_MLP"] = SklearnStrategy("MLP")
        if "RANDOMFOREST" in self.config.models:
            strategies["RANDOMFOREST"] = SklearnStrategy("RANDOMFOREST")
        
        # Filter out unavailable strategies
        available_strategies = {}
        for name, strategy in strategies.items():
            if strategy.is_available():
                available_strategies[name] = strategy
                self.logger.info(f"✅ {name} strategy available")
            else:
                self.logger.warning(f"❌ {name} strategy not available (missing dependencies)")
        
        return available_strategies
    
    def run_pipeline(self) -> PipelineResults:
        """Run the complete pipeline.
        
        Template Method implementation:
        1. Generate datasets
        2. Load and preprocess data
        3. Train models
        4. Evaluate and save results
        """
        start_time = time.time()
        
        self.logger.info(f"🚀 Starting multi-framework pipeline: {self.config.experiment_name}")
        
        # Step 1: Generate datasets
        self.logger.info("📊 Step 1: Generating datasets...")
        dataset_paths = self._generate_datasets()
        
        # Step 2: Load and preprocess data
        self.logger.info("🔄 Step 2: Loading and preprocessing data...")
        X_train, X_val, y_train, y_val, dataset_stats = self._load_and_preprocess_data(dataset_paths)
        
        # Step 3: Train models
        self.logger.info("🧠 Step 3: Training models...")
        model_results = self._train_models(X_train, X_val, y_train, y_val)
        
        # Step 4: Save results
        self.logger.info("💾 Step 4: Saving results...")
        execution_time = time.time() - start_time
        results = PipelineResults(
            config=self.config,
            dataset_stats=dataset_stats,
            model_results=model_results,
            execution_time=execution_time,
            output_dir=self.output_dir,
            experiment_name=self.config.experiment_name
        )
        
        self._save_pipeline_results(results)
        
        self.logger.info(f"✅ Pipeline completed in {execution_time:.1f} seconds")
        return results
    
    def _generate_datasets(self) -> List[Path]:
        """Generate datasets for all algorithms."""
        dataset_paths = []
        
        for algorithm in self.config.algorithms:
            self.logger.info(f"  Generating {algorithm} dataset...")
            
            cmd = [
                "python", "-m", "extensions.common.dataset_generator_cli",
                "--algorithm", algorithm,
                "--games", str(self.config.games_per_algorithm),
                "--format", "csv",
                "--grid-size", str(self.config.grid_size),
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Find the generated dataset
                dataset_dir = DatasetDirectoryManager.grid_size_dir(self.config.grid_size)
                csv_files = sorted(
                    dataset_dir.glob(f"*{algorithm.lower()}*.csv"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                
                if csv_files:
                    dataset_paths.append(csv_files[0])
                    self.logger.info(f"    ✅ Generated: {csv_files[0].name}")
                else:
                    self.logger.error(f"    ❌ Failed to find dataset for {algorithm}")
                    
            except subprocess.CalledProcessError as e:
                self.logger.error(f"    ❌ Dataset generation failed for {algorithm}: {e}")
        
        return dataset_paths
    
    def _load_and_preprocess_data(self, dataset_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load and preprocess datasets."""
        all_dataframes = []
        
        for dataset_path in dataset_paths:
            df = pd.read_csv(dataset_path)
            all_dataframes.append(df)
            self.logger.info(f"  Loaded {len(df)} samples from {dataset_path.name}")
        
        # Combine all datasets
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        self.logger.info(f"  Combined dataset: {len(combined_df)} total samples")
        
        # Prepare features and labels
        label_column = "target_move"
        feature_columns = [col for col in combined_df.columns if col != label_column]
        
        X = combined_df[feature_columns].values
        y = combined_df[label_column].map({
            "UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3
        }).values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        dataset_stats = {
            "total_samples": len(combined_df),
            "num_features": len(feature_columns),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "algorithms": self.config.algorithms,
            "class_distribution": {
                str(i): int(np.sum(y == i)) for i in range(4)
            }
        }
        
        return X_train, X_val, y_train, y_val, dataset_stats
    
    def _train_models(self, X_train: np.ndarray, X_val: np.ndarray, 
                     y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Train all configured models."""
        model_results = {}
        
        for model_name, strategy in self.strategies.items():
            self.logger.info(f"  Training {model_name}...")
            
            try:
                results = strategy.train(X_train, y_train, X_val, y_val, self.config)
                results["input_size"] = X_train.shape[1]
                
                # Save model
                model_path = self.output_dir / f"{model_name.lower()}_model"
                strategy.save_model(results["model"], model_path, self.config, results)
                results["model_path"] = str(model_path)
                
                # Remove the actual model object to make results JSON serializable
                del results["model"]
                
                model_results[model_name] = results
                
                accuracy = results["validation_accuracy"]
                training_time = results["training_time"]
                self.logger.info(f"    ✅ {model_name}: {accuracy:.3f} accuracy ({training_time:.1f}s)")
                
            except Exception as e:
                self.logger.error(f"    ❌ Failed to train {model_name}: {e}")
                model_results[model_name] = {
                    "error": str(e),
                    "training_failed": True
                }
        
        return model_results
    
    def _save_pipeline_results(self, results: PipelineResults):
        """Save pipeline results and configuration."""
        # Save main results
        results_path = self.output_dir / "pipeline_results.json"
        results.save(results_path)
        
        # Save configuration
        config_path = self.output_dir / "pipeline_config.json"
        self.config.save(config_path)
        
        # Save summary report
        summary_path = self.output_dir / "summary_report.txt"
        with open(summary_path, 'w') as f:
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            results.print_summary()
            sys.stdout = old_stdout
        
        self.logger.info(f"  💾 Results saved to: {self.output_dir}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Heuristics → Supervised Learning Integration v0.02",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--save-config", help="Save current configuration to file")
    
    # Template configurations
    parser.add_argument("--template", choices=["dev", "production", "research"],
                       help="Use configuration template")
    
    # Dataset generation parameters
    parser.add_argument("--algorithms", nargs="+", default=["BFS", "ASTAR"],
                       help="Heuristic algorithms to use")
    parser.add_argument("--games-per-algorithm", type=int, default=1000,
                       help="Number of games per algorithm")
    parser.add_argument("--grid-size", type=int, default=10,
                       help="Grid size for snake game")
    
    # Model training parameters
    parser.add_argument("--models", nargs="+", default=["MLP", "XGBOOST"],
                       help="Models to train")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set proportion")
    
    # Training hyperparameters
    parser.add_argument("--pytorch-epochs", type=int, default=100,
                       help="PyTorch training epochs")
    parser.add_argument("--pytorch-lr", type=float, default=0.001,
                       help="PyTorch learning rate")
    parser.add_argument("--xgboost-estimators", type=int, default=100,
                       help="XGBoost number of estimators")
    parser.add_argument("--lightgbm-estimators", type=int, default=100,
                       help="LightGBM number of estimators")
    
    # Pipeline settings
    parser.add_argument("--output-dir", default="output/heuristics-supervised-v0.02",
                       help="Output directory")
    parser.add_argument("--experiment-name", help="Experiment name")
    parser.add_argument("--dev-mode", action="store_true",
                       help="Use development mode (smaller parameters)")
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load or create configuration
        if args.config:
            config = MultiFrameworkConfig.load(args.config)
            logger.info(f"📋 Loaded configuration from: {args.config}")
        elif args.template:
            if args.template == "dev":
                config = MultiFrameworkConfig.development_config()
            elif args.template == "production":
                config = MultiFrameworkConfig.production_config()
            elif args.template == "research":
                config = MultiFrameworkConfig.research_config()
            logger.info(f"📋 Using {args.template} configuration template")
        else:
            # Create configuration from arguments
            config = MultiFrameworkConfig(
                algorithms=args.algorithms,
                games_per_algorithm=args.games_per_algorithm,
                grid_size=args.grid_size,
                models=args.models,
                test_size=args.test_size,
                pytorch_epochs=args.pytorch_epochs,
                pytorch_learning_rate=args.pytorch_lr,
                xgboost_n_estimators=args.xgboost_estimators,
                lightgbm_n_estimators=args.lightgbm_estimators,
                output_dir=args.output_dir,
                experiment_name=args.experiment_name,
                dev_mode=args.dev_mode,
            )
            logger.info("📋 Created configuration from command line arguments")
        
        # Save configuration if requested
        if args.save_config:
            config.save(args.save_config)
            logger.info(f"💾 Configuration saved to: {args.save_config}")
            return 0
        
        # Create and run pipeline
        pipeline = MultiFrameworkPipeline(config)
        results = pipeline.run_pipeline()
        
        # Print summary
        results.print_summary()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline cancelled by user")
        return 130
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 