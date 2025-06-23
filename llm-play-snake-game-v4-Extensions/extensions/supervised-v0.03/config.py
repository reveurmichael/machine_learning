"""
Supervised Learning v0.03 - Configuration Management
==================================================

Centralized configuration for supervised learning models.
Uses Pydantic for type safety and validation.

Design Pattern: Configuration Object Pattern
- Centralized configuration management
- Type-safe settings with validation
- Environment variable support
- Default values for all parameters
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseSettings, Field, validator
import os


class ModelConfig(BaseSettings):
    """Configuration for individual model types."""
    
    # Neural Network Config
    hidden_size: int = Field(default=256, description="Hidden layer size for neural networks")
    learning_rate: float = Field(default=0.001, description="Learning rate for training")
    batch_size: int = Field(default=32, description="Batch size for training")
    epochs: int = Field(default=100, description="Number of training epochs")
    dropout_rate: float = Field(default=0.2, description="Dropout rate for regularization")
    
    # Tree Model Config
    max_depth: int = Field(default=6, description="Maximum depth for tree models")
    n_estimators: int = Field(default=100, description="Number of estimators for ensemble models")
    min_samples_split: int = Field(default=2, description="Minimum samples for split")
    
    # Training Config
    validation_split: float = Field(default=0.2, description="Validation split ratio")
    early_stopping_patience: int = Field(default=10, description="Early stopping patience")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    
    # Model Saving Config
    save_best_only: bool = Field(default=True, description="Save only best model")
    export_onnx: bool = Field(default=True, description="Export PyTorch models to ONNX")
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if v <= 0 or v > 1:
            raise ValueError('Learning rate must be between 0 and 1')
        return v
    
    @validator('validation_split')
    def validate_validation_split(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError('Validation split must be between 0 and 1')
        return v


class TrainingConfig(BaseSettings):
    """Configuration for training pipeline."""
    
    # Dataset Config
    dataset_path: Optional[Path] = Field(default=None, description="Path to training dataset")
    dataset_format: str = Field(default="csv", description="Dataset format (csv, npz, parquet)")
    feature_columns: Optional[List[str]] = Field(default=None, description="Feature column names")
    target_column: str = Field(default="target_move", description="Target column name")
    
    # Training Pipeline Config
    max_games: int = Field(default=1000, description="Maximum number of games to generate")
    grid_size: int = Field(default=10, description="Grid size for the game")
    use_gui: bool = Field(default=False, description="Use GUI during training")
    verbose: bool = Field(default=True, description="Verbose output during training")
    
    # Output Config
    output_dir: Optional[Path] = Field(default=None, description="Output directory for models")
    experiment_name: Optional[str] = Field(default=None, description="Experiment name")
    save_frequency: int = Field(default=100, description="Save model every N epochs")
    
    @validator('grid_size')
    def validate_grid_size(cls, v):
        if v < 5 or v > 50:
            raise ValueError('Grid size must be between 5 and 50')
        return v
    
    @validator('max_games')
    def validate_max_games(cls, v):
        if v <= 0:
            raise ValueError('Max games must be positive')
        return v


class EvaluationConfig(BaseSettings):
    """Configuration for model evaluation."""
    
    # Evaluation Metrics
    metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1"],
        description="Evaluation metrics to compute"
    )
    
    # Test Config
    test_size: float = Field(default=0.2, description="Test set size ratio")
    cross_validation_folds: int = Field(default=5, description="Cross-validation folds")
    
    # Performance Config
    inference_batch_size: int = Field(default=64, description="Batch size for inference")
    measure_latency: bool = Field(default=True, description="Measure inference latency")
    
    @validator('test_size')
    def validate_test_size(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError('Test size must be between 0 and 1')
        return v


class SupervisedConfig(BaseSettings):
    """Main configuration class for supervised learning v0.03."""
    
    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")
    
    # Global Config
    log_level: str = Field(default="INFO", description="Logging level")
    num_workers: int = Field(default=4, description="Number of worker processes")
    device: str = Field(default="auto", description="Device for training (auto, cpu, cuda)")
    
    class Config:
        env_prefix = "SUPERVISED_"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup default paths if not provided."""
        if self.training.output_dir is None:
            timestamp = os.environ.get("SUPERVISED_TIMESTAMP", "latest")
            self.training.output_dir = Path(f"logs/extensions/models/grid-size-{self.training.grid_size}")
        
        if self.training.experiment_name is None:
            self.training.experiment_name = f"supervised_{self.training.grid_size}x{self.training.grid_size}"
    
    def get_model_directory(self, model_type: str) -> Path:
        """Get model directory for specific model type."""
        return self.training.output_dir / model_type.lower()
    
    def get_experiment_directory(self) -> Path:
        """Get experiment-specific directory."""
        return self.training.output_dir / self.training.experiment_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.dict(),
            "training": self.training.dict(),
            "evaluation": self.evaluation.dict(),
            "log_level": self.log_level,
            "num_workers": self.num_workers,
            "device": self.device
        }
    
    def save_config(self, filepath: Path):
        """Save configuration to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_config(cls, filepath: Path) -> 'SupervisedConfig':
        """Load configuration from file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# Default configuration instance
default_config = SupervisedConfig() 