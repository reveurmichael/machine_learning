"""training_config.py - Advanced Training Configuration Management for v0.02

Evolution from v0.01: Sophisticated configuration management with templates,
builders, and validation for production-ready training workflows.

Key Features:
- Configuration templates for common scenarios
- Builder pattern for custom configurations
- Validation and constraint checking
- Framework-specific parameter optimization
- Hyperparameter search space definition

Design Patterns:
- Builder Pattern: Flexible configuration construction
- Template Method Pattern: Configuration template structure
- Strategy Pattern: Different validation strategies
- Factory Pattern: Configuration creation from templates

Usage Examples:
    # Create from template
    config = TrainingConfigBuilder.from_template("production")
    
    # Custom configuration
    config = (TrainingConfigBuilder()
              .add_algorithms(["BFS", "ASTAR"])
              .add_models(["MLP", "XGBOOST"])
              .set_training_params(epochs=150)
              .build())
    
    # Hyperparameter optimization
    search_space = HyperparameterSearchSpace.create_for_model("XGBOOST")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple

# Import pipeline components
from .pipeline import MultiFrameworkConfig

__all__ = [
    "TrainingConfigManager",
    "TrainingConfigBuilder", 
    "ConfigurationTemplate",
    "HyperparameterSearchSpace",
    "ConfigurationValidator",
]

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSearchSpace:
    """Defines search space for hyperparameter optimization.
    
    Design Pattern: Value Object
    - Encapsulates hyperparameter ranges and constraints
    - Provides sampling methods for optimization
    - Framework-agnostic parameter definitions
    """
    
    pytorch_lr_range: Tuple[float, float] = (1e-5, 1e-1)
    pytorch_epochs_range: Tuple[int, int] = (50, 300)
    pytorch_batch_size_options: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    pytorch_hidden_sizes_options: List[List[int]] = field(default_factory=lambda: [
        [64], [128], [256], [128, 64], [256, 128], [512, 256, 128]
    ])
    
    xgboost_n_estimators_range: Tuple[int, int] = (50, 1000)
    xgboost_max_depth_range: Tuple[int, int] = (3, 15)
    xgboost_lr_range: Tuple[float, float] = (0.01, 0.3)
    
    lightgbm_n_estimators_range: Tuple[int, int] = (50, 1000)
    lightgbm_max_depth_range: Tuple[int, int] = (3, 15)
    lightgbm_lr_range: Tuple[float, float] = (0.01, 0.3)
    
    sklearn_hidden_layers_options: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (64,), (128,), (256,), (128, 64), (256, 128), (512, 256)
    ])
    sklearn_max_iter_range: Tuple[int, int] = (100, 500)
    sklearn_lr_range: Tuple[float, float] = (1e-4, 1e-1)
    
    @classmethod
    def create_for_model(cls, model_name: str) -> 'HyperparameterSearchSpace':
        """Create search space optimized for specific model."""
        if model_name in ["MLP", "CNN"]:
            # PyTorch models
            return cls(
                pytorch_epochs_range=(100, 200),
                pytorch_lr_range=(1e-4, 1e-2),
                pytorch_batch_size_options=[32, 64, 128]
            )
        elif model_name == "XGBOOST":
            return cls(
                xgboost_n_estimators_range=(100, 500),
                xgboost_max_depth_range=(4, 12),
                xgboost_lr_range=(0.05, 0.2)
            )
        elif model_name == "LIGHTGBM":
            return cls(
                lightgbm_n_estimators_range=(100, 500),
                lightgbm_max_depth_range=(4, 12),
                lightgbm_lr_range=(0.05, 0.2)
            )
        else:
            return cls()
    
    def sample_pytorch_params(self) -> Dict[str, Any]:
        """Sample PyTorch hyperparameters."""
        import random
        
        return {
            "learning_rate": random.uniform(*self.pytorch_lr_range),
            "epochs": random.randint(*self.pytorch_epochs_range),
            "batch_size": random.choice(self.pytorch_batch_size_options),
            "hidden_sizes": random.choice(self.pytorch_hidden_sizes_options),
        }
    
    def sample_xgboost_params(self) -> Dict[str, Any]:
        """Sample XGBoost hyperparameters."""
        import random
        
        return {
            "n_estimators": random.randint(*self.xgboost_n_estimators_range),
            "max_depth": random.randint(*self.xgboost_max_depth_range),
            "learning_rate": random.uniform(*self.xgboost_lr_range),
        }
    
    def sample_lightgbm_params(self) -> Dict[str, Any]:
        """Sample LightGBM hyperparameters."""
        import random
        
        return {
            "n_estimators": random.randint(*self.lightgbm_n_estimators_range),
            "max_depth": random.randint(*self.lightgbm_max_depth_range),
            "learning_rate": random.uniform(*self.lightgbm_lr_range),
        }


class ConfigurationValidator:
    """Validates training configurations.
    
    Design Pattern: Strategy Pattern
    - Different validation strategies for different scenarios
    - Comprehensive constraint checking
    - Performance optimization recommendations
    """
    
    @staticmethod
    def validate_basic(config: MultiFrameworkConfig) -> List[str]:
        """Basic configuration validation."""
        issues = []
        
        # Check algorithms
        valid_algorithms = ["BFS", "ASTAR", "HAMILTONIAN", "DFS"]
        for algorithm in config.algorithms:
            if algorithm not in valid_algorithms:
                issues.append(f"Unknown algorithm: {algorithm}")
        
        # Check models
        valid_models = ["MLP", "CNN", "XGBOOST", "LIGHTGBM", "SKLEARN_MLP", "RANDOMFOREST"]
        for model in config.models:
            if model not in valid_models:
                issues.append(f"Unknown model: {model}")
        
        # Check numeric constraints
        if config.games_per_algorithm < 1:
            issues.append("games_per_algorithm must be positive")
        
        if config.grid_size < 5 or config.grid_size > 50:
            issues.append("grid_size should be between 5 and 50")
        
        if not 0.1 <= config.test_size <= 0.5:
            issues.append("test_size should be between 0.1 and 0.5")
        
        return issues
    
    @staticmethod
    def validate_performance(config: MultiFrameworkConfig) -> List[str]:
        """Performance-oriented validation."""
        warnings = []
        
        # Check for performance issues
        if config.games_per_algorithm > 10000:
            warnings.append("Large dataset size may cause memory issues")
        
        if config.pytorch_epochs > 500:
            warnings.append("Very high epoch count may lead to overfitting")
        
        if len(config.algorithms) * len(config.models) > 12:
            warnings.append("Training many algorithm-model combinations will take significant time")
        
        # Check memory-intensive combinations
        memory_intensive_models = ["CNN", "LSTM"]
        if any(model in memory_intensive_models for model in config.models):
            if config.pytorch_batch_size > 64:
                warnings.append("Large batch size with CNN/LSTM may cause GPU memory issues")
        
        return warnings
    
    @staticmethod
    def validate_compatibility(config: MultiFrameworkConfig) -> List[str]:
        """Framework compatibility validation."""
        issues = []
        
        # Import availability checks
        from . import (check_pytorch_availability, check_xgboost_availability,
                      check_lightgbm_availability, check_sklearn_availability)
        
        # Check PyTorch models
        pytorch_models = ["MLP", "CNN", "LSTM"]
        if any(model in pytorch_models for model in config.models):
            if not check_pytorch_availability():
                issues.append("PyTorch models selected but PyTorch not available")
        
        # Check XGBoost
        if "XGBOOST" in config.models and not check_xgboost_availability():
            issues.append("XGBoost selected but not available")
        
        # Check LightGBM
        if "LIGHTGBM" in config.models and not check_lightgbm_availability():
            issues.append("LightGBM selected but not available")
        
        # Check scikit-learn
        sklearn_models = ["SKLEARN_MLP", "RANDOMFOREST"]
        if any(model in sklearn_models for model in config.models):
            if not check_sklearn_availability():
                issues.append("Scikit-learn models selected but scikit-learn not available")
        
        return issues


class ConfigurationTemplate:
    """Configuration templates for common scenarios.
    
    Design Pattern: Template Method Pattern
    - Defines structure for different configuration types
    - Provides customization points for specific needs
    - Encapsulates best practices and optimizations
    """
    
    @staticmethod
    def quick_development() -> MultiFrameworkConfig:
        """Quick development configuration for testing."""
        return MultiFrameworkConfig(
            algorithms=["BFS"],
            games_per_algorithm=50,
            models=["MLP"],
            pytorch_epochs=10,
            pytorch_batch_size=16,
            dev_mode=True,
            experiment_name="quick_dev"
        )
    
    @staticmethod
    def comprehensive_comparison() -> MultiFrameworkConfig:
        """Comprehensive comparison of all algorithms and models."""
        return MultiFrameworkConfig(
            algorithms=["BFS", "ASTAR", "HAMILTONIAN"],
            games_per_algorithm=2000,
            models=["MLP", "XGBOOST", "LIGHTGBM"],
            pytorch_epochs=150,
            xgboost_n_estimators=300,
            lightgbm_n_estimators=300,
            experiment_name="comprehensive_comparison"
        )
    
    @staticmethod
    def production_pytorch() -> MultiFrameworkConfig:
        """Production-ready PyTorch configuration."""
        return MultiFrameworkConfig(
            algorithms=["BFS", "ASTAR", "HAMILTONIAN"],
            games_per_algorithm=5000,
            models=["MLP", "CNN"],
            pytorch_epochs=200,
            pytorch_batch_size=64,
            pytorch_learning_rate=0.001,
            pytorch_hidden_sizes=[256, 128, 64],
            enable_checkpoints=True,
            experiment_name="production_pytorch"
        )
    
    @staticmethod
    def production_tree_models() -> MultiFrameworkConfig:
        """Production-ready tree model configuration."""
        return MultiFrameworkConfig(
            algorithms=["BFS", "ASTAR", "HAMILTONIAN", "DFS"],
            games_per_algorithm=10000,
            models=["XGBOOST", "LIGHTGBM", "RANDOMFOREST"],
            xgboost_n_estimators=500,
            xgboost_max_depth=8,
            lightgbm_n_estimators=500,
            lightgbm_max_depth=8,
            experiment_name="production_tree_models"
        )
    
    @staticmethod
    def research_baseline() -> MultiFrameworkConfig:
        """Research baseline with all algorithms and models."""
        return MultiFrameworkConfig(
            algorithms=["BFS", "ASTAR", "HAMILTONIAN", "DFS"],
            games_per_algorithm=3000,
            models=["MLP", "CNN", "XGBOOST", "LIGHTGBM", "SKLEARN_MLP", "RANDOMFOREST"],
            pytorch_epochs=150,
            xgboost_n_estimators=300,
            lightgbm_n_estimators=300,
            experiment_name="research_baseline"
        )
    
    @staticmethod
    def memory_efficient() -> MultiFrameworkConfig:
        """Memory-efficient configuration for limited resources."""
        return MultiFrameworkConfig(
            algorithms=["BFS", "ASTAR"],
            games_per_algorithm=1000,
            models=["SKLEARN_MLP", "XGBOOST"],
            pytorch_batch_size=16,
            pytorch_epochs=50,
            xgboost_n_estimators=100,
            experiment_name="memory_efficient"
        )


class TrainingConfigBuilder:
    """Builder for creating custom training configurations.
    
    Design Pattern: Builder Pattern
    - Provides fluent interface for configuration construction
    - Allows step-by-step customization
    - Supports method chaining and validation
    """
    
    def __init__(self):
        self.config = MultiFrameworkConfig()
        self.search_space = HyperparameterSearchSpace()
    
    def add_algorithms(self, algorithms: List[str]) -> 'TrainingConfigBuilder':
        """Add algorithms to configuration."""
        self.config.algorithms = algorithms
        return self
    
    def add_models(self, models: List[str]) -> 'TrainingConfigBuilder':
        """Add models to configuration."""
        self.config.models = models
        return self
    
    def set_dataset_params(self, games_per_algorithm: int = None, 
                          grid_size: int = None) -> 'TrainingConfigBuilder':
        """Set dataset generation parameters."""
        if games_per_algorithm is not None:
            self.config.games_per_algorithm = games_per_algorithm
        if grid_size is not None:
            self.config.grid_size = grid_size
        return self
    
    def set_training_params(self, test_size: float = None,
                           random_state: int = None) -> 'TrainingConfigBuilder':
        """Set general training parameters."""
        if test_size is not None:
            self.config.test_size = test_size
        if random_state is not None:
            self.config.random_state = random_state
        return self
    
    def set_pytorch_params(self, epochs: int = None, batch_size: int = None,
                          learning_rate: float = None, 
                          hidden_sizes: List[int] = None) -> 'TrainingConfigBuilder':
        """Set PyTorch-specific parameters."""
        if epochs is not None:
            self.config.pytorch_epochs = epochs
        if batch_size is not None:
            self.config.pytorch_batch_size = batch_size
        if learning_rate is not None:
            self.config.pytorch_learning_rate = learning_rate
        if hidden_sizes is not None:
            self.config.pytorch_hidden_sizes = hidden_sizes
        return self
    
    def set_xgboost_params(self, n_estimators: int = None, max_depth: int = None,
                          learning_rate: float = None) -> 'TrainingConfigBuilder':
        """Set XGBoost-specific parameters."""
        if n_estimators is not None:
            self.config.xgboost_n_estimators = n_estimators
        if max_depth is not None:
            self.config.xgboost_max_depth = max_depth
        if learning_rate is not None:
            self.config.xgboost_learning_rate = learning_rate
        return self
    
    def set_lightgbm_params(self, n_estimators: int = None, max_depth: int = None,
                           learning_rate: float = None) -> 'TrainingConfigBuilder':
        """Set LightGBM-specific parameters."""
        if n_estimators is not None:
            self.config.lightgbm_n_estimators = n_estimators
        if max_depth is not None:
            self.config.lightgbm_max_depth = max_depth
        if learning_rate is not None:
            self.config.lightgbm_learning_rate = learning_rate
        return self
    
    def set_output_params(self, output_dir: str = None,
                         experiment_name: str = None,
                         dev_mode: bool = None) -> 'TrainingConfigBuilder':
        """Set output and experiment parameters."""
        if output_dir is not None:
            self.config.output_dir = output_dir
        if experiment_name is not None:
            self.config.experiment_name = experiment_name
        if dev_mode is not None:
            self.config.dev_mode = dev_mode
        return self
    
    def optimize_for_performance(self) -> 'TrainingConfigBuilder':
        """Apply performance optimizations."""
        # Increase batch sizes for efficiency
        self.config.pytorch_batch_size = 64
        
        # Use more estimators for tree models
        self.config.xgboost_n_estimators = max(200, self.config.xgboost_n_estimators)
        self.config.lightgbm_n_estimators = max(200, self.config.lightgbm_n_estimators)
        
        # Enable checkpoints
        self.config.enable_checkpoints = True
        
        return self
    
    def optimize_for_memory(self) -> 'TrainingConfigBuilder':
        """Apply memory optimizations."""
        # Reduce batch sizes
        self.config.pytorch_batch_size = min(32, self.config.pytorch_batch_size)
        
        # Reduce model complexity
        self.config.pytorch_hidden_sizes = [128, 64]
        self.config.xgboost_max_depth = min(6, self.config.xgboost_max_depth)
        self.config.lightgbm_max_depth = min(6, self.config.lightgbm_max_depth)
        
        return self
    
    def sample_hyperparameters(self, model_name: str) -> 'TrainingConfigBuilder':
        """Sample hyperparameters for specific model."""
        search_space = HyperparameterSearchSpace.create_for_model(model_name)
        
        if model_name in ["MLP", "CNN"]:
            params = search_space.sample_pytorch_params()
            self.set_pytorch_params(**params)
        elif model_name == "XGBOOST":
            params = search_space.sample_xgboost_params()
            self.set_xgboost_params(**params)
        elif model_name == "LIGHTGBM":
            params = search_space.sample_lightgbm_params()
            self.set_lightgbm_params(**params)
        
        return self
    
    def validate(self) -> List[str]:
        """Validate current configuration."""
        issues = []
        issues.extend(ConfigurationValidator.validate_basic(self.config))
        issues.extend(ConfigurationValidator.validate_performance(self.config))
        issues.extend(ConfigurationValidator.validate_compatibility(self.config))
        return issues
    
    def build(self) -> MultiFrameworkConfig:
        """Build and validate final configuration."""
        # Validate configuration
        issues = self.validate()
        if issues:
            logger.warning("Configuration issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return self.config
    
    @classmethod
    def from_template(cls, template_name: str) -> 'TrainingConfigBuilder':
        """Create builder from configuration template."""
        templates = {
            "quick_dev": ConfigurationTemplate.quick_development,
            "comprehensive": ConfigurationTemplate.comprehensive_comparison,
            "production_pytorch": ConfigurationTemplate.production_pytorch,
            "production_trees": ConfigurationTemplate.production_tree_models,
            "research": ConfigurationTemplate.research_baseline,
            "memory_efficient": ConfigurationTemplate.memory_efficient,
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        config = templates[template_name]()
        builder = cls()
        builder.config = config
        return builder


class TrainingConfigManager:
    """Manages training configurations with persistence and templates.
    
    Design Pattern: Facade Pattern
    - Provides unified interface to configuration system
    - Manages configuration persistence and loading
    - Coordinates validation and optimization
    """
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_config(self, config: MultiFrameworkConfig, name: str):
        """Save configuration with given name."""
        config_path = self.config_dir / f"{name}.json"
        config.save(config_path)
        self.logger.info(f"Configuration saved: {config_path}")
    
    def load_config(self, name: str) -> MultiFrameworkConfig:
        """Load configuration by name."""
        config_path = self.config_dir / f"{name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {config_path}")
        
        config = MultiFrameworkConfig.load(config_path)
        self.logger.info(f"Configuration loaded: {config_path}")
        return config
    
    def list_configs(self) -> List[str]:
        """List available configuration names."""
        config_files = list(self.config_dir.glob("*.json"))
        return [f.stem for f in config_files]
    
    def create_template_configs(self):
        """Create all template configurations."""
        templates = {
            "quick_development": ConfigurationTemplate.quick_development,
            "comprehensive_comparison": ConfigurationTemplate.comprehensive_comparison,
            "production_pytorch": ConfigurationTemplate.production_pytorch,
            "production_tree_models": ConfigurationTemplate.production_tree_models,
            "research_baseline": ConfigurationTemplate.research_baseline,
            "memory_efficient": ConfigurationTemplate.memory_efficient,
        }
        
        for name, template_func in templates.items():
            config = template_func()
            self.save_config(config, name)
        
        self.logger.info(f"Created {len(templates)} template configurations")
    
    def validate_config(self, config: MultiFrameworkConfig) -> Dict[str, List[str]]:
        """Comprehensive configuration validation."""
        return {
            "basic_issues": ConfigurationValidator.validate_basic(config),
            "performance_warnings": ConfigurationValidator.validate_performance(config),
            "compatibility_issues": ConfigurationValidator.validate_compatibility(config),
        }
    
    def optimize_config(self, config: MultiFrameworkConfig, 
                       optimization_type: str = "balanced") -> MultiFrameworkConfig:
        """Optimize configuration for specific goals."""
        builder = TrainingConfigBuilder()
        builder.config = config
        
        if optimization_type == "performance":
            builder.optimize_for_performance()
        elif optimization_type == "memory":
            builder.optimize_for_memory()
        elif optimization_type == "balanced":
            # Apply moderate optimizations
            if config.pytorch_batch_size < 32:
                builder.set_pytorch_params(batch_size=32)
            if config.xgboost_n_estimators < 100:
                builder.set_xgboost_params(n_estimators=100)
        
        return builder.build()
    
    def generate_hyperparameter_configs(self, base_config: MultiFrameworkConfig,
                                       model_name: str, num_configs: int = 5) -> List[MultiFrameworkConfig]:
        """Generate multiple configurations with sampled hyperparameters."""
        configs = []
        
        for i in range(num_configs):
            builder = TrainingConfigBuilder()
            builder.config = MultiFrameworkConfig(**asdict(base_config))
            builder.sample_hyperparameters(model_name)
            
            # Update experiment name
            base_name = base_config.experiment_name or "hyperopt"
            builder.config.experiment_name = f"{base_name}_sample_{i+1}"
            
            configs.append(builder.build())
        
        return configs 