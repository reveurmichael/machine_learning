"""training_config.py - Advanced Training Configuration for v0.02

Evolution from v0.01: Provides comprehensive training configuration management
with support for multiple training strategies, hyperparameter optimization,
and experiment tracking.

Key Features:
- Multiple training strategies (LoRA, QLoRA, full fine-tuning)
- Hyperparameter validation and optimization
- Experiment configuration templates
- Integration with common utilities
- Serializable configuration objects

Design Patterns:
- Builder Pattern: Flexible configuration construction
- Strategy Pattern: Different training approaches
- Template Method: Configuration validation pipeline
- Factory Pattern: Configuration template creation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

# Import common utilities

__all__ = [
    "AdvancedTrainingConfig",
    "HyperparameterConfig",
    "LoRAConfig",
    "QLoRAConfig",
    "TrainingConfigBuilder",
    "ConfigurationTemplate",
]

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration with validation and optimization support.
    
    Design Pattern: Value Object
    - Immutable configuration with validation
    - Type-safe parameter definitions
    - Default values based on research best practices
    """
    
    # Learning rate configuration
    learning_rate: float = 2e-4
    learning_rate_scheduler: str = "cosine"  # linear, cosine, polynomial
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None
    
    # Batch size and gradient configuration
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Training epochs and steps
    num_train_epochs: int = 3
    max_steps: Optional[int] = None
    
    # Evaluation and logging
    eval_strategy: str = "steps"  # steps, epoch, no
    eval_steps: int = 250
    logging_steps: int = 50
    save_steps: int = 500
    
    # Optimization
    optimizer: str = "adamw_torch"  # adamw_torch, adamw_hf, sgd
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    
    def __post_init__(self):
        """Validate hyperparameters after initialization."""
        self._validate_learning_rate()
        self._validate_batch_sizes()
        self._validate_training_steps()
        self._validate_evaluation_config()
    
    def _validate_learning_rate(self):
        """Validate learning rate configuration."""
        if not 1e-6 <= self.learning_rate <= 1e-1:
            raise ValueError(f"Learning rate {self.learning_rate} outside recommended range [1e-6, 1e-1]")
        
        if self.learning_rate_scheduler not in ["linear", "cosine", "polynomial", "constant"]:
            raise ValueError(f"Unknown scheduler: {self.learning_rate_scheduler}")
        
        if not 0 <= self.warmup_ratio <= 0.5:
            raise ValueError(f"Warmup ratio {self.warmup_ratio} outside valid range [0, 0.5]")
    
    def _validate_batch_sizes(self):
        """Validate batch size configuration."""
        if self.per_device_train_batch_size < 1:
            raise ValueError("Training batch size must be >= 1")
        
        if self.per_device_eval_batch_size < 1:
            raise ValueError("Evaluation batch size must be >= 1")
        
        if self.gradient_accumulation_steps < 1:
            raise ValueError("Gradient accumulation steps must be >= 1")
        
        # Check for reasonable effective batch size
        effective_batch_size = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if effective_batch_size > 128:
            logger.warning(f"Large effective batch size: {effective_batch_size}")
    
    def _validate_training_steps(self):
        """Validate training step configuration."""
        if self.num_train_epochs <= 0:
            raise ValueError("Number of epochs must be > 0")
        
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("Max steps must be > 0 if specified")
    
    def _validate_evaluation_config(self):
        """Validate evaluation configuration."""
        if self.eval_strategy not in ["steps", "epoch", "no"]:
            raise ValueError(f"Unknown evaluation strategy: {self.eval_strategy}")
        
        if self.eval_strategy == "steps" and self.eval_steps <= 0:
            raise ValueError("Eval steps must be > 0 when using 'steps' strategy")


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration.
    
    Design Pattern: Configuration Object
    - Encapsulates LoRA-specific parameters
    - Provides validation and defaults
    - Supports different LoRA variants
    """
    
    # LoRA rank and scaling
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Target modules for LoRA
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # LoRA variant configuration
    use_rslora: bool = False
    use_dora: bool = False
    
    # Bias configuration
    bias: str = "none"  # none, all, lora_only
    
    # Task type
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        """Validate LoRA configuration."""
        if not 1 <= self.r <= 256:
            raise ValueError(f"LoRA rank {self.r} outside valid range [1, 256]")
        
        if not 1 <= self.lora_alpha <= 512:
            raise ValueError(f"LoRA alpha {self.lora_alpha} outside valid range [1, 512]")
        
        if not 0 <= self.lora_dropout <= 0.5:
            raise ValueError(f"LoRA dropout {self.lora_dropout} outside valid range [0, 0.5]")
        
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"Unknown bias configuration: {self.bias}")
    
    def get_effective_alpha(self) -> float:
        """Calculate effective alpha for scaling."""
        return self.lora_alpha / self.r
    
    def estimate_parameters(self, base_model_params: int) -> Tuple[int, float]:
        """Estimate LoRA parameter count and reduction ratio."""
        # Simplified estimation
        target_module_count = len(self.target_modules)
        estimated_lora_params = target_module_count * 2 * self.r * 4096  # Rough estimate
        
        reduction_ratio = estimated_lora_params / base_model_params
        return estimated_lora_params, reduction_ratio


@dataclass
class QLoRAConfig(LoRAConfig):
    """QLoRA (Quantized LoRA) configuration.
    
    Extends LoRA with quantization parameters for memory efficiency.
    """
    
    # Quantization configuration
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"  # float16, bfloat16, float32
    bnb_4bit_quant_type: str = "nf4"  # fp4, nf4
    bnb_4bit_use_double_quant: bool = True
    
    def __post_init__(self):
        """Validate QLoRA configuration."""
        super().__post_init__()
        
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")
        
        if not (self.load_in_4bit or self.load_in_8bit):
            logger.warning("QLoRA configuration without quantization - consider using LoRA instead")
        
        if self.bnb_4bit_compute_dtype not in ["float16", "bfloat16", "float32"]:
            raise ValueError(f"Unknown compute dtype: {self.bnb_4bit_compute_dtype}")
        
        if self.bnb_4bit_quant_type not in ["fp4", "nf4"]:
            raise ValueError(f"Unknown quantization type: {self.bnb_4bit_quant_type}")


@dataclass
class AdvancedTrainingConfig:
    """Advanced training configuration for v0.02.
    
    Evolution from v0.01: Comprehensive configuration with multiple strategies,
    validation, and integration with common utilities.
    
    Design Pattern: Facade
    - Provides unified interface to all training configurations
    - Manages complex configuration interactions
    - Simplifies configuration for common use cases
    """
    
    # Strategy configuration
    training_strategy: str = "lora"  # lora, qlora, full
    
    # Model configuration
    base_model_name: str = "microsoft/DialoGPT-small"
    model_max_length: int = 512
    use_fast_tokenizer: bool = True
    
    # Data configuration
    train_test_split: float = 0.8
    max_samples: Optional[int] = None
    data_seed: int = 42
    
    # Training configuration
    hyperparameters: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    lora_config: Optional[LoRAConfig] = None
    qlora_config: Optional[QLoRAConfig] = None
    
    # Output configuration
    output_dir: str = "output/heuristics-llm-v0.02"
    experiment_name: Optional[str] = None
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Hardware and performance
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    
    # Experiment tracking
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    logging_dir: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    data_seed: int = 42
    
    def __post_init__(self):
        """Validate and setup configuration after initialization."""
        self._validate_strategy()
        self._setup_strategy_configs()
        self._validate_output_config()
        self._setup_hardware_optimization()
    
    def _validate_strategy(self):
        """Validate training strategy configuration."""
        if self.training_strategy not in ["lora", "qlora", "full"]:
            raise ValueError(f"Unknown training strategy: {self.training_strategy}")
    
    def _setup_strategy_configs(self):
        """Setup strategy-specific configurations."""
        if self.training_strategy == "lora" and self.lora_config is None:
            self.lora_config = LoRAConfig()
            logger.info("Created default LoRA configuration")
        
        elif self.training_strategy == "qlora" and self.qlora_config is None:
            self.qlora_config = QLoRAConfig()
            logger.info("Created default QLoRA configuration")
    
    def _validate_output_config(self):
        """Validate output configuration."""
        if not 0 < self.train_test_split < 1:
            raise ValueError(f"Train-test split {self.train_test_split} must be in (0, 1)")
        
        if self.save_strategy not in ["steps", "epoch", "no"]:
            raise ValueError(f"Unknown save strategy: {self.save_strategy}")
    
    def _setup_hardware_optimization(self):
        """Setup hardware-specific optimizations."""
        try:
            import torch
            
            # Auto-detect hardware capabilities
            if torch.cuda.is_available():
                # Enable mixed precision for CUDA
                if torch.cuda.get_device_capability()[0] >= 7:  # Volta or newer
                    if not self.fp16 and not self.bf16:
                        self.bf16 = True
                        logger.info("Enabled bf16 for compatible CUDA device")
                
                # Enable gradient checkpointing for memory efficiency
                if not self.gradient_checkpointing:
                    self.gradient_checkpointing = True
                    logger.info("Enabled gradient checkpointing for memory efficiency")
        
        except ImportError:
            logger.warning("PyTorch not available - hardware optimization disabled")
    
    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments dictionary for HuggingFace Trainer.
        
        Returns:
            Dictionary of training arguments compatible with TrainingArguments
        """
        args = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.hyperparameters.num_train_epochs,
            "per_device_train_batch_size": self.hyperparameters.per_device_train_batch_size,
            "per_device_eval_batch_size": self.hyperparameters.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.hyperparameters.gradient_accumulation_steps,
            "learning_rate": self.hyperparameters.learning_rate,
            "weight_decay": self.hyperparameters.weight_decay,
            "warmup_ratio": self.hyperparameters.warmup_ratio,
            "lr_scheduler_type": self.hyperparameters.learning_rate_scheduler,
            "logging_steps": self.hyperparameters.logging_steps,
            "eval_strategy": self.hyperparameters.eval_strategy,
            "eval_steps": self.hyperparameters.eval_steps,
            "save_steps": self.hyperparameters.save_steps,
            "save_strategy": self.save_strategy,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "report_to": self.report_to,
            "seed": self.seed,
        }
        
        # Add optional arguments
        if self.hyperparameters.max_steps is not None:
            args["max_steps"] = self.hyperparameters.max_steps
        
        if self.hyperparameters.warmup_steps is not None:
            args["warmup_steps"] = self.hyperparameters.warmup_steps
        
        if self.experiment_name is not None:
            args["run_name"] = self.experiment_name
        
        if self.logging_dir is not None:
            args["logging_dir"] = self.logging_dir
        
        return args
    
    def get_lora_config_dict(self) -> Optional[Dict[str, Any]]:
        """Get LoRA configuration dictionary for PEFT."""
        if self.training_strategy == "lora" and self.lora_config:
            return asdict(self.lora_config)
        elif self.training_strategy == "qlora" and self.qlora_config:
            return asdict(self.qlora_config)
        return None
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> AdvancedTrainingConfig:
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested objects
        if "hyperparameters" in config_dict:
            config_dict["hyperparameters"] = HyperparameterConfig(**config_dict["hyperparameters"])
        
        if "lora_config" in config_dict and config_dict["lora_config"]:
            config_dict["lora_config"] = LoRAConfig(**config_dict["lora_config"])
        
        if "qlora_config" in config_dict and config_dict["qlora_config"]:
            config_dict["qlora_config"] = QLoRAConfig(**config_dict["qlora_config"])
        
        return cls(**config_dict)


class TrainingConfigBuilder:
    """Builder for creating training configurations.
    
    Design Pattern: Builder
    - Provides fluent interface for configuration construction
    - Supports step-by-step configuration building
    - Validates configuration at build time
    """
    
    def __init__(self):
        self._config = AdvancedTrainingConfig()
    
    def with_strategy(self, strategy: str) -> TrainingConfigBuilder:
        """Set training strategy."""
        self._config.training_strategy = strategy
        return self
    
    def with_model(self, model_name: str, max_length: int = 512) -> TrainingConfigBuilder:
        """Set model configuration."""
        self._config.base_model_name = model_name
        self._config.model_max_length = max_length
        return self
    
    def with_lora(self, rank: int = 16, alpha: int = 32, dropout: float = 0.1) -> TrainingConfigBuilder:
        """Configure LoRA parameters."""
        self._config.lora_config = LoRAConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout)
        return self
    
    def with_qlora(self, rank: int = 16, alpha: int = 32, dropout: float = 0.1) -> TrainingConfigBuilder:
        """Configure QLoRA parameters."""
        self._config.qlora_config = QLoRAConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout)
        return self
    
    def with_training(self, epochs: int = 3, learning_rate: float = 2e-4, 
                     batch_size: int = 4) -> TrainingConfigBuilder:
        """Set training hyperparameters."""
        self._config.hyperparameters.num_train_epochs = epochs
        self._config.hyperparameters.learning_rate = learning_rate
        self._config.hyperparameters.per_device_train_batch_size = batch_size
        return self
    
    def with_output(self, output_dir: str, experiment_name: str = None) -> TrainingConfigBuilder:
        """Set output configuration."""
        self._config.output_dir = output_dir
        self._config.experiment_name = experiment_name
        return self
    
    def build(self) -> AdvancedTrainingConfig:
        """Build and validate the configuration."""
        # Trigger validation
        self._config.__post_init__()
        return self._config


class ConfigurationTemplate:
    """Pre-defined configuration templates for common use cases.
    
    Design Pattern: Factory Method
    - Provides factory methods for common configurations
    - Encapsulates configuration best practices
    - Simplifies setup for typical scenarios
    """
    
    @staticmethod
    def quick_lora() -> AdvancedTrainingConfig:
        """Quick LoRA configuration for fast experimentation."""
        return (TrainingConfigBuilder()
                .with_strategy("lora")
                .with_lora(rank=8, alpha=16)
                .with_training(epochs=1, batch_size=8)
                .build())
    
    @staticmethod
    def production_lora() -> AdvancedTrainingConfig:
        """Production LoRA configuration with optimal settings."""
        return (TrainingConfigBuilder()
                .with_strategy("lora")
                .with_lora(rank=16, alpha=32, dropout=0.1)
                .with_training(epochs=3, learning_rate=2e-4, batch_size=4)
                .build())
    
    @staticmethod
    def memory_efficient_qlora() -> AdvancedTrainingConfig:
        """Memory-efficient QLoRA configuration for large models."""
        config = (TrainingConfigBuilder()
                 .with_strategy("qlora")
                 .with_qlora(rank=32, alpha=64)
                 .with_training(epochs=3, batch_size=2)
                 .build())
        
        # Enable additional memory optimizations
        config.gradient_checkpointing = True
        config.hyperparameters.gradient_accumulation_steps = 8
        
        return config
    
    @staticmethod
    def full_finetune_small() -> AdvancedTrainingConfig:
        """Full fine-tuning configuration for small models."""
        return (TrainingConfigBuilder()
                .with_strategy("full")
                .with_training(epochs=2, learning_rate=1e-5, batch_size=2)
                .build())
    
    @staticmethod
    def research_config() -> AdvancedTrainingConfig:
        """Research configuration with comprehensive logging."""
        config = ConfigurationTemplate.production_lora()
        config.hyperparameters.logging_steps = 10
        config.hyperparameters.eval_steps = 100
        config.hyperparameters.save_steps = 200
        config.report_to = ["tensorboard", "wandb"]
        
        return config


def main():
    """Demo of configuration system."""
    print("🔧 Advanced Training Configuration Demo")
    
    # Create configurations using different methods
    configs = {
        "Quick LoRA": ConfigurationTemplate.quick_lora(),
        "Production LoRA": ConfigurationTemplate.production_lora(),
        "QLoRA Memory Efficient": ConfigurationTemplate.memory_efficient_qlora(),
        "Research Config": ConfigurationTemplate.research_config(),
    }
    
    for name, config in configs.items():
        print(f"\n📋 {name}:")
        print(f"  Strategy: {config.training_strategy}")
        print(f"  Epochs: {config.hyperparameters.num_train_epochs}")
        print(f"  Learning Rate: {config.hyperparameters.learning_rate}")
        print(f"  Batch Size: {config.hyperparameters.per_device_train_batch_size}")
        
        if config.lora_config:
            print(f"  LoRA Rank: {config.lora_config.r}")
        if config.qlora_config:
            print(f"  QLoRA Rank: {config.qlora_config.r}")


if __name__ == "__main__":
    main() 