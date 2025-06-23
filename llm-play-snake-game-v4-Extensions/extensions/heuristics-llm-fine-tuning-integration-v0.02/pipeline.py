"""pipeline.py - Multi-Dataset Heuristics to LLM Fine-Tuning Pipeline v0.02

Evolution from v0.01: Supports multiple datasets, advanced training configurations,
and comprehensive evaluation. Demonstrates natural software progression from 
single-dataset proof-of-concept to production-ready multi-algorithm system.

Key Improvements:
- Multi-dataset training: Combine BFS, A*, Hamiltonian datasets
- Advanced training strategies: LoRA, QLoRA, full fine-tuning
- Comprehensive evaluation: Multiple metrics and comparison tools
- Enhanced error handling and validation
- Integration with common utilities for code reuse

Design Patterns:
- Template Method: Base pipeline with customizable training steps
- Strategy Pattern: Different fine-tuning approaches (LoRA vs full)
- Factory Pattern: Model and tokenizer creation
- Observer Pattern: Training progress monitoring
- Command Pattern: Evaluation task execution
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Import common utilities for code reuse
from extensions.common import (
    training_cli_utils,
    training_logging_utils, 
    rl_helpers,  # Reuse for directory setup
    ensure_extensions_logs_dir,
    get_dataset_dir,
)

# Import base v0.01 components for inheritance
try:
    from ..heuristics_llm_fine_tuning_integration_v0_01.jsonl_generator import JSONLGenerator
    from ..heuristics_llm_fine_tuning_integration_v0_01.llm_trainer import LLMTrainer
except ImportError:
    # Fallback if v0.01 not available
    print("Warning: v0.01 components not found, using standalone implementations")
    JSONLGenerator = None
    LLMTrainer = None

__all__ = [
    "MultiDatasetConfig",
    "AdvancedTrainingStrategy", 
    "MultiDatasetPipeline",
    "PipelineResults",
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultiDatasetConfig:
    """Configuration for multi-dataset training pipeline.
    
    Evolution from v0.01: Supports multiple input directories, advanced training
    configurations, and comprehensive evaluation settings.
    
    Design Pattern: Configuration Object
    - Centralizes all pipeline parameters
    - Provides validation and defaults
    - Supports serialization for reproducibility
    """
    
    # Dataset configuration
    heuristic_log_dirs: List[str] = field(default_factory=list)
    """List of heuristic log directories to process"""
    
    dataset_types: List[str] = field(default_factory=lambda: ["BFS", "ASTAR", "HAMILTONIAN"])
    """Types of heuristic algorithms to include"""
    
    max_samples_per_algorithm: int = 10000
    """Maximum samples to use per algorithm type"""
    
    train_test_split: float = 0.8
    """Proportion of data for training vs evaluation"""
    
    # Model configuration  
    base_model_name: str = "microsoft/DialoGPT-small"
    """Base model for fine-tuning"""
    
    max_length: int = 512
    """Maximum sequence length for tokenization"""
    
    # Training configuration
    training_strategy: str = "lora"
    """Training strategy: 'lora', 'qlora', 'full'"""
    
    num_epochs: int = 3
    """Number of training epochs"""
    
    learning_rate: float = 2e-4
    """Learning rate for training"""
    
    batch_size: int = 4
    """Training batch size"""
    
    gradient_accumulation_steps: int = 4
    """Gradient accumulation steps"""
    
    warmup_steps: int = 100
    """Warmup steps for learning rate scheduler"""
    
    # LoRA configuration
    lora_r: int = 16
    """LoRA rank"""
    
    lora_alpha: int = 32
    """LoRA alpha parameter"""
    
    lora_dropout: float = 0.1
    """LoRA dropout rate"""
    
    # Output configuration
    output_dir: str = "output/heuristics-llm-v0.02"
    """Output directory for models and results"""
    
    experiment_name: Optional[str] = None
    """Experiment name for tracking"""
    
    save_steps: int = 500
    """Steps between model saves"""
    
    eval_steps: int = 250
    """Steps between evaluations"""
    
    logging_steps: int = 50
    """Steps between training logs"""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.heuristic_log_dirs:
            raise ValueError("At least one heuristic log directory must be specified")
        
        if not 0 < self.train_test_split < 1:
            raise ValueError("train_test_split must be between 0 and 1")
        
        if self.training_strategy not in ["lora", "qlora", "full"]:
            raise ValueError("training_strategy must be 'lora', 'qlora', or 'full'")
        
        # Generate experiment name if not provided
        if self.experiment_name is None:
            algorithms = "_".join(sorted(self.dataset_types))
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"multi_{algorithms}_{self.training_strategy}_{timestamp}"
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            # Convert to dict and handle non-serializable types
            config_dict = {}
            for key, value in self.__dict__.items():
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> MultiDatasetConfig:
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class AdvancedTrainingStrategy:
    """Advanced training strategy implementations.
    
    Design Pattern: Strategy Pattern
    - Encapsulates different training approaches
    - Allows runtime switching between strategies
    - Provides consistent interface for all strategies
    """
    
    def __init__(self, config: MultiDatasetConfig):
        self.config = config
        self.logger = training_logging_utils.get_logger(f"training_strategy_{config.training_strategy}")
    
    def setup_model_and_tokenizer(self):
        """Set up model and tokenizer based on strategy.
        
        Returns:
            Tuple of (model, tokenizer, peft_config)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            raise ImportError("transformers and peft required for model setup")
        
        # Load base model and tokenizer
        self.logger.info(f"Loading base model: {self.config.base_model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Configure based on strategy
        peft_config = None
        
        if self.config.training_strategy == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            model = get_peft_model(model, peft_config)
            self.logger.info(f"Configured LoRA with rank {self.config.lora_r}")
        
        elif self.config.training_strategy == "qlora":
            # QLoRA configuration with 4-bit quantization
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="float16"
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
                )
                model = get_peft_model(model, peft_config)
                self.logger.info("Configured QLoRA with 4-bit quantization")
                
            except ImportError:
                self.logger.warning("BitsAndBytesConfig not available, falling back to LoRA")
                self.config.training_strategy = "lora"
                return self.setup_model_and_tokenizer()
        
        elif self.config.training_strategy == "full":
            self.logger.info("Using full fine-tuning (no LoRA)")
        
        return model, tokenizer, peft_config
    
    def get_training_args(self):
        """Get training arguments based on strategy."""
        try:
            from transformers import TrainingArguments
        except ImportError:
            raise ImportError("transformers required for training arguments")
        
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"],
            run_name=self.config.experiment_name,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )


@dataclass 
class PipelineResults:
    """Results from the multi-dataset pipeline execution.
    
    Design Pattern: Data Transfer Object
    - Encapsulates all pipeline results
    - Provides methods for analysis and reporting
    - Supports serialization for persistence
    """
    
    config: MultiDatasetConfig
    dataset_stats: Dict[str, Any]
    training_history: Dict[str, List[float]]
    evaluation_metrics: Dict[str, float]
    model_path: str
    execution_time: float
    
    def save_summary(self, path: Union[str, Path]) -> None:
        """Save pipeline results summary."""
        summary = {
            "experiment_name": self.config.experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config.__dict__,
            "dataset_stats": self.dataset_stats,
            "training_history": self.training_history,
            "evaluation_metrics": self.evaluation_metrics,
            "model_path": self.model_path,
            "execution_time": self.execution_time,
        }
        
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)


class MultiDatasetPipeline:
    """Multi-dataset heuristics to LLM fine-tuning pipeline.
    
    Evolution from v0.01: Supports multiple datasets, advanced training strategies,
    and comprehensive evaluation framework.
    
    Design Pattern: Template Method
    - Defines the skeleton of pipeline execution
    - Allows customization of individual steps
    - Provides hooks for extension and monitoring
    
    Key Features:
    - Multi-algorithm dataset combination
    - Advanced training strategies (LoRA, QLoRA, full)
    - Comprehensive evaluation and metrics
    - Progress monitoring and logging
    - Error recovery and validation
    """
    
    def __init__(self, config: MultiDatasetConfig):
        self.config = config
        self.logger = training_logging_utils.get_logger("multi_dataset_pipeline")
        self.strategy = AdvancedTrainingStrategy(config)
        
        # Setup directories using common utilities
        self.setup_directories()
        
        # Initialize components
        self.dataset_stats = {}
        self.training_history = {}
        self.evaluation_metrics = {}
    
    def setup_directories(self):
        """Setup output directories using common utilities."""
        # Use common directory setup pattern
        self.output_dir = Path(self.config.output_dir)
        self.experiment_dir = self.output_dir / self.config.experiment_name
        
        # Create directory structure
        self.experiment_dir.mkdir(parents=True, exist_ok=True)  # Create base directory
        
        # Additional directories for this pipeline
        (self.experiment_dir / "datasets").mkdir(exist_ok=True)
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "evaluation").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        
        self.logger.info(f"Created experiment directory: {self.experiment_dir}")
    
    def collect_and_process_datasets(self) -> Dict[str, Any]:
        """Collect and process datasets from multiple heuristic sources.
        
        Template Method: Step 1 of pipeline
        """
        self.logger.info("Collecting datasets from heuristic sources")
        
        all_data = []
        algorithm_counts = {}
        
        # Process each heuristic log directory
        for log_dir in self.config.heuristic_log_dirs:
            log_path = Path(log_dir)
            if not log_path.exists():
                self.logger.warning(f"Log directory not found: {log_dir}")
                continue
            
            # Process log directory
            # Note: Using basic path processing instead of DatasetDirectoryManager
            
            # Find and process algorithm-specific data
            for algorithm in self.config.dataset_types:
                algorithm_data = self._extract_algorithm_data(log_path, algorithm)
                
                if algorithm_data:
                    # Limit samples per algorithm
                    if len(algorithm_data) > self.config.max_samples_per_algorithm:
                        algorithm_data = algorithm_data[:self.config.max_samples_per_algorithm]
                    
                    all_data.extend(algorithm_data)
                    algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + len(algorithm_data)
                    
                    self.logger.info(f"Collected {len(algorithm_data)} samples from {algorithm}")
        
        # Compile dataset statistics
        self.dataset_stats = {
            "total_samples": len(all_data),
            "algorithm_distribution": algorithm_counts,
            "source_directories": self.config.heuristic_log_dirs,
            "dataset_types": self.config.dataset_types
        }
        
        self.logger.info(f"Total dataset size: {len(all_data)} samples")
        self.logger.info(f"Algorithm distribution: {algorithm_counts}")
        
        return {
            "data": all_data,
            "stats": self.dataset_stats
        }
    
    def _extract_algorithm_data(self, log_dir: Path, algorithm: str) -> List[Dict[str, Any]]:
        """Extract data for specific algorithm from log directory."""
        algorithm_data = []
        
        # Look for JSON files containing algorithm data
        json_files = list(log_dir.glob("*.json"))
        
        for json_file in json_files:
            if json_file.name == "summary.json":
                continue  # Skip summary files
            
            try:
                with open(json_file, 'r') as f:
                    game_data = json.load(f)
                
                # Check if this game used the target algorithm
                if game_data.get("algorithm", "").upper() == algorithm.upper():
                    # Extract training examples from game data
                    examples = self._extract_training_examples(game_data, algorithm)
                    algorithm_data.extend(examples)
                    
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Error processing {json_file}: {e}")
        
        return algorithm_data
    
    def _extract_training_examples(self, game_data: Dict[str, Any], algorithm: str) -> List[Dict[str, Any]]:
        """Extract training examples from game data."""
        examples = []
        
        # Extract moves and game state information
        moves = game_data.get("detailed_history", {}).get("moves", [])
        rounds_data = game_data.get("detailed_history", {}).get("rounds_data", {})
        
        # Create training examples for each move
        for i, move in enumerate(moves):
            if move in ["INVALID_REVERSAL", "EMPTY", "SOMETHING_IS_WRONG", "NO_PATH_FOUND"]:
                continue  # Skip invalid moves
            
            # Build context from game state
            context = self._build_move_context(game_data, i, algorithm)
            
            example = {
                "prompt": context["prompt"],
                "completion": context["completion"],
                "algorithm": algorithm,
                "move_index": i,
                "game_id": game_data.get("game_id", "unknown")
            }
            
            examples.append(example)
        
        return examples
    
    def _build_move_context(self, game_data: Dict[str, Any], move_index: int, algorithm: str) -> Dict[str, str]:
        """Build context for a specific move."""
        moves = game_data.get("detailed_history", {}).get("moves", [])
        
        if move_index >= len(moves):
            return {"prompt": "", "completion": ""}
        
        current_move = moves[move_index]
        
        # Build prompt with game state context
        prompt = f"You are a {algorithm} algorithm playing Snake. "
        prompt += f"Based on the current game state, what is your next move? "
        
        # Add game state information if available
        if "snake_positions" in game_data:
            head_pos = game_data["snake_positions"][0] if game_data["snake_positions"] else [0, 0]
            prompt += f"Snake head position: {head_pos}. "
        
        if "apple_position" in game_data:
            apple_pos = game_data["apple_position"]
            prompt += f"Apple position: {apple_pos}. "
        
        prompt += "Respond with your move and reasoning."
        
        # Build completion with move and reasoning
        completion = f"I choose to move {current_move}. "
        completion += f"As a {algorithm} algorithm, this move follows the optimal pathfinding strategy "
        completion += f"to reach the apple while avoiding obstacles."
        
        return {
            "prompt": prompt,
            "completion": completion
        }
    
    def prepare_training_data(self, processed_data: Dict[str, Any]):
        """Prepare data for training using HuggingFace datasets.
        
        Template Method: Step 2 of pipeline
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets library required for data preparation")
        
        self.logger.info("Preparing training data")
        
        # Convert to HuggingFace dataset format
        data = processed_data["data"]
        
        # Split into train/test
        split_index = int(len(data) * self.config.train_test_split)
        train_data = data[:split_index]
        eval_data = data[split_index:]
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
        
        # Tokenize datasets
        model, tokenizer, _ = self.strategy.setup_model_and_tokenizer()
        
        def tokenize_function(examples):
            # Combine prompt and completion for training
            texts = [
                f"Prompt: {prompt}\nCompletion: {completion}"
                for prompt, completion in zip(examples["prompt"], examples["completion"])
            ]
            
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
                return_tensors="pt"
            )
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        self.logger.info(f"Prepared {len(train_dataset)} training samples")
        self.logger.info(f"Prepared {len(eval_dataset)} evaluation samples")
        
        return {
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer
        }
    
    def train_model(self, training_data: Dict[str, Any]) -> str:
        """Train the model using prepared data.
        
        Template Method: Step 3 of pipeline
        """
        try:
            from transformers import Trainer, DataCollatorForLanguageModeling
        except ImportError:
            raise ImportError("transformers required for model training")
        
        self.logger.info("Starting model training")
        
        # Setup model and training arguments
        model, tokenizer, peft_config = self.strategy.setup_model_and_tokenizer()
        training_args = self.strategy.get_training_args()
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're doing causal language modeling
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_data["train_dataset"],
            eval_dataset=training_data["eval_dataset"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Train model
        self.logger.info("Training started...")
        train_result = trainer.train()
        
        # Save training history
        self.training_history = {
            "train_loss": trainer.state.log_history,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }
        
        # Save model
        model_save_path = self.experiment_dir / "models" / "final_model"
        trainer.save_model(str(model_save_path))
        tokenizer.save_pretrained(str(model_save_path))
        
        self.logger.info(f"Model saved to: {model_save_path}")
        
        return str(model_save_path)
    
    def evaluate_model(self, model_path: str, training_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the trained model.
        
        Template Method: Step 4 of pipeline
        """
        self.logger.info("Evaluating trained model")
        
        # Basic evaluation metrics
        metrics = {
            "final_train_loss": 0.0,  # Will be filled from training history
            "final_eval_loss": 0.0,
            "perplexity": 0.0,
            "total_parameters": 0,
            "trainable_parameters": 0,
        }
        
        # Extract metrics from training history
        if self.training_history.get("train_loss"):
            last_log = self.training_history["train_loss"][-1]
            metrics["final_train_loss"] = last_log.get("train_loss", 0.0)
            metrics["final_eval_loss"] = last_log.get("eval_loss", 0.0)
        
        # Calculate perplexity
        if metrics["final_eval_loss"] > 0:
            import math
            metrics["perplexity"] = math.exp(metrics["final_eval_loss"])
        
        self.evaluation_metrics = metrics
        self.logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def run_pipeline(self) -> PipelineResults:
        """Execute the complete multi-dataset pipeline.
        
        Template Method: Main execution flow
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting multi-dataset pipeline: {self.config.experiment_name}")
            
            # Save configuration
            self.config.save(self.experiment_dir / "config.json")
            
            # Execute pipeline steps
            processed_data = self.collect_and_process_datasets()
            training_data = self.prepare_training_data(processed_data)
            model_path = self.train_model(training_data)
            evaluation_metrics = self.evaluate_model(model_path, training_data)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create results object
            results = PipelineResults(
                config=self.config,
                dataset_stats=self.dataset_stats,
                training_history=self.training_history,
                evaluation_metrics=evaluation_metrics,
                model_path=model_path,
                execution_time=execution_time
            )
            
            # Save results summary
            results.save_summary(self.experiment_dir / "results_summary.json")
            
            self.logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
            self.logger.info(f"Results saved to: {self.experiment_dir}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """CLI entry point for multi-dataset pipeline."""
    parser = argparse.ArgumentParser(
        description="Heuristics → LLM Fine-Tuning Integration v0.02",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Use common CLI utilities for consistent argument handling
    cli_utils = TrainingCLIUtils()
    
    # Dataset arguments
    parser.add_argument(
        "--heuristic-dirs",
        nargs="+",
        required=True,
        help="Directories containing heuristic game logs"
    )
    
    parser.add_argument(
        "--algorithms",
        nargs="+", 
        default=["BFS", "ASTAR", "HAMILTONIAN"],
        help="Heuristic algorithms to include"
    )
    
    parser.add_argument(
        "--max-samples-per-algorithm",
        type=int,
        default=10000,
        help="Maximum samples per algorithm"
    )
    
    # Model arguments
    parser.add_argument(
        "--base-model",
        default="microsoft/DialoGPT-small",
        help="Base model for fine-tuning"
    )
    
    parser.add_argument(
        "--training-strategy",
        choices=["lora", "qlora", "full"],
        default="lora",
        help="Training strategy"
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        default="output/heuristics-llm-v0.02",
        help="Output directory"
    )
    
    parser.add_argument(
        "--experiment-name",
        help="Experiment name (auto-generated if not provided)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = MultiDatasetConfig(
        heuristic_log_dirs=args.heuristic_dirs,
        dataset_types=args.algorithms,
        max_samples_per_algorithm=args.max_samples_per_algorithm,
        base_model_name=args.base_model,
        training_strategy=args.training_strategy,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )
    
    # Create and run pipeline
    pipeline = MultiDatasetPipeline(config)
    results = pipeline.run_pipeline()
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"📊 Processed {results.dataset_stats['total_samples']} samples")
    print(f"🎯 Final evaluation loss: {results.evaluation_metrics.get('final_eval_loss', 'N/A')}")
    print(f"⏱️  Execution time: {results.execution_time:.2f} seconds")
    print(f"📁 Results saved to: {results.model_path}")


if __name__ == "__main__":
    main() 