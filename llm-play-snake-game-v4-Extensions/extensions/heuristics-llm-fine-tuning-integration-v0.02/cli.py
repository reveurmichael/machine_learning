#!/usr/bin/env python3
"""cli.py - Command Line Interface for Heuristics → LLM Fine-Tuning Integration v0.02

Comprehensive CLI demonstrating all v0.02 capabilities:
- Multi-dataset training pipeline
- Advanced configuration management
- Model comparison and evaluation
- Statistical analysis and reporting

Evolution from v0.01: Enhanced CLI with multiple commands, better error handling,
and integration with common utilities.

Usage Examples:
    # Basic multi-dataset training
    python cli.py train --heuristic-dirs logs/extensions/heuristics-* --algorithms BFS ASTAR

    # Advanced configuration
    python cli.py train --config advanced_qlora.json

    # Model evaluation
    python cli.py evaluate --model-path output/model --test-data data/test.jsonl

    # Model comparison
    python cli.py compare --model-a output/lora --model-b output/qlora

    # Configuration templates
    python cli.py config --template production_lora --output config.json

Design Patterns:
- Command Pattern: Different CLI commands as separate classes
- Factory Pattern: Configuration and model creation
- Strategy Pattern: Different execution strategies
- Template Method: Common CLI workflow structure
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Fix Python path for extensions
import os
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure we're working from project root
if os.getcwd() != str(project_root):
    os.chdir(str(project_root))

# Import v0.02 components
from extensions.heuristics_llm_fine_tuning_integration_v0_02 import (
    MultiDatasetConfig,
    MultiDatasetPipeline,
    AdvancedTrainingConfig,
    ConfigurationTemplate,
    TrainingConfigBuilder,
    EvaluationSuite,
    ModelComparator,
)

# Import common utilities
from extensions.common import training_logging_utils

__all__ = ["main", "CLICommands"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLICommands:
    """Container for CLI command implementations.
    
    Design Pattern: Command Pattern
    - Each command is a separate method
    - Consistent interface and error handling
    - Supports command composition and chaining
    """
    
    def __init__(self):
        self.logger = training_logging_utils.get_logger("cli_commands")
    
    def train(self, args) -> int:
        """Execute multi-dataset training pipeline."""
        self.logger.info("🚀 Starting multi-dataset training pipeline")
        
        try:
            # Load or create configuration
            if args.config:
                config = self._load_config(args.config)
            else:
                config = self._create_config_from_args(args)
            
            # Validate configuration
            self._validate_training_config(config)
            
            # Create and run pipeline
            pipeline = MultiDatasetPipeline(config)
            results = pipeline.run_pipeline()
            
            # Report results
            self._report_training_results(results)
            
            self.logger.info("✅ Training completed successfully!")
            return 0
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    def evaluate(self, args) -> int:
        """Evaluate trained model(s)."""
        self.logger.info("📊 Starting model evaluation")
        
        try:
            # Load test data
            test_data = self._load_test_data(args.test_data)
            
            # Setup evaluation suite
            evaluator = EvaluationSuite(args.output_dir)
            
            if args.model_paths:
                # Multiple model evaluation
                models = self._load_multiple_models(args.model_paths)
                report = evaluator.evaluate_multiple_models(models, test_data)
                
                self.logger.info("🏆 Model Rankings:")
                for i, (model_name, score) in enumerate(report.model_rankings, 1):
                    self.logger.info(f"  {i}. {model_name}: {score:.4f}")
            
            else:
                # Single model evaluation
                model, tokenizer = self._load_single_model(args.model_path)
                metrics = evaluator.evaluate_single_model(
                    model, tokenizer, test_data, args.model_name or "model"
                )
                
                self.logger.info("📈 Evaluation Results:")
                self.logger.info(f"  Win Rate: {metrics.snake_win_rate:.2%}")
                self.logger.info(f"  Avg Score: {metrics.snake_avg_score:.1f}")
                self.logger.info(f"  Perplexity: {metrics.perplexity:.2f}")
            
            self.logger.info("✅ Evaluation completed successfully!")
            return 0
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    def compare(self, args) -> int:
        """Compare two or more models."""
        self.logger.info("🔍 Starting model comparison")
        
        try:
            # Load models for comparison
            if args.model_a and args.model_b:
                # Two-model comparison
                model_a_results = self._load_model_results(args.model_a)
                model_b_results = self._load_model_results(args.model_b)
                
                comparator = ModelComparator(args.output_dir)
                report = comparator.compare_two_models(
                    model_a_results, model_b_results,
                    args.model_a_name or "Model A",
                    args.model_b_name or "Model B",
                    args.strategy or "statistical"
                )
                
                self._report_comparison_results(report)
            
            elif args.strategy_comparison:
                # Training strategy comparison
                lora_results = self._load_model_results(args.lora_path)
                qlora_results = self._load_model_results(args.qlora_path)
                full_results = self._load_model_results(args.full_path)
                
                comparator = ModelComparator(args.output_dir)
                reports = comparator.compare_training_strategies(
                    lora_results, qlora_results, full_results
                )
                
                self._report_strategy_comparison(reports)
            
            else:
                raise ValueError("Must specify either --model-a/--model-b or --strategy-comparison")
            
            self.logger.info("✅ Comparison completed successfully!")
            return 0
            
        except Exception as e:
            self.logger.error(f"❌ Comparison failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    def config(self, args) -> int:
        """Generate configuration files."""
        self.logger.info("⚙️  Generating configuration")
        
        try:
            if args.template:
                # Use predefined template
                config = self._create_template_config(args.template)
            else:
                # Create custom configuration
                config = self._create_custom_config(args)
            
            # Save configuration
            output_path = Path(args.output)
            config.save(output_path)
            
            self.logger.info(f"💾 Configuration saved to: {output_path}")
            
            # Print summary
            self._print_config_summary(config)
            
            self.logger.info("✅ Configuration generation completed!")
            return 0
            
        except Exception as e:
            self.logger.error(f"❌ Configuration generation failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    def demo(self, args) -> int:
        """Run demonstration of v0.02 capabilities."""
        self.logger.info("🎭 Running v0.02 demonstration")
        
        try:
            # Create demo configurations
            self.logger.info("📋 Creating demonstration configurations...")
            
            configs = {
                "Quick LoRA": ConfigurationTemplate.quick_lora(),
                "Production LoRA": ConfigurationTemplate.production_lora(),
                "Memory Efficient QLoRA": ConfigurationTemplate.memory_efficient_qlora(),
                "Research Config": ConfigurationTemplate.research_config(),
            }
            
            demo_dir = Path(args.output_dir) / "demo"
            demo_dir.mkdir(parents=True, exist_ok=True)
            
            # Save demo configurations
            for name, config in configs.items():
                config_path = demo_dir / f"{name.lower().replace(' ', '_')}.json"
                config.save(config_path)
                self.logger.info(f"  💾 {name}: {config_path}")
            
            # Create demo dataset
            self.logger.info("📊 Creating demo dataset...")
            demo_data = self._create_demo_dataset()
            
            dataset_path = demo_dir / "demo_test_data.json"
            with open(dataset_path, 'w') as f:
                json.dump(demo_data, f, indent=2)
            
            self.logger.info(f"  💾 Demo dataset: {dataset_path}")
            
            # Print usage examples
            self._print_demo_usage_examples(demo_dir)
            
            self.logger.info("✅ Demonstration setup completed!")
            return 0
            
        except Exception as e:
            self.logger.error(f"❌ Demonstration failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    # Helper methods
    
    def _load_config(self, config_path: str) -> MultiDatasetConfig:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return MultiDatasetConfig(**config_dict)
    
    def _create_config_from_args(self, args) -> MultiDatasetConfig:
        """Create configuration from command line arguments."""
        return MultiDatasetConfig(
            heuristic_log_dirs=args.heuristic_dirs,
            dataset_types=args.algorithms,
            max_samples_per_algorithm=args.max_samples,
            base_model_name=args.base_model,
            training_strategy=args.training_strategy,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )
    
    def _validate_training_config(self, config: MultiDatasetConfig):
        """Validate training configuration."""
        # Check if heuristic directories exist
        for log_dir in config.heuristic_log_dirs:
            if not Path(log_dir).exists():
                raise FileNotFoundError(f"Heuristic log directory not found: {log_dir}")
        
        # Validate training strategy
        if config.training_strategy not in ["lora", "qlora", "full"]:
            raise ValueError(f"Invalid training strategy: {config.training_strategy}")
        
        self.logger.info("✅ Configuration validation passed")
    
    def _load_test_data(self, test_data_path: str) -> List[Dict[str, Any]]:
        """Load test data from file."""
        with open(test_data_path, 'r') as f:
            if test_data_path.endswith('.jsonl'):
                # JSONL format
                data = [json.loads(line) for line in f]
            else:
                # JSON format
                data = json.load(f)
        
        self.logger.info(f"📊 Loaded {len(data)} test examples")
        return data
    
    def _load_multiple_models(self, model_paths: List[str]) -> Dict[str, tuple]:
        """Load multiple models for comparison."""
        models = {}
        for path in model_paths:
            model_name = Path(path).name
            model, tokenizer = self._load_single_model(path)
            models[model_name] = (model, tokenizer)
        
        self.logger.info(f"📚 Loaded {len(models)} models for comparison")
        return models
    
    def _load_single_model(self, model_path: str) -> tuple:
        """Load a single model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.logger.info(f"📚 Loaded model from: {model_path}")
            return model, tokenizer
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def _load_model_results(self, results_path: str) -> Dict[str, Any]:
        """Load model evaluation results."""
        results_file = Path(results_path) / "results_summary.json"
        if not results_file.exists():
            results_file = Path(results_path + "_evaluation.json")
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def _create_template_config(self, template_name: str) -> AdvancedTrainingConfig:
        """Create configuration from template."""
        templates = {
            "quick_lora": ConfigurationTemplate.quick_lora,
            "production_lora": ConfigurationTemplate.production_lora,
            "memory_efficient_qlora": ConfigurationTemplate.memory_efficient_qlora,
            "research_config": ConfigurationTemplate.research_config,
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        return templates[template_name]()
    
    def _create_custom_config(self, args) -> AdvancedTrainingConfig:
        """Create custom configuration from arguments."""
        builder = TrainingConfigBuilder()
        
        if hasattr(args, 'strategy') and args.strategy:
            builder.with_strategy(args.strategy)
        
        if hasattr(args, 'model') and args.model:
            builder.with_model(args.model)
        
        if hasattr(args, 'epochs') and args.epochs:
            builder.with_training(epochs=args.epochs)
        
        if hasattr(args, 'output') and args.output:
            builder.with_output(args.output)
        
        return builder.build()
    
    def _create_demo_dataset(self) -> List[Dict[str, Any]]:
        """Create demonstration dataset."""
        return [
            {
                "prompt": "Snake head at (5,5), apple at (7,7). What move do you make?",
                "completion": "I choose to move RIGHT because it brings me closer to the apple at (7,7) while maintaining a safe path.",
                "algorithm": "BFS"
            },
            {
                "prompt": "Snake head at (3,3), apple at (1,3). What move do you make?", 
                "completion": "I choose to move LEFT to directly approach the apple using the optimal path found by A* algorithm.",
                "algorithm": "ASTAR"
            },
            {
                "prompt": "Snake head at (8,2), apple at (8,8). What move do you make?",
                "completion": "I choose to move UP following the Hamiltonian path to ensure complete board coverage while reaching the apple.",
                "algorithm": "HAMILTONIAN"
            }
        ]
    
    def _report_training_results(self, results):
        """Report training results."""
        self.logger.info("📊 Training Results Summary:")
        self.logger.info(f"  📈 Dataset Size: {results.dataset_stats['total_samples']} samples")
        self.logger.info(f"  🏆 Final Loss: {results.evaluation_metrics.get('final_eval_loss', 'N/A')}")
        self.logger.info(f"  ⏱️  Execution Time: {results.execution_time:.2f} seconds")
        self.logger.info(f"  💾 Model Path: {results.model_path}")
    
    def _report_comparison_results(self, report):
        """Report model comparison results."""
        self.logger.info("🔍 Comparison Results:")
        self.logger.info(f"  🥇 Better Model: {report.summary_metrics['overall_better_model']}")
        self.logger.info(f"  📈 Avg Improvement: {report.summary_metrics['average_relative_improvement']:.2f}%")
        self.logger.info(f"  📊 Significant Improvements: {report.summary_metrics['significant_improvements']}")
        
        # Show top improvements
        top_improvements = report.get_top_improvements(3)
        if top_improvements:
            self.logger.info("  🔝 Top Improvements:")
            for metric, improvement in top_improvements:
                self.logger.info(f"    • {metric}: {improvement:.2f}%")
    
    def _report_strategy_comparison(self, reports):
        """Report training strategy comparison results."""
        self.logger.info("🏗️  Training Strategy Comparison:")
        
        for comparison, report in reports.items():
            better_model = report.summary_metrics['overall_better_model']
            improvement = report.summary_metrics['average_relative_improvement']
            self.logger.info(f"  {comparison}: {better_model} wins by {improvement:.2f}%")
    
    def _print_config_summary(self, config):
        """Print configuration summary."""
        self.logger.info("📋 Configuration Summary:")
        self.logger.info(f"  Strategy: {config.training_strategy}")
        self.logger.info(f"  Base Model: {config.base_model_name}")
        self.logger.info(f"  Epochs: {config.hyperparameters.num_train_epochs}")
        self.logger.info(f"  Learning Rate: {config.hyperparameters.learning_rate}")
        self.logger.info(f"  Batch Size: {config.hyperparameters.per_device_train_batch_size}")
    
    def _print_demo_usage_examples(self, demo_dir: Path):
        """Print demonstration usage examples."""
        self.logger.info("\n🎯 Demo Usage Examples:")
        self.logger.info("\n1. Train with quick LoRA configuration:")
        self.logger.info(f"   python cli.py train --config {demo_dir}/quick_lora.json")
        
        self.logger.info("\n2. Evaluate a model:")
        self.logger.info(f"   python cli.py evaluate --model-path output/model --test-data {demo_dir}/demo_test_data.json")
        
        self.logger.info("\n3. Compare two models:")
        self.logger.info("   python cli.py compare --model-a output/lora --model-b output/qlora")
        
        self.logger.info("\n4. Generate production configuration:")
        self.logger.info("   python cli.py config --template production_lora --output production.json")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Heuristics → LLM Fine-Tuning Integration v0.02",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output-dir", default="output/v0.02", help="Output directory")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train models using multi-dataset pipeline")
    train_parser.add_argument("--config", help="Configuration file path")
    train_parser.add_argument("--heuristic-dirs", nargs="+", help="Heuristic log directories")
    train_parser.add_argument("--algorithms", nargs="+", default=["BFS", "ASTAR", "HAMILTONIAN"], 
                             help="Algorithms to include")
    train_parser.add_argument("--max-samples", type=int, default=10000, help="Max samples per algorithm")
    train_parser.add_argument("--base-model", default="microsoft/DialoGPT-small", help="Base model")
    train_parser.add_argument("--training-strategy", choices=["lora", "qlora", "full"], 
                             default="lora", help="Training strategy")
    train_parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    train_parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--experiment-name", help="Experiment name")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.add_argument("--model-path", help="Single model path")
    eval_parser.add_argument("--model-paths", nargs="+", help="Multiple model paths")
    eval_parser.add_argument("--model-name", help="Model name for single evaluation")
    eval_parser.add_argument("--test-data", required=True, help="Test data file")
    
    # Compare command  
    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument("--model-a", help="First model path")
    compare_parser.add_argument("--model-b", help="Second model path")
    compare_parser.add_argument("--model-a-name", help="First model name")
    compare_parser.add_argument("--model-b-name", help="Second model name")
    compare_parser.add_argument("--strategy", choices=["statistical", "performance"], 
                               default="statistical", help="Comparison strategy")
    compare_parser.add_argument("--strategy-comparison", action="store_true", 
                               help="Compare training strategies")
    compare_parser.add_argument("--lora-path", help="LoRA model path")
    compare_parser.add_argument("--qlora-path", help="QLoRA model path")
    compare_parser.add_argument("--full-path", help="Full fine-tuning model path")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Generate configuration files")
    config_parser.add_argument("--template", choices=[
        "quick_lora", "production_lora", "memory_efficient_qlora", "research_config"
    ], help="Configuration template")
    config_parser.add_argument("--strategy", choices=["lora", "qlora", "full"], help="Training strategy")
    config_parser.add_argument("--model", help="Base model name")
    config_parser.add_argument("--epochs", type=int, help="Training epochs")
    config_parser.add_argument("--output", default="config.json", help="Output config file")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run v0.02 demonstration")
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate command
    if not args.command:
        parser.print_help()
        return 1
    
    # Create CLI commands instance
    commands = CLICommands()
    
    # Execute command
    try:
        if args.command == "train":
            return commands.train(args)
        elif args.command == "evaluate":
            return commands.evaluate(args)
        elif args.command == "compare":
            return commands.compare(args)
        elif args.command == "config":
            return commands.config(args)
        elif args.command == "demo":
            return commands.demo(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("🛑 Operation cancelled by user")
        return 130
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 