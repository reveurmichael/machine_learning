"""cli.py - Command Line Interface for v0.03

Enhanced CLI that supports both traditional command-line operations
and launching the interactive web interface.

Evolution from v0.02: CLI-only → CLI + Web interface launcher
"""

from __future__ import annotations

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure we're working from project root
if os.getcwd() != str(project_root):
    os.chdir(str(project_root))

from extensions.common import training_logging_utils

logger = training_logging_utils.TrainingLogger("cli_v0.03")


def launch_streamlit_app(port: int = 8501, host: str = "localhost") -> None:
    """Launch the Streamlit web application."""
    try:
        app_path = Path(__file__).parent / "app.py"
        
        if not app_path.exists():
            logger.error(f"Streamlit app not found at {app_path}")
            return
        
        logger.info(f"Launching Streamlit app on {host}:{port}")
        
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("Streamlit app stopped by user")
    except Exception as e:
        logger.error(f"Failed to launch Streamlit app: {e}")


def launch_flask_dashboard(port: int = 5000, debug: bool = False) -> None:
    """Launch the Flask web dashboard."""
    try:
        # Import here to avoid issues if Flask not available
        from .web_interface import WebDashboard
        
        dashboard = WebDashboard(port=port, debug=debug)
        dashboard.run()
        
    except ImportError:
        logger.error("Flask not available, cannot launch web dashboard")
    except Exception as e:
        logger.error(f"Failed to launch web dashboard: {e}")


def list_datasets(grid_size: Optional[int] = None) -> None:
    """List available datasets."""
    try:
        from .dataset_manager import DatasetManager
        
        manager = DatasetManager()
        
        if grid_size:
            datasets = manager.get_datasets_by_grid_size(grid_size)
            print(f"\n📁 Datasets for grid size {grid_size}:")
        else:
            datasets = manager.discover_datasets()
            print("\n📁 All available datasets:")
        
        if not datasets:
            print("No datasets found.")
            return
        
        # Print dataset table
        print(f"{'Name':<30} {'Algorithm':<12} {'Format':<8} {'Samples':<8} {'Size (MB)':<10} {'Modified':<12}")
        print("-" * 90)
        
        for dataset in datasets:
            print(f"{dataset.name:<30} {dataset.algorithm:<12} {dataset.format.value:<8} "
                  f"{dataset.sample_count:<8} {dataset.size_mb:<10.2f} {dataset.modified_date.strftime('%Y-%m-%d'):<12}")
        
        # Print summary
        summary = manager.get_dataset_summary()
        print(f"\nSummary: {summary['total_count']} datasets, {summary['total_size_mb']:.2f} MB total")
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")


def run_training(config_file: Optional[str] = None, **kwargs) -> None:
    """Run training pipeline."""
    try:
        if not config_file:
            logger.info("No config file provided, using default settings")
            config = {
                'strategy': kwargs.get('strategy', 'LoRA'),
                'model_name': kwargs.get('model', 'microsoft/DialoGPT-small'),
                'num_epochs': kwargs.get('epochs', 3),
                'grid_size': kwargs.get('grid_size', 10),
            }
        else:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Import and run pipeline from v0.02
        from extensions.heuristics_llm_fine_tuning_integration_v0_02.pipeline import MultiDatasetPipeline
        from extensions.heuristics_llm_fine_tuning_integration_v0_02.training_config import TrainingConfigBuilder
        
        # Build training configuration
        training_config = TrainingConfigBuilder() \
            .with_model(config['model_name']) \
            .with_strategy(config['strategy']) \
            .with_epochs(config['num_epochs']) \
            .build()
        
        # Run pipeline
        pipeline = MultiDatasetPipeline(
            output_dir="logs/extensions/models",
            config=training_config
        )
        
        logger.info(f"Starting training with config: {config}")
        results = pipeline.run()
        
        logger.info(f"Training completed: {results}")
        
    except ImportError as e:
        logger.error(f"v0.02 components not available: {e}")
    except Exception as e:
        logger.error(f"Training failed: {e}")


def run_evaluation(model_path: str, dataset_path: str, **kwargs) -> None:
    """Run model evaluation."""
    try:
        # Import from v0.02
        from extensions.heuristics_llm_fine_tuning_integration_v0_02.evaluation import EvaluationSuite
        
        evaluator = EvaluationSuite()
        
        logger.info(f"Evaluating model {model_path} on dataset {dataset_path}")
        
        results = evaluator.evaluate_model(
            model_path=model_path,
            test_datasets=[dataset_path],
            metrics=kwargs.get('metrics', ['win_rate', 'avg_score'])
        )
        
        logger.info(f"Evaluation results: {results}")
        
    except ImportError as e:
        logger.error(f"v0.02 components not available: {e}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


def preprocess_datasets(dataset_names: List[str], **kwargs) -> None:
    """Preprocess datasets."""
    try:
        from .dataset_manager import DatasetManager, PreprocessingConfig
        
        manager = DatasetManager()
        
        # Create preprocessing config
        config = PreprocessingConfig(
            output_format=kwargs.get('format', 'jsonl'),
            max_samples=kwargs.get('max_samples', 10000),
            train_split=kwargs.get('train_split', 0.8),
        )
        
        logger.info(f"Preprocessing datasets: {dataset_names}")
        
        results = manager.preprocess_datasets(dataset_names, config)
        
        if results['success']:
            logger.info(f"Preprocessing completed: {results['statistics']}")
        else:
            logger.error(f"Preprocessing failed: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")


def show_system_info() -> None:
    """Show system information and extension status."""
    print("\n🚀 LLM Fine-tuning Integration v0.03")
    print("=" * 50)
    
    # Extension info
    print("Extension Type: Web Interface")
    print(f"Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check dependencies
    dependencies = {
        'streamlit': 'Streamlit web interface',
        'flask': 'Flask web dashboard',
        'pandas': 'Data processing',
        'transformers': 'Model training',
        'torch': 'PyTorch backend'
    }
    
    print("\n📦 Dependencies:")
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            status = "✅ Available"
        except ImportError:
            status = "❌ Missing"
        
        print(f"  {dep:<15} {status:<12} {description}")
    
    # Check v0.02 availability
    try:
        from extensions.heuristics_llm_fine_tuning_integration_v0_02 import pipeline
        v0_02_status = "✅ Available"
    except ImportError:
        v0_02_status = "❌ Missing"
    
    print(f"\n🔗 v0.02 Components: {v0_02_status}")
    
    # Dataset summary
    try:
        from .dataset_manager import DatasetManager
        manager = DatasetManager()
        summary = manager.get_dataset_summary()
        
        print("\n💾 Dataset Summary:")
        print(f"  Total Datasets: {summary['total_count']}")
        print(f"  Total Size: {summary['total_size_mb']:.2f} MB")
        print(f"  Formats: {', '.join(summary['formats'].keys())}")
        print(f"  Grid Sizes: {', '.join(map(str, summary['grid_sizes'].keys()))}")
        
    except Exception as e:
        print(f"\n💾 Dataset Summary: Error loading ({e})")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Fine-tuning Integration v0.03 - Interactive Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Streamlit web app (recommended)
  python cli.py web --port 8501
  
  # Launch Flask dashboard
  python cli.py dashboard --port 5000
  
  # List available datasets
  python cli.py datasets --list
  
  # Run training (CLI mode)
  python cli.py train --strategy LoRA --model gpt2 --epochs 3
  
  # Show system information
  python cli.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Web interface command
    web_parser = subparsers.add_parser('web', help='Launch Streamlit web interface')
    web_parser.add_argument('--port', type=int, default=8501, help='Port number (default: 8501)')
    web_parser.add_argument('--host', default='localhost', help='Host address (default: localhost)')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch Flask web dashboard')
    dashboard_parser.add_argument('--port', type=int, default=5000, help='Port number (default: 5000)')
    dashboard_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Dataset management commands
    dataset_parser = subparsers.add_parser('datasets', help='Dataset management')
    dataset_parser.add_argument('--list', action='store_true', help='List available datasets')
    dataset_parser.add_argument('--grid-size', type=int, help='Filter by grid size')
    dataset_parser.add_argument('--preprocess', nargs='+', help='Preprocess datasets')
    dataset_parser.add_argument('--format', choices=['csv', 'jsonl', 'parquet'], default='jsonl',
                               help='Output format for preprocessing')
    dataset_parser.add_argument('--max-samples', type=int, default=10000,
                               help='Maximum samples per dataset')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Run training pipeline')
    train_parser.add_argument('--config', help='Training configuration file (JSON)')
    train_parser.add_argument('--strategy', choices=['LoRA', 'QLoRA', 'Full'], default='LoRA',
                             help='Fine-tuning strategy')
    train_parser.add_argument('--model', default='microsoft/DialoGPT-small',
                             help='Base model name')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    train_parser.add_argument('--grid-size', type=int, default=10, help='Grid size for datasets')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Run model evaluation')
    eval_parser.add_argument('model_path', help='Path to trained model')
    eval_parser.add_argument('dataset_path', help='Path to evaluation dataset')
    eval_parser.add_argument('--metrics', nargs='+', default=['win_rate', 'avg_score'],
                            help='Evaluation metrics')
    
    # Info command
    subparsers.add_parser('info', help='Show system information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'web':
            launch_streamlit_app(port=args.port, host=args.host)
        
        elif args.command == 'dashboard':
            launch_flask_dashboard(port=args.port, debug=args.debug)
        
        elif args.command == 'datasets':
            if args.list:
                list_datasets(grid_size=args.grid_size)
            elif args.preprocess:
                preprocess_datasets(
                    args.preprocess,
                    format=args.format,
                    max_samples=args.max_samples
                )
            else:
                dataset_parser.print_help()
        
        elif args.command == 'train':
            run_training(
                config_file=args.config,
                strategy=args.strategy,
                model=args.model,
                epochs=args.epochs,
                grid_size=args.grid_size
            )
        
        elif args.command == 'eval':
            run_evaluation(
                model_path=args.model_path,
                dataset_path=args.dataset_path,
                metrics=args.metrics
            )
        
        elif args.command == 'info':
            show_system_info()
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 