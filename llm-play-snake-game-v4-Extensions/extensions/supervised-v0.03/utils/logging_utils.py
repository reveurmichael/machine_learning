"""
Logging utilities for supervised learning v0.03.

Design Pattern: Singleton Pattern
- Centralized logging configuration
- Consistent log format across modules
- Performance monitoring capabilities
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class TrainingLogger:
    """Centralized logger for training operations."""
    
    def __init__(self, name: str = "supervised_v0.03", log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(log_dir)
    
    def _setup_handlers(self, log_dir: Optional[Path]):
        """Setup console and file handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if log_dir provided)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"training_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


class MetricsLogger:
    """Logger for training metrics and performance."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.metrics_file = log_dir / "metrics.json"
        self.metrics = []
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  train_acc: float, val_acc: float, **kwargs):
        """Log epoch metrics."""
        metric = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.metrics.append(metric)
    
    def log_training_complete(self, final_metrics: dict):
        """Log final training metrics."""
        final_metrics["training_completed"] = datetime.now().isoformat()
        self.metrics.append(final_metrics)
    
    def save_metrics(self):
        """Save metrics to file."""
        import json
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_best_epoch(self, metric: str = "val_acc") -> Optional[int]:
        """Get epoch with best metric value."""
        if not self.metrics:
            return None
        
        best_metric = max(self.metrics, key=lambda x: x.get(metric, 0))
        return best_metric.get("epoch")


def setup_logging(log_level: str = "INFO", log_dir: Optional[Path] = None) -> TrainingLogger:
    """Setup logging for the application."""
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create training logger
    return TrainingLogger(log_dir=log_dir)


def log_experiment_start(config: dict, log_dir: Path):
    """Log experiment start with configuration."""
    logger = TrainingLogger(log_dir=log_dir)
    
    logger.info("=" * 60)
    logger.info("Supervised Learning v0.03 - Experiment Started")
    logger.info("=" * 60)
    
    # Log configuration
    logger.info(f"Grid Size: {config.get('training', {}).get('grid_size', 'N/A')}")
    logger.info(f"Model: {config.get('model', {}).get('type', 'N/A')}")
    logger.info(f"Learning Rate: {config.get('model', {}).get('learning_rate', 'N/A')}")
    logger.info(f"Batch Size: {config.get('model', {}).get('batch_size', 'N/A')}")
    logger.info(f"Epochs: {config.get('model', {}).get('epochs', 'N/A')}")
    logger.info("=" * 60)
    
    return logger


def log_experiment_complete(logger: TrainingLogger, final_results: dict):
    """Log experiment completion with results."""
    logger.info("=" * 60)
    logger.info("Experiment Completed")
    logger.info("=" * 60)
    
    # Log final results
    for key, value in final_results.items():
        logger.info(f"{key}: {value}")
    
    logger.info("=" * 60) 