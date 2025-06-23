"""training_logging_utils.py – Central log helpers for ML training scripts.

Moved out of `supervised-v0.03` to give other extensions the same convenience
APIs without copy-paste.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

__all__ = [
    "TrainingLogger",
    "setup_logging",
    "log_experiment_start",
    "log_experiment_complete",
]


class TrainingLogger:
    """Thin wrapper around the stdlib *logging* with optional file sink."""

    def __init__(self, name: str = "snake-ml", log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self._add_console_handler()
            if log_dir:
                self._add_file_handler(log_dir)

    # ---------------------
    def _add_console_handler(self) -> None:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        self.logger.addHandler(h)

    def _add_file_handler(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"train_{ts}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"))
        self.logger.addHandler(fh)

    # Passthrough helpers
    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)


# ---------------------
# Convenience wrappers
# ---------------------


def setup_logging(level: str = "INFO", log_dir: Optional[Path] = None) -> TrainingLogger:
    logging.basicConfig(level=getattr(logging, level.upper(), 20))
    return TrainingLogger(log_dir=log_dir)


def log_experiment_start(cfg: Dict[str, Any], log_dir: Path) -> TrainingLogger:
    logger = TrainingLogger(log_dir=log_dir)
    logger.info("=" * 50)
    logger.info("Training started")
    logger.info("=" * 50)
    logger.info(f"Grid-size: {cfg['training']['grid_size']}")
    logger.info(f"Model: {cfg['model']['type']}")
    logger.info(f"Epochs: {cfg['model']['epochs']}")
    logger.info("=" * 50)
    return logger


def log_experiment_complete(logger: TrainingLogger, results: Dict[str, Any]):
    logger.info("=" * 50)
    logger.info("Training finished")
    for k, v in results.items():
        logger.info(f"{k}: {v}")
    logger.info("=" * 50) 