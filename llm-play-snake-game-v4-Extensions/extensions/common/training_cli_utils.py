"""training_cli_utils.py – Generic CLI helpers for ML extensions

Extracted from *supervised-v0.03* so the logic is **single-source-of-truth**
inside `extensions/common`.  All supervised/RL/evolutionary training scripts
should import from here instead of duplicating argument parsers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

__all__ = [
    "create_parser",
    "validate_args",
    "parse_model_list",
    "args_to_config",
]


# ---------------------
# Parser construction
# ---------------------


def create_parser() -> argparse.ArgumentParser:  # noqa: D401
    """Return an *opinionated* ArgumentParser shared by ML training scripts."""
    parser = argparse.ArgumentParser(
        description="Snake-AI Training CLI (generic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model MLP --grid-size 15 --epochs 200
  python train.py --model XGBOOST --grid-size 10 --max-depth 8
        """,
    )

    # Model family
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "MLP",
            "CNN",
            "LSTM",
            "GRU",
            "XGBOOST",
            "LIGHTGBM",
            "RANDOMFOREST",
            "GCN",
        ],
        help="Model type to train/evaluate",
    )

    # Generic game params
    parser.add_argument("--grid-size", type=int, default=10, choices=range(5, 51))

    # Optim/training params (superset across frameworks)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Tree params
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=100)

    # Data params
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--validation-split", type=float, default=0.2)

    # Output / logging
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-name")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")

    # Extra control
    parser.add_argument("--max-games", type=int, default=1000)

    return parser


# ---------------------
# Validation helpers
# ---------------------


def validate_args(args: argparse.Namespace) -> bool:  # noqa: D401
    """Return *True* if *args* are semantically valid, else print errors."""
    valid = True

    if not 5 <= args.grid_size <= 50:
        print("[CLI] Grid-size must be 5-50")
        valid = False
    if not 0.0 < args.learning_rate <= 1.0:
        print("[CLI] Learning-rate must be 0-1")
        valid = False
    if not 0 < args.validation_split < 1:
        print("[CLI] validation-split must be 0-1")
        valid = False
    if args.epochs <= 0:
        print("[CLI] epochs must be >0")
        valid = False
    return valid


# ---------------------
# Convenience converters
# ---------------------


def parse_model_list(models_str: str) -> List[str]:
    return [m.strip().upper() for m in models_str.split(",") if m.strip()]



def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert *argparse* namespace to nested config dict."""
    model_cfg = {
        "type": args.model.upper(),
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "dropout_rate": args.dropout,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "validation_split": args.validation_split,
    }

    training_cfg = {
        "grid_size": args.grid_size,
        "max_games": args.max_games,
        "use_gui": False,
        "verbose": args.verbose and not args.quiet,
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "experiment_name": args.experiment_name,
    }

    return {"model": model_cfg, "training": training_cfg} 