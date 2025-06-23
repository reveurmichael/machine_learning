"""pipeline.py – End-to-End Heuristic → Supervised demo (v0.01)

Usage example:

    python -m extensions.heuristics_supervised_integration_v0_01.pipeline \
        --algorithm BFS --games 500 --grid-size 10 \
        --model MLP --epochs 50

The script performs three steps:
1. Calls the *dataset generator* from `extensions.common` to create a CSV
   dataset under `logs/extensions/datasets/grid-size-N/`.
2. Loads the CSV via `pandas` and splits into train/val.
3. Trains a **scikit-learn MLPClassifier** and saves it with metadata using
   `extensions.common.model_utils`.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from extensions.common.dataset_directory_manager import DatasetDirectoryManager
from extensions.common.model_utils import save_model_standardized
from extensions.common.training_logging_utils import TrainingLogger
from extensions.common.path_utils import setup_extension_paths

setup_extension_paths()

# ---------------------
# CLI
# ---------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run heuristic→supervised pipeline")
    p.add_argument("--algorithm", default="BFS", help="Heuristic algorithm (BFS, ASTAR, etc.)")
    p.add_argument("--games", type=int, default=500, help="Number of games to generate")
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--model", default="MLP", choices=["MLP"], help="Supervised model type (only MLP for v0.01)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden-size", type=int, default=128)
    return p


# ---------------------
# Helpers
# ---------------------

def _generate_dataset(alg: str, games: int, grid: int) -> Path:
    """Call the common dataset generator CLI via subprocess."""
    cmd = [
        "python",
        "-m",
        "extensions.common.dataset_generator_cli",
        "--algorithm", alg,
        "--games", str(games),
        "--format", "csv",
        "--grid-size", str(grid),
    ]
    print(f"[PIPELINE] Generating dataset → {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    dataset_dir = DatasetDirectoryManager.grid_size_dir(grid)
    # pick most recent CSV for algorithm
    csv_files = sorted(dataset_dir.glob(f"*{alg.lower()}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        raise FileNotFoundError("Dataset generation failed – CSV not found")
    return csv_files[0]


def _load_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["target_move"])  # label column from schema
    y = df["target_move"].map({"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}).values
    return X.values, y


# ---------------------
# Main
# ---------------------

def main():  # noqa: D401
    args = _build_parser().parse_args()

    log = TrainingLogger("integration-v0.01")
    log.info("Step 1/3 – Dataset generation …")
    csv_path = _generate_dataset(args.algorithm, args.games, args.grid_size)
    log.info(f"Dataset ready: {csv_path}")

    log.info("Step 2/3 – Loading & splitting …")
    X, y = _load_csv(csv_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    log.info("Step 3/3 – Training MLPClassifier …")
    clf = MLPClassifier(hidden_layer_sizes=(args.hidden_size, args.hidden_size // 2), max_iter=args.epochs, learning_rate_init=1e-3)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_val, clf.predict(X_val))
    log.info(f"Validation accuracy: {acc:.3f}")

    save_model_standardized(
        model=clf,
        framework="sklearn",
        grid_size=args.grid_size,
        model_name=f"mlp_{args.algorithm.lower()}_{args.grid_size}",
        model_class="MLPClassifier",
        input_size=X.shape[1],
        output_size=4,
        training_params={"epochs": args.epochs, "hidden": args.hidden_size, "val_acc": acc},
    )
    log.info("✅ Pipeline finished.")


if __name__ == "__main__":
    main() 