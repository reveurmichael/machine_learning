"""pipeline.py – Heuristic JSONL generation + LLM fine-tuning (v0.01)

Example:
    python -m extensions.heuristics_llm_fine_tuning_integration_v0_01.pipeline \
        --algorithm BFS --games 800 --grid-size 10 \
        --model mistralai/Mistral-7B-v0.1 \
        --output-dir logs/extensions/models/grid-size-N/mistral_snake_sft
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from datetime import datetime

from extensions.common.dataset_directory_manager import DatasetDirectoryManager
from extensions.common.training_logging_utils import TrainingLogger
from extensions.common.path_utils import setup_extension_paths

setup_extension_paths()

# ---------------------
# CLI
# ---------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end heuristic JSONL → LLM fine-tune")
    p.add_argument("--algorithm", default="BFS", help="Heuristic algorithm (BFS, ASTAR, …)")
    p.add_argument("--games", type=int, default=1000, help="Number of games to generate")
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--model", required=True, help="Base LLM identifier for fine-tuning")
    p.add_argument("--output-dir", required=True, help="Where to save the fine-tuned model")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=2)
    return p

# ---------------------
# Helpers
# ---------------------

def _call(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _generate_jsonl(alg: str, games: int, grid: int) -> Path:
    cmd = [
        "python", "-m", "extensions.common.dataset_generator_cli",
        "--algorithm", alg,
        "--games", str(games),
        "--format", "jsonl",
        "--grid-size", str(grid),
    ]
    _call(cmd)
    ds_dir = DatasetDirectoryManager.grid_size_dir(grid)
    jsonl_files = sorted(ds_dir.glob(f"*{alg.lower()}*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not jsonl_files:
        raise FileNotFoundError("JSONL dataset not found after generation")
    return jsonl_files[0]


def _fine_tune(jsonl_path: Path, base_model: str, out_dir: str, epochs: int, batch: int):
    cmd = [
        "python", "-m", "extensions.llm_finetune_v0_01.finetune",
        "--dataset", str(jsonl_path),
        "--model", base_model,
        "--output-dir", out_dir,
        "--epochs", str(epochs),
        "--batch", str(batch),
    ]
    _call(cmd)

# ---------------------
# Main
# ---------------------

def main() -> None:  # noqa: D401
    args = _parser().parse_args()
    log = TrainingLogger("heuristic→LLM")

    log.info("[1/2] Generating explanation JSONL …")
    jsonl_path = _generate_jsonl(args.algorithm, args.games, args.grid_size)
    log.info(f"JSONL ready: {jsonl_path}")

    log.info("[2/2] Fine-tuning LLM … (this may take a while)")
    output_dir = Path(args.output_dir) / f"{args.algorithm.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _fine_tune(jsonl_path, args.model, str(output_dir), args.epochs, args.batch)
    log.info("✅ All done.")


if __name__ == "__main__":
    main() 