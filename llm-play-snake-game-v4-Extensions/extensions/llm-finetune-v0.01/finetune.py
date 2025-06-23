"""finetune.py – Minimal Hugging Face trainer for Snake‐heuristic JSONL

Usage example (single-GPU):

    python -m extensions.llm_finetune_v0_01.finetune \
        --dataset path/to/dataset.jsonl \
        --model mistralai/Mistral-7B-v0.1 \
        --output-dir logs/extensions/models/finetuned_mistral_snake/

The script is intentionally *tiny* – it exists to document the data pipeline
from heuristic explanations → LLM.  Hyper-parameters are exposed via CLI for
quick iteration; advanced features (LoRA, 8-bit) can land in v0.02.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import datasets  # type: ignore
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from extensions.common.dataset_directory_manager import DatasetDirectoryManager
from extensions.common.path_utils import setup_extension_paths

setup_extension_paths()

# ---------------------
# CLI helpers
# ---------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on heuristic JSONL")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset or directory")
    parser.add_argument("--model", type=str, required=True, help="Base HF model name or local path")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save checkpoints & logs")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--block-size", type=int, default=512, help="Max sequence length after tokenisation")
    return parser

# ---------------------
# Dataset loading
# ---------------------

def _load_jsonl(path: Path) -> datasets.Dataset:
    """Load JSONL with `prompt` & `completion` keys into HF Dataset."""
    if path.is_dir():
        jsonl_files = list(path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {path}")
        data = []
        for file in jsonl_files:
            with file.open("r", encoding="utf-8") as fh:
                data.extend(json.loads(line) for line in fh)
    else:
        data = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return datasets.Dataset.from_list(data)

# ---------------------
# Main
# ---------------------

def main() -> None:  # noqa: D401
    args = _build_arg_parser().parse_args()

    dataset_path = Path(args.dataset)
    DatasetDirectoryManager.validate_dataset_path(dataset_path)

    print(f"📚 Loading dataset from {dataset_path}")
    raw_ds = _load_jsonl(dataset_path)

    print(f"📦 Loading model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token

    def _tokenise(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            [p + "\n" + c for p, c in zip(batch["prompt"], batch["completion"])],
            truncation=True,
            max_length=args.block_size,
            padding="max_length",
        )

    ds = raw_ds.shuffle().train_test_split(test_size=0.05)
    tokenised = ds.map(_tokenise, batched=True, remove_columns=raw_ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(args.model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=50,
        save_total_limit=2,
        fp16=True,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["test"],
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Fine-tuning complete – model saved.")


if __name__ == "__main__":
    main() 