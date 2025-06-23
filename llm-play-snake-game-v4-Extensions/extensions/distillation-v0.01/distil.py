"""distil.py – Knowledge Distillation of Snake-LLM Teacher → Student

Example usage:

    python -m extensions.distillation_v0_01.distil \
        --dataset logs/extensions/datasets/grid-size-10/language_mixed_data.jsonl \
        --teacher logs/extensions/models/finetuned_mistral_snake \
        --student google/gemma-2b \
        --output-dir logs/extensions/models/distilled_gemma_snake

This script implements the *vanilla* distillation objective described in
Hinton et al. (2015):

    L = α * CE(student, labels) + (1−α) * T² * KL(student‖teacher)

where labels are the teacher's predictions at temperature *T*.

For v0.01 we use `α = 0.1`, `T = 2.0` and train for a handful of epochs.  No
LoRA / PEFT yet – that lands in v0.02.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import datasets  # type: ignore
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from extensions.common.dataset_directory_manager import DatasetDirectoryManager
from extensions.common.path_utils import setup_extension_paths

setup_extension_paths()

# ---------------------
# CLI
# ---------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Knowledge-distil a fine-tuned teacher into a small student.")
    p.add_argument("--dataset", required=True, help="JSONL file or folder (heuristic explanations)")
    p.add_argument("--teacher", required=True, help="Path / HF-hub id of fine-tuned teacher")
    p.add_argument("--student", required=True, help="Base model name for student")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--alpha", type=float, default=0.1, help="CE weight; KL weight becomes 1-alpha")
    p.add_argument("--temperature", type=float, default=2.0)
    return p

# ---------------------
# Dataset loading
# ---------------------

def _load_jsonl(path: Path) -> datasets.Dataset:
    if path.is_dir():
        files = list(path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError("No JSONL in directory")
        data = []
        for f in files:
            data.extend(json.loads(l) for l in f.read_text("utf-8").splitlines() if l.strip())
    else:
        data = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    return datasets.Dataset.from_list(data)

# ---------------------
# Distillation loss via Trainer callback
# ---------------------

class DistilLossWrapper(nn.Module):
    """Wraps student & teacher to compute combined loss."""

    def __init__(self, student: nn.Module, teacher: nn.Module, alpha: float, temperature: float):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval()  # freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.alpha = alpha
        self.T = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):  # type: ignore[override]
        student_out = self.student(input_ids=input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            teacher_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)

        s_logits = student_out.logits / self.T
        t_logits = teacher_out.logits / self.T

        loss_kl = self.kl_loss(nn.functional.log_softmax(s_logits, dim=-1), nn.functional.softmax(t_logits, dim=-1)) * (self.T ** 2)

        if labels is not None:
            loss_ce = self.ce_loss(s_logits.view(-1, s_logits.size(-1)), labels.view(-1))
            loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kl
        else:
            loss = loss_kl
        return {"loss": loss, "logits": student_out.logits}

# ---------------------
# Main
# ---------------------

def main() -> None:  # noqa: D401
    args = _build_parser().parse_args()

    ds_path = Path(args.dataset)
    DatasetDirectoryManager.validate_dataset_path(ds_path)

    raw_ds = _load_jsonl(ds_path)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token

    def _tok(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            [p + "\n" + c for p, c in zip(batch["prompt"], batch["completion"])],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    ds = raw_ds.map(_tok, batched=True, remove_columns=raw_ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher)
    student = AutoModelForCausalLM.from_pretrained(args.student)

    distil_model = DistilLossWrapper(student, teacher, args.alpha, args.temperature)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=50,
        save_steps=200,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=True,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=distil_model,
        args=train_args,
        train_dataset=ds,
        eval_dataset=None,
        data_collator=collator,
    )

    trainer.train()
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Distillation complete – student saved.")


if __name__ == "__main__":
    main() 