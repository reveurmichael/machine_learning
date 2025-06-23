from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import datasets  # type: ignore
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

try:
    import peft  # type: ignore
    from peft import LoraConfig, get_peft_model
except ModuleNotFoundError:
    peft = None  # Student will be full fine-tune if PEFT missing

from . import DistilConfig
from extensions.common.dataset_directory_manager import DatasetDirectoryManager
from extensions.common.path_utils import setup_extension_paths

setup_extension_paths()

__all__ = ["DistillationPipeline"]


class _DistilLoss(nn.Module):
    """Hinton-style distillation loss wrapper (supports LoRA student)."""

    def __init__(self, student: nn.Module, teacher: nn.Module, alpha: float, T: float):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.alpha = alpha
        self.T = T
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, input_ids=None, attention_mask=None, labels=None):  # noqa: D401
        s_out = self.student(input_ids=input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
        s_logits = s_out.logits / self.T
        t_logits = t_out.logits / self.T
        loss_kl = self.kl(torch.nn.functional.log_softmax(s_logits, dim=-1), torch.nn.functional.softmax(t_logits, dim=-1)) * (self.T ** 2)
        if labels is not None:
            loss_ce = self.ce(s_logits.view(-1, s_logits.size(-1)), labels.view(-1))
            loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kl
        else:
            loss = loss_kl
        return {"loss": loss, "logits": s_out.logits}


class DistillationPipeline:
    """End-to-end knowledge-distillation runner (future-proof)."""

    def __init__(self, cfg: DistilConfig):
        self.cfg = cfg
        self.dataset = self._load_dataset(cfg.dataset_paths)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_path, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.pad_token

    # ------------------------------------------------------------
    def run(self) -> None:  # noqa: D401
        teacher = AutoModelForCausalLM.from_pretrained(self.cfg.teacher_path)
        student = self._prepare_student()
        distil_model = _DistilLoss(student, teacher, self.cfg.alpha, self.cfg.temperature)

        train_args = TrainingArguments(
            output_dir=str(self.cfg.output_dir),
            num_train_epochs=self.cfg.num_train_epochs,
            per_device_train_batch_size=self.cfg.per_device_batch_size,
            learning_rate=self.cfg.learning_rate,
            logging_steps=50,
            save_steps=200,
            evaluation_strategy="no",
            save_total_limit=2,
            fp16=True,
            dataloader_pin_memory=False,
        )

        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        trainer = Trainer(model=distil_model, args=train_args, train_dataset=self.dataset, data_collator=collator)

        print("🚀 Starting distillation …")
        trainer.train()
        print("✅ Done – saving artefacts …")
        student.save_pretrained(str(self.cfg.output_dir))
        self.tokenizer.save_pretrained(str(self.cfg.output_dir))

    # ------------------------------------------------------------
    def _prepare_student(self):
        student = AutoModelForCausalLM.from_pretrained(self.cfg.student_model)
        if self.cfg.use_lora_student:
            if peft is None:
                raise RuntimeError("peft not installed, install or set use_lora_student=False")
            l_cfg = LoraConfig(r=self.cfg.lora_rank, lora_alpha=self.cfg.lora_rank * 2, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
            student = get_peft_model(student, l_cfg)
        student.resize_token_embeddings(len(self.tokenizer))
        student.config.pad_token_id = self.tokenizer.pad_token_id
        return student

    # ------------------------------------------------------------
    def _load_dataset(self, paths: List[Path]):
        all_records: List[Dict[str, Any]] = []
        for p in paths:
            DatasetDirectoryManager.validate_dataset_path(p)
            files = list(p.glob("*.jsonl")) if p.is_dir() else [p]
            for f in files:
                all_records.extend(json.loads(l) for l in f.read_text("utf-8").splitlines() if l.strip())
        if not all_records:
            raise RuntimeError("No records found for distillation")
        ds = datasets.Dataset.from_list(all_records)

        def _tok(batch):
            return self.tokenizer(
                [f"{p}\n{c}" for p, c in zip(batch["prompt"], batch["completion"])],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

        return ds.map(_tok, batched=True, remove_columns=ds.column_names) 