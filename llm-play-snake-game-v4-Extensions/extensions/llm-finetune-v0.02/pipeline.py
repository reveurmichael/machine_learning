from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import datasets  # type: ignore
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

try:
    import peft  # noqa: F401 – optional dependency
    from peft import LoraConfig, get_peft_model
except ModuleNotFoundError:  # Soft-dependency – fall back to full fine-tune
    peft = None

from . import FineTuneConfig
from extensions.common.path_utils import setup_extension_paths
from extensions.common.dataset_directory_manager import DatasetDirectoryManager

setup_extension_paths()

__all__ = ["FineTunePipeline"]


class FineTunePipeline:  # noqa: D101 – docstring below
    """Simple *but future-oriented* fine-tuning pipeline.

    It supports:
    • Multiple JSONL datasets (``prompt`` + ``completion``)
    • Automatic shuffling / train–validation split
    • Optional LoRA (or QLoRA if you pass 4-bit model outside)

    We purposefully keep the implementation short – the goal is educational
    clarity, not production completeness.
    """

    def __init__(self, cfg: FineTuneConfig) -> None:  # noqa: D401
        self.cfg = cfg
        self.raw_dataset = self._load_datasets(cfg.dataset_paths)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
        # Handle pad token for chat models
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.pad_token

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401
        """Run the full fine-tuning pipeline – train & save."""
        print("📚 Tokenising dataset …")
        tokenised = self._tokenise(self.raw_dataset)

        model = self._prepare_model()
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        training_args = self._build_training_args()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenised["train"],
            eval_dataset=tokenised["test"],
            data_collator=collator,
        )

        print("🚀 Starting training …")
        trainer.train()
        trainer.save_model(str(self.cfg.output_dir))
        self.tokenizer.save_pretrained(str(self.cfg.output_dir))
        print("✅ Training complete – artefacts saved to", self.cfg.output_dir)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_datasets(self, paths: List[Path]) -> datasets.Dataset:
        all_records: List[Dict[str, Any]] = []
        for p in paths:
            DatasetDirectoryManager.validate_dataset_path(p)
            if p.is_dir():
                jsonl_files = list(p.glob("*.jsonl"))
                for f in jsonl_files:
                    all_records.extend(self._read_jsonl(f))
            else:
                all_records.extend(self._read_jsonl(p))
        if not all_records:
            raise RuntimeError("No JSONL examples found in provided paths")
        ds = datasets.Dataset.from_list(all_records)
        # 95/5 split
        return ds.shuffle(seed=42).train_test_split(test_size=0.05)

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]

    # ------------------------------------------------------------------
    def _tokenise(self, dataset: datasets.DatasetDict) -> datasets.DatasetDict:
        def _tok(batch: Dict[str, Any]) -> Dict[str, Any]:
            return self.tokenizer(
                [f"{p}\n{c}" for p, c in zip(batch["prompt"], batch["completion"])],
                truncation=True,
                max_length=self.cfg.block_size,
                padding="max_length",
            )

        return dataset.map(_tok, batched=True, remove_columns=list(dataset["train"].column_names))

    # ------------------------------------------------------------------
    def _prepare_model(self):  # noqa: D401
        print("🧩 Loading base model …")
        model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name_or_path)
        if self.cfg.use_lora:
            if peft is None:
                raise RuntimeError("peft not installed – install `peft` or set use_lora=False")
            print("🔌 Applying LoRA adapters …")
            lora_cfg = LoraConfig(r=self.cfg.lora_rank, lora_alpha=self.cfg.lora_rank * 2, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
            model = get_peft_model(model, lora_cfg)
        # Fix pad token id
        model.resize_token_embeddings(len(self.tokenizer))
        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model

    # ------------------------------------------------------------------
    def _build_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(self.cfg.output_dir),
            num_train_epochs=self.cfg.num_train_epochs,
            per_device_train_batch_size=self.cfg.per_device_batch_size,
            per_device_eval_batch_size=self.cfg.per_device_batch_size,
            learning_rate=self.cfg.learning_rate,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=200,
            logging_steps=50,
            save_total_limit=2,
            fp16=True,
            dataloader_pin_memory=False,
        ) 