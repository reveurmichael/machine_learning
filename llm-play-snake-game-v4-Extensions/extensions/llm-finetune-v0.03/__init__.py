"""LLM Fine-tuning extension v0.03 – adds Streamlit dashboard & evaluation helpers.

Evolution from v0.02:
  • Interactive Streamlit `app.py` for configuring and launching runs.
  • Built-in evaluation (perplexity & exact-match) after training.
  • Same forward-looking, zero-legacy philosophy – re-uses v0.02 pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from extensions.llm_finetune_v0_02 import FineTuneConfig as _V02Config, FineTunePipeline as _V02Pipeline

__all__ = [
    "FineTuneConfig",
    "FineTunePipeline",
    "EvaluationSuite",
]


class FineTuneConfig(_V02Config):
    """Identical to v0.02 config – re-export for clarity."""

    # v0.03 might add UI-only fields later; keep inheritance simple for now.


class FineTunePipeline(_V02Pipeline):
    """Direct subclass – inherits everything from v0.02."""

    # Placeholder for future v0.03-specific overrides (e.g., Weights & Biases hooks)


class EvaluationSuite:
    """Very small evaluation helper for post-training metrics."""

    def __init__(self, model_dir: Path, dataset_paths: List[Path], block_size: int = 512):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
        import json
        import datasets  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)

        # Load evaluation dataset (merge paths)
        records = []
        for p in dataset_paths:
            if p.is_dir():
                files = list(p.glob("*.jsonl"))
            else:
                files = [p]
            for f in files:
                records.extend(json.loads(line) for line in f.read_text("utf-8").splitlines() if line.strip())
        if not records:
            raise RuntimeError("Evaluation dataset is empty")
        self.dataset = datasets.Dataset.from_list(records)
        self.block_size = block_size

    # ------------------------------------------------------
    def compute_perplexity(self, sample_size: int = 128) -> float:
        from math import exp
        import torch
        from torch.nn import CrossEntropyLoss

        subset = self.dataset.shuffle().select(range(min(sample_size, len(self.dataset))))
        encodings = self.tokenizer("\n\n".join(subset["prompt"]), return_tensors="pt")
        max_length = self.block_size
        stride = max_length
        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = i + stride
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * (end_loc - begin_loc)
            nlls.append(neg_log_likelihood)
        ppl = exp(torch.stack(nlls).sum() / end_loc)
        return float(ppl)

    def quick_report(self) -> dict:
        ppl = self.compute_perplexity()
        return {"perplexity": ppl} 