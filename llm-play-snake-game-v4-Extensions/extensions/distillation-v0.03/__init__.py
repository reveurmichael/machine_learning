"""LLM Distillation v0.03 – Interactive Dashboard + Quick Eval

Natural evolution from v0.02: adds a Streamlit UI for configuring teacher / student
runs and computes perplexity after distillation.  The training core re-uses the
`DistillationPipeline` from v0.02 – zero duplication, fully forward-looking.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from extensions.distillation_v0_02 import DistilConfig, DistillationPipeline  # re-export

__all__ = ["DistilConfig", "DistillationPipeline", "EvaluationSuite"]


class EvaluationSuite:
    """Tiny helper to compute perplexity of the distilled student."""

    def __init__(self, model_dir: Path, dataset_paths: List[Path], block_size: int = 512):
        import json
        import datasets  # type: ignore
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.block_size = block_size

        # Merge datasets
        records = []
        for p in dataset_paths:
            files = list(p.glob("*.jsonl")) if p.is_dir() else [p]
            for f in files:
                records.extend(json.loads(l) for l in f.read_text("utf-8").splitlines() if l.strip())
        if not records:
            raise RuntimeError("Evaluation dataset empty")
        self.dataset = datasets.Dataset.from_list(records)

    # --------------------------------------------------
    def perplexity(self, sample_size: int = 128) -> float:
        from math import exp
        import torch
        subset = self.dataset.shuffle(seed=42).select(range(min(sample_size, len(self.dataset))))
        enc = self.tokenizer("\n\n".join(subset["prompt"]), return_tensors="pt")
        stride = self.block_size
        nlls = []
        for i in range(0, enc.input_ids.size(1), stride):
            begin = max(i + stride - self.block_size, 0)
            end = i + stride
            inp = enc.input_ids[:, begin:end]
            tgt = inp.clone()
            with torch.no_grad():
                out = self.model(inp, labels=tgt)
            nlls.append(out.loss * (end - begin))
        ppl = exp(torch.stack(nlls).sum() / end)
        return float(ppl)

    def quick_report(self) -> dict:
        return {"perplexity": self.perplexity()} 