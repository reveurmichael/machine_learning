# LLM Fine-Tune v0.02 – Multi-Dataset + PEFT

v0.02 evolves the minimal **v0.01** script into a reusable *pipeline* that can
fine-tune any causal-LM on **multiple heuristic JSONL datasets** with optional
LoRA/QLoRA adapters.

## 🚀 Key Features

* **Typed `FineTuneConfig`** – all hyper-parameters captured in one dataclass.
* **`FineTunePipeline`** – small, readable class (≈140 loc) handling:
  * 95 / 5 split, shuffling, tokenisation.
  * Hugging-Face `Trainer` with sensible defaults.
  * Optional LoRA (rank & adapters configurable).
* **CLI (`cli.py`)** – thin wrapper around the config & pipeline.
* **Zero legacy code** – completely forward-looking; no compatibility flags.
* **Single-source-of-truth** – all paths validated by
  `extensions.common.dataset_directory_manager`.

## 🏃‍♂️ Quick Start

```bash
# Train with LoRA on two datasets
python -m extensions.llm_finetune_v0_02.cli \
    train \
    --model mistralai/Mistral-7B-v0.1 \
    --datasets logs/extensions/datasets/grid-size-10/bfs.jsonl \
               logs/extensions/datasets/grid-size-10/astar.jsonl \
    --output logs/extensions/models/grid-size-10/finetune_v0.02 \
    --epochs 3 --batch 4
```

Disable LoRA:

```bash
python -m extensions.llm_finetune_v0_02.cli train \
       --no-lora …
```

## 📂 Directory Layout After Training

```
finetune_v0.02/
├── adapter_model.bin   # (if LoRA) lightweight Δ-weights
├── config.json
├── tokenizer.json
└── …
```

## ✨ Future Evolution (v0.03)

v0.03 adds a Streamlit dashboard & evaluation helpers but re-uses the exact
pipeline and config of v0.02 – zero duplication. 