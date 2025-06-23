# Heuristics → LLM Fine-Tuning Integration v0.01

This extension automates the *two-step* workflow that converts games played by
heuristic agents into a language-rich **JSONL dataset** and immediately
fine-tunes an open-weight Large Language Model (LLM) on that data.

The purpose is to bridge **symbolic** path-finding (BFS, A*, Hamiltonian, …)
with **neural** language models, producing a model that can *explain* and
*imitate* classical heuristics in natural language.

---

## 1  Pipeline Overview

```
heuristic logs  ─▶  JSONL generator  ─▶  dataset.jsonl
                                 │
                                 └──▶  📊  CSV/NPZ/Parquet (kept for v0.03)

 dataset.jsonl  + base-model  ─▶  SFT / LoRA fine-tuner  ─▶  tuned LLM
```

Implementation lives in `pipeline.py` and uses only **public CLI** contracts
so that every moving part stays *single-source-of-truth*:

* `extensions.common.dataset_generator_cli` – grid-aware dataset generator
* `extensions.llm_finetune_v0_01.finetune` – lightweight SFT trainer
* `extensions.common.training_logging_utils` – unified, colourful logging

---

## 2  Quick Start

```bash
# (1) Generate dataset + (2) Fine-tune LLM – all in one command
python -m extensions.heuristics_llm_fine_tuning_integration_v0_01.pipeline \
    --algorithm BFS \
    --games 800 \
    --grid-size 10 \
    --model mistralai/Mistral-7B-v0.1 \
    --output-dir logs/extensions/models/mistral_snake_sft 
```

Flags:

| Argument        | Default | Description                                         |
|-----------------|---------|-----------------------------------------------------|
| `--algorithm`   | BFS     | Heuristic agent name (BFS, ASTAR, …)                |
| `--games`       | 1000    | Number of games to sample                           |
| `--grid-size`   | 10      | Board dimension (auto-detected if you pass `logs`)  |
| `--model`       | ‑       | Base LLM HF ID (e.g. `deepseek-ai/deepseek-7b`)     |
| `--output-dir`  | ‑       | Where to store the fine-tuned checkpoints           |
| `--epochs`      | 2       | Fine-tuning epochs                                  |
| `--batch`       | 2       | Per-device batch size                               |

---

## 3  Design Principles

1. **Zero duplication** – all heavy-lifting lives in the *common* package or
   in the already existing heuristic / LLM extensions.  This file only
   orchestrates.
2. **Single Source of Truth** – path handling delegates to
   `utils.path_utils.ensure_project_root()` *via* the thin façade in
   `extensions.common.path_utils`.
3. **Grid-aware datasets** – every dataset is saved in
   `logs/extensions/datasets/grid-size-N/` to guarantee experimental
   reproducibility.
4. **Composable CLIs** – you can always run the two stages independently:
   ```bash
   # 1. Dataset only
   python -m extensions.common.dataset_generator_cli jsonl --log-dir logs/extensions/heuristics-bfs_20250623_090525

   # 2. Fine-tune only
   python -m extensions.llm_finetune_v0_01.finetune \
          --dataset logs/extensions/datasets/grid-size-10/heuristics_bfs_*.jsonl \
          --model mistralai/Mistral-7B-v0.1 \
          --output-dir saved_models/sft_bfs
   ```

---

## 4  Future Work

* **v0.02** – add LoRA & QLoRA adapters, mixed-precision training, and WandB
  tracking.
* **v0.03** – Streamlit dashboard for real-time fine-tuning metrics and model
  evaluation on held-out heuristic games.

---

© 2025 Snake-GTP Authors — MIT License 