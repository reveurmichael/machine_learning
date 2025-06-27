# Heuristics to Supervised Learning Pipeline

> **Status:** Alpha – Works end-to-end on PyTorch; LightGBM/XGBoost planned.<br/>
> **Audience:** Practitioners who want to reproduce the full data-generation & training loop on a fresh clone.

> **SUPREME_RULES**: Both `heuristics-v0.03` and `heuristics-v0.04` are widely used depending on use cases and scenarios. For supervised learning and other general purposes, both versions can be used. For LLM fine-tuning, only `heuristics-v0.04` will be used. The CSV format is **NOT legacy** - it's actively used and valuable for supervised learning.

## 🎯 **Pipeline Overview**

This document describes the complete pipeline from heuristic algorithm execution to supervised learning model training, demonstrating how different extensions work together to create a comprehensive machine learning workflow.

---

## 1. Overview

The pipeline converts classical heuristic play-throughs of Snake into tabular CSV datasets and then trains supervised models (MLP, CNN, LSTM) that learn to imitate those heuristics.  The _same_ data later powers **Task-3 (RL pre-training)** and **Task-4 (LLM fine-tuning)** – so data lineage and reproducibility are crucial.

```
┌─────────────┐    game_*.json      ┌────────────────┐   CSV / NPZ / Parquet   ┌──────────────┐
│ heuristics  │───────────────────▶│ common/dataset │────────────────────────▶│ supervised   │
│  v0.04+     │                    │   generators    │                          │  v0.02/03    │
└─────────────┘                    └────────────────┘                          └──────────────┘
          ▲                                     │                                         │
          │                                     │ .pt / .onnx                            │
          └──────────────── evaluation ─────────┘                                         │
                                                   trained_models/                         │
```

**Key design rules**  (see `project-structure-plan.md`):
1. Every dataset lives under `logs/extensions/datasets/grid-size-N/{extension_type}_v{version}_{timestamp}/{algorithm_name}/processed_data/` – never mix grid sizes. Structure follows final-decision-1.md with clear separation of game_logs/ and processed_data/ folders.
2. `extensions/common/` holds *all* shared helpers; extensions must **not** import each other.
3. Output models are saved with simple, local save functions (SUPREME_RULE NO.3: avoid over-engineered utilities) – portable across OS & frameworks. 


---

## 2. Generating the dataset

```
# Example: BFS 10×10 – CSV + JSONL (rich explanations)
python -m extensions.common.dataset_generator_cli \
    both                                \  # csv|jsonl|both
    --log-dir logs/extensions/heuristics_v0.04_20250625_143022/ \  # Use v0.04 (definitive)
    --prompt-format detailed             \  # jsonl prompt style
    --output-dir logs/extensions/datasets/grid-size-10/heuristics_v0.04_20250625_143022/bfs/processed_data/ \  # Use v0.04
    --verbose
```

The CLI will:
1. Auto-detect grid size (`GridSizeDetector`) – supports 8-50.
2. Enforce `grid-size-N/` directory creation via `DatasetDirectoryManager`.
3. Produce:
   * `tabular_data.csv` - Standardized 16-feature tabular format (ACTIVE, NOT legacy)
   * `reasoning_data.jsonl` - Language-rich explanations for LLM fine-tuning
   * `metadata.json` - Schema information, git SHA, generation timestamp

**Naming Convention**: Dataset files follow standardized naming patterns enforced by validation utilities in `extensions/common/validation/`. The naming format ensures consistency across all extensions and grid sizes.

> **Tip:** `--all-algorithms` will sweep *every* `heuristics-*` log folder and batch-generate datasets.

> **SUPREME_RULES**: Use heuristics-v0.04 for dataset generation - it's a superset with no downsides.


---

## 3. Training a supervised model (PyTorch)

### 3.1 Minimal command

```
python -m extensions.supervised_v0_02.training.train_neural \
  --dataset-paths logs/extensions/datasets/grid-size-10/heuristics_v0.04_20250625_143022/bfs/processed_data/tabular_data.csv  # Use v0.04
  --model MLP                  \  # choices: MLP|CNN|LSTM|GRU
  --epochs 50 --batch-size 64  \  # quick smoke-test
  --output-dir trained_models/mlp_bfs
```

### 3.2 What happens under the hood?
 
1. `DatasetLoader` – handles train/val/test split (default 60/20/20), one-hot encodes moves.
2. `agent_mlp.MLPAgent` – simple 3-layer feed-forward **(input size = 16 engineered features)**.
3. Training loop logs loss every epoch (see stdout).  Early stopping coming in v0.03.
4. Final metrics are printed and returned as JSON:
   ```json
   {
     "validation_accuracy": 0.94,
     "test_accuracy": 0.93,
     "grid_size": 10,
     "model_path": "logs/extensions/models/grid-size-10/supervised_v0.02_20250625_143022/mlp/model_artifacts/model.pth"
   }
   ```
5. Model is saved in both **PyTorch** (`.pth`) and **ONNX** (`.onnx`) format with rich metadata using simple, local save functions (SUPREME_RULE NO.3: lightweight approach).

### 3.3 Expected baselines  (10×10 grid)

| Heuristic (teacher) | Student MLP | Student CNN | Student LSTM |
|---------------------|------------:|------------:|-------------:|
| BFS                | ≥0.92 acc | ≥0.94 acc | ≥0.94 acc |
| A*                 | ≥0.89 | ≥0.92 | ≥0.93 |
| Hamiltonian        | 1.00* | 1.00* | 1.00* |

*Hamiltonian path yields deterministic "RIGHT"/"DOWN" cycles after startup – even linear models hit 100 %.*

If your numbers are low, check:
* `--epochs` too small (start with 100 for serious runs)
* Dataset imbalance (ensure `--all-algorithms` isn't mixing incompatible agents)


---

## 4. Evaluating trained agents in-game

```
python -m extensions.supervised_v0_03.scripts.replay_web --model logs/extensions/models/grid-size-10/supervised_v0.02_20250625_143022/mlp/model_artifacts/model.pth --grid-size 10
```

*Opens a Flask replay page under <http://localhost:5000/supervised/replay> (extends ROOT/web infrastructure)*


---

## 5. Implementation notes

### 5.1 `DatasetDirectoryManager`
* **Singleton facade** – guarantees all path logic lives in one place.
* `ensure_datasets_dir()` creates dirs lazily and returns a **`Path`** object.

### 5.2 `DatasetLoader`
* Hands-off scaling: detects grid size from CSV header and rescales if needed.
* Returns *NumPy* arrays – Torch `DataLoader` wrapping handled inside each `agent_*` class.

### 5.3 Model saving
* Wrapper adds checksum, git SHA, & ONNX export.

---

## 6. We are not finished yet.

* **Cross-grid generalisation** study: train on 8×8, test on 12×12. So this is curriculum learning? 

---

## 7. Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| `ModuleNotFoundError: extensions` in Streamlit | Working directory changed | Ensure `path_utils.setup_extension_paths()` at top of script |

## 🎯 **SUPREME_RULES: Version Selection Guidelines**

- **For heuristics**: Always use v0.04 - it's a superset with no downsides
- **For supervised learning**: Use CSV from heuristics-v0.04
- **For LLM fine-tuning**: Use JSONL from heuristics-v0.04
- **For research**: Use both formats from heuristics-v0.04
- **CSV is ACTIVE**: Not legacy - actively used for supervised learning
- **JSONL is ADDITIONAL**: New capability for LLM fine-tuning
