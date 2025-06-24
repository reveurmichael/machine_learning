# Heuristics → Supervised Learning Pipeline (v0.04)

> **Status:** Alpha – Works end-to-end on PyTorch; LightGBM/XGBoost planned.<br/>
> **Audience:** Practitioners who want to reproduce the full data-generation & training loop on a fresh clone.

---

## 1. Overview

The pipeline converts classical heuristic play-throughs of Snake into tabular CSV datasets and then trains supervised models (MLP, CNN, LSTM) that learn to imitate those heuristics.  The _same_ data later powers **Task-3 (RL pre-training)** and **Task-4 (LLM fine-tuning)** – so data lineage and reproducibility are crucial.

```
┌─────────────┐    game_*.json      ┌────────────────┐   CSV / NPZ / Parquet   ┌──────────────┐
│ heuristics  │───────────────────▶│ common/dataset │────────────────────────▶│ supervised   │
│  v0.03+     │                    │   generators    │                          │  v0.02/03    │
└─────────────┘                    └────────────────┘                          └──────────────┘
          ▲                                     │                                         │
          │                                     │ .pt / .onnx                            │
          └──────────────── evaluation ─────────┘                                         │
                                                   trained_models/                         │
```

**Key design rules**  (see `project-structure-plan.md`):
1. Every dataset lives under `logs/extensions/datasets/grid-size-N/` – never mix grid sizes. #TODO: and then subfolders, subsubfolders, subsubsubfolders, etc. and then the name of the dataset files (csv, summary.json, game_N.json, jsonl, npz, parquet, etc.), which is not decided yet, because we are having a ongoing discussion on this folder and file naming conventions.
2. `extensions/common/` holds *all* shared helpers; extensions must **not** import each other.
3. Output models are saved with [`common.model_utils.save_model_standardized`](../extensions/common/model_utils.py) – portable across OS & frameworks. VITAL: THIS IS VERY IMPORTANT. I WANT THOSE FILES TO BE SHARED TO A LOT OF DIFFERENT PEOPLE WITH DIFFERENT OPERATING SYSTEMS. AND I WANT THEM TO BE TIME-PROOF, HENCE STILL USABLE IN THE FUTURE, AT LEAST OF THE NEXT 10 YEARS.


---

## 2. Generating the dataset

```
# Example: BFS 10×10 – CSV + JSONL (rich explanations)
python -m extensions.common.dataset_generator_cli \
    both                                \  # csv|jsonl|both
    --log-dir log_folder_path_for_one_experiment/blablabla_folder_or_sub_or_subsub_folders_or_file_whose_naming_is_not_decided_yet # TODO: check this, update blablabla. \
    --prompt-format detailed             \  # jsonl prompt style
    --output-dir logs/extensions/datasets/grid-size-N/or_maybe_blablabla_folder_or_sub_or_subsub_folders_or_file_whose_naming_is_not_decided_yet # TODO: check this, update blablabla. \
    --verbose
```

The CLI will:
1. Auto-detect grid size (`GridSizeDetector`) – supports 8-50.
2. Enforce `grid-size-N/` directory creation via `DatasetDirectoryManager`.
3. Produce:
   * `tabular_bfs_data_<timestamp>.csv`
   * `rich_bfs_data_<timestamp>.jsonl` *(for Task-4)*
   * `metadata_*.json` (schema, git SHA, etc.)

TODO: is this really what we want for naming? Double check. I am not sure. We will have to discuss this. After the discussion, we will have to update this documentation. After the discussion, things will be fixed, and we will really enforce the naming conventions, not by our brain memory, but by the code (validation, etc.), as well as by documentions everywhere in the codebase.

> **Tip:** `--all-algorithms` will sweep *every* `heuristics-*` log folder and batch-generate datasets.


---

## 3. Training a supervised model (PyTorch)

### 3.1 Minimal command

```
python -m extensions.supervised_v0_02.training.train_neural \
  --dataset-paths blablabla
  --model MLP                  \  # choices: MLP|CNN|LSTM|GRU
  --epochs 50 --batch-size 64  \  # quick smoke-test
  --output-dir trained_models/mlp_bfs
```

### 3.2 What happens under the hood?

TODO: things are not decided yet. We will have to discuss this. After the discussion, we will have to update this documentation. After the discussion, things will be fixed, and we will really enforce the naming conventions, not by our brain memory, but by the code (validation, etc.), as well as by documentions everywhere in the codebase.

 
1. `DatasetLoader` – handles train/val/test split (default 60/20/20), one-hot encodes moves.
2. `agent_mlp.MLPAgent` – simple 3-layer feed-forward **(input size = 16 engineered features)**.
3. Training loop logs loss every epoch (see stdout).  Early stopping coming in v0.03.
4. Final metrics are printed and returned as JSON:
   ```json
   {
     "validation_accuracy": 0.94,
     "test_accuracy": 0.93,
     "grid_size": 10,
     "model_path": "trained_models/mlp_bfs/blablabla.pth"
   }
   ```
5. Model is saved in both **PyTorch** (`.pth`) and **ONNX** (`.onnx`) format with rich metadata – see `extensions/common/model_utils.py`.

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
python -m extensions.supervised_v0_03.scripts.replay_web --model trained_models/mlp_bfs/blablabla --grid-size 10
```

*Opens a Flask replay page under <http://localhost:5000/supervised/replay>*


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

* **LightGBM / XGBoost** trainer script – tabular CSV already compatible. # TODO: make sure.
* **Cross-grid generalisation** study: train on 8×8, test on 12×12. # TODO: so this is curriculum learning? 

---

## 7. Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| `ModuleNotFoundError: extensions` in Streamlit | Working directory changed | Ensure `path_utils.setup_extension_paths()` at top of script |
