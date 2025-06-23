# `extensions.common` – Shared Utilities

This package hosts *plumbing-level* helpers that are reused by **multiple**
second-citizen extensions (heuristics, supervised, RL, …).  The guiding rule is
simple:

> If a function is useful to more than one extension **and** does **not** hide
the conceptual core of that extension, then it belongs here.

## Directory Map

| Module | Concern | Notes |
| ------ | ------- | ----- |
| `path_utils.py` | `sys.path` bootstrap helpers | Keeps entry-scripts one-liner. |
| `config.py` | constants & global paths | Single-source-of-truth for logs/datasets. |
| `dataset_*` | dataset generation / schema | Heuristic CSV & JSONL helpers. |
| `model_utils.py` | framework-agnostic save/load | Saves PyTorch, XGBoost, … with metadata. |
| `training_cli_utils.py` | Argument parsing & validation | Used by supervised / RL training scripts. |
| `training_config_utils.py` | JSON (de)serialization & defaults | Pure dataclass wrappers. |
| `training_logging_utils.py` | Lightweight experiment logger | No 3rd-party deps. |
| `rl_utils.py` | Minimal replay buffer & ε-schedule | Shared by DQN agents. |
| `rl_helpers.py` | Seed, target-net sync, reward tracker | Non-essential but convenient. |

Everything else should stay inside its extension package so that the
*algorithmic idea* remains visible to the reader.

## Heuristic Utilities (NEW)

The common package now includes specialized utilities for heuristic extensions to eliminate code duplication while keeping core algorithms visible:

### 🧠 Core Heuristic Support
- **`heuristic_utils.py`**: Session management, logging, performance tracking
- **`heuristic_replay_utils.py`**: Replay data processing, algorithm metadata
- **`heuristic_web_utils.py`**: Streamlit/Flask web interface components

### 📚 Usage Philosophy
- **Infrastructure code** → `extensions/common/` (logging, metrics, UI components)
- **Algorithm logic** → Extensions (BFS, A*, DFS, Hamiltonian pathfinding)
- **Educational value** → Clear separation makes learning easier

### 🎯 Benefits for Heuristic Extensions
1. **Consistency**: All heuristic extensions (v0.01-v0.04) use same patterns
2. **Focus**: Extension code highlights the algorithmic concepts
3. **Maintainability**: Bug fixes and improvements benefit all extensions
4. **DRY compliance**: Single source of truth for common patterns

### 📖 Documentation
See `README_HEURISTICS.md` for detailed examples and migration guide.

---

### Contribution checklist

1. **Is the helper reused by ≥2 extensions?** If *no*, keep it local.
2. **Does it obscure the core ML/RL/heuristic idea?** If *yes*, keep it local.
3. Add a focused docstring that explains the *why*, not just the *what*.
4. Export the symbol in `extensions/common/__init__.py`.
5. Write a unit-test if behaviour is non-trivial. 