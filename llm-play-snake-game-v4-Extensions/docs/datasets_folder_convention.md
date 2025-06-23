# Grid-Size Aware Dataset Folders

*Single-source-of-truth documentation – applies to **all** extensions*

---

## Rule Statement  
Every dataset produced by **any** extension must be saved under:

```
logs/extensions/datasets/grid-size-N/
```

where **`N`** is the side length of the board from which the data was
collected.  Examples:

```
logs/extensions/datasets/
├── grid-size-8/
│   └── tabular_bfs_20250701T090101.csv
├── grid-size-10/
│   ├── tabular_mixed_20250701T090305.csv
│   └── language_astar_20250701T090522.jsonl
└── grid-size-16/
    └── sequential_rl_base_20250701T091000.npz
```

### Why is this mandatory?

| Benefit                           | Explanation                                                    |
| -------------------------------- | -------------------------------------------------------------- |
| **Experiment hygiene**           | Prevents accidental mixing of different spatial complexities. |
| **Discoverability**              | You can instantly locate all 12×12 datasets, for example.     |
| **Scalability**                  | New grid sizes plug in without touching existing code.        |
| **Reproducibility**              | Training scripts document grid size via their path alone.     |

---

## Helper API (Python)
All extensions SHOULD rely on the helpers in
`extensions.common.dataset_directory_manager` **instead** of hard-coding
paths.

```python
from extensions.common.dataset_directory_manager import (
    DatasetDirectoryManager as DDM,
)

# Create/ensure directory
path = DDM.grid_size_dir(grid_size=12)

# Build a standardised filename
csv_file = DDM.make_filename(
    algorithm="BFS",
    grid_size=12,
    data_structure="tabular",
    data_format="csv",
)

# Validate an arbitrary path (raises DatasetPathError on violation)
DDM.validate_dataset_path(csv_file)
```

These helpers internally delegate to `extensions.common.config`, keeping the
rule completely **centralised** (single-source-of-truth).

---

## Integration Checklist for Extension Authors

1. **Import the helper** – never construct `Path("logs/…")` strings by hand.
2. **Detect grid size** from logs or CLI args rather than assuming *10*.
3. **Call** `DatasetDirectoryManager.grid_size_dir()` **before writing files**.
4. **Include grid size** in every CLI that generates datasets
   (`--grid-size` flag or automatic detection).
5. **Add unit tests** that fail if a file is created outside the rule.

Failing to follow the rule should raise an exception early – *silence is a bug*.

---

_Last updated: 2025-06-23_ 