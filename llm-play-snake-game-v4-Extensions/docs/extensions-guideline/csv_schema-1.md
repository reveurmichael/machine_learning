# CSV Schema for Snake Game Extensions

> **Important — Authoritative Reference:** This document supplements Final Decision 2 (Configuration Standards). **For conflicts, Final Decision 2 prevails.**

## 🎯 **Core Philosophy: Grid-Size Agnostic Design**

The CSV schema uses a **fixed set of 16 engineered features** that work for any grid size (8x8, 10x10, 12x12, 16x16, 20x20, etc.), ensuring consistency across all extensions and enabling cross-grid-size comparisons.

## 📊 **Standardized Schema Structure**

### **Fixed Feature Set (16 features)**

| Feature Category | Features | Description |
|------------------|----------|-------------|
| **Position** | `head_x`, `head_y`, `apple_x`, `apple_y` | Absolute coordinates |
| **Game State** | `snake_length` | Current snake length |
| **Apple Direction** | `apple_dir_up`, `apple_dir_down`, `apple_dir_left`, `apple_dir_right` | Binary directional flags |
| **Danger Detection** | `danger_straight`, `danger_left`, `danger_right` | Binary collision risk flags |
| **Free Space** | `free_space_up`, `free_space_down`, `free_space_left`, `free_space_right` | Free cell counts per direction |

### **Metadata & Target (3 columns)**
- `game_id`: Unique game session identifier
- `step_in_game`: Step number within the game
- `target_move`: The move taken (UP, DOWN, LEFT, RIGHT)

**Total: 19 columns** (2 metadata + 16 features + 1 target)

## 🧠 **Design Benefits**

### **Grid-Size Independence**
- **Universal Features**: Same 16 features work for any board size
- **Scalable Performance**: Efficient feature extraction across all grid configurations
- **Cross-Grid Compatibility**: Enables training on one size, testing on another

### **ML Compatibility**
- **Tabular Models**: Optimal for XGBoost, LightGBM, Random Forest
- **Neural Networks**: Appropriate input dimensions for MLPs and deep networks
- **Feature Engineering**: Rich enough feature set for effective learning

### **Cross-Extension Integration**
- **Heuristics v0.03**: Generates standardized CSV datasets for supervised learning
- **Supervised v0.02+**: Consumes CSV datasets for training all model types
- **Evaluation**: Consistent comparison framework across all algorithm types

## 📁 **Path Integration**

Uses standardized path management from Final Decision 6:

```python
from extensions.common.path_utils import get_dataset_path

# Grid-size agnostic path generation
dataset_path = get_dataset_path(
    extension_type="heuristics", 
    version="0.03",
    grid_size=grid_size,  # Any supported size
    algorithm="bfs",
    timestamp=timestamp
)
```

## 🔧 **Usage Examples**

### **Dataset Generation**
```python
from extensions.common.csv_schema import create_csv_row

csv_row = create_csv_row(
    game_state=current_state,
    target_move="RIGHT", 
    game_id=1,
    step_in_game=5,
    grid_size=grid_size  # Works with any grid size
)
```

### **Dataset Loading**
```python
from extensions.common.dataset_loader import load_dataset_for_training

X_train, X_val, X_test, y_train, y_val, y_test, info = load_dataset_for_training(
    dataset_paths=["path/to/dataset.csv"],
    grid_size=grid_size  # Validates compatibility
)
```

## 🎯 **Extension Benefits**

### **Heuristics Extensions**
- Consistent dataset generation across all algorithms (BFS, A*, Hamiltonian)
- Grid-size independence enables flexible experimentation
- Rich feature capture of algorithmic decision patterns

### **Supervised Learning Extensions**
- Standardized input format across all model types (MLP, CNN, XGBoost, etc.)
- Efficient training with appropriately sized feature vectors
- Cross-algorithm comparison using identical feature space

### **Research Applications**
- Reproducible experiments across different grid sizes
- Consistent evaluation metrics and feature interpretability
- Transfer learning possibilities between different configurations

---

**This grid-size agnostic CSV schema ensures consistent, scalable datasets for supervised learning across all Snake Game AI extensions while maintaining cross-extension compatibility.**
