> **Important — Authoritative Reference:** This CSV schema guide supplements the Final Decision Series. For conflicts, defer to Final Decision 2.

# CSV Schema for Snake Game Extensions

This document defines the **grid-size agnostic CSV schema** for supervised learning across all Snake game configurations, following the standards established in Final Decision 2.

## 🎯 **Core Philosophy: Grid-Size Independence**

The CSV schema uses a **fixed set of 16 engineered features** that work for any grid size (8x8, 10x10, 12x12, 16x16, 20x20, etc.), ensuring consistency across all extensions.

## 📊 **Standardized Schema Structure**

### **Fixed Feature Set (16 features)**

| Feature Category | Features | Description |
|------------------|----------|-------------|
| **Position** | `head_x`, `head_y`, `apple_x`, `apple_y` | Absolute coordinates |
| **Game State** | `snake_length` | Current snake length |
| **Apple Direction** | `apple_dir_up`, `apple_dir_down`, `apple_dir_left`, `apple_dir_right` | Binary flags for apple position relative to head |
| **Danger Detection** | `danger_straight`, `danger_left`, `danger_right` | Binary flags for immediate collision risk |
| **Free Space** | `free_space_up`, `free_space_down`, `free_space_left`, `free_space_right` | Count of free cells in each direction |

### **Metadata Columns (2 columns)**
- `game_id`: Unique game session identifier
- `step_in_game`: Step number within the game

### **Target Column (1 column)**
- `target_move`: The move taken (UP, DOWN, LEFT, RIGHT)

**Total: 19 columns** (2 metadata + 16 features + 1 target)

## 🧠 **Design Benefits**

### **Scalability**
- **Grid-Size Agnostic**: Same 16 features regardless of board size
- **Performance**: Efficient feature extraction across all grid sizes
- **Consistency**: Uniform training data across different configurations

### **ML Compatibility**
- **Tabular Models**: Ready for XGBoost, LightGBM, Random Forest
- **Neural Networks**: Appropriate input size for MLPs
- **Feature Engineering**: Rich enough for effective learning

### **Cross-Extension Use**
- **Heuristics v0.03**: Generates CSV datasets for supervised learning
- **Supervised v0.02+**: Consumes CSV datasets for training
- **Comparison**: Consistent evaluation across all algorithm types

## 📁 **Implementation Standards**

### **Feature Engineering**
```python
# Grid-size agnostic feature extraction
features = {
    'head_x': head_position[0],
    'head_y': head_position[1], 
    'apple_x': apple_position[0],
    'apple_y': apple_position[1],
    'snake_length': len(snake_positions),
    'apple_dir_up': 1 if apple_position[1] > head_position[1] else 0,
    'danger_straight': 1 if next_cell_blocked else 0,
    'free_space_up': count_free_cells_in_direction('UP'),
    # ... remaining features
}
```

### **Path Integration**
Uses standardized paths from Final Decision 6:
```python
from extensions.common.path_utils import get_dataset_path

dataset_path = get_dataset_path(
    extension_type="heuristics", 
    version="0.03",
    grid_size=grid_size,  # Any size supported
    algorithm="bfs",
    timestamp=timestamp
)
```

## 🔧 **Usage Examples**

### **Dataset Generation (Heuristics v0.03)**
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

### **Dataset Loading (Supervised v0.02+)**
```python
from extensions.common.dataset_loader import load_dataset_for_training

X_train, X_val, X_test, y_train, y_val, y_test, info = load_dataset_for_training(
    dataset_paths=["path/to/dataset.csv"],
    grid_size=grid_size  # Validates compatibility
)
```

## 🎓 **Benefits for Extensions**

### **Heuristics Extensions**
- Consistent dataset generation across all algorithms
- Grid-size independence enables flexible experimentation
- Rich feature set captures algorithmic decision patterns

### **Supervised Learning Extensions**
- Standardized input format across all model types
- Efficient training with appropriately sized feature vectors
- Cross-algorithm comparison using same feature space

### **Research Applications**
- Reproducible experiments across different grid sizes
- Consistent evaluation metrics and feature interpretability
- Transfer learning possibilities between different configurations

---

**This CSV schema ensures consistent, scalable, and grid-size agnostic datasets for supervised learning across all Snake Game AI extensions.**
