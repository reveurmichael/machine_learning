# CSV Schema for Snake Game Extensions

# CSV Schema for Snake Game Extensions

> **Important**: For complete data format decisions, see `data-format-decision-guide.md` - the authoritative reference for all format choices.

## 🎯 **Core Philosophy: Grid-Size Agnostic Design**

The CSV schema uses a **fixed set of 16 engineered features** that work for any grid size (8x8, 10x10, 12x12, 16x16, 20x20, etc.), ensuring consistency across all extensions and enabling cross-grid-size comparisons.

**When to Use CSV**: This format is optimal for tree-based models (XGBoost, LightGBM) and simple MLPs. For other model types, see the data format decision guide.

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

### **State Representation Decision Matrix**
The 16-feature tabular schema is **specifically designed** for certain algorithm types:

| Representation Type | Best For | Data Format | Use Cases |
|-------------------|----------|-------------|-----------|
| **16-Feature Tabular** | XGBoost, LightGBM, Random Forest, Simple MLP | CSV | Tree-based models, traditional ML |
| **Sequential (NPZ)** | LSTM, GRU, Temporal Models | NPZ arrays | Time-series analysis, RNN architectures |
| **Spatial (2D Arrays)** | CNN, Computer Vision Models | 2D numpy arrays | Image-like processing, spatial patterns |
| **Graph Structures** | GNN, Relationship-based Models | Graph formats | Complex relationships, network analysis |
| **Raw Board State** | Evolutionary Algorithms, GP | Full grid matrices | Population-based optimization |

### **When to Use 16-Feature Schema**
- ✅ **Tree-based models**: XGBoost, LightGBM, Random Forest
- ✅ **Simple MLPs**: Fully connected networks with tabular input
- ✅ **Traditional ML**: SVM, Logistic Regression, Naive Bayes
- ✅ **Fast inference**: Minimal preprocessing required
- ❌ **CNNs**: Use spatial 2D arrays instead
- ❌ **RNNs**: Use sequential NPZ format instead
- ❌ **GNNs**: Use graph representation instead

### **Cross-Extension Integration**
- **Heuristics v0.03**: Generates standardized CSV datasets for supervised learning
- **Supervised v0.02+**: Consumes CSV datasets for training all model types
- **Evaluation**: Consistent comparison framework across all algorithm types

## 📁 **Path Integration**

Uses standardized path management from `unified-path-management-guide.md`:

```python
from extensions.common.path_utils import get_dataset_path

# Standardized path generation with enforced format
dataset_path = get_dataset_path(
    extension_type="heuristics", 
    version="0.03",
    grid_size=grid_size,  # Any supported size
    algorithm="bfs",
    timestamp=timestamp  # Format: YYYYMMDD_HHMMSS
)
# Result: logs/extensions/datasets/grid-size-{grid_size}/heuristics_v0.03_{timestamp}/
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

## 🔗 **See Also**

- **`csv-schema-2.md`**: Detailed utilities and implementation guide
- **`evolutionary.md`**: Alternative state representations for evolutionary algorithms
- **`datasets_folder.md`**: Directory structure and organization standards

---

**This grid-size agnostic CSV schema ensures consistent, scalable datasets for supervised learning across all Snake Game AI extensions while maintaining cross-extension compatibility.**
