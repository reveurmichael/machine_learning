# Versioned Directory Structure - Migration Examples

This document provides practical examples of how to update existing extensions to use the new versioned directory structure.

## Quick Reference

### Before and After Directory Structure

**Before (Legacy, and should be removed, should be totally abandoned, should not be having legacy code or backward compatibility)**:
```
logs/extensions/models/grid-size-10/pytorch/mlp_model.pth
logs/extensions/models/grid-size-10/xgboost/xgb_model.json
logs/extensions/datasets/grid-size-10/tabular_data.csv
```

**After (Versioned, code should be only for this version, and should not be having legacy code or backward compatibility, per the code, it should look so fresh, so new, so future-oriented. Per the code, it has no past, no legay, nothing to look back to)**:
```
logs/extensions/models/grid-size-10/supervised_v0.02_20250625_143022/pytorch/mlp_model.pth (TODO: along with the mlp_model.onnx file? npz file? parquet file? or other file?)
logs/extensions/models/grid-size-10/supervised_v0.02_20250625_143022/xgboost/xgb_model.json (TODO: along with the xgb_model.onnx file? npz file? parquet file? or other file?)
logs/extensions/datasets/grid-size-10/heuristics_v0.03_20250625_142015/bfs/tabular_data.csv (TODO: along with the game_N.json and summary.json files)
logs/extensions/datasets/grid-size-10/heuristics_v0.03_20250625_142015/bfs/tabular_data.csv (TODO: along with the game_N.json and summary.json files. Importantly, along with the JSONL files. VITAL !)
```

## 1. Updating Model Saving in Training Scripts

### Example 1: Supervised Learning Neural Networks

**File**: `extensions/supervised-v0.02/training/train_neural.py`

**Before**:
```python
# Old way - uses legacy directory structure
saved_path = save_model_standardized(
    model=agent.model,
    framework='PyTorch',
    grid_size=grid_size,
    model_name=model_filename,
    model_class=agent.__class__.__name__,
    input_size=agent.input_size,
    output_size=4,
    training_params=training_params,
    export_onnx=True
)
```

**After**:
```python
# New way - uses versioned directory structure
saved_path = save_model_standardized(
    model=agent.model,
    framework='PyTorch',
    grid_size=grid_size,
    model_name=model_filename,
    model_class=agent.__class__.__name__,
    input_size=agent.input_size,
    output_size=4,
    training_params=training_params,
    export_onnx=True,
    extension_type="supervised",    # NEW: Extension type
    version="v0.02"                # NEW: Extension version
)
# Creates: logs/extensions/models/grid-size-{grid_size}/supervised_v0.02_{timestamp}/pytorch/
```

### Example 2: Reinforcement Learning Agents

**File**: `extensions/reinforcement-v0.02/scripts/train.py`

**Before**:
```python
# Manual model saving - not using standardized utility
model_dir = Path("logs/extensions/models") / f"grid-size-{grid_size}" / "pytorch"
model_dir.mkdir(parents=True, exist_ok=True)
torch.save(agent.state_dict(), model_dir / "dqn_agent.pth")
```

**After**:
```python
# Use versioned directory structure
from extensions.common.model_utils import save_model_standardized

saved_path = save_model_standardized(
    model=agent,
    framework="PyTorch",
    grid_size=grid_size,
    model_name="dqn_agent",
    model_class="DQNAgent",
    input_size=agent.input_size,
    output_size=agent.action_size,
    training_params={
        "learning_rate": agent.learning_rate,
        "gamma": agent.gamma,
        "epsilon": agent.epsilon,
        "episodes": episodes
    },
    extension_type="reinforcement",  # NEW: Extension type
    version="v0.02"                  # NEW: Extension version
)
# Creates: logs/extensions/models/grid-size-{grid_size}/reinforcement_v0.02_{timestamp}/pytorch/
```

## 2. Updating Dataset Generation

### Example 3: Heuristics Dataset Generation

**File**: `extensions/heuristics-v0.03/scripts/generate_dataset.py`

**Before**:
```python
# Manual directory creation
output_dir = Path("logs/extensions/datasets") / f"grid-size-{grid_size}"
output_dir.mkdir(parents=True, exist_ok=True)
csv_file = output_dir / f"tabular_{algorithm_name}_data.csv"
```

**After**:
```python
# Use versioned directory structure
from extensions.common.versioned_directory_manager import create_dataset_directory

output_dir = create_dataset_directory(
    extension_type="heuristics",
    version="v0.03",
    grid_size=grid_size,
    algorithm=algorithm_name
)
csv_file = output_dir / "tabular_data.csv"
json_file = output_dir / "game_logs.json"
# Creates: logs/extensions/datasets/grid-size-{grid_size}/heuristics_v0.03_{timestamp}/bfs/
```

## 3. Extension-Specific Patterns

### Pattern A: Get Extension Version Automatically

TODO: this is what we want to have, and this is so COOL ! Make sure in the code base of extensions we use it everywhere.

Create a utility function to extract version from extension directory:

```python
# Add to extension's utils or __init__.py
def get_extension_version() -> str:
    """Extract version from extension directory name."""
    current_file = Path(__file__).resolve()
    extension_dir = current_file.parent
    
    # Extract version from directory name (e.g., "supervised-v0.02")
    if "-v" in extension_dir.name:
        return extension_dir.name.split("-v")[-1]
    return "v0.01"  # Default fallback

# Usage in training scripts
extension_version = get_extension_version()
```

### Pattern B: Configuration-Based Extension Info

TODO: this can be good as well, though I prefer Pattern A.

**File**: `extensions/supervised-v0.03/config.py`

```python
# Extension configuration
EXTENSION_CONFIG = {
    "extension_type": "supervised",
    "version": get_extension_version(),
    "supported_models": ["MLP", "CNN", "LSTM", "XGBoost", "LightGBM"],
    "default_grid_size": 10
}

def get_extension_info():
    """Get extension type and version from config."""
    return EXTENSION_CONFIG["extension_type"], EXTENSION_CONFIG["version"]
```

**Usage in training scripts**:
```python
from .config import get_extension_info

extension_type, version = get_extension_info()

saved_path = save_model_standardized(
    # ... other parameters ...
    extension_type=extension_type,
    version=version
)
```

## 4. Complete Migration Example

### Before: `extensions/supervised-v0.02/training/train_neural.py`

```python
def train_model(model_type: str, dataset_paths: List[str], output_dir: str,
               grid_size: int = None, epochs: int = 100, **kwargs):
    
    # Load data
    X_train, y_train = load_training_data(dataset_paths)
    
    # Create and train model
    agent = create_agent(model_type, grid_size)
    training_results = agent.train(X_train, y_train, epochs=epochs)
    
    # Save model - OLD WAY
    output_path = Path(output_dir) / f"grid-size-{grid_size}" / "pytorch"
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_path = save_model_standardized(
        model=agent.model,
        framework='PyTorch',
        grid_size=grid_size,
        model_name=f"{model_type.lower()}_model",
        model_class=agent.__class__.__name__,
        input_size=agent.input_size,
        output_size=4,
        training_params={'epochs': epochs, **kwargs}
    )
    
    return {"model_path": saved_path, **training_results}
```

### After: Using Versioned Structure

```python
def train_model(model_type: str, dataset_paths: List[str], output_dir: str,
               grid_size: int = None, epochs: int = 100, **kwargs):
    
    # Load data
    X_train, y_train = load_training_data(dataset_paths)
    
    # Create and train model
    agent = create_agent(model_type, grid_size)
    training_results = agent.train(X_train, y_train, epochs=epochs)
    
    # Get extension information
    extension_type = "supervised"
    version = "v0.02"
    
    # Save model - NEW WAY with versioned structure
    saved_path = save_model_standardized(
        model=agent.model,
        framework='PyTorch',
        grid_size=grid_size,
        model_name=f"{model_type.lower()}_model",
        model_class=agent.__class__.__name__,
        input_size=agent.input_size,
        output_size=4,
        training_params={'epochs': epochs, **kwargs},
        extension_type=extension_type,  # NEW: Enables versioned directories
        version=version                 # NEW: Enables versioned directories
    )
    
    return {"model_path": saved_path, **training_results}
```

## 5. Agent Class Integration

### Example: Update Agent's `save_model` Method

**Before**:
```python
class MLPAgent(SnakeAgent):
    def save_model(self, model_name: str, export_onnx: bool = True) -> str:
        metadata = {...}
        
        model_path = save_model_standardized(
            model_name=model_name,
            model=self.model,
            metadata=metadata,
            framework="pytorch",
            grid_size=self.grid_size,
            export_onnx=export_onnx
        )
        return model_path
```

**After**:
```python
class MLPAgent(SnakeAgent):
    def save_model(self, model_name: str, export_onnx: bool = True, 
                   extension_type: str = "supervised", version: str = "v0.02") -> str:
        metadata = {...}
        
        model_path = save_model_standardized(
            model_name=model_name,
            model=self.model,
            metadata=metadata,
            framework="pytorch",
            grid_size=self.grid_size,
            export_onnx=export_onnx,
            extension_type=extension_type,  # NEW: Pass through to versioned structure
            version=version                 # NEW: Pass through to versioned structure
        )
        return model_path
```

## 6. Migration Checklist

### For Each Extension Directory:

- [ ] **Identify Extension Type**: Determine the extension type (`heuristics`, `supervised`, `reinforcement`, etc.)
- [ ] **Identify Version**: Extract version from directory name (`v0.01`, `v0.02`, etc.)
- [ ] **Update Training Scripts**: Add `extension_type` and `version` to `save_model_standardized` calls
- [ ] **Update Dataset Generation**: Use `create_dataset_directory` for dataset outputs
- [ ] **Update Agent Classes**: Modify `save_model` methods to accept and pass through version info
- [ ] **Add Configuration**: Create extension config with type and version info
- [ ] **Test Migration**: Verify versioned directories are created correctly

### For Common Utilities:

- [x] **Enhanced model_utils.py**: Added support for versioned directories
- [x] **Created versioned_directory_manager.py**: Central management for versioned structure
- [x] **Updated config.py**: Added MODELS_ROOT constant
- [x] **Updated __init__.py**: Exposed versioned directory utilities

## 7. Validation

After migration, use the compliance validator:

```bash
# Check that all extensions follow grid-size structure
python scripts/validate_grid_size_compliance.py

# Verify versioned directories are created
ls -la logs/extensions/models/grid-size-10/
ls -la logs/extensions/datasets/grid-size-10/
```

Expected output:
```
logs/extensions/models/grid-size-10/
├── supervised_v0.02_20250625_143022/
│   ├── pytorch/
│   └── .directory_metadata.json
└── reinforcement_v0.01_20250625_144500/
    ├── pytorch/
    └── .directory_metadata.json

logs/extensions/datasets/grid-size-10/
├── heuristics_v0.03_20250625_142015/
│   ├── bfs/
│   ├── astar/
│   └── .directory_metadata.json
└── heuristics_v0.04_20250625_145000/
    ├── bfs/
    ├── astar/
    └── .directory_metadata.json
```

## 8. Benefits Achieved

After migration, you will have:

1. **Temporal Separation**: Each training run gets unique timestamped directory
2. **Version Tracking**: Clear separation between v0.01, v0.02, v0.03 outputs
3. **Grid-Size Compliance**: Automatic enforcement of spatial complexity separation
4. **Automatic Metadata**: Each directory includes creation metadata
5. **Easy Discovery**: Built-in utilities to find latest models/datasets
6. **Scientific Rigor**: Prevents accidental contamination between experiments

The versioned directory structure transforms chaotic file organization into a systematic, scientific, and maintainable experimental framework. 