# Versioned Directory Structure Philosophy and Implementation

## Overview

This document outlines the **critical organizational principle** that governs how datasets and models are stored across all extensions in this machine learning project. The system enforces a **mandatory directory structure** that prevents contamination between different spatial complexities and algorithmic evolutions.

## The Fundamental Rule

### Directory Structure Mandate

**ALL** datasets and models must follow this exact structure:

```
logs/extensions/
├── datasets/
│   └── grid-size-N/
│       └── extension_v0.0M_YYYYMMDD_HHMMSS/
│           ├── algorithm1/
│           ├── algorithm2/
│           └── ...
└── models/
    └── grid-size-N/
        └── extension_v0.0M_YYYYMMDD_HHMMSS/
            ├── pytorch/
            ├── xgboost/
            ├── lightgbm/
            └── ...
```

Where:
- **N** = Grid size (8, 10, 12, 15, 20, etc.)
- **extension** = Extension name (heuristics, supervised, reinforcement, etc.)
- **M** = Extension version (01, 02, 03, 04, etc.)
- **YYYYMMDD_HHMMSS** = Timestamp of creation

### Real Examples

```
logs/extensions/datasets/grid-size-10/heuristics_v0.03_20250625_143022/
├── bfs/
│   ├── game_logs.json
│   └── tabular_data.csv
├── astar/
│   ├── game_logs.json
│   └── tabular_data.csv
└── .directory_metadata.json

logs/extensions/models/grid-size-10/supervised_v0.02_20250625_150830/
├── pytorch/
│   ├── mlp_model.pth
│   ├── cnn_model.pth
│   └── lstm_model.pth
├── xgboost/
│   └── bfs_classifier.json
└── .directory_metadata.json
```

## Philosophy and Design Principles

### 1. **Spatial Complexity Separation**

**Problem**: Different grid sizes represent fundamentally different problem complexities.

- **8×8 grid**: 64 positions, simpler spatial relationships
- **20×20 grid**: 400 positions, complex spatial dynamics

**Solution**: Strict separation prevents models trained on simple grids from contaminating experiments on complex grids.

**Implication**: A model trained on 8×8 data cannot accidentally be evaluated on 20×20 data, ensuring scientific integrity.

### 2. **Algorithmic Evolution Tracking**

**Problem**: Extensions evolve through versions (v0.01 → v0.02 → v0.03) with different algorithms and capabilities.

**Solution**: Version-aware directory structure preserves the evolutionary lineage.

**Benefits**:
- **Reproducibility**: Can always recreate results from v0.02 even after v0.03 is developed
- **Comparison**: Direct performance comparison between algorithmic generations
- **Educational**: Students can see how algorithms evolved over time

### 3. **Temporal Organization**

**Problem**: Multiple training runs need organization and identification.

**Solution**: Timestamp-based directories create temporal separation.

**Benefits**:
- **Experiment Tracking**: Each training run has unique identifier
- **Chronological Analysis**: Can analyze how performance changed over time
- **Debugging**: Easy to identify which run produced specific results

### 4. **Single Source of Truth**

**Problem**: Inconsistent directory creation across extensions leads to chaos.

**Solution**: Centralized `VersionedDirectoryManager` in `extensions/common/`.

**Benefits**:
- **Consistency**: All extensions use identical structure
- **Maintainability**: Changes propagate across all extensions
- **Validation**: Automatic compliance checking

## Implementation Architecture

### Core Component: VersionedDirectoryManager

```python
from extensions.common.versioned_directory_manager import (
    create_dataset_directory,
    create_model_directory,
    VersionedDirectoryManager
)

# Create dataset directory
dataset_dir = create_dataset_directory(
    extension_type="heuristics",
    version="v0.03",
    grid_size=10,
    algorithm="bfs"
)
# Returns: logs/extensions/datasets/grid-size-10/heuristics_v0.03_20250625_143022/bfs/

# Create model directory
model_dir = create_model_directory(
    extension_type="supervised",
    version="v0.02", 
    grid_size=15,
    framework="pytorch",
    model_name="mlp"
)
# Returns: logs/extensions/models/grid-size-15/supervised_v0.02_20250625_143530/pytorch/mlp/
```

### Integration with Model Utilities

Enhanced `model_utils.py` automatically uses versioned structure:

```python
from extensions.common.model_utils import save_model_standardized

# Automatically uses versioned directory if extension info provided
saved_path = save_model_standardized(
    model=trained_model,
    framework="PyTorch",
    grid_size=10,
    model_name="bfs_classifier",
    model_class="MLPAgent",
    input_size=104,
    output_size=4,
    training_params=config,
    extension_type="supervised",  # NEW: Enables versioned structure
    version="v0.02"               # NEW: Enables versioned structure
)
# Automatically creates: logs/extensions/models/grid-size-10/supervised_v0.02_TIMESTAMP/pytorch/
```

### Automatic Metadata Generation

Each directory gets metadata file for tracking:

```json
{
  "extension_type": "heuristics",
  "version": "v0.03",
  "grid_size": 10,
  "content_type": "dataset",
  "created_at": "2025-06-25T14:30:22.123456",
  "created_by": "VersionedDirectoryManager",
  "extra_info": "bfs"
}
```

## Enforcement and Compliance

### 1. Centralized Creation

**Rule**: All extensions MUST use `VersionedDirectoryManager` instead of manual directory creation.

```python
# ❌ WRONG - Manual directory creation
os.makedirs("logs/extensions/datasets/my_data/")

# ✅ CORRECT - Centralized creation
dataset_dir = create_dataset_directory("heuristics", "v0.03", 10, "bfs")
```

### 2. Validation and Parsing

Built-in validation ensures compliance:

```python
from extensions.common.versioned_directory_manager import parse_versioned_path

# Parse existing directory
path = Path("logs/extensions/datasets/grid-size-10/heuristics_v0.03_20250625_143022/")
extension, version, grid_size, timestamp = parse_versioned_path(path)
# Returns: ("heuristics", "v0.03", 10, "20250625_143022")
```

### 3. Discovery and Listing

Easy discovery of datasets and models:

```python
# Find all heuristics datasets for grid size 10
versions = VersionedDirectoryManager.list_dataset_versions("heuristics", 10)
# Returns: [("v0.03", "20250625_143022", Path(...)), ("v0.02", "20250624_120000", Path(...))]

# Get latest model
latest_model = VersionedDirectoryManager.get_latest_model("supervised", 10, "pytorch")
# Returns: Path to most recent supervised model directory
```

## Extension-Specific Integration

### Heuristics Extensions

```python
# In heuristics-v0.03 dataset generation
from extensions.common.versioned_directory_manager import create_dataset_directory

# Create versioned output directory
output_dir = create_dataset_directory(
    extension_type="heuristics",
    version="v0.03",
    grid_size=config.grid_size,
    algorithm=algorithm_name
)

# Save datasets to versioned directory
save_csv_data(output_dir / "tabular_data.csv")
save_json_logs(output_dir / "game_logs.json")
```

### Supervised Learning Extensions

```python
# In supervised-v0.02 training
from extensions.common.model_utils import save_model_standardized

# Model automatically saved to versioned directory
saved_path = save_model_standardized(
    model=trained_model,
    framework="PyTorch",
    grid_size=args.grid_size,
    model_name=f"{algorithm}_{model_type}",
    model_class="MLPAgent",
    input_size=input_features,
    output_size=4,
    training_params=training_config,
    extension_type="supervised",
    version="v0.02"
)
```

### Reinforcement Learning Extensions

```python
# In reinforcement-v0.01 training
model_dir = create_model_directory(
    extension_type="reinforcement",
    version="v0.01", 
    grid_size=args.grid_size,
    framework="pytorch",
    model_name=f"{args.agent_type.lower()}_agent"
)

# Save RL agent to versioned directory
torch.save(agent.state_dict(), model_dir / "agent_weights.pth")
torch.save(optimizer.state_dict(), model_dir / "optimizer_state.pth")
```

## Benefits and Implications

### 1. **Scientific Rigor**

- **No Contamination**: Impossible to accidentally mix different grid sizes
- **Reproducibility**: Every experiment has unique, traceable identifier
- **Comparison**: Fair comparison between algorithm versions

### 2. **Educational Value**

- **Evolution Tracking**: Students see how algorithms evolve over time
- **Historical Preservation**: All versions remain accessible
- **Learning Progression**: Clear progression from simple to complex

### 3. **Engineering Excellence**

- **Maintainability**: Centralized structure logic
- **Scalability**: New grid sizes and versions integrate seamlessly
- **Debuggability**: Easy to locate specific experiment artifacts

### 4. **Research Efficiency**

- **Discovery**: Quick finding of relevant datasets and models
- **Collaboration**: Shared understanding of organization
- **Documentation**: Self-documenting through structure

## Migration Strategy

### Phase 1: Adopt in New Code

All new extensions must use versioned structure:

```python
# Required imports in all new training scripts
from extensions.common.versioned_directory_manager import create_model_directory
from extensions.common.model_utils import save_model_standardized

# Required parameters in all training functions
def train_model(extension_type: str, version: str, ...):
    # Use versioned structure
    pass
```

### Phase 2: Enhance Common Utilities

Update existing utilities to support versioned structure:

- ✅ `model_utils.py` - Enhanced with version support
- ✅ `dataset_directory_manager.py` - Integration with versioned manager
- ✅ `config.py` - Added MODELS_ROOT constant

### Phase 3: Update Extensions

Gradually update existing extensions to use versioned structure while maintaining backward compatibility.

### Phase 4: Validation and Cleanup

Use compliance validator to ensure all extensions follow the rule:

```bash
python scripts/validate_grid_size_compliance.py
python scripts/validate_versioned_structure.py  # Future enhancement
```

## Common Patterns and Best Practices

### 1. **Extension Development Pattern**

Every extension should follow this pattern:

```python
def main():
    # Parse arguments including extension_type and version
    args = parse_args()
    
    # Create versioned directories
    if args.mode == "generate_data":
        output_dir = create_dataset_directory(
            extension_type=args.extension_type,
            version=args.version,
            grid_size=args.grid_size,
            algorithm=args.algorithm
        )
        generate_data(output_dir)
    
    elif args.mode == "train_model":
        model_dir = create_model_directory(
            extension_type=args.extension_type,
            version=args.version,
            grid_size=args.grid_size,
            framework=args.framework
        )
        train_and_save_model(model_dir)
```

### 2. **Version Management Pattern**

```python
# Get version from extension directory name
def get_extension_version() -> str:
    current_file = Path(__file__).resolve()
    extension_dir = current_file.parent
    
    # Extract version from directory name (e.g., "supervised-v0.02")
    if "-v" in extension_dir.name:
        return extension_dir.name.split("-v")[-1]
    return "v0.01"  # Default
```

### 3. **Discovery Pattern**

```python
def load_latest_model(extension_type: str, grid_size: int, framework: str):
    """Load the most recent model for given parameters."""
    latest_dir = VersionedDirectoryManager.get_latest_model(
        extension_type, grid_size, framework
    )
    
    if latest_dir:
        model_file = latest_dir / "model.pth"
        if model_file.exists():
            return torch.load(model_file)
    
    raise FileNotFoundError(f"No model found for {extension_type} grid-{grid_size}")
```

## Future Enhancements

### 1. **Automated Cleanup**

```python
# Clean old versions keeping only latest 3
VersionedDirectoryManager.clean_old_versions(
    extension_type="heuristics",
    grid_size=10,
    keep_latest=3,
    dry_run=False
)
```

### 2. **Cross-Extension Dependencies**

```python
# Find compatible datasets for training
compatible_datasets = VersionedDirectoryManager.find_compatible_datasets(
    grid_size=10,
    min_samples=1000,
    algorithms=["bfs", "astar"]
)
```

### 3. **Performance Analytics**

```python
# Track performance across versions
performance_trend = VersionedDirectoryManager.analyze_performance_trend(
    extension_type="supervised",
    grid_size=10,
    metric="test_accuracy"
)
```

## Conclusion

The versioned directory structure is not just an organizational tool—it's a **fundamental design principle** that ensures:

- **Scientific integrity** through spatial complexity separation
- **Evolutionary tracking** of algorithmic development
- **Reproducible research** through temporal organization
- **Engineering excellence** through centralized management

By enforcing this structure across all extensions, we create a **maintainable**, **scalable**, and **educationally valuable** machine learning framework that serves both current research needs and future educational requirements.

The system transforms what could be chaotic file organization into a **self-documenting**, **discoverable**, and **validated** structure that grows intelligently with the project's evolution.

**Remember**: This is not optional. Every extension MUST follow this structure to maintain the project's integrity and educational value. 