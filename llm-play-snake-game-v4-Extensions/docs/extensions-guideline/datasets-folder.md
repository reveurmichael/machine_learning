# Datasets Folder Standards for Snake Game AI

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for datasets folder standards and supplements the _Final Decision Series_ (`` → `final-decision.md`).

> **See also:** `data-format-decision-guide.md`, `final-decision.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Grid-Size Agnostic Organization**

The datasets folder uses a **grid-size agnostic organization** that ensures consistent dataset storage across all extensions and grid sizes. This system provides predictable dataset locations and enables cross-grid-size comparisons, strictly following SUPREME_RULES from `final-decision.md`.

### **Educational Value**
- **Dataset Organization**: Understanding consistent dataset storage
- **Grid-Size Independence**: Learning grid-size agnostic design
- **Version Management**: Clear versioning and timestamping
- **Cross-Extension Compatibility**: Enabling dataset sharing between extensions

## 🏗️ **Standardized Directory Structure**

### **Root Datasets Directory**
```
logs/extensions/datasets/
├── grid-size-8/
│   ├── heuristics_v0.03_20240101_120000/
│   ├── heuristics_v0.04_20240101_120000/
│   ├── supervised_v0.03_20240101_120000/
│   └── reinforcement_v0.02_20240101_120000/
├── grid-size-10/
│   ├── heuristics_v0.03_20240101_120000/
│   ├── heuristics_v0.04_20240101_120000/
│   ├── supervised_v0.03_20240101_120000/
│   └── reinforcement_v0.02_20240101_120000/
├── grid-size-12/
│   └── [extension]_v[version]_[timestamp]/
├── grid-size-16/
│   └── [extension]_v[version]_[timestamp]/
└── grid-size-20/
    └── [extension]_v[version]_[timestamp]/
```

### **Extension Dataset Directory Structure**
```
logs/extensions/datasets/grid-size-{N}/{extension}_v{version}_{timestamp}/
├── metadata.json                    # Dataset metadata and configuration
├── {algorithm}/                     # Algorithm-specific datasets (NEW STRUCTURE)
│   ├── game_1.json                 # Original game execution logs
│   ├── game_2.json                 # Original game execution logs  
│   ├── ...
│   ├── summary.json                # Session summary file
│   ├── {algorithm}_dataset.csv     # CSV format (ACTIVE, NOT legacy)
│   ├── {algorithm}_dataset.jsonl   # JSONL format (heuristics-v0.04 only)
│   └── prompts/                    # LLM prompts (Task-0 only)
│       ├── game_1_round_1_prompt.txt
│       ├── game_1_round_2_prompt.txt
│       └── ...
├── {algorithm2}/                   # Additional algorithm datasets
│   ├── game_1.json
│   ├── game_2.json
│   ├── ...
│   ├── summary.json
│   ├── {algorithm2}_dataset.csv
│   └── {algorithm2}_dataset.jsonl
└── evaluation/                     # Cross-algorithm evaluation results
    ├── performance_metrics.json
    ├── comparison_results.json
    └── visualization_data.json
```

## 📊 **Naming Convention Standards**

### **Grid Size Directory**
```
grid-size-{N}
```
- **N**: Grid size (8, 10, 12, 16, 20, etc.)
- **Format**: Always use `grid-size-` prefix
- **Examples**: `grid-size-8`, `grid-size-10`, `grid-size-16`

### **Extension Dataset Directory**
```
{extension}_v{version}_{timestamp}
```
- **extension**: Extension type (heuristics, supervised, reinforcement, evolutionary)
- **version**: Version number (0.01, 0.02, 0.03, 0.04)
- **timestamp**: Format YYYYMMDD_HHMMSS
- **Examples**: 
  - `heuristics_v0.03_20240101_120000`
  - `supervised_v0.02_20240101_120000`
  - `reinforcement_v0.01_20240101_120000`

### **Important Guidelines: Version Selection**
- **For supervised learning**: Use CSV from either heuristics-v0.03 or heuristics-v0.04 (both widely used)
- **For LLM fine-tuning**: Use JSONL from heuristics-v0.04 only
- **For research**: Use both formats from heuristics-v0.04
- **CSV is ACTIVE**: Not legacy - actively used for supervised learning
- **JSONL is ADDITIONAL**: New capability for LLM fine-tuning (heuristics-v0.04 only)

## 🔧 **Path Management Implementation**

### **Path Generation Utilities**
```python
from extensions.common.utils.path_utils import get_dataset_path, get_datasets_root

def create_dataset_directory(extension_type: str, version: str, grid_size: int, 
                           timestamp: str, algorithm: str) -> Path:
    """Create standardized dataset directory with algorithm subdirectory"""
    datasets_root = get_datasets_root()
    dataset_path = datasets_root / f"grid-size-{grid_size}" / f"{extension_type}_v{version}_{timestamp}"
    algorithm_path = dataset_path / algorithm.lower()
    
    # Create directory structure
    dataset_path.mkdir(parents=True, exist_ok=True)
    algorithm_path.mkdir(parents=True, exist_ok=True)
    (dataset_path / "evaluation").mkdir(exist_ok=True)
    
            print_info(f"[DatasetUtils] Created dataset directory: {algorithm_path}")  # SUPREME_RULES compliant logging
    return algorithm_path

def get_dataset_path(extension_type: str, version: str, grid_size: int, 
                    timestamp: str, algorithm: str = None) -> Path:
    """Get standardized dataset path with optional algorithm subdirectory"""
    datasets_root = get_datasets_root()
    base_path = datasets_root / f"grid-size-{grid_size}" / f"{extension_type}_v{version}_{timestamp}"
    
    if algorithm:
        return base_path / algorithm.lower()
    return base_path

def get_algorithm_dataset_path(extension_type: str, version: str, grid_size: int, 
                              timestamp: str, algorithm: str) -> Path:
    """Get algorithm-specific dataset path"""
    return get_dataset_path(extension_type, version, grid_size, timestamp, algorithm)
```

### **Metadata Management**
```python
def create_dataset_metadata(extension_type: str, version: str, grid_size: int, 
                          algorithm: str, num_games: int, **kwargs) -> dict:
    """Create standardized dataset metadata"""
    metadata = {
        "extension_type": extension_type,
        "version": version,
        "grid_size": grid_size,
        "algorithm": algorithm,
        "num_games": num_games,
        "created_at": datetime.now().isoformat(),
        "data_formats": get_supported_formats(extension_type, version),
        "configuration": kwargs
    }
    
    return metadata

def save_dataset_metadata(metadata: dict, dataset_path: Path):
    """Save metadata to dataset directory"""
    metadata_file = dataset_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
            print_info(f"[DatasetUtils] Saved metadata: {metadata_file}")  # SUPREME_RULES compliant logging
```

## 📋 **Dataset Format Standards**

### **CSV Format (heuristics-v0.03 and heuristics-v0.04)**
```python
def save_csv_dataset(data: pd.DataFrame, algorithm_path: Path, algorithm: str):
    """Save CSV dataset to algorithm-specific directory"""
    csv_file = algorithm_path / f"{algorithm.lower()}_dataset.csv"
    data.to_csv(csv_file, index=False)
    
            print_info(f"[DatasetUtils] Saved CSV dataset: {csv_file}")  # SUPREME_RULES compliant logging

def save_game_logs(game_logs: List[dict], summary: dict, algorithm_path: Path):
    """Save game logs and summary to algorithm-specific directory"""
    # Save individual game logs with proper numbering
    for i, game_log in enumerate(game_logs, 1):
        game_file = algorithm_path / f"game_{i}.json"
        with open(game_file, 'w') as f:
            json.dump(game_log, f, indent=2)
    
    # Save summary
    summary_file = algorithm_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
            print_info(f"[DatasetUtils] Saved {len(game_logs)} game logs and summary to {algorithm_path}")  # SUPREME_RULES compliant logging
```

### **JSONL Format (heuristics-v0.04 only)**
```python
def save_jsonl_dataset(data: list, algorithm_path: Path, algorithm: str):
    """Save JSONL dataset to algorithm-specific directory"""
    jsonl_file = algorithm_path / f"{algorithm.lower()}_dataset.jsonl"
    
    with open(jsonl_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    
            print_info(f"[DatasetUtils] Saved JSONL dataset: {jsonl_file}")  # SUPREME_RULES compliant logging
```

### **NPZ Format (for future extensions)**
```python
def save_npz_dataset(data_dict: dict, algorithm_path: Path, algorithm: str, format_type: str):
    """Save NPZ dataset to algorithm-specific directory"""
    npz_file = algorithm_path / f"{algorithm.lower()}_{format_type}_data.npz"
    np.savez(npz_file, **data_dict)
    
            print_info(f"[DatasetUtils] Saved NPZ dataset: {npz_file}")  # SUPREME_RULES compliant logging
```

## 🎓 **Educational Applications with Canonical Patterns**

### **Dataset Organization Benefits**
- **Consistency**: Same organization across all extensions
- **Scalability**: Works with any grid size and algorithm
- **Algorithm Separation**: Clear algorithm-specific organization
- **Educational Value**: Learn dataset organization through consistent patterns

### **Cross-Extension Benefits**
- **Compatibility**: Datasets can be shared between extensions
- **Reusability**: Train on one extension, test on another
- **Comparison**: Consistent evaluation across extensions and algorithms
- **Educational Value**: Learn cross-extension compatibility

## 📋 **SUPREME_RULES Implementation Checklist**

### **Mandatory Requirements**
- [ ] **Grid-Size Agnostic**: Works with any grid size (SUPREME_RULES from final-decision.md compliance)
- [ ] **Algorithm-Specific**: Clear algorithm separation in directory structure
- [ ] **Simple Logging**: Uses utils/print_utils.py functions only for all operations
- [ ] **GOOD_RULES Reference**: References SUPREME_RULES from final-decision.md in all documentation
- [ ] **Pattern Consistency**: Follows canonical patterns across all implementations

### **Dataset-Specific Standards**
- [ ] **Directory Structure**: Standardized directory organization with algorithm subdirectories
- [ ] **Naming Conventions**: Consistent naming patterns for algorithms and files
- [ ] **Metadata Management**: Proper metadata creation and storage
- [ ] **Format Support**: Support for multiple data formats (CSV, JSONL, NPZ)
- [ ] **Game Log Integration**: Game logs and processed datasets in same algorithm directory

---

**Datasets folder standards ensure consistent dataset organization with algorithm-specific separation while maintaining SUPREME_RULES compliance and educational value across all Snake Game AI extensions.**

## 🔗 **See Also**

- **`data-format-decision-guide.md`**: Authoritative reference for data format decisions
- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure standards

### **Example Directory Structure**
```
logs/extensions/datasets/grid-size-10/heuristics-v0.04_20250703_043703/
├── metadata.json
├── bfs/
│   ├── game_1.json                  # First game log
│   ├── game_2.json                  # Second game log
│   ├── game_3.json                  # Third game log (if N=3)
│   ├── summary.json                 # Session summary
│   ├── bfs_dataset.csv              # Aggregated CSV dataset from all games
│   └── bfs_dataset.jsonl            # Aggregated JSONL dataset from all games
├── astar/
│   ├── game_1.json                  # First game log
│   ├── game_2.json                  # Second game log
│   ├── summary.json                 # Session summary
│   ├── astar_dataset.csv            # Aggregated CSV dataset from all games
│   └── astar_dataset.jsonl          # Aggregated JSONL dataset from all games
├── hamiltonian/
│   ├── game_1.json                  # Single game log (if N=1)
│   ├── summary.json                 # Session summary
│   ├── hamiltonian_dataset.csv      # CSV dataset
│   └── hamiltonian_dataset.jsonl    # JSONL dataset
└── evaluation/
    ├── performance_metrics.json
    ├── comparison_results.json
    └── visualization_data.json
```

### **Multiple Games Handling**
The system elegantly handles any number of games (N games):
- **N=1**: Single `game_1.json` file
- **N=3**: Files `game_1.json`, `game_2.json`, `game_3.json`
- **N=10**: Files `game_1.json` through `game_10.json`
- **Aggregated Datasets**: CSV and JSONL files contain data from all N games combined
- **Session Summary**: Single `summary.json` contains aggregate statistics from all games