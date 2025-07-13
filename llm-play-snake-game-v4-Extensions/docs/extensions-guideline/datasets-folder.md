# Datasets Folder Standards for Snake Game AI

## 🎯 **Core Philosophy: Grid-Size Agnostic Organization**

The datasets folder uses a **grid-size agnostic organization** that ensures consistent dataset storage across all extensions and grid sizes. This system provides predictable dataset locations and enables cross-grid-size comparisons.

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
├── {algorithm}/                     # Algorithm-specific datasets 
│   ├── game_1.json                 # Original game execution logs
│   ├── game_2.json                 # Original game execution logs  
│   ├── ...
│   ├── summary.json                # Session summary file
│   ├── {algorithm}_dataset.csv     # CSV format (ACTIVE, NOT legacy)
│   ├── {algorithm}_dataset.jsonl   # JSONL format (heuristics-v0.04 only)
├── {algorithm2}/                   # Additional algorithm datasets
│   ├── game_1.json
│   ├── game_2.json
│   ├── ...
│   ├── summary.json
│   ├── {algorithm2}_dataset.csv
│   └── {algorithm2}_dataset.jsonl

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

## ✅ **Success Indicators**

### **Working Implementation Examples**
- **Heuristics v0.04**: Successfully generates datasets in standardized locations
- **File Organization**: Proper directory structure with algorithm subdirectories
- **Dataset Generation**: Both CSV and JSONL files generated correctly
- **Metadata Management**: Comprehensive metadata tracking
- **Cross-Algorithm Support**: Multiple algorithms supported in same session

### **Quality Standards**
- **Directory Structure**: Consistent and predictable organization
- **File Naming**: Clear, descriptive file names
- **Data Integrity**: Generated data accurately represents game state
- **Cross-Platform Compatibility**: UTF-8 encoding and path handling

## 🔗 **Cross-Extension Integration**

### **Dataset Sharing**
- **Heuristics v0.04**: Generates standardized datasets for other extensions
- **Supervised Learning**: Consumes CSV datasets from heuristics-v0.04
- **LLM Fine-tuning**: Consumes JSONL datasets from heuristics-v0.04
- **Evaluation**: Consistent comparison framework across all extensions

### **Path Management**
- **Standardized Paths**: Consistent dataset storage locations
- **Grid-Size Agnostic**: Works with any supported grid size
- **Version Management**: Clear versioning and timestamping
- **Cross-Platform**: UTF-8 encoding and proper path handling

---

**The datasets folder standards ensure consistent, organized dataset storage across all Snake Game AI extensions while maintaining cross-extension compatibility and following forward-looking architecture principles.**

## 🔗 **See Also**

- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards
- **`csv-schema.md`**: CSV schema standards and format specifications
- **`data-format-decision-guide.md`**: Data format selection guidelines
- **`extensions-v0.04.md`**: Advanced data generation patterns