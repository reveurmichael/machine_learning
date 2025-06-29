# Datasets Folder Standards for Snake Game AI

> **Important — Authoritative Reference:** This document serves as a **GOOD_RULES** authoritative reference for datasets folder standards and supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`).

> **See also:** `data-format-decision-guide.md`, `final-decision-10.md`, `project-structure-plan.md`.

## 🎯 **Core Philosophy: Grid-Size Agnostic Organization**

The datasets folder uses a **grid-size agnostic organization** that ensures consistent dataset storage across all extensions and grid sizes. This system provides predictable dataset locations and enables cross-grid-size comparisons, strictly following `final-decision-10.md` SUPREME_RULES.

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
├── processed_data/                  # Processed datasets in various formats
│   ├── tabular_data.csv            # CSV format (ACTIVE, NOT legacy)
│   ├── reasoning_data.jsonl        # JSONL format (heuristics-v0.04 only)
│   ├── sequential_data.npz         # NPZ Sequential format
│   ├── spatial_data.npz            # NPZ 2D Arrays format
│   ├── raw_data.npz                # NPZ Raw Arrays format (general)
│   └── evolutionary_data.npz       # NPZ Raw Arrays format (evolutionary specific)
├── game_logs/                      # Original game execution logs
│   ├── game_1.json
│   ├── game_2.json
│   ├── ...
│   ├── prompts/
│   │   ├── game_1_round_1_prompt.txt
│   │   ├── game_1_round_2_prompt.txt
│   │   └── ...
│   ├── responses/
│   │   ├── game_1_round_1_raw_response.txt
│   │   ├── game_1_round_2_raw_response.txt
│   │   └── ...
│   └── summary.json
└── evaluation/                     # Evaluation results and metrics
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

def create_dataset_directory(extension_type: str, version: str, grid_size: int, timestamp: str) -> Path:
    """Create standardized dataset directory"""
    datasets_root = get_datasets_root()
    dataset_path = datasets_root / f"grid-size-{grid_size}" / f"{extension_type}_v{version}_{timestamp}"
    
    # Create directory structure
    dataset_path.mkdir(parents=True, exist_ok=True)
    (dataset_path / "processed_data").mkdir(exist_ok=True)
    (dataset_path / "game_logs").mkdir(exist_ok=True)
    (dataset_path / "evaluation").mkdir(exist_ok=True)
    
    print(f"[DatasetUtils] Created dataset directory: {dataset_path}")  # Simple logging
    return dataset_path

def get_dataset_path(extension_type: str, version: str, grid_size: int, timestamp: str) -> Path:
    """Get standardized dataset path"""
    datasets_root = get_datasets_root()
    return datasets_root / f"grid-size-{grid_size}" / f"{extension_type}_v{version}_{timestamp}"
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
    
    print(f"[DatasetUtils] Saved metadata: {metadata_file}")  # Simple logging
```

## 📋 **Dataset Format Standards**

### **CSV Format (heuristics-v0.03 and heuristics-v0.04)**
```python
def save_csv_dataset(data: pd.DataFrame, dataset_path: Path, algorithm: str):
    """Save CSV dataset to processed_data directory"""
    csv_file = dataset_path / "processed_data" / f"{algorithm}_data.csv"
    data.to_csv(csv_file, index=False)
    
    print(f"[DatasetUtils] Saved CSV dataset: {csv_file}")  # Simple logging
```

### **JSONL Format (heuristics-v0.04 only)**
```python
def save_jsonl_dataset(data: list, dataset_path: Path, algorithm: str):
    """Save JSONL dataset to processed_data directory"""
    jsonl_file = dataset_path / "processed_data" / f"{algorithm}_reasoning.jsonl"
    
    with open(jsonl_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    
    print(f"[DatasetUtils] Saved JSONL dataset: {jsonl_file}")  # Simple logging
```

### **NPZ Format**
```python
def save_npz_dataset(data_dict: dict, dataset_path: Path, format_type: str):
    """Save NPZ dataset to processed_data directory"""
    npz_file = dataset_path / "processed_data" / f"{format_type}_data.npz"
    np.savez(npz_file, **data_dict)
    
    print(f"[DatasetUtils] Saved NPZ dataset: {npz_file}")  # Simple logging
```

## 🚀 **Cross-Extension Dataset Usage**

### **Loading Datasets for Training**
```python
def load_training_dataset(extension_type: str, version: str, grid_size: int, 
                         timestamp: str, format_type: str = "csv") -> tuple:
    """Load dataset for training"""
    dataset_path = get_dataset_path(extension_type, version, grid_size, timestamp)
    
    if format_type == "csv":
        data_file = dataset_path / "processed_data" / f"tabular_data.csv"
        data = pd.read_csv(data_file)
        return data
    elif format_type == "jsonl":
        data_file = dataset_path / "processed_data" / f"reasoning_data.jsonl"
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    elif format_type == "npz":
        data_file = dataset_path / "processed_data" / f"{format_type}_data.npz"
        data = np.load(data_file)
        return data
    else:
        raise ValueError(f"Unsupported format: {format_type}")
```

### **Dataset Validation**
```python
def validate_dataset_structure(dataset_path: Path) -> bool:
    """Validate dataset directory structure"""
    required_dirs = ["processed_data", "game_logs", "evaluation"]
    required_files = ["metadata.json"]
    
    # Check directories
    for dir_name in required_dirs:
        if not (dataset_path / dir_name).exists():
            print(f"[DatasetUtils] ERROR: Missing directory: {dir_name}")  # Simple logging
            return False
    
    # Check files
    for file_name in required_files:
        if not (dataset_path / file_name).exists():
            print(f"[DatasetUtils] ERROR: Missing file: {file_name}")  # Simple logging
            return False
    
    print(f"[DatasetUtils] Dataset structure validated: {dataset_path}")  # Simple logging
    return True
```

## 📊 **Extension-Specific Dataset Patterns**

### **Heuristics Extensions**
```python
# heuristics-v0.03: CSV format only
dataset_path = create_dataset_directory("heuristics", "0.03", 10, timestamp)
save_csv_dataset(tabular_data, dataset_path, "bfs")

# heuristics-v0.04: CSV + JSONL formats
dataset_path = create_dataset_directory("heuristics", "0.04", 10, timestamp)
save_csv_dataset(tabular_data, dataset_path, "bfs")
save_jsonl_dataset(reasoning_data, dataset_path, "bfs")
```

### **Supervised Learning Extensions**
```python
# Load datasets from heuristics-v0.03 or heuristics-v0.04 (both widely used)
csv_data = load_training_dataset("heuristics", "0.04", 10, timestamp, "csv")
jsonl_data = load_training_dataset("heuristics", "0.04", 10, timestamp, "jsonl")
```

### **Reinforcement Learning Extensions**
```python
# Save experience replay data
dataset_path = create_dataset_directory("reinforcement", "0.02", 10, timestamp)
save_npz_dataset(experience_data, dataset_path, "sequential")
```

### **Evolutionary Extensions**
```python
# Save evolutionary population data
dataset_path = create_dataset_directory("evolutionary", "0.02", 10, timestamp)
save_npz_dataset(evolutionary_data, dataset_path, "evolutionary")
```

## 📋 **Implementation Checklist**

### **Directory Structure**
- [ ] **Grid-Size Organization**: Proper grid-size directory structure
- [ ] **Extension Naming**: Consistent extension naming convention
- [ ] **Timestamp Usage**: Proper timestamp format (YYYYMMDD_HHMMSS)
- [ ] **Subdirectory Creation**: All required subdirectories created

### **Data Formats**
- [ ] **CSV Support**: CSV format for heuristics-v0.03 and heuristics-v0.04
- [ ] **JSONL Support**: JSONL format for heuristics-v0.04 only
- [ ] **NPZ Support**: NPZ formats for specialized use cases
- [ ] **Metadata**: Proper metadata file creation

### **Cross-Extension Compatibility**
- [ ] **Path Consistency**: Consistent path generation across extensions
- [ ] **Format Validation**: Proper format validation and error handling
- [ ] **Loading Utilities**: Standardized dataset loading utilities
- [ ] **Error Handling**: Graceful error handling for missing datasets

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Dataset Organization**: Understanding consistent dataset storage
- **Grid-Size Independence**: Learning grid-size agnostic design
- **Version Management**: Clear versioning and timestamping
- **Cross-Extension Compatibility**: Enabling dataset sharing between extensions

### **Best Practices**
- **Consistency**: Consistent dataset organization across all extensions
- **Scalability**: Grid-size agnostic design for scalability
- **Maintainability**: Clear structure for easy maintenance
- **Interoperability**: Standardized formats for cross-extension usage

---

**The datasets folder standards ensure consistent, scalable, and interoperable dataset storage across all Snake Game AI extensions while maintaining grid-size independence and cross-extension compatibility.**

## 🔗 **See Also**

- **`data-format-decision-guide.md`**: Authoritative reference for data format decisions
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure and organization

