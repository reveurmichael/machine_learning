# Large-Scale Heuristics to ML Pipeline Tutorial

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines. This tutorial demonstrates the complete data pipeline following established architectural patterns.

## 🎯 **Core Philosophy: Comprehensive ML Pipeline**

This tutorial demonstrates the complete journey from heuristic algorithm execution to trained ML models, showcasing the multi-directional data ecosystem established in the GOODRULES architecture.

### **Pipeline Overview**
- **Heuristics v0.04**: Generates 10,000+ games per algorithm with language-rich explanations
- **Supervised v0.02+**: Trains neural networks and tree models on generated datasets  
- **Cross-Extension Data Flow**: Demonstrates the benefits of the standardized architecture

## 📋 **Prerequisites and Setup**

### **Technical Requirements**
- Python 3.8+
- PyTorch installed (`pip install torch`)
- Sufficient disk space (~10GB for full pipeline)
- Time allocation: 4-6 hours for complete execution

### **Mandatory Path Management**
Following Final Decision 6, all scripts must use standardized path utilities:

```python
from extensions.common.path_utils import ensure_project_root, get_dataset_path
from extensions.common.path_utils import get_model_path

# MANDATORY: Ensure correct working directory
project_root = ensure_project_root()
```

## 🧠 **Part 1: Understanding Heuristic Algorithms**

### **Available Algorithms in heuristics-v0.04**
Following Final Decision 4 agent naming conventions:

1. **BFS** (`agent_bfs.py` → `BFSAgent`) - Basic breadth-first search
2. **BFS-SAFE-GREEDY** (`agent_bfs_safe_greedy.py`) - Enhanced BFS with safety validation
3. **BFS-HAMILTONIAN** (`agent_bfs_hamiltonian.py`) - BFS with Hamiltonian cycle fallback
4. **DFS** (`agent_dfs.py` → `DFSAgent`) - Depth-first search (educational comparison)
5. **ASTAR** (`agent_astar.py` → `AStarAgent`) - A* pathfinding with Manhattan heuristic
6. **ASTAR-HAMILTONIAN** (`agent_astar_hamiltonian.py`) - A* with Hamiltonian fallback
7. **HAMILTONIAN** (`agent_hamiltonian.py` → `HamiltonianAgent`) - Pure Hamiltonian cycle

## 📊 **Part 2: Large-Scale Dataset Generation**

### **Path Management Integration**
Following the mandatory path utilities from Final Decision 6:

```python
from extensions.common.path_utils import ensure_project_root

# Essential setup for all pipeline scripts
def setup_pipeline_environment():
    """Required setup following Final Decision 6"""
    project_root = ensure_project_root()
    return project_root
```

### **Dataset Generation Pipeline**
Using standardized directory structure from Final Decision 1:

```python
from extensions.common.path_utils import get_dataset_path

# Generate datasets for each algorithm
algorithms = ["BFS", "ASTAR", "DFS", "HAMILTONIAN", "BFS_SAFE_GREEDY", 
              "BFS_HAMILTONIAN", "ASTAR_HAMILTONIAN"]

for algorithm in algorithms:
    dataset_path = get_dataset_path(
        extension_type="heuristics",
        version="0.04",
        grid_size=10,  # Grid-size agnostic as per Final Decision 1
        algorithm=algorithm.lower(),
        timestamp=current_timestamp
    )
    
    # Execute large-scale generation (10,000 games per algorithm)
    subprocess.run([
        "python", "scripts/main.py",
        "--algorithm", algorithm,
        "--max-games", "10000",
        "--grid-size", "10",
        "--output-path", str(dataset_path)
    ])
```

### **Expected Output Structure**
Following Final Decision 1 directory organization:

```
logs/extensions/datasets/grid-size-10/heuristics_v0.04_timestamp/
├── bfs/
│   ├── game_logs/                 # Original game JSON files
│   └── processed_data/
│       ├── tabular_data.csv       # For supervised learning
│       ├── reasoning_data.jsonl   # For LLM fine-tuning
│       └── metadata.json
├── astar/
│   └── [same structure]
└── [additional algorithms...]
```

## 🤖 **Part 3: Supervised Learning Pipeline**

### **Model Training with Generated Datasets**
Following the standardized CSV schema from csv-schema-1.md:

```python
from extensions.common.dataset_loader import load_dataset_for_training

# Load datasets using standardized 16-feature schema
dataset_paths = [
    "logs/extensions/datasets/grid-size-10/heuristics_v0.04_timestamp/bfs/processed_data/tabular_data.csv",
    "logs/extensions/datasets/grid-size-10/heuristics_v0.04_timestamp/astar/processed_data/tabular_data.csv"
]

X_train, X_val, X_test, y_train, y_val, y_test, info = load_dataset_for_training(
    dataset_paths=dataset_paths,
    grid_size=10
)
```

### **Multi-Model Training Pipeline**
Using the factory pattern from Final Decision 7-8:

```python
from extensions.supervised_v0_02.agents import SupervisedAgentFactory

# Train multiple model types
model_types = ["MLP", "CNN", "LSTM", "XGBOOST", "LIGHTGBM"]

for model_type in model_types:
    # Create agent using factory pattern
    agent = SupervisedAgentFactory.create_agent(model_type)
    
    # Train model
    trained_agent = train_model(agent, X_train, y_train, X_val, y_val)
    
    # Save model using standardized paths
    model_path = get_model_path(
        extension_type="supervised",
        version="0.02",
        grid_size=10,
        algorithm=model_type.lower(),
        timestamp=current_timestamp
    )
    
    save_model(trained_agent, model_path)
```

## 🔤 **Part 4: LLM Fine-tuning Pipeline**

### **JSONL Dataset Preparation**
Following the language-rich data format from extensions-v0.04.md:

```python
# Load JSONL datasets generated by heuristics-v0.04
jsonl_files = []
for algorithm in algorithms:
    jsonl_path = get_dataset_path(
        extension_type="heuristics",
        version="0.04",
        grid_size=10,
        algorithm=algorithm,
        timestamp=timestamp
    ) / "processed_data" / "reasoning_data.jsonl"
    
    jsonl_files.append(jsonl_path)

# Combine all JSONL files for comprehensive fine-tuning dataset
combined_dataset = combine_jsonl_files(jsonl_files)
```

### **Fine-tuning Execution**
Using standardized model paths:

```python
# Fine-tune using combined heuristic reasoning data
fine_tuned_model_path = get_model_path(
    extension_type="llm_finetune",
    version="0.02",
    grid_size=10,
    algorithm="heuristics_combined",
    timestamp=current_timestamp
)

# Execute fine-tuning pipeline
fine_tune_llm(
    dataset_path=combined_dataset,
    output_dir=fine_tuned_model_path,
    epochs=3,
    batch_size=8
)
```

## 📈 **Expected Results and Performance**

### **Dataset Characteristics**
- **Total Games**: ~70,000 games (10K per algorithm)
- **CSV Files**: Grid-size agnostic 16-feature format (~35MB per algorithm)
- **JSONL Files**: Language-rich explanations (~50MB per algorithm)
- **Total Storage**: ~600MB for complete multi-format datasets

### **Model Performance Expectations**
Following the performance hierarchy from datasets_folder.md:

- **BFS Models**: 85-90% accuracy (optimal pathfinding)
- **Hamiltonian Models**: 95%+ accuracy (guaranteed safety)
- **ASTAR Models**: 90-95% accuracy (informed search)
- **Fine-tuned LLMs**: Enhanced reasoning about game states

### **Training Infrastructure**
- **Supervised Training**: 2-3 hours total across all model types
- **LLM Fine-tuning**: 6-8 hours (GPU acceleration recommended)
- **Total Pipeline**: 12-17 hours for complete execution

## 🎯 **Architectural Benefits Demonstrated**

### **Multi-Directional Data Ecosystem**
This pipeline showcases the benefits established in Final Decision 1:
- **Heuristics** provide baseline datasets with algorithmic traces
- **Supervised models** learn from expert demonstrations efficiently
- **LLM fine-tuning** gains from language-grounded reasoning
- **Cross-extension compatibility** through standardized formats

### **Educational Value**
- **Complete ML Workflow**: From data generation to model deployment
- **Multi-paradigm Integration**: Symbolic, neural, and language-based AI
- **Architecture Patterns**: Factory, path management, and configuration standards
- **Scalable Design**: Grid-size agnostic and extensible to new algorithms

---

**This pipeline demonstrates the power of the architectural decisions established in the Final Decision series, showing how proper abstractions and standardized interfaces enable sophisticated multi-algorithm workflows while maintaining educational clarity and system coherence.**
