Each extension folder of v0.03 (not for v0.01, not for v0.02), will have a folder named "scripts".

For v0.03, it's really important because streamlit app.py will call those scripts extensively. 

### **Direct Script Execution**
```bash
# Train supervised models
python scripts/training/train_supervised.py \
    --dataset-paths logs/extensions/datasets/grid-size-N/heuristics_v0.03_20250625_143022/bfs/processed_data/tabular_data.csv \
    --model-types all \
    --hyperparameter-tuning \
    --output-dir logs/extensions/models/grid-size-N/supervised_v0.02_20250625_143022/mlp/

# Generate heuristic datasets
python scripts/data_generation/generate_heuristic_data.py \
    --algorithms bfs astar hamiltonian \
    --grid-size 10 \
    --num-games 1000 \
    --output-format csv

# Evaluate model performance
python scripts/evaluation/evaluate_models.py \
    --model-dir logs/extensions/models/grid-size-N/supervised_v0.02_20250625_143022/mlp/ \
--test-data logs/extensions/datasets/grid-size-N/heuristics_v0.03_20250625_143022/bfs/processed_data/tabular_data.csv \
--output-dir evaluation_results
```



# IMPORTANT
streamlit app.py is not for visualization of game states, is not for real time showing progress, is 
not 
for showing snake moves.

It's main idea is to launch scripts in the folder "scripts" with adjustable params, with 
subprocess. That's why for extensions v0.03 we will have a folder "dashboard" in the first place.

# Script Architecture for v0.03 Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines the mandatory script architecture for all v0.03 extensions.

> **See also:** `final-decision-10.md`, `app.md`, `dashboard.md`, `standalone.md`.

## 🎯 **Core Philosophy: Script-Runner Architecture**

The `scripts/` directory is **mandatory for all v0.03 extensions** and implements the "script-runner" architecture where Streamlit applications serve as **interactive frontends** that launch specialized scripts via subprocess. This separation ensures clean architecture and enables both CLI and web-based operation.

### **Educational Value**
- **Separation of Concerns**: UI logic separate from core algorithm implementation
- **Dual Interface**: Same functionality available via CLI and web interface
- **Subprocess Safety**: Isolated execution with proper error handling
- **Canonical Patterns**: Demonstrates factory patterns and simple logging throughout

## 🏗️ **Mandatory Directory Structure**

### **Universal v0.03 Script Organization**
```
extensions/{algorithm}-v0.03/
├── app.py                      # Streamlit application (launches scripts)
├── dashboard/                  # UI components (orchestrates script execution)
├── scripts/                    # 🎯 MANDATORY: Backend execution scripts
│   ├── __init__.py
│   ├── main.py                 # Primary algorithm execution
│   ├── generate_dataset.py     # Dataset generation
│   ├── train.py                # Training (ML extensions only)
│   ├── evaluate.py             # Performance evaluation
│   └── replay.py               # Game replay functionality
└── agents/                     # Core algorithm implementations
```

**Critical Rule**: Every v0.03 extension MUST implement this script structure.

## 📜 **Standard Script Implementations**

### **1. main.py - Primary Algorithm Execution**
```python
#!/usr/bin/env python3
"""
Primary algorithm execution script for {algorithm} extension.

This script can be executed directly via CLI or launched by Streamlit app.py
through subprocess calls with parameter passing.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from extensions.common.utils.path_utils import ensure_project_root

def main():
    """Main execution function for heuristic algorithms"""
    # Ensure proper working directory
    ensure_project_root()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run heuristic pathfinding algorithms")
    parser.add_argument("--algorithm", required=True, choices=["BFS", "ASTAR", "DFS", "HAMILTONIAN"])
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-games", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Import and execute algorithm using canonical factory patterns
    from extensions.common.utils.factory_utils import SimpleFactory
    from game_manager import HeuristicGameManager
    
    # Use canonical factory pattern
    factory = SimpleFactory()
    factory.register("heuristic", HeuristicGameManager)
    
    manager = factory.create("heuristic", args)  # CANONICAL create() method - SUPREME_RULES
    results = manager.run()
    
    print(f"Execution completed. Results: {results}")  # Simple logging - SUPREME_RULES
    return results

if __name__ == "__main__":
    main()
```

### **2. generate_dataset.py - Dataset Generation**
```python
#!/usr/bin/env python3
"""
Dataset generation script for supervised learning training data.

Generates standardized datasets in multiple formats (CSV, JSONL, NPZ) following
the data format decisions from `data-format-decision-guide.md`.
"""

import argparse
from pathlib import Path

def main():
    """Generate training datasets from algorithm execution"""
    parser = argparse.ArgumentParser(description="Generate training datasets")
    parser.add_argument("--algorithms", nargs="+", default=["BFS", "ASTAR"])
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--output-format", choices=["csv", "jsonl", "npz", "all"], default="csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    
    args = parser.parse_args()
    
    # Generate datasets for each algorithm using canonical factory patterns
    from extensions.common.utils.factory_utils import SimpleFactory
    from extensions.common.utils.dataset_utils import DatasetGenerator
    
    # Use canonical factory pattern
    factory = SimpleFactory()
    factory.register("dataset", DatasetGenerator)
    
    generator = factory.create("dataset", args.grid_size, args.output_format)  # Canonical
    
    for algorithm in args.algorithms:
        print(f"Generating dataset for {algorithm}...")  # Simple logging - SUPREME_RULES
        
        # Execute games and collect data
        game_args = argparse.Namespace(
            algorithm=algorithm,
            grid_size=args.grid_size,
            max_games=args.num_games,
            output_dir=args.output_dir / algorithm
        )
        
        # Use canonical factory pattern for game manager
        game_factory = SimpleFactory()
        game_factory.register("game", HeuristicGameManager)
        
        manager = game_factory.create("game", game_args)  # Canonical
        results = manager.run()
        
        # Convert to training format
        dataset = generator.create_dataset(results, algorithm)
        generator.save_dataset(dataset, args.output_dir / algorithm)
        
        print(f"Dataset saved for {algorithm} at {args.output_dir / algorithm}")  # Simple logging

if __name__ == "__main__":
    main()
```

### **3. train.py - Training Script (ML Extensions)**
```python
#!/usr/bin/env python3
"""
Training script for supervised learning models.

Loads datasets from heuristics extensions and trains multiple model types
with configurable hyperparameters and evaluation metrics.
"""

import argparse
from pathlib import Path

def main():
    """Train supervised learning models"""
    parser = argparse.ArgumentParser(description="Train supervised learning models")
    parser.add_argument("--dataset-paths", nargs="+", required=True)
    parser.add_argument("--model-types", nargs="+", default=["MLP", "XGBOOST"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hyperparameter-tuning", action="store_true")
    
    args = parser.parse_args()
    
    # Use canonical factory pattern for training
    from extensions.common.utils.factory_utils import SimpleFactory
    from extensions.common.utils.dataset_utils import DatasetLoader
    
    # Load datasets
    loader = DatasetLoader(grid_size=10)
    datasets = []
    for path in args.dataset_paths:
        df = loader.load_csv_dataset(path)
        datasets.append(df)
        print(f"Loaded dataset: {path} with {len(df)} rows")  # Simple logging - SUPREME_RULES
    
    # Train models using canonical factory pattern
    factory = SimpleFactory()
    factory.register("trainer", ModelTrainer)
    
    for model_type in args.model_types:
        print(f"Training {model_type} model...")  # Simple logging - SUPREME_RULES
        
        trainer = factory.create("trainer", model_type, args.output_dir)  # Canonical
        results = trainer.train(datasets, args.hyperparameter_tuning)
        
        print(f"{model_type} training completed: {results}")  # Simple logging

if __name__ == "__main__":
    main()
```

### **4. evaluate.py - Performance Evaluation**
```python
#!/usr/bin/env python3
"""
Model evaluation and comparison script.

Loads trained models and evaluates performance on test datasets with
comprehensive metrics and comparison analysis.
"""

import argparse
from pathlib import Path
import json

def main():
    """Evaluate model performance with comprehensive metrics"""
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-mode", action="store_true")
    
    args = parser.parse_args()
    
    # Import evaluation infrastructure
    from evaluation.model_evaluator import ModelEvaluator
    from extensions.common.utils.dataset_utils import load_test_dataset
    
    # Load test data
    X_test, y_test = load_test_dataset(args.test_data)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.output_dir)
    
    # Find and evaluate all models in directory
    model_files = list(args.model_dir.glob("**/*.pth"))  # PyTorch models
    model_files.extend(args.model_dir.glob("**/*.pkl"))  # Scikit-learn models
    
    results = {}
    
    for model_file in model_files:
        print(f"Evaluating {model_file.name}...")  # Simple logging
        
        model = evaluator.load_model(model_file)
        metrics = evaluator.evaluate_comprehensive(model, X_test, y_test)
        
        results[model_file.name] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.3f}")  # Simple logging
        print(f"  F1-Score: {metrics['f1_score']:.3f}")  # Simple logging
    
    # Save results
    with open(args.output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate comparison report if requested
    if args.comparison_mode:
        evaluator.generate_comparison_report(results)

if __name__ == "__main__":
    main()
```

## 🔗 **Streamlit Integration Pattern**

### **Script Launching from Streamlit**
```python
# dashboard/tab_main.py
import subprocess
import streamlit as st
from pathlib import Path

class MainTab:
    """Main algorithm execution tab that launches scripts"""
    
    def render(self, session_state):
        """Render main algorithm interface"""
        st.header("🎮 Algorithm Execution")
        
        # Parameter controls
        col1, col2 = st.columns(2)
        
        with col1:
            algorithm = st.selectbox("Algorithm", ["BFS", "ASTAR", "DFS"])
            grid_size = st.slider("Grid Size", 8, 20, 10)
            
        with col2:
            max_games = st.number_input("Max Games", 1, 100, 1)
            visualization = st.checkbox("Enable Visualization")
        
        # Launch script button
        if st.button("🚀 Run Algorithm"):
            with st.spinner("Executing algorithm..."):
                result = self.launch_main_script(
                    algorithm=algorithm,
                    grid_size=grid_size,
                    max_games=max_games,
                    visualization=visualization
                )
                
                if result.returncode == 0:
                    st.success("Algorithm execution completed!")
                    st.json(result.stdout)
                else:
                    st.error(f"Execution failed: {result.stderr}")
    
    def launch_main_script(self, **kwargs):
        """Launch main.py script with parameters"""
        script_path = Path(__file__).parent.parent / "scripts" / "main.py"
        
        cmd = ["python", str(script_path)]
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key.replace('_', '-')}")
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return subprocess.run(cmd, capture_output=True, text=True)
```

## 📋 **Implementation Checklist**

### **Required Components**
- [ ] **main.py**: Primary algorithm execution script
- [ ] **generate_dataset.py**: Dataset generation (if applicable)
- [ ] **train.py**: Training script (ML extensions only)
- [ ] **evaluate.py**: Performance evaluation script
- [ ] **replay.py**: Game replay functionality
- [ ] **Path Management**: Proper working directory handling
- [ ] **Argument Parsing**: Standardized command-line interface
- [ ] **Error Handling**: Graceful failure recovery

### **Quality Standards**
- [ ] **Modularity**: Clean separation of concerns
- [ ] **Reusability**: Scripts work independently and via subprocess
- [ ] **Error Handling**: Proper error messages and recovery
- [ ] **Documentation**: Clear docstrings and usage examples
- [ ] **Testing**: Comprehensive test coverage

### **Integration Requirements**
- [ ] **Streamlit Integration**: Seamless subprocess launching
- [ ] **Factory Pattern**: Dynamic algorithm selection
- [ ] **Configuration**: Support for configurable parameters
- [ ] **Logging**: Simple print statements for debugging
- [ ] **Data Management**: Safe handling of datasets and models

---

**Script architecture ensures clean separation between UI and core algorithm implementation, enabling both interactive web interfaces and command-line operation while maintaining educational value and technical excellence.**

## 🔗 **See Also**

- **`app.md`**: Streamlit application architecture
- **`final-decision-10.md`**: final-decision-10.md governance system
- **`standalone.md`**: Standalone principle and extension independence
