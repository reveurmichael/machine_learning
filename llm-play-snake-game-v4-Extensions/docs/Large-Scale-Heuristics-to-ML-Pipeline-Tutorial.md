# Large-Scale Heuristics to ML Pipeline Tutorial

## Overview

This tutorial demonstrates how to generate large-scale datasets from heuristic algorithms and use them for:
1. **Supervised Learning** (PyTorch neural networks, tree models)
2. **LLM Fine-tuning** (using generated text descriptions)
3. **Model Distillation** (knowledge transfer between models)

The pipeline generates **10,000 games per algorithm** across 7 different heuristic strategies, creating comprehensive datasets in multiple formats (JSON, CSV, JSONL).

## Prerequisites

- Python 3.8+
- PyTorch installed (`pip install torch`)
- Sufficient disk space (~10GB for full pipeline)
- Time: 4-6 hours for complete execution

## Part 1: Understanding the Heuristic Algorithms

### Available Algorithms in heuristics-v0.04

1. **BFS** - Basic breadth-first search
2. **BFS-SAFE-GREEDY** - Enhanced BFS with safety validation
3. **BFS-HAMILTONIAN** - BFS with Hamiltonian cycle fallback
4. **DFS** - Depth-first search (educational comparison)
5. **ASTAR** - A* pathfinding with Manhattan heuristic
6. **ASTAR-HAMILTONIAN** - A* with Hamiltonian fallback
7. **HAMILTONIAN** - Pure Hamiltonian cycle (guaranteed safety)

### Algorithm Hierarchy (Inheritance Design)
```
BaseAgent (core)
├── BFSAgent
│   ├── BFSSafeGreedyAgent
│   │   └── BFSHamiltonianAgent
├── DFSAgent
├── AStarAgent
│   └── AStarHamiltonianAgent
└── HamiltonianAgent
```

## Part 2: Large-Scale Dataset Generation

### Step 1: Create Symbolic Links for Import Compatibility

```bash
# Create symbolic links to fix import issues
cd extensions
ln -sf heuristics-v0.04 heuristics_v0_04
ln -sf supervised-v0.02 supervised_v0_02
```

### Step 2: Generate Heuristic Game Data (10,000 games per algorithm)

Create the master data generation script:

```bash
# Create the batch generation script
cat > scripts/generate_large_scale_datasets.py << 'EOF'
#!/usr/bin/env python3
"""
Large-Scale Dataset Generation Pipeline
=====================================

Generates 10,000 games per heuristic algorithm and converts to multiple formats.
Total expected: ~70,000 games across 7 algorithms.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import List

# Available algorithms from heuristics-v0.04
ALGORITHMS = [
    "BFS",
    "BFS-SAFE-GREEDY", 
    "BFS-HAMILTONIAN",
    "DFS",
    "ASTAR",
    "ASTAR-HAMILTONIAN",
    "HAMILTONIAN"
]

def run_heuristic_games(algorithm: str, max_games: int = 10000) -> str:
    """Run heuristic games and return log directory."""
    print(f"\\n{'='*60}")
    print(f"RUNNING {algorithm} - {max_games} games")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 
        "extensions/heuristics-v0.04/scripts/main.py",
        "--algorithm", algorithm,
        "--max-games", str(max_games),
        "--max-steps", "800",
        "--grid-size", "10",
        "--no-gui"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    print(f"✅ {algorithm} completed in {elapsed:.1f}s")
    
    # Parse log directory from output
    for line in result.stdout.split('\\n'):
        if 'Log directory:' in line:
            return line.split(': ')[1].strip()
    
    raise RuntimeError(f"Could not find log directory for {algorithm}")

def generate_datasets(log_dir: str, algorithm: str):
    """Generate CSV and JSONL datasets from game logs."""
    print(f"\\n📊 Generating datasets for {algorithm}")
    
    # Generate CSV (tabular format for supervised learning)
    # Use dynamic grid-size from the log directory analysis or config
    grid_size = 10  # This should be extracted from the actual game logs
    csv_output = f"logs/extensions/datasets/grid-size-{grid_size}/{algorithm.lower()}_10k_tabular.csv"
    cmd_csv = [
        sys.executable,
        "extensions/heuristics-v0.04/scripts/generate_datasets.py",
        "--log-dir", log_dir,
        "--output", csv_output,
        "--format", "csv",
        "--grid-size", "10"
    ]
    subprocess.run(cmd_csv, check=True)
    print(f"  ✅ CSV: {csv_output}")
    
    # Generate JSONL (for LLM fine-tuning)
    jsonl_output = f"logs/extensions/datasets/grid-size-10/{algorithm.lower()}_10k_language.jsonl"
    cmd_jsonl = [
        sys.executable,
        "extensions/heuristics-v0.04/scripts/generate_datasets.py", 
        "--log-dir", log_dir,
        "--output", jsonl_output,
        "--format", "jsonl",
        "--grid-size", "10"
    ]
    subprocess.run(cmd_jsonl, check=True)
    print(f"  ✅ JSONL: {jsonl_output}")

def main():
    """Main pipeline execution."""
    print("🚀 Starting Large-Scale Heuristics Dataset Generation")
    print(f"Target: {len(ALGORITHMS)} algorithms × 10,000 games = 70,000 total games")
    
    # Ensure output directory exists
    Path("logs/extensions/datasets/grid-size-10").mkdir(parents=True, exist_ok=True)
    
    total_start = time.time()
    
    for i, algorithm in enumerate(ALGORITHMS, 1):
        print(f"\\n🎯 Algorithm {i}/{len(ALGORITHMS)}: {algorithm}")
        
        # Step 1: Generate game data
        log_dir = run_heuristic_games(algorithm)
        
        # Step 2: Convert to datasets
        generate_datasets(log_dir, algorithm)
        
        elapsed = time.time() - total_start
        print(f"📈 Progress: {i}/{len(ALGORITHMS)} algorithms completed ({elapsed/60:.1f} min)")
    
    total_time = time.time() - total_start
    print(f"\\n🎉 PIPELINE COMPLETE!")
    print(f"⏱️  Total time: {total_time/3600:.1f} hours")
    print(f"📁 Datasets saved to: logs/extensions/datasets/grid-size-10/")

if __name__ == "__main__":
    main()
EOF
```

### Step 3: Execute Large-Scale Generation

```bash
# Run the full pipeline (4-6 hours)
python scripts/generate_large_scale_datasets.py
```

**Expected Output Structure:**
```
logs/extensions/datasets/grid-size-{N}/     # N = actual grid size used
├── bfs_10k_tabular.csv                    # 10K BFS decisions 
├── bfs_10k_language.jsonl                 # 10K BFS text descriptions
├── bfs-safe-greedy_10k_tabular.csv        # 10K Enhanced BFS decisions
├── bfs-safe-greedy_10k_language.jsonl     # 10K Enhanced BFS descriptions
├── ...                                    # Similar for all 7 algorithms
└── hamiltonian_10k_language.jsonl         # 10K Hamiltonian descriptions
```

**Models will be saved to:**
```
logs/extensions/models/grid-size-{N}/       # N = actual grid size used
├── pytorch/                               # PyTorch neural networks
│   ├── bfs_mlp.pth
│   ├── astar_mlp.pth
│   └── ...
├── xgboost/                               # Tree-based models
│   ├── bfs_xgb.json
│   └── ...
└── distilled/                             # Distilled models
    └── ...
```

## Part 3: Supervised Learning Pipeline

### Step 1: Train Neural Network Models

```bash
# Create training script for all datasets
cat > scripts/train_supervised_models.py << 'EOF'
#!/usr/bin/env python3
"""
Supervised Learning Training Pipeline
===================================

Trains neural networks on heuristic datasets and evaluates performance.
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd

ALGORITHMS = ["bfs", "bfs-safe-greedy", "bfs-hamiltonian", "dfs", "astar", "astar-hamiltonian", "hamiltonian"]

def train_model(dataset_path: str, algorithm: str, model_type: str = "MLP"):
    """Train a single model."""
    print(f"\\n🧠 Training {model_type} on {algorithm} dataset")
    
    output_dir = f"logs/extensions/models/{algorithm}_{model_type.lower()}"
    
    cmd = [
        sys.executable, "-m", "extensions.supervised_v0_02.training.train_neural",
        "--dataset-paths", dataset_path,
        "--model", model_type,
        "--epochs", "50",
        "--batch-size", "32", 
        "--learning-rate", "0.001",
        "--output-dir", output_dir,
        "--validation-split", "0.15",
        "--test-split", "0.15"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✅ {algorithm} {model_type} training completed")
        # Extract performance metrics from output
        for line in result.stdout.split('\\n'):
            if 'Test accuracy:' in line:
                print(f"  📊 {line.strip()}")
    else:
        print(f"  ❌ {algorithm} {model_type} training failed:")
        print(f"  {result.stderr}")

def main():
    """Train models on all heuristic datasets."""
    print("🎯 Starting Supervised Learning Pipeline")
    
    dataset_dir = Path("logs/extensions/datasets/grid-size-10")
    
    for algorithm in ALGORITHMS:
        dataset_path = dataset_dir / f"{algorithm}_10k_tabular.csv"
        
        if dataset_path.exists():
            # Check dataset size
            df = pd.read_csv(dataset_path)
            print(f"\\n📈 {algorithm.upper()}: {len(df):,} training samples")
            
            # Train MLP model
            train_model(str(dataset_path), algorithm, "MLP")
            
        else:
            print(f"⚠️  Dataset not found: {dataset_path}")

if __name__ == "__main__":
    main()
EOF

# Execute supervised learning training
python scripts/train_supervised_models.py
```

### Step 2: Model Performance Analysis

```bash
# Create evaluation script
cat > scripts/evaluate_models.py << 'EOF'
#!/usr/bin/env python3
"""
Model Performance Evaluation
===========================

Evaluates trained models and generates performance report.
"""

import pandas as pd
from pathlib import Path
import json

def analyze_performance():
    """Analyze model performance across algorithms."""
    results = []
    
    models_dir = Path("logs/extensions/models")
    
    for model_dir in models_dir.glob("*_mlp"):
        algorithm = model_dir.name.replace("_mlp", "")
        
        # Look for evaluation results
        eval_file = model_dir / "evaluation_results.json"
        if eval_file.exists():
            with open(eval_file) as f:
                metrics = json.load(f)
                results.append({
                    "Algorithm": algorithm.upper(),
                    "Test_Accuracy": metrics.get("test_accuracy", 0),
                    "Training_Samples": metrics.get("training_samples", 0),
                    "Model_Size": metrics.get("model_parameters", 0)
                })
    
    # Create performance report
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Test_Accuracy", ascending=False)
        
        print("\\n📊 SUPERVISED LEARNING PERFORMANCE REPORT")
        print("=" * 60)
        print(df.to_string(index=False))
        
        # Save report
        df.to_csv("logs/extensions/supervised_learning_report.csv", index=False)
        print(f"\\n💾 Report saved to: logs/extensions/supervised_learning_report.csv")

if __name__ == "__main__":
    analyze_performance()
EOF

python scripts/evaluate_models.py
```

## Part 4: LLM Fine-tuning Pipeline

### Step 1: Prepare Fine-tuning Data

```bash
# Create LLM fine-tuning preparation script
cat > scripts/prepare_llm_finetuning.py << 'EOF'
#!/usr/bin/env python3
"""
LLM Fine-tuning Data Preparation
===============================

Combines JSONL datasets and prepares for fine-tuning.
"""

import json
from pathlib import Path

def combine_jsonl_datasets():
    """Combine all algorithm JSONL files for comprehensive training."""
    
    dataset_dir = Path("logs/extensions/datasets/grid-size-10")
    output_file = "logs/extensions/datasets/combined_heuristics_10k_per_algorithm.jsonl"
    
    combined_data = []
    
    for jsonl_file in dataset_dir.glob("*_language.jsonl"):
        algorithm = jsonl_file.stem.replace("_10k_language", "")
        
        print(f"📖 Processing {algorithm} dataset...")
        
        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line)
                # Add algorithm context
                data["algorithm"] = algorithm.upper()
                combined_data.append(data)
    
    # Shuffle and save
    import random
    random.shuffle(combined_data)
    
    with open(output_file, "w") as f:
        for item in combined_data:
            f.write(json.dumps(item) + "\\n")
    
    print(f"\\n✅ Combined dataset created: {output_file}")
    print(f"📊 Total training examples: {len(combined_data):,}")
    
    return output_file

def create_finetuning_config():
    """Create configuration for LLM fine-tuning."""
    
    config = {
        "model_name": "microsoft/DialoGPT-small",  # or your preferred base model
        "dataset_path": "logs/extensions/datasets/combined_heuristics_10k_per_algorithm.jsonl",
        "output_dir": "logs/extensions/models/finetuned_heuristics_llm",
        "num_epochs": 3,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "warmup_steps": 500,
        "max_length": 512,
        "task_type": "snake_game_reasoning"
    }
    
    with open("logs/extensions/llm_finetuning_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("🔧 Fine-tuning config saved to: logs/extensions/llm_finetuning_config.json")

def main():
    combine_jsonl_datasets()
    create_finetuning_config()

if __name__ == "__main__":
    main()
EOF

python scripts/prepare_llm_finetuning.py
```

### Step 2: Execute LLM Fine-tuning

```bash
# Use the heuristics-llm-fine-tuning-integration-v0.03 extension
python -m extensions.heuristics-llm-fine-tuning-integration-v0.03.cli \\
    --dataset logs/extensions/datasets/combined_heuristics_10k_per_algorithm.jsonl \\
    --output-dir logs/extensions/models/finetuned_heuristics_llm \\
    --epochs 3 \\
    --batch-size 8
```

## Part 5: Model Distillation Pipeline

### Step 1: Teacher-Student Distillation

```bash
# Use the distillation extension
python -m extensions.distillation-v0.01.distil \\
    --teacher-model logs/extensions/models/hamiltonian_mlp \\
    --student-model logs/extensions/models/bfs_mlp \\
    --distillation-datasets logs/extensions/datasets/grid-size-10/ \\
    --output logs/extensions/models/distilled_models \\
    --temperature 3.0 \\
    --alpha 0.7
```

## Part 6: Performance Analysis and Reporting

### Create Comprehensive Analysis

```bash
cat > scripts/generate_final_report.py << 'EOF'
#!/usr/bin/env python3
"""
Final Performance Analysis Report
===============================

Generates comprehensive analysis of the entire pipeline.
"""

import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_heuristic_performance():
    """Analyze raw heuristic algorithm performance."""
    
    results = []
    datasets_dir = Path("logs/extensions/datasets/grid-size-10")
    
    for csv_file in datasets_dir.glob("*_tabular.csv"):
        algorithm = csv_file.stem.replace("_10k_tabular", "")
        
        # Load and analyze dataset
        df = pd.read_csv(csv_file)
        
        # Calculate success metrics (if game outcome data available)
        if "game_won" in df.columns:
            win_rate = df["game_won"].mean()
            avg_score = df["final_score"].mean() if "final_score" in df.columns else 0
            avg_steps = df["step_count"].mean() if "step_count" in df.columns else 0
            
            results.append({
                "Algorithm": algorithm.upper(),
                "Games": len(df),
                "Win_Rate": f"{win_rate:.1%}",
                "Avg_Score": f"{avg_score:.1f}",
                "Avg_Steps": f"{avg_steps:.1f}"
            })
    
    return pd.DataFrame(results)

def create_performance_dashboard():
    """Create performance visualization dashboard."""
    
    # Heuristic Performance
    heuristic_df = analyze_heuristic_performance()
    
    if not heuristic_df.empty:
        print("\\n🎯 HEURISTIC ALGORITHM PERFORMANCE")
        print("=" * 70)
        print(heuristic_df.to_string(index=False))
    
    # Supervised Learning Performance (if available)
    supervised_report = Path("logs/extensions/supervised_learning_report.csv")
    if supervised_report.exists():
        supervised_df = pd.read_csv(supervised_report)
        print("\\n🧠 SUPERVISED LEARNING PERFORMANCE")
        print("=" * 70)
        print(supervised_df.to_string(index=False))
    
    # Dataset Statistics
    print("\\n📊 DATASET STATISTICS")
    print("=" * 70)
    
    datasets_dir = Path("logs/extensions/datasets/grid-size-10")
    total_csv_samples = 0
    total_jsonl_samples = 0
    
    for csv_file in datasets_dir.glob("*_tabular.csv"):
        df = pd.read_csv(csv_file)
        total_csv_samples += len(df)
    
    for jsonl_file in datasets_dir.glob("*_language.jsonl"):
        with open(jsonl_file) as f:
            total_jsonl_samples += sum(1 for _ in f)
    
    print(f"Total CSV training samples: {total_csv_samples:,}")
    print(f"Total JSONL training samples: {total_jsonl_samples:,}")
    print(f"Storage size: ~{(total_csv_samples + total_jsonl_samples) * 0.5 / 1000:.1f} MB")
    
    # Model Statistics
    models_dir = Path("logs/extensions/models")
    if models_dir.exists():
        model_count = len(list(models_dir.glob("*")))
        print(f"Trained models: {model_count}")

def main():
    """Generate final comprehensive report."""
    print("📋 LARGE-SCALE HEURISTICS TO ML PIPELINE - FINAL REPORT")
    print("=" * 80)
    
    create_performance_dashboard()
    
    print("\\n✅ PIPELINE SUMMARY:")
    print("  • 7 heuristic algorithms × 10,000 games = 70,000 game sessions")
    print("  • CSV datasets for supervised learning")
    print("  • JSONL datasets for LLM fine-tuning")
    print("  • Trained neural network models")
    print("  • Fine-tuned language models")
    print("  • Knowledge distillation experiments")
    
    print("\\n🎯 NEXT STEPS:")
    print("  • Deploy models for live game playing")
    print("  • Compare performance against original LLM")
    print("  • Experiment with ensemble methods")
    print("  • Scale to larger grid sizes")

if __name__ == "__main__":
    main()
EOF

python scripts/generate_final_report.py
```

## Expected Results

### Dataset Sizes
- **Total Games**: ~70,000 games (10K per algorithm)
- **CSV Files**: ~35MB per algorithm (245MB total)
- **JSONL Files**: ~50MB per algorithm (350MB total)
- **Total Storage**: ~600MB for all datasets

### Model Performance Expectations
- **BFS Models**: 85-90% accuracy (optimal pathfinding)
- **Hamiltonian Models**: 95%+ accuracy (guaranteed safety)
- **ASTAR Models**: 90-95% accuracy (informed search)
- **Fine-tuned LLMs**: Improved reasoning about game states

### Training Times
- **Heuristic Generation**: 4-6 hours total
- **Supervised Training**: 2-3 hours total
- **LLM Fine-tuning**: 6-8 hours (depending on hardware)
- **Total Pipeline**: 12-17 hours

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch sizes or process algorithms sequentially
2. **Import Errors**: Ensure symbolic links are created correctly
3. **Disk Space**: Monitor available space (need ~10GB free)
4. **GPU Usage**: Fine-tuning benefits significantly from GPU acceleration

### Performance Optimization

```bash
# Monitor system resources during execution
htop  # or top on macOS

# Check disk usage
du -h logs/extensions/datasets/

# Monitor GPU usage (if available)
nvidia-smi  # Linux with NVIDIA GPU
```

## Conclusion

This pipeline demonstrates a complete machine learning workflow:

1. **Data Generation**: Large-scale synthetic data from proven algorithms
2. **Multi-format Datasets**: Both tabular (CSV) and text (JSONL) formats
3. **Model Training**: Supervised learning with performance evaluation
4. **Knowledge Transfer**: LLM fine-tuning and model distillation
5. **Comprehensive Analysis**: Performance monitoring and reporting

The resulting models can be used for:
- **Real-time Snake game playing**
- **Educational demonstrations** of different AI approaches
- **Research into** algorithm-to-neural-network knowledge transfer
- **Benchmarking** new approaches against established heuristics

Total expected output: **70,000 game sessions**, **multiple trained models**, and **comprehensive performance analysis**. 