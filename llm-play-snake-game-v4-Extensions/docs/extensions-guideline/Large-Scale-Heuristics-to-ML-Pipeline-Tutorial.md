# Large-Scale Heuristics to ML Pipeline Tutorial

## Overview

This tutorial demonstrates how to generate large-scale datasets from heuristic algorithms and use them for:
1. **Supervised Learning** (PyTorch neural networks, tree models)
2. **LLM Fine-tuning** (using generated text descriptions)
3. **Model Distillation** (knowledge transfer between models)

The pipeline generates **10,000 games per algorithm** across 7 different heuristic strategies, creating comprehensive datasets in multiple formats (JSON, CSV, JSONL).

TODO
TODO
TODO
TODO
TODO
TODO
TODO


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

## Part 2: Large-Scale Dataset Generation

### Step 1: Chdir() to the extensions folder with Python code

DON'T EVER FORGET TO CHDIR() TO THE EXTENSIONS FOLDER.

DON'T EVER DO THIS, EVEN THOUGH YOU MIGHT BE TEMPTED TO DO SO. DON'T DO THIS, BECAUSE IT'S SO BAD:
```bash
# Create symbolic links to fix import issues # DON'T EVER DO THIS.
cd extensions
ln -sf heuristics-v0.04 heuristics_v0_04 # DON'T EVER DO THIS.
ln -sf supervised-v0.02 supervised_v0_02 # DON'T EVER DO THIS.
```

INSTEAD, USE CHDIR() with Python codeWILL BE SO MUCH MUCH MUCH BETTER. CHECK common/path_utils.py for more details.

### Step 2: Generate Heuristic Game Data (10,000 games per algorithm) TODO: maybe 100000 games per heuristic algorithm? 1 million? or even more?


### Step 3: Execute Large-Scale Generation (10,000 games per heuristic algorithm). TODO: maybe 100000 games per heuristic algorithm? 1 million? or even more?

**Models will be saved to:**
```
logs/extensions/datasets/grid_size_{N}/       # N = actual grid size used
and then perhaps subfolders, subsubfolders, subsubsubfolders, etc. 

And then datasets in different formats. csv, jsonl, etc. we will also have game_N.json and summary.json files.

TODO: naming of model folders and model files to be discussed. The discussion is ongoing.


TODO: naming of datasets folders and dataset files to be discussed. The discussion is ongoing.
```

## Part 3: Supervised Learning Pipeline

### Step 1: Train Neural Network Models, with datasets generated in Step 2, by heuristics-v0.04 extension.


And then models in different formats are saved in the ./logs/extensions/models folder.  TODO: naming of model folders and subfolders  and subsubfolders, etc. and model files to be discussed. The discussion is ongoing.



## Part 4: LLM Fine-tuning Pipeline

### Step 1: Prepare Fine-tuning Data


### Step 2: Execute LLM Fine-tuning
either,
```bash
python -m extensions.heuristics-llm-fine-tuning-integration-v0.03.cli --dataset logs/extensions/datasets_path_for_heuristics_v0.04_generated_jsonl_files_or_files_or_folder_or_folders --output-dir logs/extensions/models/finetuned_heuristics_llm --epochs 3 --batch-size 8
```

TODO: or, write a new python script to do the fine-tuning.

TODO: Or, tell users to open a streamlit app, click on the right tab, adjust the parameters, and click on the "Fine-tune" button.

## Part 5: Model Distillation Pipeline

### Step 1: Teacher-Student Distillation

## Expected Results

### Dataset Sizes
- **Total Games**: ~70,000 games (10K per algorithm) (Or, maybe, much more? 1 million? or even more?)
- **CSV Files**: ~35MB per algorithm (245MB total)
- **JSONL Files**: ~50MB per algorithm (350MB total)
- **Total Storage**: ~600MB for all datasets

### Model Performance Expectations
- **BFS Models**: 85-90% accuracy (optimal pathfinding)
- **Hamiltonian Models**: 95%+ accuracy (guaranteed safety)
- **ASTAR Models**: 90-95% accuracy (informed search)
- **Fine-tuned LLMs**: Improved reasoning about game states

### Training Times
- **Supervised Training**: 2-3 hours total
- **LLM Fine-tuning**: 6-8 hours (depending on hardware)
- **Total Pipeline**: 12-17 hours

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch sizes or process algorithms sequentially
2. **Import Errors**: chdir() extensively, in the original code, and also in your newly created code for automating this task pipeline.
3. **GPU Usage**: Fine-tuning benefits significantly from GPU acceleration


## Conclusion

This pipeline demonstrates a complete machine learning workflow:

1. **Data Generation**: Large-scale synthetic data from proven algorithms
2. **Multi-format Datasets**: Both tabular (CSV) and text (JSONL) formats
3. **Model Training**: Supervised learning with performance evaluation
4. **Knowledge Transfer**: LLM fine-tuning and model distillation
