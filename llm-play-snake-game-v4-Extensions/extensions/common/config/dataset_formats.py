import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

"""
Dataset Format Specifications for Snake Game AI Extensions.

This module defines the standardized data format specifications used
across all extensions, following the data format decision guide.

Design Philosophy:
- Forward-looking: No legacy compatibility, clean and self-contained
- Consistent schema across all extensions
- Format-specific optimizations for different use cases
- Educational clarity and documentation

Reference: docs/extensions-guideline/data-format-decision-guide.md
"""

from typing import List, Dict, Any

# =============================================================================
# NPZ Format Specifications
# =============================================================================

# Sequential NPZ arrays (for RNN/RL)
NPZ_SEQUENTIAL_ARRAYS: List[str] = [
    'states',        # Shape: (timesteps, features)
    'actions',       # Shape: (timesteps,)
    'rewards',       # Shape: (timesteps,)
    'next_states',   # Shape: (timesteps, features)
    'dones'          # Shape: (timesteps,) - episode termination flags
]

# Spatial NPZ arrays (for CNN)
NPZ_SPATIAL_ARRAYS: List[str] = [
    'boards',        # Shape: (samples, height, width, channels)
    'targets',       # Shape: (samples,) - move indices
    'metadata'       # Shape: (samples, metadata_dim)
]

# Raw NPZ arrays (for evolutionary algorithms)
NPZ_RAW_ARRAYS: List[str] = [
    'population',           # Shape: (population_size, individual_length)
    'fitness_scores',       # Shape: (population_size, num_objectives)
    'generation_history',   # Shape: (num_generations, population_size, individual_length)
    'selection_pressure',   # Shape: (num_generations,)
    'diversity_metrics'     # Shape: (num_generations,)
]

# =============================================================================
# Common Dataset File Extensions
# =============================================================================

COMMON_DATASET_EXTENSIONS: Dict[str, List[str]] = {
    "csv": [".csv"],
    "jsonl": [".jsonl"],
    "npz": [".npz"],
    "json": [".json"]
}

# =============================================================================
# Format Selection Guidelines
# =============================================================================

FORMAT_USE_CASES: Dict[str, Dict[str, Any]] = {
    "csv": {
        "best_for": ["Tree models", "Simple MLPs", "Traditional ML"],
        "algorithms": ["XGBoost", "LightGBM", "Random Forest", "SVM"],
        "grid_size_support": "Universal",
        "pros": ["Fast loading", "Human readable", "Small size"],
        "cons": ["Limited to tabular data", "No sequence information"]
    },
    "jsonl": {
        "best_for": ["LLM fine-tuning", "Language models"],
        "algorithms": ["GPT", "Claude", "LLaMA", "T5"],
        "grid_size_support": "Universal", 
        "pros": ["Rich explanations", "Human readable", "Flexible"],
        "cons": ["Large file size", "Requires processing"]
    },
    "npz": {
        "best_for": ["Deep learning", "Sequential models", "Spatial models"],
        "algorithms": ["CNN", "RNN", "LSTM", "RL agents"],
        "grid_size_support": "Universal",
        "pros": ["Efficient storage", "Native numpy", "Multiple arrays"],
        "cons": ["Binary format", "Requires numpy"]
    }
}

# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # NPZ format specifications
    "NPZ_SEQUENTIAL_ARRAYS",
    "NPZ_SPATIAL_ARRAYS", 
    "NPZ_RAW_ARRAYS",
    
    # Common extensions
    "COMMON_DATASET_EXTENSIONS",
    
    # Format specifications
    "FORMAT_USE_CASES",
]

