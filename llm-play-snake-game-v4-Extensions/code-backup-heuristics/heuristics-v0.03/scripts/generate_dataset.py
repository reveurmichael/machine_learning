#!/usr/bin/env python3
"""
Dataset Generation Script for Heuristics v0.03
--------------------

Generates training datasets from heuristic game logs for supervised learning models.
Supports multiple data formats (CSV, NPZ, Parquet) and structures (tabular, sequential, graph).

This script demonstrates the progression from v0.02 (pure heuristics) to v0.03 (data generation),
showing how heuristic algorithms can be used to generate high-quality training data for
supervised learning models.

Usage:
    # Generate tabular dataset for XGBoost/LightGBM
    python scripts/generate_dataset.py --data-structure tabular --data-format csv \
        --output datasets/tabular_heuristics.csv \
        --dataset-path ../../logs/extensions/heuristics-bfs_20250623_102805 \
        --dataset-path ../../logs/extensions/heuristics-astar_20250623_103000

    # Generate sequential dataset for neural networks
    python scripts/generate_dataset.py --data-structure sequential --data-format npz \
        --output datasets/sequential_heuristics.npz \
        --sequence-length 15 \
        --dataset-path ../../logs/extensions/heuristics-bfs_20250623_102805

    # Generate graph dataset for GNNs
    python scripts/generate_dataset.py --data-structure graph --data-format npz \
        --output datasets/graph_heuristics.npz \
        --dataset-path ../../logs/extensions/heuristics-hamiltonian_20250623_103500

Features:
- Multiple data structures: tabular, sequential, graph
- Multiple data formats: CSV, NPZ, Parquet
- Configurable sequence lengths for RNN/LSTM models
- Automatic feature engineering for each model type
- Comprehensive metadata preservation
- Extensive reuse of Task-0 utilities and common extensions
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))

from extensions.common import generate_training_dataset, get_dataset_path, ensure_datasets_dir, DEFAULT_GRID_SIZE, validate_grid_size


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for dataset generation.
    
    Design Pattern: Command Pattern
    - Encapsulates dataset generation requests as objects
    - Allows parameterization of different dataset types
    - Supports queuing and logging of generation operations
    """
    parser = argparse.ArgumentParser(
        description="Generate training datasets from heuristic game logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tabular dataset for tree-based models
  python scripts/generate_dataset.py --data-structure tabular --data-format csv \\
      --output datasets/xgboost_data.csv \\
      --dataset-path ../../logs/extensions/heuristics-bfs_20250623_102805

  # Sequential dataset for RNNs
  python scripts/generate_dataset.py --data-structure sequential --data-format npz \\
      --output datasets/rnn_data.npz --sequence-length 20 \\
      --dataset-path ../../logs/extensions/heuristics-astar_20250623_103000

  # Graph dataset for GNNs
  python scripts/generate_dataset.py --data-structure graph --data-format npz \\
      --output datasets/gnn_data.npz \\
      --dataset-path ../../logs/extensions/heuristics-hamiltonian_20250623_103500
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--data-structure",
        choices=["tabular", "sequential", "graph"],
        required=True,
        help="Data structure for the dataset (tabular: XGBoost/LightGBM, sequential: RNN/LSTM/CNN, graph: GNN)"
    )
    
    parser.add_argument(
        "--data-format",
        choices=["csv", "npz", "parquet"],
        required=True,
        help="Output format for the dataset (csv: human-readable, npz: numpy arrays, parquet: efficient storage)"
    )
    
    parser.add_argument(
        "--output",
        required=False,
        help="Output path for the generated dataset (optional, auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--dataset-path",
        action="append",
        required=True,
        help="Path to heuristic log directory (can be specified multiple times)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Sequence length for sequential data structure (default: 10)"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Grid size for the dataset (default: {DEFAULT_GRID_SIZE})"
    )
    
    parser.add_argument(
        "--algorithm",
        default="mixed",
        help="Algorithm name for dataset filename (default: mixed)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually generating the dataset"
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if arguments are valid, False otherwise
    """
    # Validate grid size
    if not validate_grid_size(args.grid_size):
        from extensions.common import SUPPORTED_GRID_SIZES
        print(f"❌ Unsupported grid size: {args.grid_size}")
        print(f"Supported sizes: {SUPPORTED_GRID_SIZES}")
        return False
    
    # Auto-generate output path if not provided
    if not args.output:
        args.output = str(get_dataset_path(
            args.data_structure, 
            args.data_format, 
            args.algorithm,
            args.grid_size
        ))
        print(f"📁 Auto-generated output path: {args.output}")
    
    # Check if dataset paths exist
    for dataset_path in args.dataset_path:
        path = Path(dataset_path)
        if not path.exists():
            print(f"❌ Dataset path does not exist: {dataset_path}")
            return False
        
        # Check if it contains game JSON files
        game_files = list(path.glob("game_*.json"))
        if not game_files:
            print(f"❌ No game_*.json files found in: {dataset_path}")
            return False
        
        if args.verbose:
            print(f"✅ Found {len(game_files)} game files in: {dataset_path}")
    
    # Ensure output directory exists
    output_path = Path(args.output)
    if not output_path.parent.exists():
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if args.verbose:
                print(f"✅ Created output directory: {output_path.parent}")
        except Exception as e:
            print(f"❌ Cannot create output directory: {e}")
            return False
    
    # Validate data structure and format combinations
    if args.data_structure == "tabular" and args.data_format not in ["csv", "parquet"]:
        print("⚠️  Warning: Tabular data is typically saved as CSV or Parquet")
    
    if args.data_structure in ["sequential", "graph"] and args.data_format == "csv":
        print("⚠️  Warning: Sequential/Graph data is typically saved as NPZ")
    
    return True


def print_dataset_info(args: argparse.Namespace) -> None:
    """
    Print information about the dataset generation task.
    
    Args:
        args: Parsed command line arguments
    """
    print("🎯 Dataset Generation Configuration:")
    print(f"   Data Structure: {args.data_structure}")
    print(f"   Data Format: {args.data_format}")
    print(f"   Grid Size: {args.grid_size}x{args.grid_size}")
    print(f"   Algorithm Filter: {args.algorithm}")
    print(f"   Output Path: {args.output}")
    print(f"   Input Directories: {len(args.dataset_path)}")
    
    for i, path in enumerate(args.dataset_path, 1):
        print(f"     {i}. {path}")
    
    if args.data_structure == "sequential":
        print(f"   Sequence Length: {args.sequence_length}")
    
    print()


def main() -> None:
    """
    Main entry point for dataset generation.
    
    Demonstrates the evolution from heuristics-v0.02 (pure algorithms) to 
    heuristics-v0.03 (data generation for supervised learning).
    """
    print("🚀 Heuristics v0.03 - Dataset Generation")
    print("=" * 50)
    
    # Parse and validate arguments
    args = parse_arguments()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    # Print configuration
    print_dataset_info(args)
    
    # Dry run mode
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be generated")
        print("✅ All validations passed!")
        return
    
    try:
        # Ensure dataset directory exists
        ensure_datasets_dir(args.grid_size)
        
        # Generate dataset using common utilities
        kwargs = {}
        if args.data_structure == "sequential":
            kwargs["sequence_length"] = args.sequence_length
        
        # Add grid size to kwargs for feature extraction
        kwargs["grid_size"] = args.grid_size
        
        generate_training_dataset(
            input_dirs=args.dataset_path,
            output_path=args.output,
            data_structure=args.data_structure,
            data_format=args.data_format,
            **kwargs
        )
        
        print("🎉 Dataset generation completed successfully!")
        
        # Print next steps
        print("\n📋 Next Steps:")
        print("1. Use the generated dataset to train supervised learning models")
        print("2. Experiment with different algorithms (XGBoost, LightGBM, Neural Networks)")
        print("3. Compare performance against the original heuristic algorithms")
        print("4. Consider ensemble methods combining multiple approaches")
        
        if args.data_structure == "tabular":
            print("\n💡 Suggested Models for Tabular Data:")
            print("   - XGBoost: High performance, interpretable")
            print("   - LightGBM: Fast training, memory efficient")
            print("   - Random Forest: Robust, handles missing values")
            
        elif args.data_structure == "sequential":
            print("\n💡 Suggested Models for Sequential Data:")
            print("   - LSTM: Good for long sequences")
            print("   - GRU: Faster than LSTM, similar performance")
            print("   - CNN: Efficient for pattern recognition")
            
        elif args.data_structure == "graph":
            print("\n💡 Suggested Models for Graph Data:")
            print("   - GCN: Graph Convolutional Networks")
            print("   - GraphSAGE: Scalable graph neural networks")
            print("   - GAT: Graph Attention Networks")
        
    except Exception as e:
        print(f"❌ Error during dataset generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 