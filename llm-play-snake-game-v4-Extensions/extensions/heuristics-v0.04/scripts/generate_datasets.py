#!/usr/bin/env python3
"""
Heuristics v0.04 Dataset Generation Script

This script provides v0.04 specific dataset generation by leveraging the
unified dataset generation utilities in the common folder. It supports both
CSV (v0.03 compatibility) and JSONL (v0.04 language-rich) formats.

v0.04 Features:
- JSONL format with rich natural language explanations
- CSV format for v0.03 compatibility
- Language-rich datasets for LLM fine-tuning
- Maintains all v0.03 functionality

Design Philosophy:
This script is a thin adapter that focuses on v0.04 specific needs while
delegating actual generation to the well-organized common utilities. It
demonstrates the standalone nature of v0.04 + common.

Usage Examples:
    # Generate JSONL datasets for all algorithms (v0.04 specialty)
    python generate_datasets.py --format jsonl --all-algorithms
    
    # Generate both CSV and JSONL for specific algorithm
    python generate_datasets.py --format both --algorithm BFS --max-games 100
    
    # Generate CSV for v0.03 compatibility
    python generate_datasets.py --format csv --all-algorithms
"""

import sys
import os
from pathlib import Path

# Add project root and common to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
common_dir = project_root / "extensions" / "common"

for path in [str(project_root), str(common_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Change to project root for consistent relative paths
os.chdir(str(project_root))

# Import and delegate to the unified CLI
try:
    from dataset_generator_cli import main as cli_main
    
    if __name__ == "__main__":
        print("Heuristics v0.04 Dataset Generator")
        print("=" * 40)
        print("Delegating to unified CLI in common folder...")
        print()
        
        # Simply delegate to the unified CLI
        cli_main()
        
except ImportError as e:
    print(f"Error: Could not import unified CLI: {e}")
    print("Make sure you're running from the project root and common utilities are available.")
    sys.exit(1)
except Exception as e:
    print(f"Dataset generation failed: {e}")
    sys.exit(1) 