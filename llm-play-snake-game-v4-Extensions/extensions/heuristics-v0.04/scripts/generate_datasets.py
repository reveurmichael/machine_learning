"""
Heuristics v0.04 Dataset Generation Script

This script provides v0.04 specific dataset generation by leveraging the
unified dataset generation utilities in the common folder. It supports both
CSV (numerical features) and JSONL (v0.04 language-rich) formats.

v0.04 Features:
- JSONL format with rich natural language explanations
- CSV format for numerical features
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
    
            # Generate CSV for numerical features
    python generate_datasets.py --format csv --all-algorithms
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ---------------------------------------------------------------------------
# Ensure project root and common utilities are on sys.path BEFORE any imports
# ---------------------------------------------------------------------------

from utils.path_utils import ensure_project_root
ensure_project_root()

# Now that paths are set up we can safely import project modules
from utils.print_utils import print_info, print_error

# Import and delegate to the unified CLI
try:
    from extensions.common.utils.dataset_generator_cli import main as cli_main
    
    if __name__ == "__main__":
        print_info("Heuristics v0.04 Dataset Generator")
        print_info("=" * 40)
        print_info("Delegating to unified CLI in common folder...")
        print_info("")
        
        # Simply delegate to the unified CLI
        cli_main()
        
except ImportError as e:
    print_error(f"Error: Could not import unified CLI: {e}")
    print_error("Make sure you're running from the project root and common utilities are available.")
    sys.exit(1)
except Exception as e:
    print_error(f"Dataset generation failed: {e}")
    sys.exit(1) 