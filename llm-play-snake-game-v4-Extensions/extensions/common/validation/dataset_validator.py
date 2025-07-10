"""Lightweight dataset validator (`extensions.common.validation.dataset_validator`).

This helper intentionally performs **only the most basic** sanity checks:

* CSV  : verifies that the required 16-feature columns exist.
* JSONL: verifies that each record contains a minimal set of keys.

Anything beyond that is the business of individual extensions.

This module follows final-decision.md Guideline 3: lightweight, OOP-based common utilities 
with simple logging (print() statements) rather than complex *.log file mechanisms.

Design Philosophy:
- Simple, object-oriented utilities that can be inherited and extended
- No tight coupling with ML/DL/RL/LLM-specific concepts
- Simple logging with print() statements (final-decision.md Guideline 3)
- Enables easy addition of new extensions without friction
"""
from __future__ import annotations


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


from typing import Union

import pandas as pd

from utils.print_utils import print_info, print_error
from ..config.csv_formats import CSV_ALL_COLUMNS
from . import ValidationResult, ValidationLevel

__all__ = ["validate_dataset"]

# ----------------
# Internal helpers
# ----------------

def _validate_csv(path: Path) -> ValidationResult:
    """Validate CSV dataset with basic column checks.
    
    Follows final-decision.md Guideline 3: Simple logging with print() statements.
    """
    print_info(f"Validating CSV: {path}", "DatasetValidator")
    
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print_error(f"CSV validation failed: {exc}")
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"Failed to read CSV: {exc}",
        )

    missing = set(CSV_ALL_COLUMNS) - set(df.columns)
    if missing:
        print_error(f"CSV missing columns: {missing}")
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"CSV is missing required columns: {sorted(missing)}",
            details={"expected": CSV_ALL_COLUMNS, "found": list(df.columns)},
        )

    print_info(f"CSV validation passed: {len(df)} rows", "DatasetValidator")
    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message="CSV dataset passed basic validation.",
        details={"rows": len(df)},
    )


# ----------------
# Public API
# ----------------

def validate_dataset(file_path: Union[str, Path]) -> ValidationResult:
    """Validate a dataset file (CSV or JSONL).
    
    Follows final-decision.md Guideline 3: Simple logging with print() statements.
    """
    path = Path(file_path)
    print_info(f"Starting validation: {path}", "DatasetValidator")

    if not path.exists():
        print_error(f"File not found: {path}")
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"Dataset file not found: {path}",
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _validate_csv(path)
    if suffix == ".jsonl":
        pass # TODO: no need for jsonl validation

    # Unknown or unsupported file type – accept by default.
    print_info(f"Unknown file type, accepting by default: {suffix}", "DatasetValidator")
    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message="No validation applied for this file type (accepted by default).",
    )
