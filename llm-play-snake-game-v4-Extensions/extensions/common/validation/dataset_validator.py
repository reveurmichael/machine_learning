"""Lightweight dataset validator (`extensions.common.validation.dataset_validator`).

This helper intentionally performs **only the most basic** sanity checks:

* CSV  : verifies that the required 16-feature columns exist.
* JSONL: verifies that each record contains a minimal set of keys.

Anything beyond that is the business of individual extensions.

This module follows final-decision-10.md Guideline 3: lightweight, OOP-based common utilities 
with simple logging (print() statements) rather than complex *.log file mechanisms.

Design Philosophy:
- Simple, object-oriented utilities that can be inherited and extended
- No tight coupling with ML/DL/RL/LLM-specific concepts
- Simple logging with print() statements (final-decision-10.md Guideline 3)
- Enables easy addition of new extensions without friction
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, List, Dict, Any
import json

import pandas as pd

from ..config.dataset_formats import CSV_BASIC_COLUMNS, JSONL_BASIC_KEYS
from . import ValidationResult, ValidationLevel

__all__ = ["validate_dataset"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_csv(path: Path) -> ValidationResult:
    """Validate CSV dataset with basic column checks.
    
    Follows final-decision-10.md Guideline 3: Simple logging with print() statements.
    """
    print(f"[DatasetValidator] Validating CSV: {path}")  # final-decision-10.md Guideline 3: Simple logging
    
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[DatasetValidator] CSV validation failed: {exc}")  # final-decision-10.md Guideline 3: Simple logging
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"Failed to read CSV: {exc}",
        )

    missing = set(CSV_BASIC_COLUMNS) - set(df.columns)
    if missing:
        print(f"[DatasetValidator] CSV missing columns: {missing}")  # final-decision-10.md Guideline 3: Simple logging
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"CSV is missing required columns: {sorted(missing)}",
            details={"expected": CSV_BASIC_COLUMNS, "found": list(df.columns)},
        )

    print(f"[DatasetValidator] CSV validation passed: {len(df)} rows")  # final-decision-10.md Guideline 3: Simple logging
    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message="CSV dataset passed basic validation.",
        details={"rows": len(df)},
    )


def _validate_jsonl(path: Path) -> ValidationResult:
    """Validate JSONL dataset with basic key checks.
    
    Follows final-decision-10.md Guideline 3: Simple logging with print() statements.
    """
    print(f"[DatasetValidator] Validating JSONL: {path}")  # final-decision-10.md Guideline 3: Simple logging
    
    missing_keys: Dict[int, List[str]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not (line := line.strip()):
                    continue  # skip empty lines
                try:
                    obj: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[DatasetValidator] JSONL validation failed: line {idx} - {exc}")  # final-decision-10.md Guideline 3: Simple logging
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Invalid JSON on line {idx}: {exc}",
                    )
                missing = set(JSONL_BASIC_KEYS) - set(obj)
                if missing:
                    missing_keys[idx] = sorted(missing)
    except Exception as exc:
        print(f"[DatasetValidator] JSONL validation failed: {exc}")  # final-decision-10.md Guideline 3: Simple logging
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"Failed to read JSONL: {exc}",
        )

    if missing_keys:
        print(f"[DatasetValidator] JSONL missing keys: {missing_keys}")  # final-decision-10.md Guideline 3: Simple logging
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message="Some JSONL records are missing required keys.",
            details=missing_keys,
        )

    print(f"[DatasetValidator] JSONL validation passed")  # final-decision-10.md Guideline 3: Simple logging
    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message="JSONL dataset passed basic validation.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_dataset(file_path: Union[str, Path]) -> ValidationResult:
    """Validate a dataset file (CSV or JSONL).
    
    Follows final-decision-10.md Guideline 3: Simple logging with print() statements.
    """
    path = Path(file_path)
    print(f"[DatasetValidator] Starting validation: {path}")  # final-decision-10.md Guideline 3: Simple logging

    if not path.exists():
        print(f"[DatasetValidator] File not found: {path}")  # final-decision-10.md Guideline 3: Simple logging
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"Dataset file not found: {path}",
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _validate_csv(path)
    if suffix == ".jsonl":
        return _validate_jsonl(path)

    # Unknown or unsupported file type – accept by default.
    print(f"[DatasetValidator] Unknown file type, accepting by default: {suffix}")  # final-decision-10.md Guideline 3: Simple logging
    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message="No validation applied for this file type (accepted by default).",
    )