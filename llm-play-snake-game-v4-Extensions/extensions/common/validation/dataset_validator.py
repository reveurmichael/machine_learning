"""Lightweight dataset validator (`extensions.common.validation.dataset_validator`).

This helper intentionally performs **only the most basic** sanity checks:

* CSV  : verifies that the required 16-feature columns exist.
* JSONL: verifies that each record contains a minimal set of keys.

Anything beyond that is the business of individual extensions.
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
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"Failed to read CSV: {exc}",
        )

    missing = set(CSV_BASIC_COLUMNS) - set(df.columns)
    if missing:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"CSV is missing required columns: {sorted(missing)}",
            details={"expected": CSV_BASIC_COLUMNS, "found": list(df.columns)},
        )

    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message="CSV dataset passed basic validation.",
        details={"rows": len(df)},
    )


def _validate_jsonl(path: Path) -> ValidationResult:
    missing_keys: Dict[int, List[str]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not (line := line.strip()):
                    continue  # skip empty lines
                try:
                    obj: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as exc:
                    return ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Invalid JSON on line {idx}: {exc}",
                    )
                missing = set(JSONL_BASIC_KEYS) - set(obj)
                if missing:
                    missing_keys[idx] = sorted(missing)
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"Failed to read JSONL: {exc}",
        )

    if missing_keys:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message="Some JSONL records are missing required keys.",
            details=missing_keys,
        )

    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message="JSONL dataset passed basic validation.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_dataset(file_path: Union[str, Path]) -> ValidationResult:
    """Validate a dataset file (CSV or JSONL)."""
    path = Path(file_path)

    if not path.exists():
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
    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message="No validation applied for this file type (accepted by default).",
    )