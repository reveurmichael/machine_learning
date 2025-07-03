"""Lightweight path validator (`extensions.common.validation.path_validator`).

Checks only for the most obvious issues:
1. The path exists on disk.
2. The path is located somewhere inside an *extensions* directory (either the
   source tree or `logs/extensions`).

Anything more sophisticated is intentionally left to individual extensions.
"""
from __future__ import annotations


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


from pathlib import Path
from typing import Union

from . import ValidationResult, ValidationLevel

__all__ = ["validate_extension_path"]


def validate_extension_path(path: Union[str, Path]) -> ValidationResult:
    """Return a ValidationResult summarising whether *path* looks reasonable."""
    p = Path(path)

    if not p.exists():
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"Path does not exist: {p}",
        )

    # A very soft check that encourages (but does not strictly enforce) the
    # canonical directory structure described in the documentation.
    if "extensions" not in p.parts:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.WARNING,
            message="Path does not appear to belong to an extension directory.",
            details={"parts": p.parts},
        )

    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message="Path looks good.",
    ) 
