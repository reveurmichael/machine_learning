"""Lightweight path validator (`extensions.common.validation.path_validator`).

Guarantees that a given path exists and looks like an extension directory or a
sub-directory inside *logs/extensions*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from . import ValidationResult, ValidationLevel

__all__ = ["validate_extension_path"]


def validate_extension_path(path: Union[str, Path]) -> ValidationResult:
    """Very small helper to spot the most obvious path issues."""
    p = Path(path)

    if not p.exists():
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"Path does not exist: {p}",
        )

    # Heuristic: an extension directory always lives somewhere inside the
    # `extensions` folder or in the `logs/extensions` hierarchy.  This keeps the
    # rule flexible while still catching typos like `extenstions/`.
    if "extensions" not in p.parts:
        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.WARNING,
            message="Path does not seem to belong to an extension directory.",
            details={"path_parts": p.parts},
        )

    return ValidationResult(
        is_valid=True,
        level=ValidationLevel.INFO,
        message="Path looks good.",
    ) 