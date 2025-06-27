"""extensions.common.validation
================================
A *very* small collection of helper functions that perform lightweight sanity
checks on datasets and paths. The idea is to catch obvious issues while
remaining completely unopinionated – extensions are free to perform stricter
validation in their own codebases.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict

# ---------------------------------------------------------------------------
# Core result data structure
# ---------------------------------------------------------------------------

class ValidationLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ValidationResult:
    is_valid: bool
    message: str
    level: ValidationLevel = ValidationLevel.INFO
    details: Optional[Dict] = None

    def __bool__(self) -> bool:  # pragma: no cover – syntactic sugar only
        return self.is_valid


# ---------------------------------------------------------------------------
# Re-export the two lightweight validators
# ---------------------------------------------------------------------------
from .dataset_validator import validate_dataset  # noqa: E402  (import after dataclass)
from .path_validator import validate_extension_path  # noqa: E402

__all__ = [
    "ValidationLevel",
    "ValidationResult",
    "validate_dataset",
    "validate_extension_path",
]
