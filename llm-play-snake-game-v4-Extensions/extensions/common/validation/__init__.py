import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

"""extensions.common.validation
================================
A *very* small collection of helper functions that perform lightweight sanity
checks on datasets and paths. The idea is to catch obvious issues while
remaining completely unopinionated – extensions are free to perform stricter
validation in their own codebases.

This package follows final-decision-10.md Guideline 3: lightweight, OOP-based common utilities 
with simple logging (print() statements) rather than complex *.log file mechanisms.

Design Philosophy:
- Simple, object-oriented utilities that can be inherited and extended
- No tight coupling with ML/DL/RL/LLM-specific concepts
- Simple logging with print() statements (final-decision-10.md Guideline 3)
- Enables easy addition of new extensions without friction
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict

# ----------------
# Core result data structure
# ----------------

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


# ----------------
# Re-export the two lightweight validators
# ----------------
from .dataset_validator import validate_dataset  # noqa: E402  (import after dataclass)
from .path_validator import validate_extension_path  # noqa: E402

__all__ = [
    "ValidationLevel",
    "ValidationResult",
    "validate_dataset",
    "validate_extension_path",
]
