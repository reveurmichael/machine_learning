"""
Common validation types and result classes.

This module defines the shared validation result types used across
all validation modules in the package.

Design Patterns:
- Value Object Pattern: Immutable validation results
- Enum Pattern: Well-defined validation levels

Educational Value:
Shows how to design consistent error reporting and validation
result structures that provide clear feedback to users.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None

@dataclass
class ValidationReport:
    """Complete validation report with multiple results."""
    results: List[ValidationResult]
    overall_valid: bool
    
    @property
    def errors(self) -> List[ValidationResult]:
        """Get all error-level results."""
        return [r for r in self.results if r.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]]
    
    @property
    def warnings(self) -> List[ValidationResult]:
        """Get all warning-level results."""
        return [r for r in self.results if r.level == ValidationLevel.WARNING]
    
    def __str__(self) -> str:
        """Human-readable validation report."""
        lines = [f"Validation Report: {'PASSED' if self.overall_valid else 'FAILED'}"]
        lines.append(f"Total checks: {len(self.results)}")
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  - {error.message}")
                if error.suggestion:
                    lines.append(f"    Suggestion: {error.suggestion}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning.message}")
        
        return "\n".join(lines) 