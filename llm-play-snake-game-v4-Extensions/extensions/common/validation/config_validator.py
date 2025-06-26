"""
Configuration Validation Utilities for Snake Game AI Extensions.

This module provides comprehensive configuration validation for import restrictions,
config access control, and compliance with extension guidelines.

Design Patterns:
- Chain of Responsibility: Sequential validation checks
- Command Pattern: Encapsulated validation commands
- Observer Pattern: Configuration monitoring and compliance

Educational Value:
Demonstrates how to build robust configuration validation systems that enforce
architectural boundaries and prevent inappropriate dependencies.
"""

from typing import Dict, List, Any, Optional, Union, Set
import re
import ast
import logging
from pathlib import Path

# Import configuration constants
from ..config.validation_rules import (
    LLM_WHITELIST_EXTENSIONS, FORBIDDEN_IMPORT_PATTERNS, CONFIG_ACCESS_RULES
)

# Validation result classes
from .validation_types import ValidationResult, ValidationLevel

# =============================================================================
# Configuration Access Validator
# =============================================================================

class ConfigValidator:
    """
    Validator for configuration access compliance.
    
    Design Pattern: Strategy Pattern
    Purpose: Enforce configuration access rules and prevent pollution
    
    Educational Note:
    Shows how to implement access control validation that prevents
    inappropriate cross-domain dependencies in a modular system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ConfigValidator")
    
    def validate_config_access(
        self,
        extension_type: str,
        imported_modules: List[str]
    ) -> ValidationResult:
        """
        Validate that extension only imports allowed configuration modules.
        
        Args:
            extension_type: Type of extension (e.g., 'heuristics', 'supervised')
            imported_modules: List of imported module names
        
        Returns:
            ValidationResult with access validation outcome
        
        Educational Note:
        This function enforces the LLM constants whitelist to prevent
        Task-0 specific configurations from polluting general extensions.
        """
        violations = []
        
        # Check LLM constants access
        llm_imports = [
            module for module in imported_modules
            if module.startswith(('config.llm_constants', 'config.prompt_templates'))
        ]
        
        if llm_imports:
            # Check if extension is allowed to import LLM constants
            if not self._is_llm_whitelist_extension(extension_type):
                violations.append({
                    "type": "forbidden_llm_import",
                    "modules": llm_imports,
                    "message": f"Extension '{extension_type}' cannot import LLM-specific constants"
                })
        
        # Check for other forbidden patterns
        for module in imported_modules:
            for pattern_name, pattern in FORBIDDEN_IMPORT_PATTERNS.items():
                if re.search(pattern, module):
                    violations.append({
                        "type": "forbidden_pattern",
                        "module": module,
                        "pattern": pattern_name,
                        "message": f"Import '{module}' matches forbidden pattern '{pattern_name}'"
                    })
        
        if violations:
            violation_messages = [v["message"] for v in violations]
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Configuration access violations: {'; '.join(violation_messages)}",
                details={"violations": violations},
                suggestion=self._get_access_suggestion(extension_type, violations)
            )
        
        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Configuration access validation passed"
        )
    
    def _is_llm_whitelist_extension(self, extension_type: str) -> bool:
        """Check if extension type is whitelisted for LLM constants access."""
        # Check if extension name starts with any whitelisted prefix
        for prefix in LLM_WHITELIST_EXTENSIONS:
            if extension_type.startswith(prefix):
                return True
        return False
    
    def _get_access_suggestion(
        self,
        extension_type: str,
        violations: List[Dict[str, Any]]
    ) -> str:
        """Generate helpful suggestion for resolving access violations."""
        suggestions = []
        
        for violation in violations:
            if violation["type"] == "forbidden_llm_import":
                suggestions.append(
                    "Use extensions.common.config for ML-specific constants instead of LLM constants"
                )
            elif violation["type"] == "forbidden_pattern":
                suggestions.append(
                    f"Avoid importing {violation['pattern']} - use alternative approach"
                )
        
        return "; ".join(suggestions)

# =============================================================================
# Import Validator
# =============================================================================

class ImportValidator:
    """
    Validator for import statement compliance.
    
    Design Pattern: Chain of Responsibility
    Purpose: Sequential validation of different import restrictions
    
    Educational Note:
    Demonstrates how to parse and validate import statements to enforce
    architectural boundaries and prevent inappropriate dependencies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ImportValidator")
    
    def validate_file_imports(self, file_path: Path) -> ValidationResult:
        """
        Validate all imports in a Python file.
        
        Args:
            file_path: Path to Python file to validate
        
        Returns:
            ValidationResult with import validation outcome
        """
        try:
            # Extract extension type from file path
            extension_type = self._extract_extension_type(file_path)
            
            # Parse imports from file
            imports = self._extract_imports(file_path)
            
            # Validate imports
            return self.validate_imports(extension_type, imports)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.CRITICAL,
                message=f"Import validation failed: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            )
    
    def validate_imports(
        self,
        extension_type: str,
        imports: List[str]
    ) -> ValidationResult:
        """
        Validate import list for extension type.
        
        Args:
            extension_type: Type of extension
            imports: List of imported module names
        
        Returns:
            ValidationResult with validation outcome
        """
        config_validator = ConfigValidator()
        return config_validator.validate_config_access(extension_type, imports)
    
    def _extract_extension_type(self, file_path: Path) -> str:
        """Extract extension type from file path."""
        # Look for extension directory pattern
        path_parts = file_path.parts
        for part in path_parts:
            if '-v' in part:  # Extension directory pattern: algorithm-vX.XX
                extension_type = part.split('-v')[0]
                return extension_type
        
        # Default to 'unknown' if pattern not found
        return 'unknown'
    
    def _extract_imports(self, file_path: Path) -> List[str]:
        """Extract all import statements from Python file."""
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to extract imports
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        # Also add specific imports
                        for alias in node.names:
                            imports.append(f"{node.module}.{alias.name}")
            
        except Exception as e:
            self.logger.warning(f"Could not parse imports from {file_path}: {e}")
        
        return imports

# =============================================================================
# Extension Compliance Validator
# =============================================================================

class ExtensionComplianceValidator:
    """
    Comprehensive validator for extension compliance.
    
    Design Pattern: Facade Pattern
    Purpose: Provide unified interface for all extension validation
    
    Educational Note:
    Shows how to combine multiple validators into a comprehensive
    compliance checking system with clear reporting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ExtensionComplianceValidator")
        self.config_validator = ConfigValidator()
        self.import_validator = ImportValidator()
    
    def validate_extension_compliance(
        self,
        extension_path: Path
    ) -> ValidationResult:
        """
        Comprehensive validation of extension compliance.
        
        Args:
            extension_path: Path to extension directory
        
        Returns:
            ValidationResult with comprehensive compliance check
        """
        try:
            # Extract extension info
            extension_type = extension_path.name.split('-v')[0]
            
            # Collect all Python files
            python_files = list(extension_path.rglob("*.py"))
            
            # Validate each file
            all_violations = []
            for py_file in python_files:
                result = self.import_validator.validate_file_imports(py_file)
                if not result.is_valid:
                    all_violations.append({
                        "file": str(py_file.relative_to(extension_path)),
                        "violations": result.details.get("violations", []),
                        "message": result.message
                    })
            
            if all_violations:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Extension compliance violations found in {len(all_violations)} files",
                    details={"violations": all_violations},
                    suggestion="Fix import violations to ensure extension compliance"
                )
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"Extension compliance validation passed for {extension_type}"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.CRITICAL,
                message=f"Extension compliance validation failed: {str(e)}",
                details={"extension_path": str(extension_path), "error": str(e)}
            )

# =============================================================================
# High-Level Validation Functions
# =============================================================================

def validate_config_access(
    extension_type: str,
    imported_modules: List[str]
) -> ValidationResult:
    """
    High-level function to validate configuration access.
    
    Args:
        extension_type: Type of extension
        imported_modules: List of imported module names
    
    Returns:
        ValidationResult with validation outcome
    
    Educational Note:
    This function provides a simple interface for configuration access validation
    while hiding the complexity of the underlying validation logic.
    """
    validator = ConfigValidator()
    return validator.validate_config_access(extension_type, imported_modules)

def validate_import_restrictions(file_path: Union[str, Path]) -> ValidationResult:
    """
    Validate import restrictions for a Python file.
    
    Args:
        file_path: Path to Python file
    
    Returns:
        ValidationResult with validation outcome
    """
    file_path = Path(file_path)
    validator = ImportValidator()
    return validator.validate_file_imports(file_path)

def validate_extension_imports(extension_path: Union[str, Path]) -> ValidationResult:
    """
    Validate all imports in an extension directory.
    
    Args:
        extension_path: Path to extension directory
    
    Returns:
        ValidationResult with validation outcome
    """
    extension_path = Path(extension_path)
    validator = ExtensionComplianceValidator()
    return validator.validate_extension_compliance(extension_path) 