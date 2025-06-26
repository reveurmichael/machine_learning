"""
Extension Validator for Snake Game AI Extensions.

This module provides comprehensive validation for entire extensions,
combining all validation types into a unified compliance checking system.

Design Patterns:
- Facade Pattern: Unified interface for all validation types
- Composite Pattern: Hierarchical validation structure
- Observer Pattern: Progress monitoring for complex validations

Educational Value:
Demonstrates how to build comprehensive validation systems that combine
multiple validation strategies into a cohesive compliance framework.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field

# Import other validators
from .dataset_validator import DatasetValidatorFactory, validate_dataset_format
from .model_validator import ModelValidatorFactory, validate_model_output
from .path_validator import PathValidatorFactory, validate_path_structure
from .config_validator import ExtensionComplianceValidator, validate_config_access

# Validation result classes
from .validation_types import ValidationResult, ValidationLevel, ValidationReport

# =============================================================================
# Extension Validator
# =============================================================================

@dataclass
class ExtensionValidationContext:
    """Context information for extension validation."""
    extension_path: Path
    extension_type: str
    version: str
    grid_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExtensionValidator:
    """
    Comprehensive validator for Snake Game AI extensions.
    
    Design Pattern: Facade Pattern + Template Method Pattern
    Purpose: Provide unified interface for all extension validation
    
    Educational Note (SUPREME_RULE NO.4):
    This class demonstrates how to combine multiple validation strategies
    into a single, comprehensive validation system that can handle
    complex extension compliance requirements. The validator is designed
    to be extensible for specialized validation needs.
    
    SUPREME_RULE NO.4 Implementation:
    - Base validator provides complete functionality for most extensions
    - Protected methods allow selective validation customization
    - Virtual methods enable additional validation steps
    - Extension-specific validators can inherit and adapt
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ExtensionValidator")
        
        # Initialize component validators
        self.compliance_validator = ExtensionComplianceValidator()
        
        # Validation progress tracking
        self.validation_steps = [
            "Path Structure Validation",
            "Configuration Compliance",
            "Import Restrictions", 
            "Naming Conventions",
            "Directory Structure",
            "File Format Compliance",
            "Dataset Validation",
            "Model Validation"
        ]
        
        # Initialize validator-specific settings
        self._initialize_validator_settings()
        
    def validate_extension(
        self,
        extension_path: Union[str, Path],
        context: Optional[ExtensionValidationContext] = None
    ) -> ValidationReport:
        """
        Comprehensive extension validation.
        
        Args:
            extension_path: Path to extension directory
            context: Optional validation context with additional information
        
        Returns:
            ValidationReport with comprehensive validation results
        
        Educational Note:
        This method demonstrates how to orchestrate multiple validation
        types into a comprehensive compliance check with detailed reporting.
        """
        extension_path = Path(extension_path)
        
        if context is None:
            context = self._create_context(extension_path)
        
        self.logger.info(f"Starting comprehensive validation for {context.extension_type}")
        
        # Collect all validation results
        results = []
        
        try:
            # 1. Path Structure Validation
            results.append(self._validate_path_structure(context))
            
            # 2. Configuration Compliance
            results.append(self._validate_configuration_compliance(context))
            
            # 3. Import Restrictions
            results.append(self._validate_import_restrictions(context))
            
            # 4. Naming Conventions
            results.append(self._validate_naming_conventions(context))
            
            # 5. Directory Structure Compliance
            results.append(self._validate_directory_structure(context))
            
            # 6. File Format Compliance
            results.append(self._validate_file_formats(context))
            
            # 7. Dataset Validation (if applicable)
            dataset_result = self._validate_datasets(context)
            if dataset_result:
                results.append(dataset_result)
            
            # 8. Model Validation (if applicable)
            model_result = self._validate_models(context)
            if model_result:
                results.append(model_result)
            
            # 9. Extension-specific validation (SUPREME_RULE NO.4)
            extension_specific_results = self._validate_extension_specific(context)
            results.extend(extension_specific_results)
            
            # Determine overall validation status
            overall_valid = all(result.is_valid for result in results)
            
            return ValidationReport(
                results=results,
                overall_valid=overall_valid
            )
            
        except Exception as e:
            error_result = ValidationResult(
                is_valid=False,
                level=ValidationLevel.CRITICAL,
                message=f"Extension validation failed: {str(e)}",
                details={"extension_path": str(extension_path), "error": str(e)}
            )
            
            return ValidationReport(
                results=[error_result],
                overall_valid=False
            )
    
    def _create_context(self, extension_path: Path) -> ExtensionValidationContext:
        """Create validation context from extension path."""
        # Parse extension information from path
        extension_name = extension_path.name
        
        # Extract extension type and version
        if '-v' in extension_name:
            extension_type, version_part = extension_name.split('-v', 1)
            version = f"v{version_part}"
        else:
            extension_type = extension_name
            version = "unknown"
        
        return ExtensionValidationContext(
            extension_path=extension_path,
            extension_type=extension_type,
            version=version
        )
    
    def _validate_path_structure(self, context: ExtensionValidationContext) -> ValidationResult:
        """Validate extension path structure."""
        try:
            return validate_path_structure(
                context.extension_path,
                "extension",
                {"extension_type": context.extension_type, "version": context.version}
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Path structure validation failed: {str(e)}"
            )
    
    def _validate_configuration_compliance(self, context: ExtensionValidationContext) -> ValidationResult:
        """Validate configuration access compliance."""
        try:
            return self.compliance_validator.validate_extension_compliance(context.extension_path)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Configuration compliance validation failed: {str(e)}"
            )
    
    def _validate_import_restrictions(self, context: ExtensionValidationContext) -> ValidationResult:
        """Validate import restrictions across all Python files."""
        try:
            python_files = list(context.extension_path.rglob("*.py"))
            
            violations = []
            for py_file in python_files:
                # Skip __pycache__ and other irrelevant files
                if "__pycache__" in str(py_file) or py_file.name.startswith('.'):
                    continue
                
                # Simple import validation (could be enhanced)
                if self._has_forbidden_imports(py_file, context.extension_type):
                    violations.append(str(py_file.relative_to(context.extension_path)))
            
            if violations:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Import restriction violations in files: {violations}",
                    suggestion="Review import statements and ensure compliance"
                )
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="Import restrictions validation passed"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Import restrictions validation failed: {str(e)}"
            )
    
    def _validate_naming_conventions(self, context: ExtensionValidationContext) -> ValidationResult:
        """Validate naming conventions for files and directories."""
        try:
            issues = []
            
            # Check agent file naming (if agents directory exists)
            agents_dir = context.extension_path / "agents"
            if agents_dir.exists():
                for agent_file in agents_dir.glob("*.py"):
                    if not agent_file.name.startswith("agent_") and agent_file.name != "__init__.py":
                        issues.append(f"Agent file doesn't follow naming convention: {agent_file.name}")
            
            # Check dashboard file naming (if dashboard directory exists)
            dashboard_dir = context.extension_path / "dashboard"
            if dashboard_dir.exists():
                for dashboard_file in dashboard_dir.glob("*.py"):
                    if not (dashboard_file.name.startswith("tab_") or dashboard_file.name == "__init__.py"):
                        issues.append(f"Dashboard file doesn't follow naming convention: {dashboard_file.name}")
            
            if issues:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Naming convention issues: {'; '.join(issues)}",
                    suggestion="Follow established naming conventions"
                )
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="Naming conventions validation passed"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Naming conventions validation failed: {str(e)}"
            )
    
    def _validate_directory_structure(self, context: ExtensionValidationContext) -> ValidationResult:
        """Validate directory structure based on version requirements."""
        try:
            issues = []
            
            # Version-specific requirements
            if context.version.startswith("v0."):
                version_parts = context.version[1:].split(".")
                if len(version_parts) >= 2:
                    major, minor = int(version_parts[0]), int(version_parts[1])
                    
                    # v0.02+ requires agents directory
                    if major == 0 and minor >= 2:
                        if not (context.extension_path / "agents").exists():
                            issues.append("v0.02+ extensions must have 'agents' directory")
                    
                    # v0.03+ requires dashboard and scripts directories
                    if major == 0 and minor >= 3:
                        required_dirs = ["dashboard", "scripts"]
                        for req_dir in required_dirs:
                            if not (context.extension_path / req_dir).exists():
                                issues.append(f"v0.03+ extensions must have '{req_dir}' directory")
            
            if issues:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Directory structure issues: {'; '.join(issues)}",
                    suggestion="Create required directories for version compliance"
                )
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="Directory structure validation passed"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Directory structure validation failed: {str(e)}"
            )
    
    def _validate_file_formats(self, context: ExtensionValidationContext) -> ValidationResult:
        """Validate file formats and required files."""
        try:
            issues = []
            
            # Check for required core files
            core_files = ["game_logic.py", "game_manager.py"]
            for core_file in core_files:
                if not (context.extension_path / core_file).exists():
                    issues.append(f"Missing required core file: {core_file}")
            
            # Check for __init__.py files in package directories
            package_dirs = ["agents", "dashboard", "scripts"]
            for pkg_dir in package_dirs:
                pkg_path = context.extension_path / pkg_dir
                if pkg_path.exists() and pkg_path.is_dir():
                    if not (pkg_path / "__init__.py").exists():
                        issues.append(f"Missing __init__.py in {pkg_dir} directory")
            
            if issues:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"File format issues: {'; '.join(issues)}",
                    suggestion="Ensure all required files are present"
                )
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="File format validation passed"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"File format validation failed: {str(e)}"
            )
    
    def _validate_datasets(self, context: ExtensionValidationContext) -> Optional[ValidationResult]:
        """Validate datasets if present."""
        # Look for dataset directories
        logs_dir = context.extension_path.parent.parent / "logs" / "extensions" / "datasets"
        
        if not logs_dir.exists():
            return None  # No datasets to validate
        
        # Find dataset directories matching this extension
        pattern = f"{context.extension_type}_*"
        dataset_dirs = list(logs_dir.glob(f"*/grid-size-*/{pattern}"))
        
        if not dataset_dirs:
            return None  # No datasets found
        
        try:
            issues = []
            for dataset_dir in dataset_dirs[:3]:  # Limit to first 3 for performance
                # Look for data files
                data_files = list(dataset_dir.rglob("*.csv")) + list(dataset_dir.rglob("*.jsonl")) + list(dataset_dir.rglob("*.npz"))
                
                for data_file in data_files[:2]:  # Limit files per directory
                    result = validate_dataset_format(data_file)
                    if not result.is_valid:
                        issues.append(f"Dataset validation failed for {data_file.name}: {result.message}")
            
            if issues:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Dataset validation issues: {'; '.join(issues[:3])}...",
                    suggestion="Review dataset quality and format"
                )
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"Dataset validation passed for {len(dataset_dirs)} dataset directories"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Dataset validation failed: {str(e)}"
            )
    
    def _validate_models(self, context: ExtensionValidationContext) -> Optional[ValidationResult]:
        """Validate models if present."""
        # Look for model directories
        logs_dir = context.extension_path.parent.parent / "logs" / "extensions" / "models"
        
        if not logs_dir.exists():
            return None  # No models to validate
        
        # Find model directories matching this extension
        pattern = f"{context.extension_type}_*"
        model_dirs = list(logs_dir.glob(f"*/grid-size-*/{pattern}"))
        
        if not model_dirs:
            return None  # No models found
        
        try:
            issues = []
            for model_dir in model_dirs[:3]:  # Limit to first 3 for performance
                # Look for model files
                model_files = (
                    list(model_dir.rglob("*.pth")) +
                    list(model_dir.rglob("*.pkl")) +
                    list(model_dir.rglob("*.onnx"))
                )
                
                for model_file in model_files[:2]:  # Limit files per directory
                    result = validate_model_output(model_file)
                    if not result.is_valid:
                        issues.append(f"Model validation failed for {model_file.name}: {result.message}")
            
            if issues:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Model validation issues: {'; '.join(issues[:3])}...",
                    suggestion="Review model format and quality"
                )
            
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"Model validation passed for {len(model_dirs)} model directories"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Model validation failed: {str(e)}"
            )
    
    def _has_forbidden_imports(self, file_path: Path, extension_type: str) -> bool:
        """Check if file has forbidden imports for extension type."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple check for forbidden LLM imports
            if extension_type not in ["agentic-llms", "llm", "vision-language-model"]:
                forbidden_patterns = [
                    "from config.llm_constants",
                    "import config.llm_constants",
                    "from config.prompt_templates",
                    "import config.prompt_templates"
                ]
                
                for pattern in forbidden_patterns:
                    if pattern in content:
                        return True
            
            return False
            
        except Exception:
            return False  # Can't read file, assume no violations
    
    def _initialize_validator_settings(self) -> None:
        """
        Initialize validator-specific settings (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to set up extension-specific
        validation configurations, custom validators, or specialized checks.
        
        Example:
            class RLExtensionValidator(ExtensionValidator):
                def _initialize_validator_settings(self):
                    self.rl_model_validator = RLModelValidator()
                    self.environment_validator = EnvironmentValidator()
        """
        pass
    
    def _validate_extension_specific(self, context: ExtensionValidationContext) -> List[ValidationResult]:
        """
        Perform extension-specific validation (SUPREME_RULE NO.4 Extension Point).
        
        Override this method in subclasses to add custom validation logic
        specific to particular extension types or requirements.
        
        Args:
            context: Extension validation context
            
        Returns:
            List of validation results from extension-specific checks
            
        Example:
            class EvolutionaryValidator(ExtensionValidator):
                def _validate_extension_specific(self, context):
                    results = []
                    
                    # Validate genetic algorithm parameters
                    if self._has_genetic_config(context):
                        results.append(self._validate_genetic_parameters(context))
                    
                    # Validate population management
                    results.append(self._validate_population_structure(context))
                    
                    return results
        """
        return []

# =============================================================================
# High-Level Validation Functions
# =============================================================================

def validate_extension_compliance(
    extension_path: Union[str, Path],
    context: Optional[ExtensionValidationContext] = None
) -> ValidationReport:
    """
    High-level function for comprehensive extension validation.
    
    Args:
        extension_path: Path to extension directory
        context: Optional validation context
    
    Returns:
        ValidationReport with comprehensive validation results
    
    Educational Note:
    This function provides a simple interface for comprehensive extension
    validation while hiding the complexity of the underlying validation system.
    """
    validator = ExtensionValidator()
    return validator.validate_extension(extension_path, context) 