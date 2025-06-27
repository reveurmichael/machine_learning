"""
Extension Utility Functions for Snake Game AI

This module provides common utilities that all extensions need, following the principle
of DRY (Don't Repeat Yourself) while maintaining the standalone nature of extensions.

Design Patterns Used:
- Template Method Pattern: Standard extension setup workflow
- Factory Pattern: Create appropriate loggers and handlers
- Singleton Pattern: Ensure single configuration instance per extension
- Strategy Pattern: Different logging strategies for different extension types

Educational Value:
This module demonstrates how common functionality can be shared while respecting
the architectural principle that each extension + common folder = standalone unit.
"""

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from .path_utils import ensure_project_root_on_path, get_extension_path, get_dataset_path, get_model_path
from ..config.validation_rules import MIN_GRID_SIZE, MAX_GRID_SIZE


class ExtensionType(Enum):
    """
    Enumeration of supported extension types
    
    Educational Note:
    Using enums instead of strings prevents typos and enables IDE autocompletion,
    making the code more maintainable and less error-prone.
    """
    HEURISTICS = "heuristics"
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"
    EVOLUTIONARY = "evolutionary"
    AGENTIC_LLMS = "agentic-llms"
    LLM_FINETUNE = "llm-finetune"
    LLM_DISTILLATION = "llm-distillation"
    VISION_LANGUAGE_MODEL = "vision-language-model"


@dataclass(frozen=True)
class ExtensionConfig:
    """
    Immutable configuration for extension setup
    
    Design Pattern: Value Object Pattern
    - Immutable data container
    - Encapsulates related configuration data
    - Enables safe sharing between components
    
    Educational Value:
    Demonstrates how immutable objects can prevent configuration bugs
    and make the system more predictable and thread-safe.
    """
    extension_type: ExtensionType
    version: str
    grid_size: int
    algorithm: str
    timestamp: str
    debug_mode: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not (MIN_GRID_SIZE <= self.grid_size <= MAX_GRID_SIZE):
            raise ValueError(f"Grid size must be between {MIN_GRID_SIZE} and {MAX_GRID_SIZE}, got {self.grid_size}")
        
        if not self.version.replace('.', '').isdigit():
            raise ValueError(f"Version must be in format 'X.YZ', got {self.version}")


class ExtensionEnvironment:
    """
    Manages the complete environment setup for an extension
    
    Design Pattern: Facade Pattern
    - Provides simple interface to complex environment setup
    - Hides complexity of path management, logging, and validation
    - Single point of control for extension initialization
    
    Educational Note (SUPREME_RULE NO.4):
    This class is designed to be extensible through inheritance. Extensions
    can create specialized environment managers by inheriting from this base class
    and overriding specific methods for unique requirements while maintaining
    the common setup workflow.
    
    SUPREME_RULE NO.4 Implementation:
    - Base class provides complete environment setup functionality
    - Protected methods allow selective customization by subclasses
    - Virtual methods enable complete behavior replacement when needed
    - Extension-specific environments can inherit and adapt as needed
    
    Responsibilities:
    - Set up working directory and Python paths
    - Configure logging with appropriate handlers
    - Validate extension configuration
    - Create necessary directories
    """
    
    def __init__(self, config: ExtensionConfig):
        self.config = config
        self._project_root = None
        self._extension_path = None
        self._logger = None
        self._initialized = False
        self._initialize_environment_specific_settings()
    
    def setup(self) -> Tuple[Path, Path, logging.Logger]:
        """
        Complete extension environment setup
        
        Returns:
            Tuple of (project_root, extension_path, logger)
            
        Design Note:
        Returns paths and logger as tuple to enable unpacking in client code:
        project_root, extension_path, logger = environment.setup()
        """
        if self._initialized:
            return self._project_root, self._extension_path, self._logger
        
        # 1. Setup paths and working directory
        self._project_root = ensure_project_root_on_path()
        self._extension_path = self._get_extension_path()
        
        # 2. Setup logging
        self._logger = self._setup_logging()
        
        # 3. Create necessary directories
        self._create_directories()
        
        # 4. Validate environment
        self._validate_environment()
        
        self._initialized = True
        self._logger.info(f"Extension environment initialized: {self.config.extension_type.value} v{self.config.version}")
        
        return self._project_root, self._extension_path, self._logger
    
    def _initialize_environment_specific_settings(self) -> None:
        """
        Initialize environment-specific settings (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to set up extension-specific
        configurations, custom validators, or specialized environment requirements.
        
        Example:
            class RLEnvironmentManager(ExtensionEnvironment):
                def _initialize_environment_specific_settings(self):
                    self.rl_model_tracker = RLModelTracker()
                    self.tensorboard_setup = True
                    self.gpu_validation_required = True
        """
        pass
    
    def _get_extension_path(self) -> Path:
        """Get extension path based on configuration"""
        extension_name = f"{self.config.extension_type.value}-v{self.config.version}"
        return self._project_root / "extensions" / extension_name
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup extension-specific logging
        
        Design Pattern: Factory Method Pattern
        Creates appropriate logger configuration based on extension type and debug mode.
        
        Educational Value:
        Shows how to create structured logging that adapts to different use cases
        while maintaining consistent format across all extensions.
        """
        # Create logger with extension-specific name
        logger_name = f"{self.config.extension_type.value}_v{self.config.version}"
        logger = logging.getLogger(logger_name)
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Set log level based on debug mode
        log_level = logging.DEBUG if self.config.debug_mode else logging.INFO
        logger.setLevel(log_level)
        
        # Create formatter
        formatter = self._create_log_formatter()
        
        # Console handler (always present)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (for persistent logging)
        log_dir = self._project_root / "logs" / "extensions" / f"grid-size-{self.config.grid_size}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{self.config.extension_type.value}_v{self.config.version}_{self.config.timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Allow subclasses to add extension-specific logging
        self._setup_extension_specific_logging(logger)
        
        return logger
    
    def _create_log_formatter(self) -> logging.Formatter:
        """Create standardized log formatter"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _setup_extension_specific_logging(self, logger: logging.Logger) -> None:
        """
        Setup extension-specific logging (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to add specialized loggers,
        handlers, or formatting for specific extension requirements.
        
        Example:
            class MLExtensionEnvironment(ExtensionEnvironment):
                def _setup_extension_specific_logging(self, logger):
                    # Add TensorBoard logging
                    tensorboard_handler = TensorBoardHandler()
                    logger.addHandler(tensorboard_handler)
                    
                    # Add metrics logging
                    metrics_handler = MetricsHandler()
                    logger.addHandler(metrics_handler)
        """
        pass
    
    def _create_directories(self) -> None:
        """Create necessary directories for extension operation"""
        # Ensure extension directory exists
        self._extension_path.mkdir(parents=True, exist_ok=True)
        
        # Create dataset directories
        dataset_path = get_dataset_path(
            self.config.extension_type.value,
            self.config.version,
            self.config.grid_size,
            self.config.algorithm,
            self.config.timestamp
        )
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create model directories (for ML extensions)
        if self.config.extension_type in [ExtensionType.SUPERVISED, ExtensionType.REINFORCEMENT]:
            model_path = get_model_path(
                self.config.extension_type.value,
                self.config.version,
                self.config.grid_size,
                self.config.algorithm,
                self.config.timestamp
            )
            model_path.mkdir(parents=True, exist_ok=True)
        
        # Allow subclasses to create extension-specific directories
        self._create_extension_specific_directories()
    
    def _create_extension_specific_directories(self) -> None:
        """
        Create extension-specific directories (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to create directories
        specific to their extension requirements.
        
        Example:
            class RLExtensionEnvironment(ExtensionEnvironment):
                def _create_extension_specific_directories(self):
                    # Create RL-specific directories
                    checkpoint_dir = self._extension_path / "checkpoints"
                    checkpoint_dir.mkdir(exist_ok=True)
                    
                    tensorboard_dir = self._extension_path / "tensorboard"
                    tensorboard_dir.mkdir(exist_ok=True)
        """
        pass
    
    def _validate_environment(self) -> None:
        """Validate extension environment setup"""
        # Check that extension path exists
        if not self._extension_path.exists():
            raise EnvironmentError(f"Extension path does not exist: {self._extension_path}")
        
        # Validate grid size
        if not (MIN_GRID_SIZE <= self.config.grid_size <= MAX_GRID_SIZE):
            raise ValueError(f"Invalid grid size: {self.config.grid_size}")
        
        # Check Python path includes project root
        if str(self._project_root) not in sys.path:
            raise EnvironmentError(f"Project root not in Python path: {self._project_root}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_extension_environment(
    extension_type: str,
    version: str,
    grid_size: int,
    algorithm: str,
    debug_mode: bool = False
) -> Tuple[Path, Path, logging.Logger]:
    """
    Convenience function to create and setup extension environment
    
    Args:
        extension_type: Type of extension (heuristics, supervised, etc.)
        version: Version string (e.g., "0.04")
        grid_size: Grid size for the extension
        algorithm: Algorithm name
        debug_mode: Enable debug logging
        
    Returns:
        Tuple of (project_root, extension_path, logger)
        
    Example:
        >>> project_root, extension_path, logger = create_extension_environment(
        ...     "heuristics", "0.04", 10, "bfs", debug_mode=True
        ... )
    """
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create configuration
    config = ExtensionConfig(
        extension_type=ExtensionType(extension_type),
        version=version,
        grid_size=grid_size,
        algorithm=algorithm,
        timestamp=timestamp,
        debug_mode=debug_mode
    )
    
    # Create and setup environment
    environment = ExtensionEnvironment(config)
    return environment.setup()


def setup_extension_logging(
    extension_name: str,
    debug_mode: bool = False,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup logging for an extension
    
    Args:
        extension_name: Name of the extension
        debug_mode: Enable debug logging
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(extension_name)
    logger.handlers.clear()
    
    # Set log level
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_extension_info(extension_path: Path) -> Dict[str, Any]:
    """
    Extract information about an extension from its path and files
    
    Args:
        extension_path: Path to extension directory
        
    Returns:
        Dictionary with extension information
    """
    info = {
        "path": str(extension_path),
        "name": extension_path.name,
        "exists": extension_path.exists(),
        "type": None,
        "version": None,
        "agents": [],
        "scripts": [],
        "dashboard": False
    }
    
    if not extension_path.exists():
        return info
    
    # Parse extension name for type and version
    name_parts = extension_path.name.split('-')
    if len(name_parts) >= 2:
        info["type"] = name_parts[0]
        if name_parts[1].startswith('v'):
            info["version"] = name_parts[1][1:]  # Remove 'v' prefix
    
    # Check for agents directory
    agents_dir = extension_path / "agents"
    if agents_dir.exists():
        info["agents"] = [
            f.stem for f in agents_dir.glob("agent_*.py")
            if f.is_file() and not f.name.startswith('__')
        ]
    
    # Check for scripts directory
    scripts_dir = extension_path / "scripts"
    if scripts_dir.exists():
        info["scripts"] = [
            f.stem for f in scripts_dir.glob("*.py")
            if f.is_file() and not f.name.startswith('__')
        ]
    
    # Check for dashboard
    dashboard_dir = extension_path / "dashboard"
    if dashboard_dir.exists():
        info["dashboard"] = True
    
    return info


def list_available_extensions(project_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    List all available extensions in the project
    
    Args:
        project_root: Project root path (auto-detected if None)
        
    Returns:
        List of extension information dictionaries
    """
    if project_root is None:
        project_root = ensure_project_root_on_path()
    
    extensions_dir = project_root / "extensions"
    if not extensions_dir.exists():
        return []
    
    extensions = []
    for ext_path in extensions_dir.iterdir():
        if ext_path.is_dir() and not ext_path.name.startswith('.') and ext_path.name != "common":
            ext_info = get_extension_info(ext_path)
            extensions.append(ext_info)
    
    # Sort by type and version
    extensions.sort(key=lambda x: (x.get("type", ""), x.get("version", "")))
    return extensions


def validate_extension_structure(extension_path: Path) -> List[str]:
    """
    Validate extension directory structure
    
    Args:
        extension_path: Path to extension directory
        
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    if not extension_path.exists():
        warnings.append(f"Extension directory does not exist: {extension_path}")
        return warnings
    
    # Check for required files
    required_files = ["__init__.py"]
    for file_name in required_files:
        file_path = extension_path / file_name
        if not file_path.exists():
            warnings.append(f"Missing required file: {file_name}")
    
    # Check for v0.02+ structure (agents directory)
    if "v0.01" not in extension_path.name:
        agents_dir = extension_path / "agents"
        if not agents_dir.exists():
            warnings.append("Missing agents/ directory (required for v0.02+)")
    
    # Check for v0.03+ structure (dashboard directory)
    if any(version in extension_path.name for version in ["v0.03", "v0.04"]):
        dashboard_dir = extension_path / "dashboard"
        if not dashboard_dir.exists():
            warnings.append("Missing dashboard/ directory (required for v0.03+)")
        
        scripts_dir = extension_path / "scripts"
        if not scripts_dir.exists():
            warnings.append("Missing scripts/ directory (required for v0.03+)")
    
    return warnings


def create_extension_directories(extension_path: Path, version: str) -> None:
    """
    Create standard extension directory structure
    
    Args:
        extension_path: Path to extension directory
        version: Extension version (determines structure)
    """
    extension_path.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py
    init_file = extension_path / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Extension package initialization"""\n')
    
    # v0.02+ requires agents directory
    if version != "0.01":
        agents_dir = extension_path / "agents"
        agents_dir.mkdir(exist_ok=True)
        
        agents_init = agents_dir / "__init__.py"
        if not agents_init.exists():
            agents_init.write_text('"""Agent implementations package"""\n')
    
    # v0.03+ requires dashboard and scripts directories
    if version in ["0.03", "0.04"]:
        dashboard_dir = extension_path / "dashboard"
        dashboard_dir.mkdir(exist_ok=True)
        
        dashboard_init = dashboard_dir / "__init__.py"
        if not dashboard_init.exists():
            dashboard_init.write_text('"""Dashboard components package"""\n')
        
        scripts_dir = extension_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        scripts_init = scripts_dir / "__init__.py"
        if not scripts_init.exists():
            scripts_init.write_text('"""CLI scripts package"""\n') 