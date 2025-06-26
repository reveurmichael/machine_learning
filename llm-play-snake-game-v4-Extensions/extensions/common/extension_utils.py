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

from .path_utils import ensure_project_root, get_extension_path, get_dataset_path, get_model_path
from .config.validation_rules import MIN_GRID_SIZE, MAX_GRID_SIZE


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
        self._project_root = ensure_project_root()
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
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # Console handler (always present)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (for persistent logging)
        log_dir = self._project_root / "logs" / "extensions" / f"grid-size-{self.config.grid_size}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{logger_name}_{self.config.timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _create_directories(self) -> None:
        """Create necessary directories for extension operation"""
        # Dataset directory
        dataset_path = get_dataset_path(
            extension_type=self.config.extension_type.value,
            version=self.config.version,
            grid_size=self.config.grid_size,
            algorithm=self.config.algorithm,
            timestamp=self.config.timestamp
        )
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Model directory (if applicable for this extension type)
        if self.config.extension_type in [ExtensionType.SUPERVISED, ExtensionType.REINFORCEMENT, 
                                         ExtensionType.LLM_FINETUNE, ExtensionType.LLM_DISTILLATION]:
            model_path = get_model_path(
                extension_type=self.config.extension_type.value,
                version=self.config.version,
                grid_size=self.config.grid_size,
                algorithm=self.config.algorithm,
                timestamp=self.config.timestamp
            )
            model_path.mkdir(parents=True, exist_ok=True)
    
    def _validate_environment(self) -> None:
        """Validate that environment is properly set up"""
        # Check that extension directory exists
        if not self._extension_path.exists():
            raise FileNotFoundError(f"Extension directory not found: {self._extension_path}")
        
        # Check that project root is correct
        if not (self._project_root / "config").exists():
            raise FileNotFoundError(f"Invalid project root - config directory not found: {self._project_root}")
        
        # Validate grid size
        if not (MIN_GRID_SIZE <= self.config.grid_size <= MAX_GRID_SIZE):
            raise ValueError(f"Invalid grid size: {self.config.grid_size}")


def create_extension_environment(
    extension_type: str,
    version: str,
    grid_size: int,
    algorithm: str,
    debug_mode: bool = False
) -> Tuple[Path, Path, logging.Logger]:
    """
    Factory function for creating and setting up extension environments
    
    This is the main entry point that extensions should use for environment setup.
    
    Args:
        extension_type: Type of extension (heuristics, supervised, etc.)
        version: Extension version (e.g., "0.03")
        grid_size: Game grid size
        algorithm: Algorithm name
        debug_mode: Enable debug logging
        
    Returns:
        Tuple of (project_root, extension_path, logger)
        
    Example:
        project_root, ext_path, logger = create_extension_environment(
            "heuristics", "0.03", 10, "bfs"
        )
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


def get_extension_info(extension_path: Path) -> Dict[str, Any]:
    """
    Extract extension information from path and structure
    
    Args:
        extension_path: Path to extension directory
        
    Returns:
        Dictionary with extension metadata
        
    Educational Value:
    Demonstrates how to parse structured directory names and extract metadata,
    which is useful for validation and automated tooling.
    """
    extension_name = extension_path.name
    
    # Parse extension type and version from directory name
    # Expected format: {type}-v{version}
    if '-v' not in extension_name:
        raise ValueError(f"Invalid extension directory name: {extension_name}")
    
    extension_type, version_part = extension_name.rsplit('-v', 1)
    
    # Check if extension type is supported
    try:
        ext_type = ExtensionType(extension_type)
    except ValueError:
        raise ValueError(f"Unsupported extension type: {extension_type}")
    
    # Get available algorithms
    agents_dir = extension_path / "agents"
    algorithms = []
    if agents_dir.exists():
        for agent_file in agents_dir.glob("agent_*.py"):
            if agent_file.name != "__init__.py":
                algorithm = agent_file.stem.replace("agent_", "").upper()
                algorithms.append(algorithm)
    
    return {
        'extension_type': ext_type,
        'version': version_part,
        'algorithms': algorithms,
        'has_dashboard': (extension_path / "dashboard").exists(),
        'has_scripts': (extension_path / "scripts").exists(),
        'extension_path': extension_path
    }


def list_available_extensions(project_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    List all available extensions in the project
    
    Args:
        project_root: Project root path (auto-detected if None)
        
    Returns:
        List of extension information dictionaries
        
    Educational Value:
    Shows how to programmatically discover and analyze project structure,
    which is useful for automated tooling and validation.
    """
    if project_root is None:
        project_root = ensure_project_root()
    
    extensions_dir = project_root / "extensions"
    if not extensions_dir.exists():
        return []
    
    extensions = []
    for extension_dir in extensions_dir.iterdir():
        if extension_dir.is_dir() and extension_dir.name != "common":
            try:
                info = get_extension_info(extension_dir)
                extensions.append(info)
            except ValueError as e:
                # Skip invalid extension directories
                continue
    
    # Sort by extension type and version
    extensions.sort(key=lambda x: (x['extension_type'].value, x['version']))
    return extensions


def validate_extension_standalone(extension_path: Path) -> List[str]:
    """
    Validate that an extension is properly standalone
    
    Args:
        extension_path: Path to extension directory
        
    Returns:
        List of validation error messages (empty if valid)
        
    Educational Value:
    Demonstrates how to programmatically validate architectural constraints,
    ensuring that the "extension + common = standalone" principle is maintained.
    """
    errors = []
    
    # Check required files exist
    required_files = ["__init__.py", "game_logic.py", "game_manager.py"]
    for required_file in required_files:
        if not (extension_path / required_file).exists():
            errors.append(f"Missing required file: {required_file}")
    
    # Check for proper agent organization (v0.02+)
    info = get_extension_info(extension_path)
    if info['version'] >= "0.02":
        agents_dir = extension_path / "agents"
        if not agents_dir.exists():
            errors.append("v0.02+ extensions must have agents/ directory")
        elif not (agents_dir / "__init__.py").exists():
            errors.append("agents/ directory must have __init__.py")
    
    # Check for dashboard (v0.03+)
    if info['version'] >= "0.03":
        if not (extension_path / "dashboard").exists():
            errors.append("v0.03+ extensions must have dashboard/ directory")
    
    # Check for forbidden cross-extension imports
    # This would require parsing Python files, which is beyond scope here
    # but could be implemented with ast module for more thorough validation
    
    return errors 