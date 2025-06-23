"""
Versioned Directory Manager - Extension Directory Structure Enforcement
=====================================================================

This module enforces the critical directory structure rule that ensures clean
separation of datasets and models by grid size AND extension version:

DATASETS: logs/extensions/datasets/grid-size-N/extension_v0.0.M_timestamp/
MODELS:   logs/extensions/models/grid-size-N/extension_v0.0.M_timestamp/

Where:
- N = grid size (8, 10, 12, 15, 20, etc.)
- M = extension version (01, 02, 03, 04, etc.)
- extension = heuristics, supervised, reinforcement, etc.
- timestamp = YYYYMMDD_HHMMSS format

Design Philosophy:
================

1. **Spatial Complexity Separation**: Different grid sizes represent fundamentally
   different problem complexities and should never be mixed.

2. **Evolutionary Tracking**: Extension versions (v0.01, v0.02, etc.) represent
   algorithmic evolution and should be traceable.

3. **Temporal Organization**: Timestamps enable experiment reproducibility and
   chronological analysis.

4. **Single Source of Truth**: Centralized in common/ to prevent inconsistencies
   across extensions.

Design Patterns Used:
==================
- **Facade Pattern**: Simple interface hiding complex path logic
- **Factory Pattern**: Creates appropriate directory structures
- **Template Method**: Standardized directory creation workflow
- **Validation Pattern**: Ensures compliance before operations
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, Optional, Tuple, List
from datetime import datetime

from .config import (
    DATASETS_ROOT, 
    SUPPORTED_GRID_SIZES
)

# Add MODELS_ROOT if not in config
try:
    from .config import MODELS_ROOT
except ImportError:
    MODELS_ROOT = Path("logs/extensions/models")

__all__ = [
    "VersionedDirectoryError",
    "VersionedDirectoryManager", 
    "ExtensionType",
    "create_dataset_directory",
    "create_model_directory",
    "parse_versioned_path",
    "validate_directory_structure"
]

# ---------------------
# Types and Constants
# ---------------------

class ExtensionType:
    """Extension type constants for directory naming."""
    HEURISTICS = "heuristics"
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement" 
    LLM_FINETUNE = "llm-finetune"
    DISTILLATION = "distillation"
    EVOLUTIONARY = "evolutionary"
    
    # Integration extensions
    HEURISTICS_SUPERVISED = "heuristics-supervised-integration"
    HEURISTICS_LLM = "heuristics-llm-fine-tuning-integration"
    
    ALL = [
        HEURISTICS, SUPERVISED, REINFORCEMENT, LLM_FINETUNE, 
        DISTILLATION, EVOLUTIONARY, HEURISTICS_SUPERVISED, HEURISTICS_LLM
    ]

# Regex patterns for parsing versioned directories
_VERSIONED_DIR_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<extension>[a-z-]+)_v(?P<major>\d+)\.(?P<minor>\d+)_(?P<timestamp>\d{8}_\d{6})"
)

_GRID_SIZE_PATTERN: Final[re.Pattern[str]] = re.compile(r"grid-size-(\d+)")

# ---------------------
# Exceptions
# ---------------------

class VersionedDirectoryError(RuntimeError):
    """Raised when versioned directory operations fail validation."""


# ---------------------
# Core Manager Class
# ---------------------

class VersionedDirectoryManager:
    """
    Centralized manager for versioned extension directories.
    
    This class enforces the mandatory directory structure across ALL extensions,
    ensuring consistent organization and preventing spatial/temporal contamination.
    
    Design Principle: Single Source of Truth
    All extensions MUST use this manager instead of implementing their own
    directory logic to ensure consistency and compliance.
    """

    @staticmethod
    def create_dataset_directory(
        extension_type: str,
        version: str,
        grid_size: int,
        timestamp: Optional[str] = None,
        algorithm: Optional[str] = None
    ) -> Path:
        """
        Create a versioned dataset directory following the mandatory structure.
        
        Args:
            extension_type: Type of extension (e.g., "heuristics", "supervised")
            version: Version string (e.g., "v0.01", "v0.02") 
            grid_size: Grid size for spatial separation
            timestamp: Optional timestamp (auto-generated if None)
            algorithm: Optional algorithm name for sub-organization
            
        Returns:
            Path to created directory
            
        Example:
            logs/extensions/datasets/grid-size-{grid_size}/heuristics_v0.03_{timestamp}/
            logs/extensions/datasets/grid-size-{grid_size}/supervised_v0.02_{timestamp}/bfs/
        """
        # Validate inputs
        VersionedDirectoryManager._validate_extension_type(extension_type)
        VersionedDirectoryManager._validate_grid_size(grid_size)
        version = VersionedDirectoryManager._normalize_version(version)
        timestamp = timestamp or VersionedDirectoryManager._generate_timestamp()
        
        # Build directory path
        versioned_name = f"{extension_type}_{version}_{timestamp}"
        base_path = DATASETS_ROOT / f"grid-size-{grid_size}" / versioned_name
        
        # Add algorithm subdirectory if specified
        if algorithm:
            base_path = base_path / algorithm.lower()
        
        # Create directory structure
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file for tracking
        VersionedDirectoryManager._create_metadata_file(
            base_path, extension_type, version, grid_size, "dataset", algorithm
        )
        
        return base_path

    @staticmethod
    def create_model_directory(
        extension_type: str,
        version: str,
        grid_size: int,
        framework: str,
        timestamp: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Path:
        """
        Create a versioned model directory following the mandatory structure.
        
        Args:
            extension_type: Type of extension (e.g., "supervised", "reinforcement")
            version: Version string (e.g., "v0.01", "v0.02")
            grid_size: Grid size for spatial separation
            framework: ML framework (e.g., "pytorch", "xgboost", "lightgbm")
            timestamp: Optional timestamp (auto-generated if None)
            model_name: Optional model name for sub-organization
            
        Returns:
            Path to created directory
            
        Example:
            logs/extensions/models/grid-size-{grid_size}/supervised_v0.02_{timestamp}/pytorch/
            logs/extensions/models/grid-size-{grid_size}/reinforcement_v0.01_{timestamp}/pytorch/dqn/
        """
        # Validate inputs
        VersionedDirectoryManager._validate_extension_type(extension_type)
        VersionedDirectoryManager._validate_grid_size(grid_size)
        version = VersionedDirectoryManager._normalize_version(version)
        timestamp = timestamp or VersionedDirectoryManager._generate_timestamp()
        
        # Build directory path
        versioned_name = f"{extension_type}_{version}_{timestamp}"
        base_path = MODELS_ROOT / f"grid-size-{grid_size}" / versioned_name / framework.lower()
        
        # Add model subdirectory if specified
        if model_name:
            base_path = base_path / model_name.lower()
        
        # Create directory structure
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file for tracking
        VersionedDirectoryManager._create_metadata_file(
            base_path.parent, extension_type, version, grid_size, "model", framework
        )
        
        return base_path

    @staticmethod
    def parse_versioned_path(path: Path) -> Tuple[str, str, int, str]:
        """
        Parse a versioned directory path to extract components.
        
        Args:
            path: Path to parse
            
        Returns:
            Tuple of (extension_type, version, grid_size, timestamp)
            
        Raises:
            VersionedDirectoryError: If path doesn't match expected pattern
        """
        path_str = str(path)
        
        # Extract grid size
        grid_match = _GRID_SIZE_PATTERN.search(path_str)
        if not grid_match:
            raise VersionedDirectoryError(f"No grid-size found in path: {path}")
        grid_size = int(grid_match.group(1))
        
        # Extract versioned directory components
        version_match = _VERSIONED_DIR_PATTERN.search(path_str)
        if not version_match:
            raise VersionedDirectoryError(f"Invalid versioned directory format: {path}")
        
        extension_type = version_match.group("extension")
        major = version_match.group("major")
        minor = version_match.group("minor") 
        timestamp = version_match.group("timestamp")
        
        version = f"v{major}.{minor}"
        
        return extension_type, version, grid_size, timestamp

    @staticmethod
    def list_dataset_versions(
        extension_type: str,
        grid_size: int,
        algorithm: Optional[str] = None
    ) -> List[Tuple[str, str, Path]]:
        """
        List all versions of datasets for an extension type and grid size.
        
        Args:
            extension_type: Extension type to search for
            grid_size: Grid size to filter by
            algorithm: Optional algorithm filter
            
        Returns:
            List of (version, timestamp, path) tuples sorted by timestamp
        """
        grid_dir = DATASETS_ROOT / f"grid-size-{grid_size}"
        if not grid_dir.exists():
            return []
        
        versions = []
        
        for versioned_dir in grid_dir.glob(f"{extension_type}_v*"):
            if versioned_dir.is_dir():
                try:
                    _, version, _, timestamp = VersionedDirectoryManager.parse_versioned_path(versioned_dir)
                    
                    # Check algorithm subdirectory if specified
                    search_path = versioned_dir
                    if algorithm:
                        search_path = versioned_dir / algorithm.lower()
                        if not search_path.exists():
                            continue
                    
                    versions.append((version, timestamp, search_path))
                except VersionedDirectoryError:
                    continue  # Skip invalid directories
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda x: x[1], reverse=True)
        return versions

    @staticmethod
    def get_latest_dataset(
        extension_type: str,
        grid_size: int,
        algorithm: Optional[str] = None
    ) -> Optional[Path]:
        """Get the most recent dataset directory for given parameters."""
        versions = VersionedDirectoryManager.list_dataset_versions(
            extension_type, grid_size, algorithm
        )
        return versions[0][2] if versions else None

    @staticmethod
    def get_latest_model(
        extension_type: str,
        grid_size: int,
        framework: Optional[str] = None
    ) -> Optional[Path]:
        """Get the most recent model directory for given parameters."""
        versions = VersionedDirectoryManager.list_model_versions(
            extension_type, grid_size, framework
        )
        return versions[0][2] if versions else None

    # ---------------------
    # Private Helper Methods
    # ---------------------

    @staticmethod
    def _validate_extension_type(extension_type: str) -> None:
        """Validate extension type is supported."""
        if extension_type not in ExtensionType.ALL:
            raise VersionedDirectoryError(
                f"Unsupported extension type: {extension_type}. "
                f"Supported types: {ExtensionType.ALL}"
            )

    @staticmethod
    def _validate_grid_size(grid_size: int) -> None:
        """Validate grid size is supported."""
        if grid_size not in SUPPORTED_GRID_SIZES:
            raise VersionedDirectoryError(
                f"Unsupported grid size: {grid_size}. "
                f"Supported sizes: {SUPPORTED_GRID_SIZES}"
            )

    @staticmethod
    def _normalize_version(version: str) -> str:
        """Normalize version string to standard format."""
        # Remove 'v' prefix if present
        if version.startswith('v'):
            version = version[1:]
        
        # Ensure 0.0N format
        parts = version.split('.')
        if len(parts) == 2:
            major, minor = parts
            return f"v{major}.{minor:0>2}"
        else:
            raise VersionedDirectoryError(f"Invalid version format: {version}")

    @staticmethod
    def _generate_timestamp() -> str:
        """Generate timestamp in YYYYMMDD_HHMMSS format."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _create_metadata_file(
        directory: Path,
        extension_type: str,
        version: str,
        grid_size: int,
        content_type: str,
        extra_info: Optional[str] = None
    ) -> None:
        """Create metadata file for tracking directory contents."""
        import json
        
        metadata = {
            "extension_type": extension_type,
            "version": version,
            "grid_size": grid_size,
            "content_type": content_type,  # "dataset" or "model"
            "created_at": datetime.now().isoformat(),
            "created_by": "VersionedDirectoryManager",
            "extra_info": extra_info
        }
        
        metadata_file = directory / ".directory_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def list_model_versions(
        extension_type: str,
        grid_size: int,
        framework: Optional[str] = None
    ) -> List[Tuple[str, str, Path]]:
        """
        List all versions of models for an extension type and grid size.
        """
        grid_dir = MODELS_ROOT / f"grid-size-{grid_size}"
        if not grid_dir.exists():
            return []
        
        versions = []
        
        for versioned_dir in grid_dir.glob(f"{extension_type}_v*"):
            if versioned_dir.is_dir():
                try:
                    _, version, _, timestamp = VersionedDirectoryManager.parse_versioned_path(versioned_dir)
                    
                    # Check framework subdirectory if specified
                    if framework:
                        framework_path = versioned_dir / framework.lower()
                        if framework_path.exists():
                            versions.append((version, timestamp, framework_path))
                    else:
                        versions.append((version, timestamp, versioned_dir))
                        
                except VersionedDirectoryError:
                    continue  # Skip invalid directories
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda x: x[1], reverse=True)
        return versions


# ---------------------
# Convenience Functions
# ---------------------

def create_dataset_directory(
    extension_type: str,
    version: str, 
    grid_size: int,
    algorithm: Optional[str] = None,
    timestamp: Optional[str] = None
) -> Path:
    """Convenience function for creating dataset directories."""
    return VersionedDirectoryManager.create_dataset_directory(
        extension_type, version, grid_size, timestamp, algorithm
    )

def create_model_directory(
    extension_type: str,
    version: str,
    grid_size: int, 
    framework: str,
    model_name: Optional[str] = None,
    timestamp: Optional[str] = None
) -> Path:
    """Convenience function for creating model directories."""
    return VersionedDirectoryManager.create_model_directory(
        extension_type, version, grid_size, framework, timestamp, model_name
    )

def parse_versioned_path(path: Path) -> Tuple[str, str, int, str]:
    """Convenience function for parsing versioned paths."""
    return VersionedDirectoryManager.parse_versioned_path(path)

def validate_directory_structure(base_path: Path) -> bool:
    """
    Validate that a directory follows the versioned structure.
    
    Args:
        base_path: Path to validate
        
    Returns:
        True if structure is valid, False otherwise
    """
    try:
        VersionedDirectoryManager.parse_versioned_path(base_path)
        return True
    except VersionedDirectoryError:
        return False


# ---------------------
# Integration with Existing Common Utilities
# ---------------------

def update_existing_model_utils():
    """
    Update model_utils.py to use versioned directories.
    This function provides integration guidance.
    """
    integration_note = """
    INTEGRATION REQUIRED:
    
    Update extensions/common/model_utils.py:
    
    1. Import VersionedDirectoryManager
    2. Modify save_model_standardized() to use create_model_directory()
    3. Add extension_type and version parameters
    4. Ensure all model saving uses versioned structure
    
    Example:
        from .versioned_directory_manager import create_model_directory
        
        model_dir = create_model_directory(
            extension_type="supervised",
            version="v0.02", 
            grid_size=grid_size,
            framework=framework,
            model_name=model_name
        )
    """
    print(integration_note)

def update_existing_dataset_utils():
    """
    Update dataset utilities to use versioned directories.
    This function provides integration guidance.
    """
    integration_note = """
    INTEGRATION REQUIRED:
    
    Update extensions/common/dataset_directory_manager.py:
    
    1. Import VersionedDirectoryManager
    2. Add versioned dataset creation methods
    3. Update heuristics generation to use versioned structure
    4. Ensure all dataset outputs use versioned directories
    
    Example:
        from .versioned_directory_manager import create_dataset_directory
        
        dataset_dir = create_dataset_directory(
            extension_type="heuristics",
            version="v0.03",
            grid_size=grid_size,
            algorithm=algorithm
        )
    """
    print(integration_note) 