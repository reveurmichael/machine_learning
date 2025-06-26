"""
Dataset Loading Utilities for Snake Game AI Extensions.

This module provides standardized dataset loading and preprocessing utilities
for all extension types (heuristics, supervised, reinforcement, evolutionary, LLM).

Design Patterns:
- Factory Pattern: Create dataset loaders for different formats
- Strategy Pattern: Different loading strategies for different extension types
- Template Method Pattern: Common loading workflow with extension-specific customization
- Observer Pattern: Progress tracking during dataset loading

Educational Value:
Demonstrates how to design a flexible data loading system that can handle
multiple formats while maintaining consistency and performance across
different machine learning paradigms.

Key Features:
- Grid-size agnostic loading
- Multiple format support (CSV, JSONL, NPZ)
- Extension-specific preprocessing
- Memory-efficient loading for large datasets
- Comprehensive validation and error handling
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Generator
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass
from enum import Enum
import logging

# Import configuration constants
from .config.dataset_formats import (
    CSV_COLUMN_NAMES, JSONL_REQUIRED_FIELDS, NPZ_ARRAY_NAMES
)
from .config.training_defaults import (
    DEFAULT_TRAIN_SPLIT, DEFAULT_VAL_SPLIT, DEFAULT_BATCH_SIZE
)
from .config.validation_rules import DATASET_VALIDATION_RULES
from .path_utils import ensure_project_root_on_path

# =============================================================================
# Enums and Data Classes
# =============================================================================

class DatasetFormat(Enum):
    """Supported dataset formats."""
    CSV = "csv"
    JSONL = "jsonl"
    NPZ = "npz"

class ExtensionType(Enum):
    """Extension types with different data requirements."""
    HEURISTICS = "heuristics"
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"
    EVOLUTIONARY = "evolutionary"
    LLM = "llm"

@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    format: DatasetFormat
    extension_type: ExtensionType
    train_split: float = DEFAULT_TRAIN_SPLIT
    val_split: float = DEFAULT_VAL_SPLIT
    batch_size: int = DEFAULT_BATCH_SIZE
    shuffle: bool = True
    grid_size: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.train_split + self.val_split > 1.0:
            raise ValueError("Train and validation splits cannot exceed 1.0")

@dataclass
class LoadedDataset:
    """Container for loaded dataset with metadata."""
    data: Union[pd.DataFrame, List[Dict], np.ndarray]
    metadata: Dict[str, Any]
    config: DatasetConfig
    
    @property
    def size(self) -> int:
        """Get dataset size."""
        if isinstance(self.data, pd.DataFrame):
            return len(self.data)
        elif isinstance(self.data, list):
            return len(self.data)
        elif isinstance(self.data, np.ndarray):
            return self.data.shape[0]
        return 0

# =============================================================================
# Base Dataset Loader (Template Method Pattern)
# =============================================================================

class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    Design Pattern: Template Method Pattern
    Purpose: Define common loading workflow with extension-specific customization
    
    Educational Note (SUPREME_RULE NO.4):
    This class is designed to be extensible through inheritance. Extensions
    can create specialized loaders by inheriting from this base class and
    overriding specific methods to handle unique requirements while
    maintaining the common workflow.
    
    SUPREME_RULE NO.4 Implementation:
    - Base class provides complete functionality for most extensions
    - Protected methods allow selective customization by subclasses
    - Virtual methods enable complete behavior replacement when needed
    - Extension-specific loaders can inherit and adapt as needed
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_loader_specific_settings()
        
    def load_dataset(self, file_path: Path) -> LoadedDataset:
        """
        Template method for loading datasets.
        
        This method defines the common workflow:
        1. Validate file path
        2. Load raw data
        3. Validate data format
        4. Apply extension-specific preprocessing
        5. Split data if needed
        6. Return loaded dataset with metadata
        """
        # Step 1: Validate file path
        self._validate_file_path(file_path)
        
        # Step 2: Load raw data
        raw_data = self._load_raw_data(file_path)
        
        # Step 3: Validate data format
        self._validate_data_format(raw_data)
        
        # Step 4: Apply extension-specific preprocessing
        processed_data = self._preprocess_data(raw_data)
        
        # Step 5: Generate metadata
        metadata = self._generate_metadata(processed_data, file_path)
        
        # Step 6: Return loaded dataset
        return LoadedDataset(
            data=processed_data,
            metadata=metadata,
            config=self.config
        )
    
    def _initialize_loader_specific_settings(self) -> None:
        """
        Initialize loader-specific settings (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden by subclasses to set up extension-specific
        configurations, custom validators, or specialized preprocessing pipelines.
        
        Example:
            class CustomHeuristicsLoader(BaseDatasetLoader):
                def _initialize_loader_specific_settings(self):
                    self.custom_validator = HeuristicsValidator()
                    self.preprocessing_pipeline = HeuristicsPreprocessor()
        """
        pass
    
    def _validate_file_path(self, file_path: Path) -> None:
        """Validate that file exists and has correct extension."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        expected_suffix = f".{self.config.format.value}"
        if file_path.suffix != expected_suffix:
            raise ValueError(f"Expected {expected_suffix} file, got {file_path.suffix}")
    
    @abstractmethod
    def _load_raw_data(self, file_path: Path) -> Any:
        """Load raw data from file (implementation specific)."""
        pass
    
    @abstractmethod
    def _validate_data_format(self, data: Any) -> None:
        """Validate that data conforms to expected format."""
        pass
    
    @abstractmethod
    def _preprocess_data(self, data: Any) -> Any:
        """Apply extension-specific preprocessing."""
        pass
    
    def _generate_metadata(self, data: Any, file_path: Path) -> Dict[str, Any]:
        """
        Generate metadata for the loaded dataset (SUPREME_RULE NO.4 Extension Point).
        
        This method can be overridden to add extension-specific metadata.
        """
        base_metadata = {
            "source_file": str(file_path),
            "format": self.config.format.value,
            "extension_type": self.config.extension_type.value,
            "grid_size": self.config.grid_size,
            "size": self._get_data_size(data),
            "loading_config": {
                "train_split": self.config.train_split,
                "val_split": self.config.val_split,
                "batch_size": self.config.batch_size,
                "shuffle": self.config.shuffle
            }
        }
        
        # Allow subclasses to add extension-specific metadata
        extension_metadata = self._generate_extension_specific_metadata(data, file_path)
        base_metadata.update(extension_metadata)
        
        return base_metadata
    
    def _generate_extension_specific_metadata(self, data: Any, file_path: Path) -> Dict[str, Any]:
        """
        Generate extension-specific metadata (SUPREME_RULE NO.4 Extension Point).
        
        Override this method in subclasses to add custom metadata fields.
        
        Example:
            class RLDatasetLoader(BaseDatasetLoader):
                def _generate_extension_specific_metadata(self, data, file_path):
                    return {
                        "episode_count": self._count_episodes(data),
                        "reward_range": self._calculate_reward_range(data),
                        "action_distribution": self._analyze_actions(data)
                    }
        """
        return {}
    
    @abstractmethod
    def _get_data_size(self, data: Any) -> int:
        """Get the size of the dataset."""
        pass

# =============================================================================
# Concrete Dataset Loaders
# =============================================================================

class CSVDatasetLoader(BaseDatasetLoader):
    """
    CSV dataset loader for tabular data.
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle CSV-specific loading and validation
    
    Educational Note:
    Demonstrates how to implement format-specific loading while
    maintaining the common interface defined by the Template Method.
    """
    
    def _load_raw_data(self, file_path: Path) -> pd.DataFrame:
        """Load CSV data using pandas."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
    
    def _validate_data_format(self, data: pd.DataFrame) -> None:
        """Validate CSV data format."""
        # Check required columns
        missing_cols = set(CSV_COLUMN_NAMES) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check data types
        validation_rules = DATASET_VALIDATION_RULES["csv"]
        for col, expected_type in validation_rules["column_types"].items():
            if col in data.columns:
                if expected_type == "int":
                    if not pd.api.types.is_integer_dtype(data[col]):
                        raise ValueError(f"Column {col} should be integer type")
                elif expected_type == "str":
                    if not pd.api.types.is_string_dtype(data[col]) and not pd.api.types.is_object_dtype(data[col]):
                        raise ValueError(f"Column {col} should be string type")
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply CSV-specific preprocessing."""
        # Filter by grid size if specified
        if self.config.grid_size is not None:
            # Assuming grid_size column exists or can be inferred
            if "grid_size" in data.columns:
                data = data[data["grid_size"] == self.config.grid_size].copy()
        
        # Extension-specific preprocessing
        if self.config.extension_type == ExtensionType.SUPERVISED:
            # For supervised learning, ensure target column is categorical
            if "target_move" in data.columns:
                data["target_move"] = data["target_move"].astype("category")
        
        return data
    
    def _get_data_size(self, data: pd.DataFrame) -> int:
        """Get number of rows in DataFrame."""
        return len(data)

class JSONLDatasetLoader(BaseDatasetLoader):
    """
    JSONL dataset loader for sequential data.
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle JSONL-specific loading for game sequences
    
    Educational Note:
    Shows how to handle streaming JSON data while maintaining
    memory efficiency for large datasets.
    """
    
    def _load_raw_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL data line by line."""
        data = []
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")
            return data
        except Exception as e:
            raise ValueError(f"Failed to load JSONL file: {e}")
    
    def _validate_data_format(self, data: List[Dict[str, Any]]) -> None:
        """Validate JSONL data format."""
        if not data:
            raise ValueError("JSONL file is empty")
        
        # Check required fields in first few records
        sample_size = min(10, len(data))
        for i, record in enumerate(data[:sample_size]):
            missing_fields = set(JSONL_REQUIRED_FIELDS) - set(record.keys())
            if missing_fields:
                raise ValueError(f"Record {i} missing required fields: {missing_fields}")
    
    def _preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply JSONL-specific preprocessing."""
        # Filter by grid size if specified
        if self.config.grid_size is not None:
            data = [
                record for record in data 
                if record.get("grid_size") == self.config.grid_size
            ]
        
        # Extension-specific preprocessing
        if self.config.extension_type == ExtensionType.REINFORCEMENT:
            # For RL, ensure reward fields are numeric
            for record in data:
                if "reward" in record:
                    record["reward"] = float(record["reward"])
        
        return data
    
    def _get_data_size(self, data: List[Dict[str, Any]]) -> int:
        """Get number of records in list."""
        return len(data)

class NPZDatasetLoader(BaseDatasetLoader):
    """
    NPZ dataset loader for numerical arrays.
    
    Design Pattern: Strategy Pattern implementation
    Purpose: Handle NPZ-specific loading for neural network training
    
    Educational Note:
    Demonstrates efficient loading of numerical data for deep learning
    while maintaining the common interface.
    """
    
    def _load_raw_data(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load NPZ data as dictionary of arrays."""
        try:
            return dict(np.load(file_path))
        except Exception as e:
            raise ValueError(f"Failed to load NPZ file: {e}")
    
    def _validate_data_format(self, data: Dict[str, np.ndarray]) -> None:
        """Validate NPZ data format."""
        # Check required arrays
        missing_arrays = set(NPZ_ARRAY_NAMES) - set(data.keys())
        if missing_arrays:
            raise ValueError(f"Missing required arrays: {missing_arrays}")
        
        # Check array shapes are consistent
        if "features" in data and "targets" in data:
            if data["features"].shape[0] != data["targets"].shape[0]:
                raise ValueError("Features and targets must have same number of samples")
    
    def _preprocess_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply NPZ-specific preprocessing."""
        # Extension-specific preprocessing
        if self.config.extension_type == ExtensionType.SUPERVISED:
            # Normalize features for supervised learning
            if "features" in data:
                features = data["features"]
                # Simple min-max normalization
                features_min = features.min(axis=0)
                features_max = features.max(axis=0)
                features_range = features_max - features_min
                # Avoid division by zero
                features_range[features_range == 0] = 1
                data["features"] = (features - features_min) / features_range
                data["normalization_params"] = {
                    "min": features_min,
                    "max": features_max,
                    "range": features_range
                }
        
        return data
    
    def _get_data_size(self, data: Dict[str, np.ndarray]) -> int:
        """Get number of samples in arrays."""
        if "features" in data:
            return data["features"].shape[0]
        elif "states" in data:
            return data["states"].shape[0]
        else:
            return 0

# =============================================================================
# Dataset Loader Factory (Factory Pattern)
# =============================================================================

class DatasetLoaderFactory:
    """
    Factory for creating dataset loaders.
    
    Design Pattern: Factory Pattern
    Purpose: Create appropriate loader based on format and extension type
    
    Educational Note:
    Demonstrates how to use the Factory pattern to encapsulate object
    creation logic and provide a simple interface for client code.
    """
    
    _loaders = {
        DatasetFormat.CSV: CSVDatasetLoader,
        DatasetFormat.JSONL: JSONLDatasetLoader,
        DatasetFormat.NPZ: NPZDatasetLoader,
    }
    
    @classmethod
    def create_loader(cls, config: DatasetConfig) -> BaseDatasetLoader:
        """Create appropriate dataset loader based on configuration."""
        loader_class = cls._loaders.get(config.format)
        if loader_class is None:
            supported_formats = [f.value for f in DatasetFormat]
            raise ValueError(f"Unsupported format: {config.format.value}. "
                           f"Supported formats: {supported_formats}")
        
        return loader_class(config)
    
    @classmethod
    def register_loader(cls, format_type: DatasetFormat, loader_class: type) -> None:
        """Register a new loader class for a format (for extension)."""
        cls._loaders[format_type] = loader_class

# =============================================================================
# High-Level Loading Functions
# =============================================================================

def load_dataset_for_training(
    file_path: Union[str, Path],
    extension_type: ExtensionType,
    format_type: Optional[DatasetFormat] = None,
    grid_size: Optional[int] = None,
    **kwargs
) -> LoadedDataset:
    """
    High-level function to load dataset for training.
    
    Args:
        file_path: Path to dataset file
        extension_type: Type of extension (heuristics, supervised, etc.)
        format_type: Dataset format (auto-detected from file extension if None)
        grid_size: Filter by grid size if specified
        **kwargs: Additional configuration options
    
    Returns:
        LoadedDataset with data and metadata
    
    Educational Note:
    This function demonstrates how to provide a simple, high-level interface
    that hides the complexity of the factory and template method patterns
    from the client code.
    """
    ensure_project_root_on_path()
    
    file_path = Path(file_path)
    
    # Auto-detect format if not specified
    if format_type is None:
        format_map = {
            ".csv": DatasetFormat.CSV,
            ".jsonl": DatasetFormat.JSONL,
            ".npz": DatasetFormat.NPZ,
        }
        format_type = format_map.get(file_path.suffix)
        if format_type is None:
            raise ValueError(f"Cannot auto-detect format for file: {file_path}")
    
    # Create configuration
    config = DatasetConfig(
        format=format_type,
        extension_type=extension_type,
        grid_size=grid_size,
        **kwargs
    )
    
    # Create loader and load dataset
    loader = DatasetLoaderFactory.create_loader(config)
    return loader.load_dataset(file_path)

def split_dataset(
    dataset: LoadedDataset,
    train_split: Optional[float] = None,
    val_split: Optional[float] = None,
    random_seed: int = 42
) -> Tuple[LoadedDataset, LoadedDataset, Optional[LoadedDataset]]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        dataset: Loaded dataset to split
        train_split: Training set proportion (uses config default if None)
        val_split: Validation set proportion (uses config default if None)
        random_seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        test_dataset is None if train_split + val_split = 1.0
    
    Educational Note:
    Demonstrates how to handle dataset splitting while preserving
    metadata and configuration across split datasets.
    """
    np.random.seed(random_seed)
    
    # Use provided splits or defaults from config
    train_split = train_split or dataset.config.train_split
    val_split = val_split or dataset.config.val_split
    test_split = 1.0 - train_split - val_split
    
    if test_split < 0:
        raise ValueError("Train and validation splits cannot exceed 1.0")
    
    # Get data size and create indices
    data_size = dataset.size
    indices = np.random.permutation(data_size)
    
    # Calculate split points
    train_end = int(data_size * train_split)
    val_end = train_end + int(data_size * val_split)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:] if test_split > 0 else None
    
    # Split data based on type
    if isinstance(dataset.data, pd.DataFrame):
        train_data = dataset.data.iloc[train_indices].copy()
        val_data = dataset.data.iloc[val_indices].copy()
        test_data = dataset.data.iloc[test_indices].copy() if test_indices is not None else None
    elif isinstance(dataset.data, list):
        train_data = [dataset.data[i] for i in train_indices]
        val_data = [dataset.data[i] for i in val_indices]
        test_data = [dataset.data[i] for i in test_indices] if test_indices is not None else None
    elif isinstance(dataset.data, dict) and "features" in dataset.data:
        # Handle NPZ format
        train_data = {k: v[train_indices] for k, v in dataset.data.items()}
        val_data = {k: v[val_indices] for k, v in dataset.data.items()}
        test_data = {k: v[test_indices] for k, v in dataset.data.items()} if test_indices is not None else None
    else:
        raise ValueError(f"Unsupported data type for splitting: {type(dataset.data)}")
    
    # Create split datasets with updated metadata
    def create_split_dataset(data, split_name: str) -> LoadedDataset:
        metadata = dataset.metadata.copy()
        metadata["split"] = split_name
        metadata["split_size"] = len(train_indices) if split_name == "train" else (
            len(val_indices) if split_name == "val" else len(test_indices)
        )
        metadata["split_random_seed"] = random_seed
        return LoadedDataset(data=data, metadata=metadata, config=dataset.config)
    
    train_dataset = create_split_dataset(train_data, "train")
    val_dataset = create_split_dataset(val_data, "val")
    test_dataset = create_split_dataset(test_data, "test") if test_data is not None else None
    
    return train_dataset, val_dataset, test_dataset 