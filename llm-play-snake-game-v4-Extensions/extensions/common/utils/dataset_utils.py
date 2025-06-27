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
from ..config.dataset_formats import (
    CSV_COLUMN_NAMES, JSONL_REQUIRED_FIELDS, NPZ_ARRAY_NAMES
)
from ..config.training_defaults import (
    DEFAULT_TRAIN_SPLIT, DEFAULT_VAL_SPLIT, DEFAULT_BATCH_SIZE
)
from ..config.validation_rules import DATASET_VALIDATION_RULES
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
    JSONL dataset loader for language-rich data.
    
    Purpose: Handle JSONL-specific loading for LLM fine-tuning data
    """
    
    def _load_raw_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL data line by line."""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {line_num}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load JSONL file: {e}")
        
        return data
    
    def _validate_data_format(self, data: List[Dict[str, Any]]) -> None:
        """Validate JSONL data format."""
        if not data:
            raise ValueError("JSONL file is empty")
        
        # Check required fields
        for i, item in enumerate(data[:10]):  # Check first 10 items
            missing_fields = set(JSONL_REQUIRED_FIELDS) - set(item.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields in item {i}: {missing_fields}")
    
    def _preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply JSONL-specific preprocessing."""
        # Filter by extension type
        if self.config.extension_type == ExtensionType.LLM:
            # Ensure all items have valid prompt and completion
            filtered_data = []
            for item in data:
                if item.get("prompt") and item.get("completion"):
                    filtered_data.append(item)
            return filtered_data
        
        return data
    
    def _get_data_size(self, data: List[Dict[str, Any]]) -> int:
        """Get number of items in list."""
        return len(data)


class NPZDatasetLoader(BaseDatasetLoader):
    """
    NPZ dataset loader for numerical array data.
    
    Purpose: Handle NPZ-specific loading for RL/ML numerical data
    """
    
    def _load_raw_data(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load NPZ data using numpy."""
        try:
            npz_file = np.load(file_path)
            return dict(npz_file)
        except Exception as e:
            raise ValueError(f"Failed to load NPZ file: {e}")
    
    def _validate_data_format(self, data: Dict[str, np.ndarray]) -> None:
        """Validate NPZ data format."""
        if not data:
            raise ValueError("NPZ file is empty")
        
        # Check for required arrays
        required_arrays = NPZ_ARRAY_NAMES.get(self.config.extension_type.value, [])
        missing_arrays = set(required_arrays) - set(data.keys())
        if missing_arrays:
            raise ValueError(f"Missing required arrays: {missing_arrays}")
    
    def _preprocess_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply NPZ-specific preprocessing."""
        processed_data = {}
        
        for key, array in data.items():
            # Ensure all arrays are proper numpy arrays
            processed_data[key] = np.asarray(array)
        
        # Extension-specific preprocessing
        if self.config.extension_type == ExtensionType.REINFORCEMENT:
            # Normalize rewards if present
            if "rewards" in processed_data:
                rewards = processed_data["rewards"]
                if len(rewards) > 0:
                    processed_data["rewards"] = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return processed_data
    
    def _get_data_size(self, data: Dict[str, np.ndarray]) -> int:
        """Get size of the first array (assuming all arrays have same first dimension)."""
        if not data:
            return 0
        first_array = next(iter(data.values()))
        return first_array.shape[0]


# =============================================================================
# Factory Pattern Implementation
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
            available_formats = list(cls._loaders.keys())
            raise ValueError(f"Unsupported format: {config.format}. Available: {available_formats}")
        
        return loader_class(config)
    
    @classmethod
    def register_loader(cls, format_type: DatasetFormat, loader_class: type) -> None:
        """Register a new dataset loader for a specific format."""
        cls._loaders[format_type] = loader_class


# =============================================================================
# Convenience Functions
# =============================================================================

def load_dataset_for_training(
    file_path: Union[str, Path],
    extension_type: ExtensionType,
    format_type: Optional[DatasetFormat] = None,
    grid_size: Optional[int] = None,
    **kwargs
) -> LoadedDataset:
    """
    Convenience function to load a dataset for training.
    
    Args:
        file_path: Path to the dataset file
        extension_type: Type of extension (heuristics, supervised, etc.)
        format_type: Format of the dataset (auto-detected if None)
        grid_size: Grid size filter (optional)
        **kwargs: Additional configuration options
    
    Returns:
        LoadedDataset: The loaded dataset with metadata
    
    Example:
        >>> dataset = load_dataset_for_training(
        ...     "data/heuristics_v0.04_20240101_120000.csv",
        ...     ExtensionType.SUPERVISED,
        ...     grid_size=10
        ... )
    """
    file_path = Path(file_path)
    
    # Auto-detect format if not provided
    if format_type is None:
        suffix_to_format = {
            ".csv": DatasetFormat.CSV,
            ".jsonl": DatasetFormat.JSONL,
            ".npz": DatasetFormat.NPZ,
        }
        format_type = suffix_to_format.get(file_path.suffix)
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
    Split a dataset into training, validation, and test sets.
    
    Args:
        dataset: The dataset to split
        train_split: Training split ratio (uses config default if None)
        val_split: Validation split ratio (uses config default if None)
        random_seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        test_dataset is None if train_split + val_split = 1.0
    """
    if train_split is None:
        train_split = dataset.config.train_split
    if val_split is None:
        val_split = dataset.config.val_split
    
    test_split = 1.0 - train_split - val_split
    if test_split < 0:
        raise ValueError("Train and validation splits cannot exceed 1.0")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Handle different data types
    if isinstance(dataset.data, pd.DataFrame):
        data = dataset.data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        total_size = len(data)
        
        train_end = int(total_size * train_split)
        val_end = train_end + int(total_size * val_split)
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:] if test_split > 0 else None
        
    elif isinstance(dataset.data, list):
        data = dataset.data.copy()
        np.random.shuffle(data)
        total_size = len(data)
        
        train_end = int(total_size * train_split)
        val_end = train_end + int(total_size * val_split)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:] if test_split > 0 else None
        
    elif isinstance(dataset.data, dict):
        # Handle NPZ data (dict of arrays)
        # Shuffle indices
        first_key = next(iter(dataset.data.keys()))
        total_size = dataset.data[first_key].shape[0]
        indices = np.random.permutation(total_size)
        
        train_end = int(total_size * train_split)
        val_end = train_end + int(total_size * val_split)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:] if test_split > 0 else None
        
        train_data = {k: v[train_indices] for k, v in dataset.data.items()}
        val_data = {k: v[val_indices] for k, v in dataset.data.items()}
        test_data = {k: v[test_indices] for k, v in dataset.data.items()} if test_indices is not None else None
        
    else:
        raise ValueError(f"Unsupported data type for splitting: {type(dataset.data)}")
    
    # Create split datasets
    def create_split_dataset(data, split_name: str) -> LoadedDataset:
        split_metadata = dataset.metadata.copy()
        split_metadata.update({
            "split_type": split_name,
            "split_size": len(data) if hasattr(data, '__len__') else data[first_key].shape[0],
            "original_size": dataset.size
        })
        
        return LoadedDataset(
            data=data,
            metadata=split_metadata,
            config=dataset.config
        )
    
    train_dataset = create_split_dataset(train_data, "train")
    val_dataset = create_split_dataset(val_data, "validation")
    test_dataset = create_split_dataset(test_data, "test") if test_data is not None else None
    
    return train_dataset, val_dataset, test_dataset 