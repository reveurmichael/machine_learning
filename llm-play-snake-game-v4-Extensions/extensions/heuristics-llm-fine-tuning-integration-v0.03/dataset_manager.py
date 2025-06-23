"""dataset_manager.py - Dataset Management for v0.03

Provides dataset discovery, preprocessing, and management functionality
for the web interface with enhanced user experience.

Key Features:
- Automatic dataset discovery across grid sizes
- Interactive preprocessing with format conversion
- Dataset validation and quality checks
- Metadata extraction and caching
- Progress tracking for long operations

Design Patterns:
- Repository Pattern: Centralized dataset access
- Observer Pattern: Progress notifications
- Strategy Pattern: Different preprocessing strategies
- Factory Pattern: Dataset format creation
"""

from __future__ import annotations

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from extensions.common import training_logging_utils, dataset_utils

logger = training_logging_utils.TrainingLogger("dataset_manager")


class DatasetFormat(Enum):
    """Supported dataset formats."""
    CSV = "csv"
    JSONL = "jsonl"
    PARQUET = "parquet"
    NPZ = "npz"


class DatasetType(Enum):
    """Dataset type categories."""
    TABULAR = "tabular"
    SEQUENTIAL = "sequential"
    GRAPH = "graph"
    RAW = "raw"


@dataclass
class DatasetInfo:
    """Information about a dataset.
    
    Design Pattern: Value Object
    - Immutable container for dataset metadata
    - Provides serialization capabilities
    - Type-safe dataset information
    """
    
    name: str
    path: str
    format: DatasetFormat
    type: DatasetType
    grid_size: int
    sample_count: int
    feature_count: int
    size_bytes: int
    created_date: datetime
    modified_date: datetime
    algorithm: str = "unknown"
    checksum: str = ""
    validation_status: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'path': self.path,
            'format': self.format.value,
            'type': self.type.value,
            'grid_size': self.grid_size,
            'sample_count': self.sample_count,
            'feature_count': self.feature_count,
            'size_mb': round(self.size_bytes / (1024 * 1024), 2),
            'created_date': self.created_date.isoformat(),
            'modified_date': self.modified_date.isoformat(),
            'algorithm': self.algorithm,
            'checksum': self.checksum,
            'validation_status': self.validation_status
        }
    
    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def is_valid(self) -> bool:
        """Check if dataset is valid."""
        return self.validation_status == "valid"


@dataclass
class PreprocessingConfig:
    """Configuration for dataset preprocessing.
    
    Design Pattern: Configuration Object
    - Encapsulates preprocessing parameters
    - Provides validation and defaults
    - Supports serialization for persistence
    """
    
    output_format: DatasetFormat = DatasetFormat.JSONL
    max_samples: int = 10000
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    random_seed: int = 42
    include_metadata: bool = True
    normalize_features: bool = False
    remove_duplicates: bool = True
    min_sequence_length: int = 5
    max_sequence_length: int = 1000
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate split ratios
        total_split = self.train_split + self.validation_split + self.test_split
        if abs(total_split - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")
        
        # Validate other parameters
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        
        if not 0 < self.train_split < 1:
            raise ValueError("train_split must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class DatasetProgressTracker:
    """Tracks progress for long-running dataset operations.
    
    Design Pattern: Observer Pattern
    - Notifies observers of progress updates
    - Provides standardized progress interface
    - Supports cancellation and error handling
    """
    
    def __init__(self, total_steps: int, operation_name: str = "Operation"):
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.start_time = datetime.now()
        self.observers = []
        self.is_cancelled = False
        self.error_message = None
    
    def add_observer(self, observer):
        """Add progress observer."""
        self.observers.append(observer)
    
    def remove_observer(self, observer):
        """Remove progress observer."""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def update_progress(self, step: int, message: str = ""):
        """Update progress and notify observers."""
        self.current_step = step
        progress_data = {
            'operation': self.operation_name,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'percentage': (self.current_step / self.total_steps) * 100,
            'message': message,
            'elapsed_time': (datetime.now() - self.start_time).total_seconds(),
            'is_cancelled': self.is_cancelled,
            'error': self.error_message
        }
        
        for observer in self.observers:
            observer(progress_data)
    
    def cancel(self):
        """Cancel the operation."""
        self.is_cancelled = True
        self.update_progress(self.current_step, "Operation cancelled")
    
    def error(self, message: str):
        """Report an error."""
        self.error_message = message
        self.update_progress(self.current_step, f"Error: {message}")


class DatasetManager:
    """Main dataset management class for v0.03.
    
    Design Pattern: Repository Pattern
    - Provides centralized access to dataset operations
    - Abstracts underlying storage mechanisms
    - Supports caching and metadata management
    """
    
    def __init__(self, base_path: str = "logs/extensions/datasets"):
        self.base_path = Path(base_path)
        self.cache = {}
        self.metadata_cache = {}
        self.progress_tracker = None
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DatasetManager initialized with base path: {self.base_path}")
    
    def discover_datasets(self, force_refresh: bool = False) -> List[DatasetInfo]:
        """Discover all available datasets.
        
        Args:
            force_refresh: Force refresh of cached data
            
        Returns:
            List of discovered datasets
        """
        if not force_refresh and 'all_datasets' in self.cache:
            return self.cache['all_datasets']
        
        logger.info("Discovering datasets...")
        datasets = []
        
        # Scan grid size directories
        for grid_dir in self.base_path.glob("grid-size-*"):
            if not grid_dir.is_dir():
                continue
            
            try:
                grid_size = int(grid_dir.name.split("-")[-1])
            except ValueError:
                logger.warning(f"Invalid grid directory name: {grid_dir.name}")
                continue
            
            # Scan dataset files
            for file_path in grid_dir.iterdir():
                if file_path.is_file() and file_path.suffix in ['.csv', '.jsonl', '.parquet', '.npz']:
                    try:
                        dataset_info = self._analyze_dataset(file_path, grid_size)
                        datasets.append(dataset_info)
                    except Exception as e:
                        logger.warning(f"Failed to analyze dataset {file_path}: {e}")
        
        # Cache results
        self.cache['all_datasets'] = datasets
        logger.info(f"Discovered {len(datasets)} datasets")
        
        return datasets
    
    def get_datasets_by_grid_size(self, grid_size: int) -> List[DatasetInfo]:
        """Get datasets for a specific grid size."""
        all_datasets = self.discover_datasets()
        return [ds for ds in all_datasets if ds.grid_size == grid_size]
    
    def get_datasets_by_algorithm(self, algorithm: str) -> List[DatasetInfo]:
        """Get datasets for a specific algorithm."""
        all_datasets = self.discover_datasets()
        return [ds for ds in all_datasets if algorithm.lower() in ds.algorithm.lower()]
    
    def get_datasets_by_format(self, format_type: DatasetFormat) -> List[DatasetInfo]:
        """Get datasets of a specific format."""
        all_datasets = self.discover_datasets()
        return [ds for ds in all_datasets if ds.format == format_type]
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all datasets."""
        datasets = self.discover_datasets()
        
        if not datasets:
            return {
                'total_count': 0,
                'total_size_mb': 0.0,
                'formats': {},
                'grid_sizes': {},
                'algorithms': {}
            }
        
        # Calculate statistics
        total_size = sum(ds.size_bytes for ds in datasets)
        
        # Count by format
        formats = {}
        for ds in datasets:
            formats[ds.format.value] = formats.get(ds.format.value, 0) + 1
        
        # Count by grid size
        grid_sizes = {}
        for ds in datasets:
            grid_sizes[ds.grid_size] = grid_sizes.get(ds.grid_size, 0) + 1
        
        # Count by algorithm
        algorithms = {}
        for ds in datasets:
            algorithms[ds.algorithm] = algorithms.get(ds.algorithm, 0) + 1
        
        return {
            'total_count': len(datasets),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'formats': formats,
            'grid_sizes': grid_sizes,
            'algorithms': algorithms,
            'avg_samples': round(sum(ds.sample_count for ds in datasets) / len(datasets)),
            'latest_modified': max(ds.modified_date for ds in datasets).isoformat()
        }
    
    def preprocess_datasets(self, 
                          dataset_names: List[str],
                          config: PreprocessingConfig,
                          progress_callback = None) -> Dict[str, Any]:
        """Preprocess multiple datasets with given configuration.
        
        Args:
            dataset_names: List of dataset names to preprocess
            config: Preprocessing configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Results of preprocessing operation
        """
        logger.info(f"Starting preprocessing of {len(dataset_names)} datasets")
        
        # Setup progress tracking
        total_steps = len(dataset_names) * 3  # Load, process, save
        self.progress_tracker = DatasetProgressTracker(total_steps, "Dataset Preprocessing")
        
        if progress_callback:
            self.progress_tracker.add_observer(progress_callback)
        
        results = {
            'success': True,
            'processed_datasets': [],
            'failed_datasets': [],
            'output_files': [],
            'statistics': {}
        }
        
        try:
            output_dir = self.base_path / f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir.mkdir(exist_ok=True)
            
            step = 0
            
            for dataset_name in dataset_names:
                if self.progress_tracker.is_cancelled:
                    break
                
                try:
                    # Find dataset
                    datasets = self.discover_datasets()
                    dataset = next((ds for ds in datasets if ds.name == dataset_name), None)
                    
                    if not dataset:
                        raise ValueError(f"Dataset not found: {dataset_name}")
                    
                    # Load dataset
                    step += 1
                    self.progress_tracker.update_progress(step, f"Loading {dataset_name}")
                    data = self._load_dataset(dataset)
                    
                    # Process dataset
                    step += 1
                    self.progress_tracker.update_progress(step, f"Processing {dataset_name}")
                    processed_data = self._process_dataset(data, config)
                    
                    # Save processed dataset
                    step += 1
                    output_file = output_dir / f"{dataset.name.split('.')[0]}_processed.{config.output_format.value}"
                    self.progress_tracker.update_progress(step, f"Saving {output_file.name}")
                    self._save_dataset(processed_data, output_file, config.output_format)
                    
                    results['processed_datasets'].append(dataset_name)
                    results['output_files'].append(str(output_file))
                    
                except Exception as e:
                    logger.error(f"Failed to process dataset {dataset_name}: {e}")
                    results['failed_datasets'].append({
                        'name': dataset_name,
                        'error': str(e)
                    })
            
            # Calculate statistics
            results['statistics'] = {
                'total_processed': len(results['processed_datasets']),
                'total_failed': len(results['failed_datasets']),
                'output_directory': str(output_dir),
                'processing_time': (datetime.now() - self.progress_tracker.start_time).total_seconds()
            }
            
            logger.info(f"Preprocessing completed: {results['statistics']}")
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            self.progress_tracker.error(str(e))
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def validate_dataset(self, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Validate a dataset and return validation results."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check file exists
            if not Path(dataset_info.path).exists():
                validation_results['errors'].append("File does not exist")
                validation_results['is_valid'] = False
                return validation_results
            
            # Load and analyze dataset
            data = self._load_dataset(dataset_info)
            
            # Check for empty dataset
            if len(data) == 0:
                validation_results['errors'].append("Dataset is empty")
                validation_results['is_valid'] = False
            
            # Check for duplicates
            if dataset_info.format == DatasetFormat.CSV:
                duplicates = data.duplicated().sum()
                if duplicates > 0:
                    validation_results['warnings'].append(f"Found {duplicates} duplicate rows")
            
            # Check for missing values
            if dataset_info.format == DatasetFormat.CSV:
                missing_values = data.isnull().sum().sum()
                if missing_values > 0:
                    validation_results['warnings'].append(f"Found {missing_values} missing values")
            
            # Dataset-specific validation
            if 'target_move' in data.columns:
                # Validate target moves
                valid_moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                invalid_moves = data[~data['target_move'].isin(valid_moves)]['target_move'].unique()
                if len(invalid_moves) > 0:
                    validation_results['errors'].append(f"Invalid target moves: {invalid_moves}")
                    validation_results['is_valid'] = False
            
            # Calculate statistics
            validation_results['statistics'] = {
                'sample_count': len(data),
                'feature_count': len(data.columns) if hasattr(data, 'columns') else 0,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024) if hasattr(data, 'memory_usage') else 0
            }
            
        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    def _analyze_dataset(self, file_path: Path, grid_size: int) -> DatasetInfo:
        """Analyze a dataset file and extract metadata."""
        # Determine format
        format_map = {
            '.csv': DatasetFormat.CSV,
            '.jsonl': DatasetFormat.JSONL,
            '.parquet': DatasetFormat.PARQUET,
            '.npz': DatasetFormat.NPZ
        }
        
        format_type = format_map.get(file_path.suffix, DatasetFormat.CSV)
        
        # Get file statistics
        stat = file_path.stat()
        
        # Extract algorithm from filename
        algorithm = "unknown"
        name_lower = file_path.name.lower()
        algorithms = ['bfs', 'astar', 'dfs', 'hamiltonian', 'greedy']
        for alg in algorithms:
            if alg in name_lower:
                algorithm = alg.upper()
                break
        
        # Determine type based on filename
        dataset_type = DatasetType.TABULAR
        if 'sequential' in name_lower:
            dataset_type = DatasetType.SEQUENTIAL
        elif 'graph' in name_lower:
            dataset_type = DatasetType.GRAPH
        
        # Try to get sample and feature counts
        sample_count = 0
        feature_count = 0
        
        try:
            if format_type == DatasetFormat.CSV:
                # Quick scan for CSV
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                    feature_count = len(first_line.split(','))
                    sample_count = sum(1 for _ in f)  # Remaining lines
            elif format_type == DatasetFormat.JSONL:
                # Quick scan for JSONL
                with open(file_path, 'r') as f:
                    sample_count = sum(1 for _ in f)
                    # For feature count, would need to parse first line
                    feature_count = 1  # Placeholder
        except Exception:
            # If quick scan fails, leave as 0
            pass
        
        return DatasetInfo(
            name=file_path.name,
            path=str(file_path),
            format=format_type,
            type=dataset_type,
            grid_size=grid_size,
            sample_count=sample_count,
            feature_count=feature_count,
            size_bytes=stat.st_size,
            created_date=datetime.fromtimestamp(stat.st_ctime),
            modified_date=datetime.fromtimestamp(stat.st_mtime),
            algorithm=algorithm
        )
    
    def _load_dataset(self, dataset_info: DatasetInfo) -> Union[pd.DataFrame, List[Dict], np.ndarray]:
        """Load a dataset based on its format."""
        file_path = Path(dataset_info.path)
        
        if dataset_info.format == DatasetFormat.CSV:
            return pd.read_csv(file_path)
        elif dataset_info.format == DatasetFormat.JSONL:
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        elif dataset_info.format == DatasetFormat.PARQUET:
            return pd.read_parquet(file_path)
        elif dataset_info.format == DatasetFormat.NPZ:
            return np.load(file_path)
        else:
            raise ValueError(f"Unsupported format: {dataset_info.format}")
    
    def _process_dataset(self, data, config: PreprocessingConfig):
        """Process dataset according to configuration."""
        # Apply preprocessing steps based on data type
        if isinstance(data, pd.DataFrame):
            return self._process_dataframe(data, config)
        elif isinstance(data, list):
            return self._process_jsonl_data(data, config)
        else:
            # For other formats, return as-is for now
            return data
    
    def _process_dataframe(self, df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
        """Process pandas DataFrame."""
        # Remove duplicates
        if config.remove_duplicates:
            original_len = len(df)
            df = df.drop_duplicates()
            logger.info(f"Removed {original_len - len(df)} duplicate rows")
        
        # Shuffle data
        if config.shuffle:
            df = df.sample(frac=1, random_state=config.random_seed).reset_index(drop=True)
        
        # Limit samples
        if len(df) > config.max_samples:
            df = df.head(config.max_samples)
            logger.info(f"Limited to {config.max_samples} samples")
        
        # Normalize features (if specified)
        if config.normalize_features:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
        
        return df
    
    def _process_jsonl_data(self, data: List[Dict], config: PreprocessingConfig) -> List[Dict]:
        """Process JSONL data."""
        # Remove duplicates
        if config.remove_duplicates:
            seen = set()
            unique_data = []
            for item in data:
                item_str = json.dumps(item, sort_keys=True)
                if item_str not in seen:
                    seen.add(item_str)
                    unique_data.append(item)
            data = unique_data
        
        # Shuffle data
        if config.shuffle:
            np.random.seed(config.random_seed)
            np.random.shuffle(data)
        
        # Limit samples
        if len(data) > config.max_samples:
            data = data[:config.max_samples]
        
        return data
    
    def _save_dataset(self, data, output_path: Path, format_type: DatasetFormat):
        """Save processed dataset in specified format."""
        if format_type == DatasetFormat.CSV and isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        elif format_type == DatasetFormat.JSONL:
            with open(output_path, 'w') as f:
                if isinstance(data, pd.DataFrame):
                    for _, row in data.iterrows():
                        f.write(json.dumps(row.to_dict()) + '\n')
                elif isinstance(data, list):
                    for item in data:
                        f.write(json.dumps(item) + '\n')
        elif format_type == DatasetFormat.PARQUET and isinstance(data, pd.DataFrame):
            data.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Cannot save data type {type(data)} to format {format_type}")
    
    def cleanup_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.metadata_cache.clear()
        logger.info("Cache cleared")


# Factory functions for easy instantiation
def create_dataset_manager(base_path: str = None) -> DatasetManager:
    """Create a DatasetManager instance."""
    if base_path is None:
        base_path = "logs/extensions/datasets"
    return DatasetManager(base_path)


def create_preprocessing_config(**kwargs) -> PreprocessingConfig:
    """Create a preprocessing configuration with custom parameters."""
    return PreprocessingConfig(**kwargs) 