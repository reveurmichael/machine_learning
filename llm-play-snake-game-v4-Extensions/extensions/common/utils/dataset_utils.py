"""
Dataset utilities for Snake Game AI extensions.

This file follows the principles from final-decision-10.md:
- This file follows the principles from final-decision-10.md.
- All utilities must use simple print logging (simple logging).
- All utilities must be OOP, extensible, and never over-engineered.
- Reference: SimpleFactory in factory_utils.py is the canonical factory pattern for all extensions.
- See also: agents.md, core.md, config.md, factory-design-pattern.md, extension-evolution-rules.md.

This module provides dataset loading and saving utilities for the extensions package,
following simple logging with simple, OOP-based utilities that can be inherited
and extended without tight coupling to ML/DL/RL/LLM-specific concepts.

Key Functions:
    load_csv_dataset: Load CSV datasets with simple error handling
    save_csv_dataset: Save CSV datasets with simple error handling
    load_jsonl_dataset: Load JSONL datasets for LLM fine-tuning
    save_jsonl_dataset: Save JSONL datasets for LLM fine-tuning
    load_npz_dataset: Load NPZ datasets for numerical data
    save_npz_dataset: Save NPZ datasets for numerical data
    get_dataset_info: Get basic information about dataset files

Design Philosophy:
- Simple, object-oriented utilities that can be inherited and extended
- No tight coupling with ML/DL/RL/LLM-specific concepts
- Simple logging with print() statements (simple logging)
- Enables easy addition of new extensions without friction
- All code examples use print() and create() as canonical patterns.

Reference: docs/extensions-guideline/final-decision-10.md
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import json

from utils.print_utils import print_info

# Import basic format specifications
from ..config.dataset_formats import (
    COMMON_DATASET_EXTENSIONS
)


def load_csv_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load CSV dataset with basic error handling.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        pandas DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
        
    Follows simple logging: Simple logging with print() statements.
    """
    file_path = Path(file_path)
    print_info(f"Loading CSV dataset: {file_path}", "DatasetUtils")
    
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print_info(f"CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns", "DatasetUtils")
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Empty CSV file: {file_path}")


def load_jsonl_dataset(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load JSONL dataset with basic error handling.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
        
    Follows simple logging: Simple logging with print() statements.
    """
    file_path = Path(file_path)
    print_info(f"Loading JSONL dataset: {file_path}", "DatasetUtils")
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON on line {line_num}: {e.msg}",
                        e.doc, e.pos
                    )
    
    print_info(f"JSONL loaded successfully: {len(records)} records", "DatasetUtils")
    return records


def load_npz_dataset(file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load NPZ dataset with basic error handling.
    
    Args:
        file_path: Path to NPZ file
        
    Returns:
        Dictionary of arrays
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Follows simple logging: Simple logging with print() statements.
    """
    file_path = Path(file_path)
    print_info(f"Loading NPZ dataset: {file_path}", "DatasetUtils")
    
    if not file_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {file_path}")
    
    npz_file = np.load(file_path)
    arrays = dict(npz_file)
    print_info(f"NPZ loaded successfully: {len(arrays)} arrays", "DatasetUtils")
    return arrays


def save_csv_dataset(data: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """
    Save DataFrame to CSV with basic error handling.
    
    Args:
        data: pandas DataFrame to save
        file_path: Path where to save CSV file
        
    Follows simple logging: Simple logging with print() statements.
    """
    file_path = Path(file_path)
    print_info(f"Saving CSV dataset: {file_path}", "DatasetUtils")
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(file_path, index=False)
    print_info(f"CSV saved successfully: {len(data)} rows", "DatasetUtils")


def save_jsonl_dataset(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save list of dictionaries to JSONL with basic error handling.
    
    Args:
        data: List of dictionaries to save
        file_path: Path where to save JSONL file
        
    Follows simple logging: Simple logging with print() statements.
    """
    file_path = Path(file_path)
    print_info(f"Saving JSONL dataset: {file_path}", "DatasetUtils")
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    
    print_info(f"JSONL saved successfully: {len(data)} records", "DatasetUtils")


def save_npz_dataset(data: Dict[str, np.ndarray], file_path: Union[str, Path]) -> None:
    """
    Save dictionary of arrays to NPZ with basic error handling.
    
    Args:
        data: Dictionary of numpy arrays to save
        file_path: Path where to save NPZ file
        
    Follows simple logging: Simple logging with print() statements.
    """
    file_path = Path(file_path)
    print_info(f"Saving NPZ dataset: {file_path}", "DatasetUtils")
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(file_path, **data)
    print_info(f"NPZ saved successfully: {len(data)} arrays", "DatasetUtils")


def get_dataset_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get basic information about a dataset file.
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        Dictionary with basic file information
        
    Follows simple logging: Simple logging with print() statements.
    """
    file_path = Path(file_path)
    print_info(f"Getting dataset info: {file_path}", "DatasetUtils")
    
    if not file_path.exists():
        return {"exists": False, "error": f"File not found: {file_path}"}
    
    info = {
        "exists": True,
        "path": str(file_path),
        "name": file_path.name,
        "extension": file_path.suffix,
        "size_bytes": file_path.stat().st_size
    }
    
    # Try to get more specific info based on file type
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            info.update({
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns)
            })
        elif file_path.suffix == '.jsonl':
            with open(file_path, 'r') as f:
                lines = sum(1 for line in f if line.strip())
            info.update({"records": lines})
        elif file_path.suffix == '.npz':
            npz_file = np.load(file_path)
            info.update({
                "arrays": list(npz_file.keys()),
                "array_shapes": {key: npz_file[key].shape for key in npz_file.keys()}
            })
    except Exception as e:
        info["error"] = str(e)
    
    print_info(f"Dataset info retrieved: {info.get('rows', info.get('records', 'unknown'))} items", "DatasetUtils")
    return info


def guess_dataset_format(file_path: Union[str, Path]) -> Optional[str]:
    """
    Guess dataset format based on file extension.
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        Format type string or None if unknown
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    for format_type, extensions in COMMON_DATASET_EXTENSIONS.items():
        if extension in extensions:
            return format_type
    
    return None 


# ---------------------------------------------------------------------------
# Lightweight OOP façade (simple logging)
# ---------------------------------------------------------------------------
class DatasetIO:
    """Tiny wrapper class that groups dataset IO helpers.

    Design Goal (simple logging):
        • Keep the implementation *dead simple* – essentially just delegate to
          the existing standalone functions.
        • Provide an extensibility point for specialised extensions that may
          want to override or augment a single method without editing the
          common base.

    Usage (both styles work):
        >>> df = load_csv_dataset("file.csv")           # Functional
        >>> df = DatasetIO().load_csv("file.csv")       # OOP façade
    """

    # CSV -------------------------------------------------------------------
    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:  # noqa: D401
        return load_csv_dataset(file_path)

    def save_csv(self, data: pd.DataFrame, file_path: Union[str, Path]) -> None:  # noqa: D401
        save_csv_dataset(data, file_path)

    # JSONL -----------------------------------------------------------------
    def load_jsonl(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:  # noqa: D401
        return load_jsonl_dataset(file_path)

    def save_jsonl(self, data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:  # noqa: D401
        save_jsonl_dataset(data, file_path)

    # NPZ -------------------------------------------------------------------
    def load_npz(self, file_path: Union[str, Path]) -> Dict[str, np.ndarray]:  # noqa: D401
        return load_npz_dataset(file_path)

    def save_npz(self, data: Dict[str, np.ndarray], file_path: Union[str, Path]) -> None:  # noqa: D401
        save_npz_dataset(data, file_path)

    # Info ------------------------------------------------------------------
    def info(self, file_path: Union[str, Path]) -> Dict[str, Any]:  # noqa: D401
        return get_dataset_info(file_path)

    # Format guessing -------------------------------------------------------
    def guess_format(self, file_path: Union[str, Path]) -> Optional[str]:  # noqa: D401
        return guess_dataset_format(file_path)


# Default instance for quick access ------------------------------------------------
# Extensions that like an OOP style can `from ... import dataset_io` and use it.
# The object is intentionally *stateless* so sharing it is harmless.
dataset_io = DatasetIO()

# ---------------------------------------------------------------------------
# Public re-exports ----------------------------------------------------------
__all__ = [
    # Functional API
    "load_csv_dataset",
    "save_csv_dataset",
    "load_jsonl_dataset",
    "save_jsonl_dataset",
    "load_npz_dataset",
    "save_npz_dataset",
    "get_dataset_info",
    "guess_dataset_format",
    # OOP façade
    "DatasetIO",
    "dataset_io",
] 

class DatasetLoader:
    """
    Loads and preprocesses datasets for extensions.
    
    - OOP extensibility: subclass for custom behavior.
    - Logging is always simple (print()).
    - No ML/DL/RL/LLM-specific coupling.
    """
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        print_info(f"Initialized for grid size: {grid_size}", "DatasetLoader")

    def load_csv_dataset(self, path: str) -> pd.DataFrame:
        """
        Load a CSV dataset.
        Args:
            path: Path to the CSV file.
        Returns:
            DataFrame with loaded data.
        """
        print_info(f"Loading CSV dataset from: {path}", "DatasetLoader")
        df = pd.read_csv(path)
        print_info(f"Loaded {len(df)} rows.", "DatasetLoader")
        return df

    def prepare_features_and_targets(self, df: pd.DataFrame, scale_features: bool = False) -> Tuple[Any, Any]:
        """
        Prepare features and targets from DataFrame.
        Args:
            df: Input DataFrame.
            scale_features: Whether to scale features (default: False).
        Returns:
            Tuple of (features, targets).
        """
        print_info("Preparing features and targets.", "DatasetLoader")
        X = df.drop(columns=["target_move"]).values
        y = df["target_move"].values
        if scale_features:
            print_info("Scaling features (not implemented, placeholder).", "DatasetLoader")
            # Implement scaling if needed
        return X, y

    def split_dataset(self, X, y, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """
        Split dataset into train/val/test.
        Args:
            X: Features.
            y: Targets.
            val_ratio: Validation split ratio.
            test_ratio: Test split ratio.
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        print_info(f"Splitting dataset: val_ratio={val_ratio}, test_ratio={test_ratio}", "DatasetLoader")
        n = len(X)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        X_train, X_val, X_test = X[:n-n_val-n_test], X[n-n_val-n_test:n-n_test], X[n-n_test:]
        y_train, y_val, y_test = y[:n-n_val-n_test], y[n-n_val-n_test:n-n_test], y[n-n_test:]
        print_info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}", "DatasetLoader")
        return X_train, X_val, X_test, y_train, y_val, y_test 