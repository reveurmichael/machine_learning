"""
Simple Dataset Utilities for Snake Game Extensions.

This module provides basic dataset loading utilities that are
generic and can be used across different extensions.

Design Philosophy: Keep it Simple
- Basic file loading functions
- Generic utilities without complex validation
- Extensions can add their own specific logic
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Import basic format specifications
from ..config.dataset_formats import (
    CSV_BASIC_COLUMNS, JSONL_BASIC_KEYS, COMMON_DATASET_EXTENSIONS
)
from .path_utils import ensure_project_root_on_path


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
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    try:
        return pd.read_csv(file_path)
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
    """
    file_path = Path(file_path)
    
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
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {file_path}")
    
    npz_file = np.load(file_path)
    return dict(npz_file)


def save_csv_dataset(data: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """
    Save DataFrame to CSV with basic error handling.
    
    Args:
        data: pandas DataFrame to save
        file_path: Path where to save CSV file
    """
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(file_path, index=False)


def save_jsonl_dataset(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save list of dictionaries to JSONL with basic error handling.
    
    Args:
        data: List of dictionaries to save
        file_path: Path where to save JSONL file
    """
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')


def save_npz_dataset(data: Dict[str, np.ndarray], file_path: Union[str, Path]) -> None:
    """
    Save dictionary of arrays to NPZ with basic error handling.
    
    Args:
        data: Dictionary of numpy arrays to save
        file_path: Path where to save NPZ file
    """
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(file_path, **data)


def get_dataset_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get basic information about a dataset file.
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        Dictionary with basic file information
    """
    file_path = Path(file_path)
    
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
        info["warning"] = f"Could not read file details: {str(e)}"
    
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