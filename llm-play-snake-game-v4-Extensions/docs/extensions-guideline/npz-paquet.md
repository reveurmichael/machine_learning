# Multi-Format Dataset Architecture for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines. Multi-format datasets follow the same architectural patterns established in the GOODFILES.

## 🎯 **Core Philosophy: Format-Agnostic Data Pipeline**

The multi-format dataset architecture provides flexible data storage and loading capabilities to support diverse machine learning approaches. Different model types require different data representations, and this architecture ensures optimal performance for each approach.

### **Design Philosophy**
- **Format Specialization**: Each format optimized for specific model types
- **Unified Interface**: Consistent data loading regardless of format
- **Storage Efficiency**: Optimal storage characteristics for each use case
- **Cross-Extension Compatibility**: Standardized formats across all extensions

## 📊 **Format Selection Strategy**

### **Format Mapping by Model Type**
Following the established ML architecture patterns:

| Model Type | Primary Format | Secondary Format | Use Case |
|------------|---------------|------------------|----------|
| **Tree Models** (XGBoost, LightGBM) | CSV | Parquet | Tabular feature engineering |
| **Simple Neural** (MLP) | CSV | NPZ | Fixed-size feature vectors |
| **Sequential Models** (LSTM, GRU) | NPZ | Parquet | Time-series sequences |
| **Convolutional** (CNN) | NPZ | - | Grid-based spatial data |
| **Graph Models** (GNN) | Parquet | NPZ | Complex relational structures |

### **Format Characteristics and Benefits**

#### **CSV Format**
- **Best For**: Tree models, simple neural networks, debugging
- **Benefits**: Human-readable, universal compatibility, easy debugging
- **Storage**: Text-based, moderate efficiency
- **Loading**: Fast for tabular data, built-in pandas support

#### **NPZ Format**
- **Best For**: Sequential models, CNNs, high-dimensional data
- **Benefits**: Preserves array shapes, fast I/O, binary compression
- **Storage**: Highly efficient for numerical arrays
- **Loading**: Native NumPy support, preserves data types

#### **Parquet Format**
- **Best For**: Large datasets, complex schemas, graph data
- **Benefits**: Columnar storage, excellent compression, schema evolution
- **Storage**: Most efficient for heterogeneous data
- **Loading**: Partial reading, excellent for big data workflows

## 🏗️ **Factory Pattern Dataset Architecture**

### **Multi-Format Dataset Factory**
Following Final Decision 7-8 factory patterns:

```python
class DatasetFormatFactory:
    """Factory for creating format-specific dataset handlers"""
    
    _format_registry = {
        "csv": CSVDatasetHandler,
        "npz": NPZDatasetHandler,
        "parquet": ParquetDatasetHandler,
    }
    
    @classmethod
    def create_handler(cls, format_type: str, **kwargs) -> BaseDatasetHandler:
        """Create dataset handler by format type"""
        handler_class = cls._format_registry.get(format_type.lower())
        if not handler_class:
            raise ValueError(f"Unsupported format: {format_type}")
        return handler_class(**kwargs)
```

### **Universal Dataset Interface**
```python
class BaseDatasetHandler:
    """Base class for all dataset format handlers"""
    
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.supported_formats = self._get_supported_formats()
        
    @abstractmethod
    def save_dataset(self, data: Dict[str, Any], output_path: str) -> None:
        """Save dataset in format-specific manner"""
        pass
        
    @abstractmethod
    def load_dataset(self, input_path: str) -> Dict[str, Any]:
        """Load dataset from format-specific file"""
        pass
        
    @abstractmethod
    def _get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions"""
        pass
        
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate data structure before saving"""
        required_keys = ['features', 'targets', 'metadata']
        return all(key in data for key in required_keys)
```

## 🔧 **Format-Specific Implementations**

### **CSV Dataset Handler**
```python
class CSVDatasetHandler(BaseDatasetHandler):
    """Handler for CSV tabular datasets"""
    
    def save_dataset(self, data: Dict[str, Any], output_path: str) -> None:
        """Save as CSV with standardized schema"""
        import pandas as pd
        
        # Create DataFrame with standardized columns
        df = pd.DataFrame({
            **data['features'],  # Feature columns
            'target_move': data['targets'],
            'game_id': data['metadata']['game_ids'],
            'step_in_game': data['metadata']['step_numbers']
        })
        
        # Save with consistent formatting
        df.to_csv(output_path, index=False, float_format='%.6f')
        
        # Save metadata separately
        metadata_path = output_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(data['metadata'], f, indent=2)
            
    def load_dataset(self, input_path: str) -> Dict[str, Any]:
        """Load CSV with automatic feature/target separation"""
        import pandas as pd
        
        df = pd.read_csv(input_path)
        
        # Separate features and targets
        feature_cols = [col for col in df.columns 
                       if col not in ['target_move', 'game_id', 'step_in_game']]
        
        return {
            'features': df[feature_cols].values,
            'targets': df['target_move'].values,
            'metadata': {
                'game_ids': df['game_id'].values,
                'step_numbers': df['step_in_game'].values,
                'feature_names': feature_cols
            }
        }
        
    def _get_supported_formats(self) -> List[str]:
        return ['.csv']
```

### **NPZ Dataset Handler**
```python
class NPZDatasetHandler(BaseDatasetHandler):
    """Handler for NumPy array datasets"""
    
    def save_dataset(self, data: Dict[str, Any], output_path: str) -> None:
        """Save as compressed NumPy arrays"""
        np.savez_compressed(
            output_path,
            features=data['features'],
            targets=data['targets'],
            **data['metadata']
        )
        
    def load_dataset(self, input_path: str) -> Dict[str, Any]:
        """Load NPZ with automatic array reconstruction"""
        npz_data = np.load(input_path)
        
        # Separate arrays from metadata
        features = npz_data['features']
        targets = npz_data['targets']
        
        metadata = {key: npz_data[key] for key in npz_data.keys() 
                   if key not in ['features', 'targets']}
        
        return {
            'features': features,
            'targets': targets,
            'metadata': metadata
        }
        
    def _get_supported_formats(self) -> List[str]:
        return ['.npz']
```

### **Parquet Dataset Handler**
```python
class ParquetDatasetHandler(BaseDatasetHandler):
    """Handler for Parquet columnar datasets"""
    
    def save_dataset(self, data: Dict[str, Any], output_path: str) -> None:
        """Save as Parquet with schema preservation"""
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Create comprehensive DataFrame
        df = pd.DataFrame({
            **data['features'],
            'target_move': data['targets'],
            **data['metadata']
        })
        
        # Define optimized schema
        schema = pa.Schema([
            ('target_move', pa.string()),
            *[(col, pa.float32()) for col in data['features'].keys()],
            ('game_id', pa.int32()),
            ('step_in_game', pa.int32())
        ])
        
        # Write with compression
        table = pa.Table.from_pandas(df, schema=schema)
        pq.write_table(table, output_path, compression='snappy')
        
    def load_dataset(self, input_path: str) -> Dict[str, Any]:
        """Load Parquet with optimized column selection"""
        import pandas as pd
        
        df = pd.read_parquet(input_path)
        
        # Efficient column separation
        feature_cols = [col for col in df.columns 
                       if col not in ['target_move', 'game_id', 'step_in_game']]
        
        return {
            'features': df[feature_cols].to_dict('series'),
            'targets': df['target_move'].values,
            'metadata': {
                'game_ids': df['game_id'].values,
                'step_numbers': df['step_in_game'].values,
                'feature_names': feature_cols
            }
        }
        
    def _get_supported_formats(self) -> List[str]:
        return ['.parquet']
```

## 📁 **Path Integration and Storage Strategy**

### **Multi-Format Dataset Storage**
Following Final Decision 1 directory structure:

```python
from extensions.common.path_utils import get_dataset_path

def save_multi_format_dataset(
    data: Dict[str, Any],
    extension_type: str,
    version: str,
    algorithm: str,
    grid_size: int,
    timestamp: str,
    formats: List[str] = None
) -> Dict[str, str]:
    """Save dataset in multiple formats with standardized paths"""
    
    formats = formats or ['csv', 'npz', 'parquet']
    saved_paths = {}
    
    # Get base dataset directory
    base_path = get_dataset_path(
        extension_type=extension_type,
        version=version,
        grid_size=grid_size,
        algorithm=algorithm,
        timestamp=timestamp
    )
    
    # Save in each requested format
    for format_type in formats:
        handler = DatasetFormatFactory.create_handler(format_type, grid_size=grid_size)
        
        # Create format-specific filename
        output_path = base_path / f"dataset.{format_type}"
        
        # Save dataset
        handler.save_dataset(data, str(output_path))
        saved_paths[format_type] = str(output_path)
        
    return saved_paths
```

## 🚀 **Extension Integration Benefits**

### **Heuristics Extensions - Dataset Generation**
- **CSV**: Primary format for supervised learning consumption
- **NPZ**: Efficient storage for large-scale heuristic trace datasets
- **Parquet**: Optimal for complex game state sequences and metadata

### **Supervised Learning Extensions - Model Training**
- **CSV**: Direct loading for tree models (XGBoost, LightGBM)
- **NPZ**: Efficient array loading for neural networks
- **Parquet**: Schema evolution for complex feature engineering

### **Reinforcement Learning Extensions - Experience Replay**
- **NPZ**: Efficient storage for experience buffers and trajectories
- **Parquet**: Complex state-action-reward sequences with metadata
- **CSV**: Simple debugging and analysis of RL training data

### **Cross-Extension Data Flow**
- **Format Conversion**: Automatic conversion between formats as needed
- **Storage Optimization**: Choose optimal format based on data characteristics
- **Loading Performance**: Format-specific optimizations for different use cases

---

**The multi-format dataset architecture provides flexible, efficient data storage and loading capabilities while maintaining the established patterns from the Final Decision series. This enables optimal performance for diverse machine learning approaches across the entire Snake Game AI ecosystem.** 