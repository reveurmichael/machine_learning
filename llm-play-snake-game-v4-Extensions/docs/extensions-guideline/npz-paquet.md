# Multi-Format Dataset Architecture for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines. Multi-format datasets follow the same architectural patterns established in the GOOD_RULES.

## 🎯 **Core Philosophy: Format-Agnostic Data Pipeline**

The multi-format dataset architecture provides flexible data storage and loading capabilities to support diverse machine learning approaches. Different model types require different data representations, and this architecture ensures optimal performance for each approach.

### **Design Philosophy**
- **Format Specialization**: Each format optimized for specific model types
- **Unified Interface**: Consistent data loading regardless of format
- **Storage Efficiency**: Optimal storage characteristics for each use case
- **Cross-Extension Compatibility**: Standardized formats across all extensions

## 📊 **Format Selection Strategy**

### **State Representation Decision Matrix**
Choose the right combination of representation type and storage format:

| Representation Type | Storage Format | Model Types | Why This Combination |
|-------------------|---------------|-------------|---------------------|
| **16-Feature Tabular** | **CSV** | XGBoost, LightGBM, Random Forest | Simple tabular data, human readable |
| **16-Feature Tabular** | **Parquet** | Large-scale tree models | Efficient columnar storage |
| **Sequential Time Series** | **NPZ** | LSTM, GRU, RNN | Preserves temporal array shapes |
| **Spatial 2D Arrays** | **NPZ** | CNN, computer vision | Native array format, spatial structure |
| **Graph Structures** | **Parquet** | GNN, relationship models | Complex schema support |
| **Raw Board States** | **NPZ** | Evolutionary algorithms | Direct array manipulation |
| **Visual Images** | **Image Files** | VLMs (GPT-4V, LLaVA) | Native visual format |

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

# NPZ and Parquet Data Formats for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and follows established architectural patterns. For data format decisions, see `data-format-decision-guide.md`.

## 🎯 **Core Philosophy: Efficient Scientific Data Storage**

NPZ and Parquet formats represent advanced data storage solutions for different scientific computing and machine learning workflows. While CSV serves as the universal interchange format, NPZ and Parquet provide specialized storage for performance-critical applications and large-scale data processing.

### **Format Positioning in Snake Game AI**
- **CSV**: Universal format for tabular features (16-feature schema)
- **NPZ**: Scientific computing arrays and tensor data
- **Parquet**: Large-scale analytics and columnar operations
- **JSONL**: Sequential data and metadata-rich formats

## 📊 **Strategic Format Decision Matrix**

### **NPZ Format Applications**
Following scientific computing best practices:

| Data Type | Use Case | NPZ Advantages |
|-----------|----------|----------------|
| **Multi-dimensional Arrays** | Neural network weights, tensor data | Native NumPy compatibility |
| **Scientific Computing** | Mathematical transformations, statistical analysis | Zero-copy operations |
| **High-frequency Data** | Real-time game state sequences | Compressed binary storage |
| **Research Datasets** | Experimental data with complex structures | Python-native ecosystem |

### **Parquet Format Applications**
Following big data analytics patterns:

| Data Type | Use Case | Parquet Advantages |
|-----------|----------|-------------------|
| **Large Tabular Datasets** | Million+ game episodes | Columnar compression |
| **Analytics Workloads** | Cross-game performance analysis | Predicate pushdown |
| **Data Lake Storage** | Long-term archival with schema evolution | Self-describing format |
| **Cross-Language Processing** | R, Scala, Java analytics integration | Language-agnostic access |

## 🏗️ **NPZ Implementation Architecture**

### **NPZ Data Format Factory**
Following Final Decision 7-8 factory patterns:

```python
class NPZDataFactory:
    """
    Factory for creating NPZ data handlers
    
    Design Pattern: Factory Pattern
    Purpose: Create appropriate NPZ handlers for different data types
    Educational Note: Demonstrates scientific data format patterns
    """
    
    _handler_registry = {
        "game_sequences": GameSequenceNPZHandler,
        "model_weights": ModelWeightsNPZHandler,
        "feature_tensors": FeatureTensorNPZHandler,
        "training_data": TrainingDataNPZHandler,
    }
    
    @classmethod
    def create_handler(cls, data_type: str, **kwargs) -> BaseNPZHandler:
        """Create NPZ handler by data type"""
        handler_class = cls._handler_registry.get(data_type)
        if not handler_class:
            raise ValueError(f"Unsupported NPZ data type: {data_type}")
        return handler_class(**kwargs)
```

### **Game Sequence NPZ Handler**
```python
class GameSequenceNPZHandler(BaseNPZHandler):
    """
    Handle game sequence data in NPZ format
    
    Design Pattern: Strategy Pattern
    Purpose: Manage different game sequence storage strategies
    Educational Note: Shows efficient temporal data storage
    """
    
    def __init__(self, grid_size: int = 10, sequence_length: int = 1000):
        self.grid_size = grid_size
        self.sequence_length = sequence_length
        
    def save_sequences(self, sequences: List[GameSequence], 
                      output_path: Path) -> None:
        """Save game sequences with optimal compression"""
        
        # Convert sequences to structured arrays
        structured_data = self._convert_to_arrays(sequences)
        
        # Save with compression for space efficiency
        np.savez_compressed(
            output_path,
            **structured_data
        )
        
    def _convert_to_arrays(self, sequences: List[GameSequence]) -> Dict[str, np.ndarray]:
        """Convert game sequences to NumPy arrays"""
        n_sequences = len(sequences)
        max_length = max(len(seq.steps) for seq in sequences)
        
        # Preallocate arrays for efficiency
        game_states = np.zeros((n_sequences, max_length, self.grid_size, self.grid_size), dtype=np.int8)
        actions = np.zeros((n_sequences, max_length), dtype=np.int8)
        rewards = np.zeros((n_sequences, max_length), dtype=np.float32)
        sequence_lengths = np.zeros(n_sequences, dtype=np.int32)
        
        # Fill arrays with sequence data
        for i, sequence in enumerate(sequences):
            seq_len = len(sequence.steps)
            sequence_lengths[i] = seq_len
            
            for j, step in enumerate(sequence.steps):
                game_states[i, j] = step.board_state
                actions[i, j] = step.action
                rewards[i, j] = step.reward
        
        return {
            'game_states': game_states,
            'actions': actions,
            'rewards': rewards,
            'sequence_lengths': sequence_lengths,
            'metadata': self._create_metadata(sequences)
        }
        
    def load_sequences(self, npz_path: Path) -> List[GameSequence]:
        """Load game sequences from NPZ format"""
        data = np.load(npz_path)
        
        sequences = []
        n_sequences = len(data['sequence_lengths'])
        
        for i in range(n_sequences):
            seq_len = data['sequence_lengths'][i]
            
            # Extract sequence data
            sequence_steps = []
            for j in range(seq_len):
                step = GameStep(
                    board_state=data['game_states'][i, j],
                    action=data['actions'][i, j],
                    reward=data['rewards'][i, j]
                )
                sequence_steps.append(step)
            
            sequence = GameSequence(steps=sequence_steps)
            sequences.append(sequence)
        
        return sequences
```

## 📈 **Parquet Implementation Architecture**

### **Parquet Analytics Factory**
```python
class ParquetAnalyticsFactory:
    """
    Factory for creating Parquet analytics handlers
    
    Design Pattern: Factory Pattern
    Purpose: Create appropriate handlers for different analytics workloads
    Educational Note: Demonstrates big data format patterns
    """
    
    _handler_registry = {
        "game_analytics": GameAnalyticsParquetHandler,
        "performance_metrics": PerformanceMetricsParquetHandler,
        "cross_experiment": CrossExperimentParquetHandler,
        "time_series": TimeSeriesParquetHandler,
    }
    
    @classmethod
    def create_handler(cls, analytics_type: str, **kwargs) -> BaseParquetHandler:
        """Create Parquet handler by analytics type"""
        handler_class = cls._handler_registry.get(analytics_type)
        if not handler_class:
            raise ValueError(f"Unsupported analytics type: {analytics_type}")
        return handler_class(**kwargs)
```

### **Game Analytics Parquet Handler**
```python
class GameAnalyticsParquetHandler(BaseParquetHandler):
    """
    Handle large-scale game analytics in Parquet format
    
    Purpose: Enable efficient analytics on millions of game episodes
    Educational Note: Shows big data processing patterns
    """
    
    def __init__(self, partition_columns: List[str] = None):
        self.partition_columns = partition_columns or ['grid_size', 'algorithm_type']
        
    def save_analytics_data(self, df: pd.DataFrame, output_path: Path,
                           compression: str = 'snappy') -> None:
        """Save analytics data with optimal Parquet configuration"""
        
        # Optimize data types for storage efficiency
        df_optimized = self._optimize_dtypes(df)
        
        # Save with partitioning for query performance
        df_optimized.to_parquet(
            output_path,
            partition_cols=self.partition_columns,
            compression=compression,
            engine='pyarrow',
            index=False
        )
        
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for Parquet storage"""
        df_opt = df.copy()
        
        # Convert categorical columns
        categorical_columns = ['algorithm_type', 'agent_name', 'experiment_id']
        for col in categorical_columns:
            if col in df_opt.columns:
                df_opt[col] = df_opt[col].astype('category')
        
        # Optimize integer columns
        int_columns = ['score', 'steps', 'grid_size', 'episode_id']
        for col in int_columns:
            if col in df_opt.columns:
                df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')
        
        # Optimize float columns
        float_columns = ['mean_reward', 'efficiency_score', 'survival_rate']
        for col in float_columns:
            if col in df_opt.columns:
                df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')
        
        return df_opt
        
    def load_analytics_data(self, parquet_path: Path, 
                           filters: List[Tuple] = None) -> pd.DataFrame:
        """Load analytics data with optional filtering"""
        return pd.read_parquet(
            parquet_path,
            filters=filters,
            engine='pyarrow'
        )
        
    def query_performance_metrics(self, parquet_path: Path,
                                 grid_size: int = None,
                                 algorithm_type: str = None) -> pd.DataFrame:
        """Query specific performance metrics with predicate pushdown"""
        filters = []
        
        if grid_size:
            filters.append(('grid_size', '==', grid_size))
        if algorithm_type:
            filters.append(('algorithm_type', '==', algorithm_type))
        
        return self.load_analytics_data(parquet_path, filters=filters)
```

## 🔧 **Data Format Integration Patterns**

### **Multi-Format Data Pipeline**
```python
class MultiFormatDataPipeline:
    """
    Manages data flow between different formats
    
    Design Pattern: Pipeline Pattern
    Purpose: Coordinate between CSV, NPZ, and Parquet formats
    Educational Note: Shows enterprise data pipeline patterns
    """
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.csv_handler = CSVDataHandler()
        self.npz_factory = NPZDataFactory()
        self.parquet_factory = ParquetAnalyticsFactory()
        
    def convert_csv_to_npz(self, csv_path: Path, npz_path: Path,
                          data_type: str = "training_data") -> None:
        """Convert CSV tabular data to NPZ format for scientific computing"""
        
        # Load CSV data
        df = self.csv_handler.load_data(csv_path)
        
        # Convert to NumPy arrays
        features = df.drop('action', axis=1).values.astype(np.float32)
        labels = df['action'].values.astype(np.int32)
        
        # Create NPZ handler and save
        npz_handler = self.npz_factory.create_handler(data_type)
        npz_handler.save_training_data(features, labels, npz_path)
        
    def convert_csv_to_parquet(self, csv_path: Path, parquet_path: Path,
                              partition_by: List[str] = None) -> None:
        """Convert CSV data to Parquet format for analytics"""
        
        # Load CSV data
        df = self.csv_handler.load_data(csv_path)
        
        # Add analytics-friendly columns
        df = self._enhance_for_analytics(df)
        
        # Create Parquet handler and save
        parquet_handler = self.parquet_factory.create_handler(
            "game_analytics", 
            partition_columns=partition_by
        )
        parquet_handler.save_analytics_data(df, parquet_path)
        
    def _enhance_for_analytics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed columns for analytics optimization"""
        df_enhanced = df.copy()
        
        # Add derived analytics columns
        df_enhanced['efficiency_score'] = df_enhanced['score'] / df_enhanced['steps']
        df_enhanced['survival_rate'] = df_enhanced['steps'] / df_enhanced['max_possible_steps']
        
        # Add categorical analytics columns
        df_enhanced['performance_tier'] = pd.cut(
            df_enhanced['score'], 
            bins=[0, 10, 25, 50, np.inf], 
            labels=['Low', 'Medium', 'High', 'Expert']
        )
        
        return df_enhanced
```

## 📊 **Performance Comparison and Use Cases**

### **Format Performance Characteristics**

| Metric | CSV | NPZ | Parquet |
|--------|-----|-----|---------|
| **File Size** | Large (text) | Medium (binary) | Small (columnar compression) |
| **Read Speed** | Slow | Fast (arrays) | Fast (columnar) |
| **Write Speed** | Medium | Fast | Medium |
| **Query Performance** | Poor | Good (arrays) | Excellent (predicate pushdown) |
| **Cross-Language** | Excellent | Python-focused | Excellent |
| **Schema Evolution** | Manual | Manual | Built-in |

### **Recommended Usage Patterns**

```python
# ✅ RECOMMENDED FORMAT USAGE:

# CSV: Universal interchange and small datasets
csv_data = load_csv_for_visualization(game_results)

# NPZ: Scientific computing and array operations
npz_data = load_npz_for_numpy_processing(tensor_data)

# Parquet: Large-scale analytics and data warehousing
parquet_data = load_parquet_for_analytics(million_episodes)

# JSONL: Metadata-rich sequential data
jsonl_data = load_jsonl_for_llm_training(conversation_data)
```

## 🛣️ **Path Management Integration**

### **Multi-Format Path Strategy**
Following Final Decision 6 path management:

```python
from extensions.common.path_utils import get_dataset_path

class MultiFormatPathManager:
    """
    Manage paths for different data formats
    
    Design Pattern: Strategy Pattern
    Purpose: Handle path generation for different formats
    """
    
    def get_format_specific_path(self, base_path: Path, 
                                format_type: str) -> Path:
        """Generate format-specific paths within dataset directory"""
        
        format_subdirs = {
            'csv': 'tabular',
            'npz': 'arrays', 
            'parquet': 'analytics',
            'jsonl': 'sequential'
        }
        
        subdir = format_subdirs.get(format_type, 'raw')
        return base_path / subdir
        
    def create_multi_format_dataset(self, extension_type: str, version: str,
                                   grid_size: int, algorithm: str,
                                   timestamp: str) -> Dict[str, Path]:
        """Create complete multi-format dataset structure"""
        
        base_path = get_dataset_path(
            extension_type=extension_type,
            version=version,
            grid_size=grid_size,
            algorithm=algorithm,
            timestamp=timestamp
        )
        
        format_paths = {}
        for format_type in ['csv', 'npz', 'parquet', 'jsonl']:
            format_paths[format_type] = self.get_format_specific_path(
                base_path, format_type
            )
            format_paths[format_type].mkdir(parents=True, exist_ok=True)
        
        return format_paths
```

---

## 🚀 **Integration Summary and Best Practices**

### **Format Selection Guidelines**
1. **Start with CSV**: Universal compatibility and debugging ease
2. **Add NPZ**: When array operations become performance bottlenecks
3. **Scale to Parquet**: When analytics queries need optimization
4. **Include JSONL**: For metadata-rich and sequential data requirements

### **Performance Optimization Patterns**
- **NPZ**: Use compression for storage, memory mapping for large arrays
- **Parquet**: Leverage partitioning and predicate pushdown for queries
- **Multi-Format**: Maintain format-specific optimizations while preserving interoperability

### **Architectural Benefits**
- **Factory Patterns**: Consistent creation across formats
- **Strategy Patterns**: Format-specific optimizations
- **Pipeline Patterns**: Smooth data flow between formats
- **Educational Value**: Real-world data engineering practices

**NPZ and Parquet formats extend the Snake Game AI data ecosystem beyond CSV, providing specialized storage solutions for scientific computing and large-scale analytics while maintaining architectural consistency with established patterns.** 