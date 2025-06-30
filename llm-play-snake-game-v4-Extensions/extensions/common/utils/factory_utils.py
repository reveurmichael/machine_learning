"""
Factory utilities for Snake Game AI extensions.

This module follows the principles from final-decision-10.md:
- Canonical factory method is create() 
- OOP extensibility is prioritized
- Logging is always simple (print())
- No ML/DL/RL/LLM-specific coupling

Reference: docs/extensions-guideline/final-decision-10.md
"""

from typing import Dict, Type, Any, List

class SimpleFactory:
    """
    Simple factory class following final-decision-10.md principles
    
    Design Pattern: Factory Pattern
    - Simple dictionary-based registry
    - Clear create() method interface (CANONICAL)
    - Easy extension for new types
    - No over-engineering or complex inheritance
    
    Educational Value:
    Shows how factory patterns can be implemented simply
    without complex inheritance hierarchies or over-engineering.
    
    Simple logging: All logging uses simple print() statements.
    """
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
        print(f"[{self.__class__.__name__}] Factory initialized")  # Simple logging
    
    def register(self, name: str, cls: Type) -> None:
        """Register a class with the factory"""
        self._registry[name.upper()] = cls
        print(f"[{self.__class__.__name__}] Registered: {name} -> {cls.__name__}")  # Simple logging
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create instance by name - CANONICAL METHOD NAME
        
        All factory methods must be named create(),
        never create_agent(), create_model(), etc.
        """
        cls = self._registry.get(name.upper())
        if not cls:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown type: {name}. Available: {available}")
        
        print(f"[{self.__class__.__name__}] Creating: {name}")  # Simple logging
        return cls(*args, **kwargs)
    
    def list_available(self) -> List[str]:
        """List all available types"""
        return list(self._registry.keys())
    
    def get_class(self, name: str) -> Type:
        """Get class by name without instantiation"""
        cls = self._registry.get(name.upper())
        if not cls:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown type: {name}. Available: {available}")
        return cls

# =============================================================================
# Specialized Factory Classes (SUPREME_RULES Compliant)
# =============================================================================

class DatasetFactory:
    """
    Factory for creating dataset utilities
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create appropriate dataset utilities based on format type
    Educational Value: Shows how canonical factory patterns work with datasets
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    _registry = {
        "CSV": "csv_loader",
        "JSONL": "jsonl_loader", 
        "NPZ": "npz_loader"
    }
    
    @classmethod
    def create(cls, dataset_type: str, **kwargs):  # CANONICAL create() method
        """Create dataset utility using canonical create() method (SUPREME_RULES compliance)"""
        from . import dataset_utils
        
        loader_map = {
            "CSV": dataset_utils.load_csv_dataset,
            "JSONL": dataset_utils.load_jsonl_dataset,
            "NPZ": dataset_utils.load_npz_dataset
        }
        
        loader = loader_map.get(dataset_type.upper())
        if not loader:
            available = list(loader_map.keys())
            raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {available}")
        
        print(f"[DatasetFactory] Creating dataset loader: {dataset_type}")  # Simple logging
        return loader

class PathFactory:
    """
    Factory for creating path utilities
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create appropriate path utilities based on path type
    Educational Value: Shows how canonical factory patterns work with paths
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    @classmethod
    def create(cls, path_type: str, **kwargs):  # CANONICAL create() method
        """Create path utility using canonical create() method (SUPREME_RULES compliance)"""
        from . import path_utils
        
        path_map = {
            "DATASET": path_utils.get_dataset_path,
            "MODEL": path_utils.get_model_path,
            "EXTENSION": path_utils.get_extension_path,
            "PROJECT_ROOT": path_utils.ensure_project_root
        }
        
        path_func = path_map.get(path_type.upper())
        if not path_func:
            available = list(path_map.keys())
            raise ValueError(f"Unknown path type: {path_type}. Available: {available}")
        
        print(f"[PathFactory] Creating path utility: {path_type}")  # Simple logging
        return path_func

class ValidationFactory:
    """
    Factory for creating validation utilities
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create appropriate validation utilities based on data type
    Educational Value: Shows how canonical factory patterns work with validation
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    @classmethod
    def create(cls, validator_type: str, **kwargs):  # CANONICAL create() method
        """Create validator using canonical create() method (SUPREME_RULES compliance)"""
        from ..validation import validate_dataset, validate_extension_path
        
        validator_map = {
            "DATASET": validate_dataset,
            "PATH": validate_extension_path
        }
        
        validator = validator_map.get(validator_type.upper())
        if not validator:
            available = list(validator_map.keys())
            raise ValueError(f"Unknown validator type: {validator_type}. Available: {available}")
        
        print(f"[ValidationFactory] Creating validator: {validator_type}")  # Simple logging
        return validator

class FeatureExtractorFactory:
    """
    Factory for creating feature extractors
    
    Design Pattern: Factory Pattern (Canonical Implementation)
    Purpose: Create appropriate feature extractors based on extraction type
    Educational Value: Shows how canonical factory patterns work with feature extraction
    
    Reference: final-decision-10.md for canonical method naming
    """
    
    @classmethod
    def create(cls, extractor_type: str, **kwargs):  # CANONICAL create() method
        """Create feature extractor using canonical create() method (SUPREME_RULES compliance)"""
        from . import csv_schema_utils
        
        extractor_map = {
            "TABULAR": csv_schema_utils.TabularFeatureExtractor,
            "CSV": csv_schema_utils.TabularFeatureExtractor
        }
        
        extractor_class = extractor_map.get(extractor_type.upper())
        if not extractor_class:
            available = list(extractor_map.keys())
            raise ValueError(f"Unknown extractor type: {extractor_type}. Available: {available}")
        
        print(f"[FeatureExtractorFactory] Creating extractor: {extractor_type}")  # Simple logging
        return extractor_class(**kwargs)

# =============================================================================
# Simple utility functions
# =============================================================================

def create_simple_factory() -> SimpleFactory:
    """Create a simple factory instance"""
    print("[FactoryUtils] Creating simple factory")  # Simple logging
    return SimpleFactory()

def validate_factory_registry(factory: SimpleFactory, required_types: List[str]) -> bool:
    """Validate that factory has required types registered"""
    available = factory.list_available()
    missing = [t for t in required_types if t.upper() not in available]
    
    if missing:
        print(f"[FactoryUtils] WARNING: Missing types: {missing}")  # Simple logging
        return False
    
    print("[FactoryUtils] Factory validation passed")  # Simple logging
    return True

def create_dataset_factory():
    """Create dataset factory using canonical pattern"""
    print("[FactoryUtils] Creating dataset factory")  # Simple logging
    return DatasetFactory()

def create_path_factory():
    """Create path factory using canonical pattern"""
    print("[FactoryUtils] Creating path factory")  # Simple logging
    return PathFactory()

def create_validation_factory():
    """Create validation factory using canonical pattern"""
    print("[FactoryUtils] Creating validation factory")  # Simple logging
    return ValidationFactory()

def create_feature_extractor_factory():
    """Create feature extractor factory using canonical pattern"""
    print("[FactoryUtils] Creating feature extractor factory")  # Simple logging
    return FeatureExtractorFactory()

# Example usage for documentation and educational value
if __name__ == "__main__":
    class Dummy:
        def __init__(self, x):
            self.x = x
    
    # Use SimpleFactory as referenced in documentation
    f = SimpleFactory()
    f.register("dummy", Dummy)
    d = f.create("dummy", x=42)  # CANONICAL create() method
    print(f"Created: {d.__class__.__name__}, x={d.x}")
    print(f"Available: {f.list_available()}")
    
    # Use specialized factories
    dataset_factory = DatasetFactory()
    csv_loader = dataset_factory.create("CSV")  # CANONICAL create() method
    print(f"CSV loader: {csv_loader}")
    
    path_factory = PathFactory()
    dataset_path_func = path_factory.create("DATASET")  # CANONICAL create() method
    print(f"Dataset path function: {dataset_path_func}")
    
    validation_factory = ValidationFactory()
    dataset_validator = validation_factory.create("DATASET")  # CANONICAL create() method
    print(f"Dataset validator: {dataset_validator}")
    
    extractor_factory = FeatureExtractorFactory()
    tabular_extractor = extractor_factory.create("TABULAR")  # CANONICAL create() method
    print(f"Tabular extractor: {tabular_extractor}") 