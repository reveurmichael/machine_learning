# ONNX Export Architecture for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines. ONNX export follows the same architectural patterns established in the GOODFILES.

## 🎯 **Core Philosophy: Framework-Agnostic Model Deployment**

ONNX (Open Neural Network Exchange) provides a standardized format for neural network models, enabling seamless deployment across different frameworks and platforms. The Snake Game AI ecosystem leverages ONNX to achieve true framework independence and optimized inference performance.

### **Design Philosophy**
- **Framework Independence**: Models export consistently regardless of training framework
- **Deployment Optimization**: Optimized inference for production environments
- **Cross-Platform Compatibility**: Consistent model behavior across different systems
- **Educational Value**: Demonstrates industry-standard model export practices

## 🏗️ **Factory Pattern ONNX Architecture**

### **ONNX Export Factory**
Following Final Decision 7-8 factory patterns:

```python
class ONNXExportFactory:
    """Factory for creating framework-specific ONNX exporters"""
    
    _exporter_registry = {
        "pytorch": PyTorchONNXExporter,
        "sklearn": SklearnONNXExporter,
        "xgboost": XGBoostONNXExporter,
    }
    
    @classmethod
    def create_exporter(cls, framework: str, model, **kwargs) -> BaseONNXExporter:
        """Create ONNX exporter by framework type"""
        exporter_class = cls._exporter_registry.get(framework.lower())
        if not exporter_class:
            raise ValueError(f"Unsupported framework: {framework}")
        return exporter_class(model, **kwargs)
```

### **Universal ONNX Export Interface**
```python
class BaseONNXExporter:
    """Base class for all ONNX model exporters"""
    
    def __init__(self, model, grid_size: int = 10):
        self.model = model
        self.grid_size = grid_size
        self.input_spec = self._define_input_specification()
        self.output_spec = self._define_output_specification()
        
    @abstractmethod
    def export_to_onnx(self, output_path: str) -> None:
        """Export model to ONNX format with framework-specific logic"""
        pass
        
    @abstractmethod
    def _define_input_specification(self) -> Dict[str, Any]:
        """Define input tensor specifications"""
        pass
        
    @abstractmethod
    def _define_output_specification(self) -> Dict[str, Any]:
        """Define output tensor specifications"""
        pass
        
    def validate_export(self, onnx_path: str) -> bool:
        """Validate ONNX model integrity after export"""
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        return True
```

## 🔧 **Framework-Specific Export Implementations**

### **PyTorch ONNX Export Strategy**
```python
class PyTorchONNXExporter(BaseONNXExporter):
    """PyTorch to ONNX model exporter with dynamic axes support"""
    
    def export_to_onnx(self, output_path: str) -> None:
        """Export PyTorch model with comprehensive configuration"""
        import torch
        
        # Create representative input tensor
        dummy_input = self._create_dummy_input()
        
        # Export with production-ready settings
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=['game_features'],
            output_names=['action_logits'],
            dynamic_axes={
                'game_features': {0: 'batch_size'},
                'action_logits': {0: 'batch_size'}
            },
            opset_version=14,  # Latest stable ONNX opset
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
    def _define_input_specification(self) -> Dict[str, Any]:
        """Define PyTorch model input requirements"""
        if hasattr(self.model, 'input_size'):
            return {"shape": (1, self.model.input_size), "dtype": "float32"}
        return {"shape": (1, 16), "dtype": "float32"}  # Default CSV features
        
    def _create_dummy_input(self) -> torch.Tensor:
        """Create dummy input tensor for export process"""
        input_shape = self.input_spec["shape"]
        return torch.randn(*input_shape)
```

### **Multi-Framework Export Support**
```python
class XGBoostONNXExporter(BaseONNXExporter):
    """XGBoost to ONNX exporter using sklearn-onnx"""
    
    def export_to_onnx(self, output_path: str) -> None:
        """Export XGBoost model via sklearn-onnx conversion"""
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Define input type for ONNX conversion
        initial_type = [('input', FloatTensorType([None, 16]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(
            self.model,
            initial_types=initial_type,
            target_opset=14
        )
        
        # Save ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
    def _define_input_specification(self) -> Dict[str, Any]:
        """XGBoost expects tabular feature input"""
        return {"shape": (1, 16), "dtype": "float32"}
```

## 📁 **Path Integration and Model Storage**

### **ONNX Model Storage Structure**
Following Final Decision 1 directory structure:

```python
from extensions.common.path_utils import get_model_path

def export_model_to_onnx(
    model, 
    framework: str,
    algorithm: str, 
    version: str, 
    grid_size: int,
    timestamp: str
) -> str:
    """Export model with standardized path management"""
    
    # Get standardized model directory
    model_dir = get_model_path(
        extension_type="supervised",
        version=version,
        grid_size=grid_size,
        algorithm=algorithm,
        timestamp=timestamp
    )
    
    # Create ONNX-specific paths
    onnx_path = model_dir / "model_artifacts" / "model.onnx"
    metadata_path = model_dir / "model_artifacts" / "onnx_export_info.json"
    
    # Perform export using factory pattern
    exporter = ONNXExportFactory.create_exporter(framework, model, grid_size=grid_size)
    exporter.export_to_onnx(str(onnx_path))
    
    # Save comprehensive export metadata
    export_metadata = {
        "framework": framework,
        "grid_size": grid_size,
        "input_specification": exporter.input_spec,
        "output_specification": exporter.output_spec,
        "export_timestamp": datetime.now().isoformat(),
        "onnx_opset_version": 14,
        "model_validation": exporter.validate_export(str(onnx_path))
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(export_metadata, f, indent=2)
        
    return str(onnx_path)
```

## 🚀 **Universal ONNX Inference Engine**

### **Framework-Agnostic Inference**
```python
class ONNXInferenceEngine:
    """Universal ONNX model inference with optimized runtime"""
    
    def __init__(self, onnx_model_path: str, providers: List[str] = None):
        import onnxruntime as ort
        
        # Configure optimized providers (CPU, CUDA, etc.)
        providers = providers or ['CPUExecutionProvider']
        
        # Create optimized inference session
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=providers
        )
        
        # Extract model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
    def predict(self, game_features: np.ndarray) -> np.ndarray:
        """Framework-agnostic model inference with optimization"""
        # Ensure correct input format
        if game_features.ndim == 1:
            game_features = game_features.reshape(1, -1)
            
        # Run optimized inference
        return self.session.run(
            [self.output_name],
            {self.input_name: game_features.astype(np.float32)}
        )[0]
        
    def batch_predict(self, batch_features: np.ndarray) -> np.ndarray:
        """Efficient batch inference for multiple game states"""
        return self.session.run(
            [self.output_name],
            {self.input_name: batch_features.astype(np.float32)}
        )[0]
```

## 🎯 **Extension Integration Benefits**

### **Supervised Learning Extensions**
- **Model Portability**: Export trained models for deployment independence
- **Performance Optimization**: ONNX Runtime provides faster inference than training frameworks
- **Cross-Framework Evaluation**: Compare models trained in different frameworks
- **Production Deployment**: Industry-standard format for serving models

### **Cross-Extension Compatibility**
- **Heuristics Integration**: Use ONNX models alongside heuristic algorithms
- **Reinforcement Learning**: Export policy networks for deployment
- **LLM Integration**: Combine ONNX models with language model reasoning

### **Deployment Advantages**
- **Edge Computing**: Optimized inference for resource-constrained environments
- **Cloud Deployment**: Consistent model behavior across different cloud providers
- **Mobile Integration**: Potential for mobile Snake AI applications
- **Microservices**: Lightweight model serving in containerized environments

---

**The ONNX export architecture provides framework-agnostic model deployment capabilities while maintaining the established patterns from the Final Decision series. This enables true model portability and optimized inference performance across the entire Snake Game AI ecosystem.**

```python
# Export
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'model.onnx',
                  input_names=['input'], output_names=['output'],
                  opset_version=11)
```

along with other model formats (json, .pth, jsonl, etc., maybe)