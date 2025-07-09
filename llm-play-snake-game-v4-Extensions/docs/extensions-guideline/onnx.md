# ONNX Export Architecture for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision.md`) and follows the architectural patterns established across all extensions.

## 🎯 **Core Philosophy: Framework-Agnostic Model Deployment**

ONNX (Open Neural Network Exchange) provides a **unified, framework-agnostic** format for neural-network models, ensuring:

1. **Framework Independence** – seamless export from PyTorch, scikit-learn, XGBoost, etc.
2. **Deployment Optimisation** – faster inference with ONNX Runtime on CPU, CUDA, or specialised accelerators.
3. **Cross-Platform Compatibility** – identical model behaviour on servers, edge devices, and mobile.
4. **Educational Value** – industry-standard best-practices demonstrated in a research-friendly codebase.

---

## 🏗️ **Factory-Pattern Export Architecture**

### ONNXExportFactory
```python
class ONNXExportFactory:
    """Factory for creating framework-specific ONNX exporters."""

    _REGISTRY: dict[str, type["BaseONNXExporter"]] = {
        "pytorch": PyTorchONNXExporter,
        "sklearn": SklearnONNXExporter,
        "xgboost": XGBoostONNXExporter,
    }

    @classmethod
    def create_exporter(cls, framework: str, model, **kwargs) -> "BaseONNXExporter":
        exporter_cls = cls._REGISTRY.get(framework.lower())
        if exporter_cls is None:
            raise ValueError(f"Unsupported framework: {framework}")
        return exporter_cls(model, **kwargs)
```

### BaseONNXExporter (Universal Interface)
```python
class BaseONNXExporter(ABC):
    """Abstract base-class for all ONNX exporters."""

    def __init__(self, model, grid_size: int = 10):
        self.model = model
        self.grid_size = grid_size
        self.input_spec = self._define_input_specification()
        self.output_spec = self._define_output_specification()

    @abstractmethod
    def export_to_onnx(self, output_path: str) -> None: ...
    @abstractmethod
    def _define_input_specification(self) -> dict[str, Any]: ...
    @abstractmethod
    def _define_output_specification(self) -> dict[str, Any]: ...

    # Universal validation helper
    def validate_export(self, onnx_path: str) -> bool:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        return True
```

---

## 🔧 **Framework-Specific Exporters**

### PyTorchONNXExporter
```python
class PyTorchONNXExporter(BaseONNXExporter):
    """PyTorch → ONNX exporter with dynamic-axes support."""

    def export_to_onnx(self, output_path: str) -> None:
        import torch
        dummy_input = self._create_dummy_input()
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["game_features"],
            output_names=["action_logits"],
            dynamic_axes={
                "game_features": {0: "batch_size"},
                "action_logits": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    def _define_input_specification(self) -> dict[str, Any]:
        size = getattr(self.model, "input_size", 16)
        return {"shape": (1, size), "dtype": "float32"}

    def _define_output_specification(self) -> dict[str, Any]:
        return {"shape": (1, 4), "dtype": "float32"}  # logits for 4 actions

    def _create_dummy_input(self):
        import torch
        return torch.randn(*self.input_spec["shape"], dtype=torch.float32)
```

### XGBoostONNXExporter (via skl2onnx)
```python
class XGBoostONNXExporter(BaseONNXExporter):
    """XGBoost → ONNX exporter using skl2onnx."""

    def export_to_onnx(self, output_path: str) -> None:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("input", FloatTensorType([None, 16]))]
        onnx_model = convert_sklearn(self.model, initial_types=initial_type, target_opset=14)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    def _define_input_specification(self) -> dict[str, Any]:
        return {"shape": (1, 16), "dtype": "float32"}

    def _define_output_specification(self) -> dict[str, Any]:
        return {"shape": (1,), "dtype": "int64"}  # predicted action index
```

> **Note** – `SklearnONNXExporter` follows the same pattern and is omitted for brevity.

---

## 📁 **Standardised Storage Structure**

Models exported to ONNX **must** follow the path rules defined in `final-decision-1.md`:
```
logs/extensions/models/
└── grid-size-{N}/
    └── supervised_v0.02_{timestamp}/
        └── {algorithm}/
            └── model_artifacts/
                ├── model.onnx
                └── onnx_export_info.json
```

Helper function:
```python
from datetime import datetime
from extensions.common.path_utils import get_model_path


def export_model_to_onnx(model, framework: str, algorithm: str, version: str, grid_size: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = get_model_path(
        extension_type="supervised",
        version=version,
        grid_size=grid_size,
        algorithm=algorithm,
        timestamp=timestamp,
    )
    onnx_path = model_dir / "model_artifacts" / "model.onnx"
    metadata_path = model_dir / "model_artifacts" / "onnx_export_info.json"

    exporter = ONNXExportFactory.create_exporter(framework, model, grid_size=grid_size)
    exporter.export_to_onnx(str(onnx_path))

    # Save metadata
    export_metadata = {
        "framework": framework,
        "grid_size": grid_size,
        "input_specification": exporter.input_spec,
        "output_specification": exporter.output_spec,
        "export_timestamp": timestamp,
        "onnx_opset_version": 14,
        "model_validation": exporter.validate_export(str(onnx_path)),
    }
    metadata_path.write_text(json.dumps(export_metadata, indent=2))
    return str(onnx_path)
```

---

## 🚀 **Universal ONNX Inference Engine**
```python
class ONNXInferenceEngine:
    """Framework-agnostic inference wrapper using ONNX Runtime."""

    def __init__(self, onnx_model_path: str, providers: list[str] | None = None):
        import onnxruntime as ort
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.session.run([self.output_name], {self.input_name: features.astype(np.float32)})[0]
```

---

## 🎯 **Extension-Level Benefits**

| Extension | Benefit of ONNX | Example |
|-----------|-----------------|---------|
| **Supervised** | Portable, fast inference | Deploy MLP model in mobile app |
| **Reinforcement** | Deploy trained DQN policies | Run policy on edge device |
| **Heuristics** | Combine classical & learned models | Heuristic fallback to ONNX policy |
| **LLM Fine-Tuning** | Blend symbolic reasoning with neural policies | Use ONNX policy as tool for LLM | 
| **LLM Distillation** | Distil knowledge into compact ONNX models | Transfer large-model insights |

---

**The ONNX export architecture guarantees that all models in the Snake Game AI ecosystem are portable, efficient, and production-ready, while respecting the architectural decisions codified in the Final Decision documents.**

```python
# Export
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'model.onnx',
                  input_names=['input'], output_names=['output'],
                  opset_version=11)
```

along with other model formats (json, .pth, jsonl, etc.)