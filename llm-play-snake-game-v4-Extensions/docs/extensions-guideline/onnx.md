Export to **ONNX** for framework-agnostic inference

```python
# Export
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'model.onnx',
                  input_names=['input'], output_names=['output'],
                  opset_version=11)
```

along with other model formats (json, .pth, jsonl, etc., maybe)