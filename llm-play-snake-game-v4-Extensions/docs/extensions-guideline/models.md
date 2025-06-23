## grid_size

The grid_size should not be fixed to 10, because they will be stored in ./logs/extensions/models/grid-size-N


## format of the model files
 
To ensure your trained models remain **cross-platform** (Windows, macOS, Linux) and **time-proof** (readable years down the road), follow these guidelines for **saving** and **loading** in each framework, and consider exporting to **standardized interchange formats**.

---

## 1. PyTorch

### 🔒 Save the **state\_dict**, not the full object

* **Why?**  Only saves parameter tensors, not class definitions or code.
* **Format:** PyTorch’s native binary format (Pickle under the hood), but tied to `state_dict`.

```python
# Saving
import torch

model = MyModel(...)
# After training…
torch.save({
    'model_state': model.state_dict(),
    'epoch': epoch,
    'optimizer_state': optimizer.state_dict(),
    'meta': {
        'torch_version': torch.__version__,
        'model_class': 'MyModel',
        'timestamp': datetime.utcnow().isoformat(),
    }
}, 'model_pt_v1.pth')
```

```python
# Loading
import torch
from my_models import MyModel  # ensure code is available

checkpoint = torch.load('model_pt_v1.pth', map_location='cpu')
model = MyModel(**checkpoint['meta'].get('init_args', {}))
model.load_state_dict(checkpoint['model_state'])
model.eval()
```

#### 📦 Optional: Export to **ONNX** for framework-agnostic inference

```python
# Export
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'model.onnx',
                  input_names=['input'], output_names=['output'],
                  opset_version=11)
```

---

## 2. XGBoost

### 🔧 Use **JSON** or **UBJ** formats rather than binary Pickle

* **Why?**  JSON and UBJ are text-based, version-safe, and human-readable.

```python
# Training and saving
import xgboost as xgb

bst = xgb.XGBClassifier(**params)
bst.fit(X_train, y_train)

# JSON
bst.get_booster().save_model('model_xgb_v1.json')

# Or binary UBJ
bst.get_booster().save_model('model_xgb_v1.ubj')
```

```python
# Loading
import xgboost as xgb

bst = xgb.XGBClassifier()
bst.load_model('model_xgb_v1.json')
```

---

## 3. LightGBM

### 💾 Save as **JSON** or **text** model files

* LightGBM’s text format is also human-readable, version-stable.

```python
# Training and saving
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
gbm = lgb.train(params, lgb_train)
gbm.save_model('model_lgb_v1.txt')    # default text format
```

```python
# Loading
import lightgbm as lgb

gbm = lgb.Booster(model_file='model_lgb_v1.txt')
```

---

## 4. Best Practices Across All Frameworks

1. **Pin Versions**
   Record the library and Python version in metadata. e.g.

   ```json
   { "framework": "PyTorch", "version": "2.0.1", "python": "3.10.8" }
   ```

2. **Store Initialization Args**
   If your model class takes custom `hidden_size=128`, save those in metadata so you can reconstruct the architecture before loading weights.

3. **Include Timestamps & Checksums**

   * Timestamps help audit when model was saved.
   * SHA256 checksums verify file integrity.

4. **Containerize or Virtualize When Deploying**
   For true time-proofing, bundle your model with a Docker image or Conda environment spec (`environment.yml`) capturing exact dependencies.

5. **Consider Interchange Formats**

   * **ONNX** for neural nets (PyTorch, TensorFlow, etc.).
   * **PMML** or **PFA** for tree ensembles, though JSON/UBJ is usually sufficient.

6. **Testing**
   Always write a small script to load each saved model and run a static inference to ensure future compatibility.

---

### Example Directory Layout

```
./logs/extensions/models/grid-size-N/
├── pytorch/
│   ├── model_pt_v1.pth
│   ├── model_pt_v1.onnx
│   └── metadata_pt_v1.json
├── xgboost/
│   ├── model_xgb_v1.json
│   └── metadata_xgb_v1.json
└── lightgbm/
    ├── model_lgb_v1.txt
    └── metadata_lgb_v1.json
```

With this approach, your saved models will **survive** Python upgrades, OS changes, and even language/framework migrations for years to come.





