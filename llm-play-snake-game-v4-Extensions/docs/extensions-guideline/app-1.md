# Issues you might face 


# Streamlit app in extensions

### **🔴 Problem:**
```
ModuleNotFoundError: No module named 'extensions'
```
When running `streamlit run extensions/heuristics/app.py`, Python couldn't find the `extensions` module because Streamlit changes the working directory and Python path.

### **✅ Solution:**
Added **path fixing code** at the beginning of both Streamlit apps:

```python
# Fix Python path for Streamlit
import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure we're working from project root
if os.getcwd() != str(project_root):
    os.chdir(str(project_root))
```
