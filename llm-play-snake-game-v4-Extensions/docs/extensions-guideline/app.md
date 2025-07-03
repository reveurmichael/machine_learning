# Streamlit App Pattern for Snake Game AI Extensions

> **Reference:** See `final-decision-10.md` (SUPREME_RULES), `scripts.md`, `standalone.md`.

## 🎯 Purpose

All v0.03+ extensions must provide a `Streamlit app.py` as a **script launcher** (SUPREME_RULE NO.5). The app is not for real-time visualization, but for launching backend scripts with adjustable parameters.

## 🏗️ Minimal App Structure

```python
# extensions/{algorithm}-v0.03/app.py
import streamlit as st
from utils.print_utils import print_info
from extensions.common.utils.path_utils import ensure_project_root
import subprocess
from pathlib import Path

ensure_project_root()

class SubprocessRunner:
    def __init__(self, extension_path: Path):
        self.extension_path = extension_path
        print_info(f"[SubprocessRunner] Initialized for {extension_path}")
    def run_script(self, script_name: str, args: dict):
        script_path = self.extension_path / "scripts" / script_name
        cmd = ["python", str(script_path)]
        for k, v in args.items():
            cmd.extend([f"--{k}", str(v)])
        return subprocess.Popen(cmd, cwd=self.extension_path.parent.parent.parent)

# Example UI
st.title("Snake Game AI Launcher")
ext_path = Path(__file__).parent
runner = SubprocessRunner(ext_path)

algo = st.selectbox("Algorithm", ["BFS", "ASTAR", "DFS"])
grid = st.slider("Grid Size", 5, 25, 10)
if st.button("Run Script"):
    runner.run_script("main.py", {"algorithm": algo, "grid_size": grid})
```

## ✅ Key Points
- Always use `ensure_project_root()` before imports.
- Only launch scripts, do not visualize game state.
- Use print_utils for all logging.
- No .log files, no complex logging.

---


