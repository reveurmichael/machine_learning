import sys
import subprocess
from pathlib import Path
from typing import List

import streamlit as st

from utils.path_utils import ensure_project_root

ensure_project_root()

st.set_page_config(page_title="Supervised v0.01 Runner", page_icon="🧠", layout="wide")

st.title("🧠 Supervised v0.01 Runner")

with st.sidebar:
    st.header("⚙️ Parameters")
    from agents import get_available_algorithms, DEFAULT_ALGORITHM
    algorithms = get_available_algorithms()
    default_index = max(0, algorithms.index(DEFAULT_ALGORITHM)) if DEFAULT_ALGORITHM in algorithms else 0
    algorithm = st.selectbox("Algorithm", algorithms, index=default_index)
    grid_size: int = st.slider("Grid size", min_value=5, max_value=25, value=10)
    max_games: int = st.number_input("Max games", min_value=1, max_value=1000000, value=5)
    max_steps: int = st.number_input("Max steps per game", min_value=100, max_value=10000, value=500)
    verbose: bool = st.checkbox("Verbose output", value=False)

if st.button("🚀 Run"):
    ext_dir = Path(__file__).parent
    script_path = ext_dir / "scripts" / "main.py"

    cmd: List[str] = [sys.executable, str(script_path)]
    cmd.extend(["--algorithm", algorithm])
    cmd.extend(["--grid-size", str(grid_size)])
    cmd.extend(["--max-games", str(max_games)])
    cmd.extend(["--max-steps", str(max_steps)])
    if verbose:
        cmd.append("--verbose")

    st.code(" ".join(cmd), language="bash")

    with st.spinner("Running…"):
        result = subprocess.run(cmd, cwd=str(ext_dir), capture_output=True, text=True)

    if result.returncode == 0:
        st.success("Run completed ✨")
    else:
        st.error(f"Run failed (exit code {result.returncode})")

    with st.expander("📜 Output"):
        st.text(result.stdout + "\n" + result.stderr)