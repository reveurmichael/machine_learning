import sys
import subprocess
from pathlib import Path
from typing import List

import streamlit as st


try:
    from utils.path_utils import ensure_project_root
except ModuleNotFoundError:
    # Fallback if executed in an unusual environment – resolve root manually
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
else:
    project_root = ensure_project_root()

# Ensure we run everything from project root so that relative paths work
# Note: We don't change directory here to avoid Streamlit path issues
# The path setup above ensures imports work correctly

# ---------------------------------------------------------------------------
# Import helper from unified dataset CLI to retrieve available algorithms
# ---------------------------------------------------------------------------

try:
    from extensions.common.utils.dataset_generator_cli import find_available_algorithms
except Exception:
    # Safe-fallback list in case of import issues – keeps UI usable.
    find_available_algorithms = lambda: [  # type: ignore
        "BFS",
        "BFS-SAFE-GREEDY",
    ]

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Heuristics v0.04 Dataset Generator", page_icon="🐍", layout="wide")

st.title("🐍 Heuristics v0.04 Dataset Generator")

# Sidebar – parameter selection ------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parameters")

    # Whether to process all algorithms or a single one ----------------------
    all_algorithms = st.checkbox("Process ALL algorithms", value=False)
    available_algorithms: List[str] = find_available_algorithms()

    if all_algorithms:
        algorithm = None  # Will be handled via CLI flag
    else:
        algorithm = st.selectbox("Algorithm", available_algorithms)

    dataset_format = st.selectbox(
        "Dataset format",
        ["both", "csv", "jsonl"],
        index=0,
        help="Choose which dataset files to generate."
    )

    grid_size: int = st.slider("Grid size", min_value=5, max_value=25, value=10)
    max_games: int = st.number_input("Max games", min_value=1, max_value=1000000000000, value=10)
    max_steps: int = st.number_input("Max steps per game", min_value=100, max_value=10000, value=500)
    verbose: bool = st.checkbox("Verbose output", value=False)

# Main area – action -----------------------------------------------------------

if st.button("🚀 Generate Dataset"):
    # Build command ----------------------------------------------------------
    extension_dir = Path(__file__).parent  # heuristics-v0.04 directory
    script_path = extension_dir / "scripts" / "main.py"

    cmd: List[str] = [sys.executable, str(script_path)]

    if all_algorithms:
        cmd.append("--all-algorithms")
    else:
        if algorithm is None:
            st.error("Please select an algorithm or enable ALL algorithms option.")
            st.stop()
        cmd.extend(["--algorithm", algorithm])

    cmd.extend(["--format", dataset_format])
    cmd.extend(["--grid-size", str(grid_size)])
    cmd.extend(["--max-games", str(max_games)])
    cmd.extend(["--max-steps", str(max_steps)])

    if verbose:
        cmd.append("--verbose")

    # Display the command for transparency ----------------------------------
    st.code(" ".join(cmd), language="bash")

    # Run the command --------------------------------------------------------
    with st.spinner("Running dataset generation… this may take a while 🤖"):
        result = subprocess.run(
            cmd,
            cwd=str(extension_dir),  # Execute inside the extension folder
            capture_output=True,
            text=True,
        )

    # Output handling -------------------------------------------------------
    if result.returncode == 0:
        st.success("Dataset generation completed successfully ✨")
    else:
        st.error(f"Dataset generation failed (exit code {result.returncode})")

    # Always show stdout / stderr for transparency --------------------------
    with st.expander("📜 Script output (stdout + stderr)"):
        st.text(result.stdout + "\n" + result.stderr) 