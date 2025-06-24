from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

# Ensure repo root in path ----------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.path_utils import ensure_project_root  # type: ignore

ensure_project_root()

# Lazy import manager to avoid heavy deps at startup -------------------------
RLGameManager = importlib.import_module(
    "extensions.reinforcement-v0.03.game_manager"
).RLGameManager


# ---------------------------------------------------------------------------
# Streamlit UI ---------------------------------------------------------------
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RL Snake – v0.03", layout="wide")
st.title("Reinforcement Learning – Snake (v0.03)")

with st.sidebar:
    st.header("Training parameters")
    algo = st.selectbox("Algorithm", ["DQN", "PPO", "A3C", "SAC"], index=0)
    episodes = st.number_input("Episodes", 100, 100000, 1000, 100)
    grid_size = st.number_input("Grid Size", 5, 20, 10, 1)

    if st.button("Train 🐍"):
        # Build fake argparse.Namespace so we can reuse CLI manager
        args = SimpleNamespace(
            algorithm=algo,
            episodes=episodes,
            grid_size=grid_size,
            max_steps=1000,
            output_dir=str(ROOT / "logs/extensions/models"),
            no_gui=True,
        )
        st.success("Training started (head-less)… check console logs for progress.")
        manager = RLGameManager(args)
        manager.run()
        st.success("Training completed!") 