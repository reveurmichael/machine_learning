# Reinforcement Learning v0.02 – Multi-Algorithm Expansion

> Second-citizen extension; works stand-alone together with `extensions/common/`.  
> Builds on v0.01 (DQN-only) and introduces **algorithm diversity**.

---

## 🚀 What's new compared to v0.01?

| Area | v0.01 | **v0.02** |
|------|-------|-----------|
| Algorithms | DQN | **DQN, PPO, A3C, SAC** |
| Agents folder | ❌ single file | **✅ `agents/` package** with one module per algorithm |
| CLI | `train.py` (DQN-centric) | **Unified CLI** (`scripts/train.py`) with `--algorithm` switch |
| Game logic | Heavy environment | **Lightweight gym-style env** (`game_logic.py`) – faster prototyping |
| Observability | Basic prints | Observer hooks for episode complete events |
| Docs | Minimal | **This README** 😄 |

No GUI / replay yet – those arrive in v0.03, in line with the global roadmap.

---

## 🏗️ Folder layout

```
extensions/reinforcement-v0.02/
├── __init__.py          # RLConfig + agent factory
├── README.md            # ← you are here
├── agents/
│   ├── __init__.py      # Protocol + helper
│   ├── agent_dqn.py     # Functional mini-DQN
│   ├── agent_ppo.py     # PPO stub
│   ├── agent_a3c.py     # A3C stub
│   └── agent_sac.py     # SAC stub
├── game_data.py         # RLGameData container
├── game_logic.py        # Lightweight env
├── game_manager.py      # Training loop & logging
└── scripts/
    └── train.py         # CLI – headless training
```

---

## 🔧 Quick start

1. **Install deps** (only NumPy required for stubs):
   ```bash
   pip install numpy
   ```

2. **Train DQN** for 1 000 episodes on 10×10 grid:
   ```bash
   python extensions/reinforcement-v0.02/scripts/train.py \
       --algorithm DQN --episodes 1000
   ```

3. **Switch algorithm** (e.g. PPO):
   ```bash
   python extensions/reinforcement-v0.02/scripts/train.py \
       --algorithm PPO --episodes 500
   ```

Models are saved under:
```
logs/extensions/models/grid-size-N/reinforcement-<algo>_{timestamp}/
```
Training metrics print to stdout; connect your own observer to stream to
TensorBoard.

---

## 🎨 Design patterns in play

1. **Factory Pattern** – `create_rl_agent()` selects the proper class from the
   registry; adding a new algorithm is a two-line diff.
2. **Template-Method** – `RLGameManager.run()` drives the training while agent
   strategy varies.
3. **Observer** – agents emit `episode_complete` which the manager logs; can be
   extended for dashboards.
4. **Singleton** – underlying FileManager (shared via `extensions/common`) keeps
   logs coerently organised.

All classes are documented for educational clarity.

---

## 📈 Roadmap

| Version | Focus |
|---------|-------|
| v0.03 | Streamlit dashboard, PyGame & web replay, gym-compatible wrappers |
---

