# Reinforcement Learning v0.03 – Dashboards, Replay & Web

> Evolution of v0.02; now with user-friendly visualisation and analysis tools.

---

## 🌟 Key additions

1. **Streamlit dashboard** – real-time loss / reward curves, hyper-parameter sliders.  
2. **Replay systems** – PyGame desktop and Flask web-based viewer.  
3. **Dataset export** – save trajectories to CSV / NPZ for downstream research.  
4. **Modular UI** – dashboard code lives under `dashboard/` to keep `app.py` thin.

Everything else (agents, environment, manager) is **identical to v0.02** to
ensure apples-to-apples comparisons.

---

## 📁 Folder layout

```
extensions/reinforcement-v0.03/
├── __init__.py          # re-exports RLConfig + factory from v0.02
├── README.md            # ← you are here
├── agents/              # ⚠️ copied verbatim from v0.02 – DO NOT EDIT
│   └── …
├── game_data.py         # import from v0.02
├── game_logic.py        # import from v0.02
├── game_manager.py      # Thin wrapper adding dashboard hooks
├── dashboard/
│   ├── __init__.py      # registers Streamlit tabs
│   └── training_dashboard.py
├── app.py               # Streamlit entry point
├── scripts/
│   ├── train.py         # CLI wrapper – identical flags to v0.02
│   ├── replay_pygame.py # Desktop replay
│   └── replay_web.py    # Flask replay
└── datasets/
    └── (auto-generated CSV/NPZ saved here)
```

---

## 🚀 Quick start (dashboard)

```bash
streamlit run extensions/reinforcement-v0.03/app.py --server.headless true
```
Choose algorithm & parameters in the sidebar, click **Train** – plots update
on the fly.  Replays appear under the **Replay** tab once a run finishes.

---

## 🛠️ Under the hood

* **Observer pattern** – `RLGameManager` pushes episode stats to the
  `dashboard.training_dashboard.MetricsBuffer` which emits Streamlit updates.
* **Factory pattern** – unchanged agent registry from v0.02.
* **Template-Method** – manager hooks call base training loop, injecting UI.
* **Singleton** – shared FileManager guarantees only one active log path.

---

## 📈 Upgrade path

| Version | Focus |
|---------|-------|
| v0.01 | Proof-of-concept DQN |
| v0.02 | Multi-algorithm & headless training |
| **v0.03** | Visualisation, replay, dataset export ← **YOU ARE HERE** |
| v0.04 | *Not planned* – RL stabilised |

---

*Last updated 2025-06-23* 