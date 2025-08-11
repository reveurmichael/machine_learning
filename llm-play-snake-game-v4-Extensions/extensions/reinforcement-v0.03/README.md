### RL v0.03 (Extension)

Incremental improvement over v0.02:
- Computes per-game metrics (e.g., apples per step) and writes a session `metrics.json`
- Saves canonical `game_N.json` and `game_N_features.json` per game

Algorithms (factory `create()`): Q_LEARNING, DQN, PPO (optional `model_path`).

Run:
- `python scripts/main.py --algorithm Q_LEARNING --max-games 10 --grid-size 10 [--model-path /path/to/model.pkl]`