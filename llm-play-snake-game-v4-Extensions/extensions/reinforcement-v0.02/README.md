### RL v0.02 (Extension)

Incremental improvement over v0.01:
- Records per-move features via logic `_post_move`
- Saves canonical `game_N.json` and a sidecar `game_N_features.json`

Algorithms (factory `create()`): Q_LEARNING, DQN, PPO (optional `model_path`).

Run:
- `python scripts/main.py --algorithm PPO --max-games 10 --grid-size 10 [--model-path /path/to/model.pkl]`