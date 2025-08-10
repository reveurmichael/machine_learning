### Supervised v0.03 (Extension)

Incremental improvement over v0.02:
- Adds simple evaluation metrics (apples per step, average steps per game)
- Saves per-game sidecar `features.json` and a `metrics.json` in summary folder
- Uses streamlined core APIs throughout

Run:
- `python scripts/main.py --algorithm GREEDY --max-games 10 --grid-size 10`