### Supervised v0.02 (Extension)

Incremental improvement over v0.01:
- Adds simple per-move feature extraction (for future training pipelines)
- Logs features alongside canonical `game_N.json` via the streamlined core
- Still inference-only; no training loop here

Run:
- `python scripts/main.py --algorithm GREEDY --max-games 10 --grid-size 10`