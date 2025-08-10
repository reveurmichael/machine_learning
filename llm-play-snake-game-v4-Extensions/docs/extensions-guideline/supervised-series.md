### Supervised Extensions (v0.01 → v0.03)

These extensions illustrate a clean evolution path leveraging the streamlined core API.

- v0.01: Minimal supervised playback
  - Manager uses `setup_game()`, public round API, and `save_current_game_json()`
  - GREEDY agent via canonical `create()` factory
- v0.02: Feature logging
  - Records per-move features in logic (`_post_move`)
  - Saves sidecar `game_N_features.json` next to canonical `game_N.json`
- v0.03: Metrics
  - Computes simple metrics per game and saves a `metrics.json` at the summary level

Related docs:
- `final-decision.md`: authoritative governance and conventions
- `core.md`: core architecture, base classes
- `round.md`: round public APIs and synchronization
- `game_file_manager.md`: logging conventions and locations
- `extension-skeleton.md`: template for new extensions

Keep code minimal; prefer core helpers and public APIs; avoid private access.