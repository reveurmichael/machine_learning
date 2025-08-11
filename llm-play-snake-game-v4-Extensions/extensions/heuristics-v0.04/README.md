### Heuristics v0.04 (Extension)

This extension demonstrates a clean, newly shipped extension that uses the streamlined core API.

Key points:
- Inherits `BaseGameManager`; no custom `setup_game` needed. `--grid_size` is handled by core.
- Uses round public API: `round_manager.record_round_game_state(state)` to store pre/post move snapshots.
- Saves per-game JSON via `save_current_game_json(metadata={...})` which picks the best serializer.
- Uses `display_basic_results()` and `save_simple_session_summary()` for consistent I/O.
- Avoids private attributes and legacy patterns.

Structure:
- `game_logic.py`: extends `BaseGameLogic` for heuristic planning; records planned moves with `round_manager`.
- `game_manager.py`: orchestrates games, uses core helpers for logging, saving, and reset.
- `dataset_generator.py`: handles JSONL/CSV generation specific to this extension.

Minimal flow in manager:
1) `initialize()` → `setup_logging(...)`, `setup_game()`, optional `set_agent(...)`.
2) Run loop builds pre-move state, asks agent for a move, calls `make_move(...)`.
3) On game end, call `save_current_game_json(...)`, update datasets, `reset_for_next_game()`.

Use this extension as a template for future extensions. Keep core logic in `core/` and only implement what’s extension-specific here.