## Question

what is game_data.py ? what's the relationship between game_data.py with game_N.json and summary.json ?

## Answer

1. What `core/game_data.py` is  
   • It defines the `GameData` class, which lives inside every `Game` instance (as `game.game_state`).  
   • While a game is running every event—move, apple spawn, LLM round, response-time, token-usage, error, etc.—is funneled through one of the `record_*` or `sync_round_data()` methods of this class.  
   • The class therefore holds the canonical, in-memory “ledger” of everything that happened in the current game.

2. How a *game_N.json* file is produced  

   ┌──────── play loop (dozens of ticks) ───────┐  
   │ each tick updates `GameData` via record_*() │  
   └─────────────────────────────────────────────┘  
              ↓ end-of-game  
   utils/game_manager_utils.process_game_over()  
              ↓ calls  
   GameData.save_game_summary( …path/game_7.json… )  

   `save_game_summary()` packs the current content of the `GameData` object into a dict with `generate_game_summary()` and writes it straight to disk.  
   Result: one self-contained, ground-truth file per game (`game_1.json`, `game_2.json`, …).

3. How *summary.json* is produced  

   • Still in `process_game_over()` the manager asks the just-finished `GameData` object for high-level aggregates (`get_step_stats()`, `get_time_stats()`, `get_token_stats()`, …).  
   • Those numbers are merged into running totals that live in memory inside the GameManager.  
   • After every game the helper `utils/json_utils.save_session_stats()` rewrites `summary.json` with the updated totals.

   Therefore:

   • `game_data.py`  → single-game, tick-level data (source of truth)  
   • game_N.json     ← exact dump of one `GameData` instance at “Game Over”  
   • summary.json    ← cross-game roll-up built by summing / averaging fields extracted from each finished `GameData`.

If you ever see an inconsistency in the JSON files, the bug is almost always in the way something was *recorded* into `GameData`, because both output files ultimately come from that same object.