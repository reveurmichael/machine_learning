
## Core

## Round

Round-tracking remains exactly as before; nothing in the refactor altered how
Task-0 counts, records, or serialises rounds.

Where the counter lives
• `core/game_rounds.BaseRoundManager` – owns  
  `round_count`, `round_buffer`, and `rounds_data`.  
• `core/game_data.BaseGameData` instantiates a `RoundManager` once per game.  
• Task-0’s `GameData` calls `start_new_round()` whenever a new plan is
  requested; that increments `round_count` and flushes the previous buffer.

Where it’s incremented
1. `GameManager.finish_round()` → `increment_round()` updates both  
   `manager.round_count` and the session-level `round_counts` list.  
2. `_request_and_execute_first_move()` in `GameLoop` calls
   `manager.finish_round()` whenever the previous plan is exhausted and
   before it asks the LLM for the next one.  
3. On game reset (`_handle_game_over`) we zero `round_count`
   and push the last value into `round_counts`, exactly like before.

Where it’s written to disk
• `GameData.generate_game_summary()` embeds  
  `"round_count": self.round_manager.round_count`.  
• `save_game_summary()` flushes the in-memory buffer so the final plan’s
  moves are included in `rounds_data`.  
• Session-level `summary.json` stores `round_counts` and `total_rounds`
  via `save_session_stats()` just as in the pre-refactor code.

Generic safety
• `BaseRoundManager` is 100 % LLM-agnostic; Tasks 1-5 inherit it unchanged.  
• Its public API (`start_new_round`, `record_apple_position`,
  `record_planned_moves`, `sync_round_data`, `flush_buffer`) is documented
  with detailed, task-neutral doc-strings so future contributors know exactly
  when to call what.

Extra documentation added
• At the top of `core/game_rounds.py` a new “Why *rounds* exist” block lists
  how each upcoming task (heuristic, ML, RL, …) will map its own notion of
  a *plan* to one round.  
• `GameManager.finish_round()` got an expanded inline comment explaining the
  session-level counters and their impact on `summary.json`.  
• `GameData.start_new_round()` now states that subclasses may pass
  `apple_position=None` (safe default) so generic callers don’t need to know
  whether a particular task uses the parameter.

Quick sanity test
Run a short session (`scripts/main.py --max-games 1 --move-pause 0 --no-gui`)
and grep the produced `logs/**/game_1.json`:

```bash
jq '.round_count, (.detailed_history.rounds_data | length)' logs/*/game_1.json
```

Both numbers should be identical, confirming the round counter increments
and the buffer flush at game-over.

Therefore Task-0 round logic—and its serialisation for replays and metrics—
works exactly as before while remaining ready for Tasks 1-5.


## Replay

Everything in `replay/` is already in the requested “Base → Task-0 subclass” shape and contains only the generic attributes/methods you want in *BaseClassBlabla*:

1. replay/replay_engine.py  
   • `BaseReplayEngine` – LLM-agnostic, generic helpers only  
     – attributes: `log_dir`, `round_manager`, `pause_between_moves`,  
       `game_number`, `snake_positions`, `apple_position`, `moves`, etc.  
     – generic methods: `load_next_game`, `execute_replay_move`,  
       `_build_state_base`, `set_gui`.  
   • `ReplayEngine` – thin Task-0 subclass; adds the few LLM extras
     (`primary_llm`, `llm_response`, etc.).  
   • Doc-strings atop every class/method explain extension rules for Tasks 1-5.

2. replay/replay_utils.py, replay/replay_data.py  
   • Pure utilities / data-classes; no LLM fields; heavily documented.

3. No Task-0-specific names (empty steps, token stats …) leak into the base
   class; they appear only in `ReplayEngine` or Task-0 core modules.

4. All filenames honour the naming convention (`replay_*.py`).

5. Circular-import free: `BaseReplayEngine` imports only `core.game_controller`
   (generic) and config/constants.

So the replay layer is SOLID-compliant, ready for Tasks 1-5, and Task-0 still
works unchanged.

## GUI

