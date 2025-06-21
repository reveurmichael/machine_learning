# Preventing Duplicate-Move & Round-Sync Bugs in Extensions

> **Audience:** Developers working on *Task-1 … Task-5* inside the `extensions/` folder.
>
> **Scope:** How to integrate alternative planners (heuristics, RL, fine-tuned LLMs …) **without** breaking the core invariants that keep Task-0 logs clean.

---

## 1 Core Invariants (Do **not** break these!)

| ID | Invariant | Why it matters |
|----|-----------|----------------|
| I-1 | **One LLM/Planner request ≙ one Round**  | Keeps `round_count`, prompt/response filenames and `RoundManager.rounds_data` in perfect lock-step. |
| I-2 | **`planned_moves` is the *only* source of truth for future moves** | Ensures UI, replays and analytics all observe the same queue. |
| I-3 | **Every executed move passes through `GameLoop._execute_next_planned_move`** | Guarantees exactly-once recording in `RoundBuffer.moves`. |
| I-4 | **When a plan finishes → `GameManager.finish_round()`** | Flushes buffers and bumps counters; *never* treat this as an "EMPTY" sentinel. |
| I-5 | **No physics class (controller/logic) may advance rounds** | Separation of concerns; only the loop owns gameplay flow. |

## 2 Typical Pitfalls & How to Dodge Them

| Pitfall | Symptoms | Avoid / Fix |
|---------|----------|-------------|
| Leaving the *first* element of a new plan in `planned_moves` | Duplicate first move in JSON (`["RIGHT","RIGHT",…]`) | Pop it immediately **or** (recommended) route all moves through `_execute_next_planned_move` like Task-0. |
| Calling `finish_round()` *too often* (e.g. after each apple) | Mismatch between prompt/response filenames and JSON rounds | Only call it when the plan queue is **empty**. |
| Treating normal round boundaries as `EMPTY` moves | `Empty Moves occurred` spam, premature "max empty" termination | Use the round-completion path (`finish_round`) instead of `_handle_no_move()`. |
| Bypassing `RoundManager.record_planned_moves()` | `planned_moves` missing in logs → replay breakage | Always feed the full plan through that helper. |

## 3 Implementation Checklist

1. **Subclass `BaseGameLoop`** – only override the minimal set of hooks:
   * `_get_new_plan()` – fetch/compute a new plan and store it in `game.planned_moves`.
   * Optionally, `_handle_no_move()` or `_handle_no_path_found()` for custom sentinels.
2. **Do *not* override `run()` unless you really need to.** If you do, replicate the Task-0 override logic that forces `_process_active_game`.
3. Inside `_get_new_plan()`:
   * If this is *not* the first plan of a game, call `manager.increment_round(<reason>)` **before** querying your planner (Invariant I-1).
   * Store the **full** plan via `game.game_state.round_manager.record_planned_moves(plan)`.
4. **Never return the first move.** Let `_execute_next_planned_move()` pull it; this avoids duplicate-first-move bugs (Invariant I-3).
5. When `_execute_next_planned_move()` yields `None`, call `manager.finish_round()` and set `manager.need_new_plan = True`.
6. **Do not touch `RoundBuffer.moves` directly.** Use `GameData.record_move()` or higher-level helpers.
7. **Unit-test your loop** with a mock planner that returns predictable plans; assert that:
   * `rounds_data[X]["moves"]` contains **no duplicates**.
   * The number of rounds equals the number of prompt/response files generated.

## 4 Skeleton Example (Task-2: Heuristic Planner)

```python
from core.game_loop import BaseGameLoop

class HeuristicLoop(BaseGameLoop):

    def _get_new_plan(self) -> None:
        manager = self.manager

        # --- Round bookkeeping -----------------------------------------
        if getattr(manager, "_first_plan", False):
            manager._first_plan = False  # first round of a new game
        else:
            manager.increment_round("heuristic new round")

        # --- Compute plan ----------------------------------------------
        plan = my_super_pathfinder(manager.game)  # returns ["UP","LEFT",…]

        # Persist plan for logging & UI
        manager.game.game_state.round_manager.record_planned_moves(plan)
        manager.game.planned_moves = plan

        # Signal to the loop that we can start executing
        manager.need_new_plan = False
```

## 5 Testing Tips

* Enable `--max-games 1 --max-steps 50 --no-gui` in scripts to run head-less.
* Assert log JSON via something like:
  ```python
  data = json.load(open(latest_game_json))
  for rnd, info in data["detailed_history"]["rounds_data"].items():
      assert info["moves"] == list(dict.fromkeys(info["moves"])), "duplicates!"
  ```

## 6 Further Reading

* `core/game_loop.py` – inline comments around the duplicate-first-move fix.
* `docs/2.md` – historical analysis of the bug.
* `docs/1.md` – deep dive into round-increment architecture.

---
Following these practices keeps your extension's logs perfectly aligned with Task-0's robust data contracts, ensuring replays, dashboards and analytics remain reliable across all future tasks. 