
## Core
The `core/` package is already split cleanly into “generic base” vs. “Task-0 concrete” layers and follows the naming conventions (`game_*.py`).  Every first-citizen service surface—runtime loop, data containers, controller, manager and replay—has a thin, LLM-free superclass from which the Task-0 implementation inherits.  This means Task-1 → Task-5 can plug in their own logic without touching (or even importing) the LLM-specific code.

Below is a recap of the current structure, the key generic attributes/methods that future tasks inherit out-of-the-box, and a concrete example of how a heuristic agent would drop in with **zero refactor**.

--------------------------------------------------------------------
1.  Current generic spine inside `core/`
--------------------------------------------------------------------
1.1  Controllers & game logic  
• `BaseGameController`  
  – Handles board array, collision detection, apple spawning, reversal filtering, etc.  
  – Injects the data tracker through the class attribute `GAME_DATA_CLS` (defaults to `BaseGameData`).  
• `BaseGameLogic` (sub-class of `BaseGameController`)  
  – Adds only `planned_moves` helpers and the “pop next move” utility; no LLM coupling.

1.2  Data & statistics  
• `BaseGameData`  
  – Houses every attribute you listed as “should be generic”: score, steps, need_new_plan, consecutive_invalid_reversals / no_path_found, apple history, round tracking, etc.  
  – `self.stats` is an instance of `BaseGameStatistics`, whose `BaseStepStats` tracks only `valid`, `invalid_reversals`, `no_path_found`.  
• `GameData` extends it with the LLM fields (EMPTY, SOMETHING_IS_WRONG, token/time stats, etc.).

1.3  Session orchestration  
• `BaseGameManager`  
  – Keeps session-level counters (total_score, total_rounds, game_active flag, `use_gui`, `running`, etc.).  
  – Factory hook `GAME_LOGIC_CLS` lets subclasses decide which logic to spin up.  
• `BaseGameLoop` (in `game_loop.py`)  
  – Drives the tick-based loop, frame pacing, apple-after-move logic and GUI delays.  
  – Does not import any network / LLM code.

1.4  Replay  
• `BaseReplayEngine` inherits only from `BaseGameController` and therefore stays pure.  
  – Contains GUI toggle, pause_between_moves, state snapshot builder and sentinel-move execution—again totally re-usable.

1.5  Contract for agents  
• `core.game_agents.SnakeAgent` protocol = single method `get_move(game) -> str | None`.  Any future policy (heuristic, RL, supervised) just implements that.

--------------------------------------------------------------------
2.  Proof that LLM-only state lives outside the bases
--------------------------------------------------------------------
Attributes in your “should NOT be in BaseClassBlabla” list only appear in the concrete subclasses:

• `GameManager` – `empty_steps`, `something_is_wrong_steps`, `awaiting_plan`, `time_stats`, `token_stats`, `llm_client`, …  
• `GameData`  – `max_consecutive_empty_moves_allowed`, `consecutive_empty_steps`, `record_primary_token_stats`, `record_llm_communication_*`, …  
• `ReplayEngine` – `primary_llm`, `secondary_llm`, raw `llm_response`, etc.

So a second-citizen task that imports solely the `Base*` symbols will never even import the LLM modules.

--------------------------------------------------------------------
3.  How Task-1 (Heuristic BFS) would plug in
--------------------------------------------------------------------
Step 1 – implement the agent:

```python
# extensions/heuristics/algorithms/bfs.py
from core.game_agents import SnakeAgent
from typing import Any

class BFSAgent(SnakeAgent):
    def get_move(self, game: Any) -> str | None:
        """Return 'UP'/'DOWN'/'LEFT'/'RIGHT' using BFS on game.board."""
        # • game exposes board, snake_positions, apple_position,
        #   filter_invalid_reversals(), etc. via BaseGameController
        # • Only generic attributes are accessed.
        path = bfs(game)                 # your internal BFS
        return path[0] if path else "NO_PATH_FOUND"
```

Step 2 – thin manager that reuses ALL generic infra:

```python
# extensions/heuristics/manager.py
from argparse import Namespace
from core.game_manager import BaseGameManager
from core.game_loop import run_game_loop
from core.game_logic import BaseGameLogic
from extensions.heuristics.algorithms.bfs import BFSAgent

class HeuristicGameLogic(BaseGameLogic):
    """No changes needed – BaseGameLogic already perfect for planned moves."""

class HeuristicGameManager(BaseGameManager):
    GAME_LOGIC_CLS = HeuristicGameLogic     # swap in the generic logic

    def initialize(self) -> None:
        self.agent = BFSAgent()             # plug agent into base loop
        self.setup_game()                   # comes from BaseGameManager

    def run(self) -> None:
        run_game_loop(self)                 # identical to Task-0 path

# entry-point
if __name__ == "__main__":
    args = Namespace(max_games=100, move_pause=0.0, no_gui=True,
                     max_consecutive_invalid_reversals_allowed=10,
                     max_consecutive_no_path_found_allowed=1,
                     max_steps=400)
    HeuristicGameManager(args).initialize()
    HeuristicGameManager(args).run()
```

Things reused verbatim:  
• GUI toggle (`use_gui`), `get_pause_between_moves`, sentinel handling, collision & board updates, apple generation, round tracking, end-reason caps, logging helpers from `utils.game_manager_utils`, etc.

The heuristic manager neither re-implements the game loop nor touches any LLM attribute—just overrides two factory hooks.

--------------------------------------------------------------------
4.  Why this already satisfies SOLID
--------------------------------------------------------------------
• **Open for extension** – every big chunk is decomposed into small protected hooks (`_request_and_execute_first_move`, `_post_apple_logic`, `setup_game`, `start_new_round` …) so new tasks override only what they need.  
• **Closed for modification** – Task-0 code keeps working because its concrete subclasses aren’t changed; new behaviour lives in new subclasses under `extensions/`.  
• **Liskov substitution** – anything expecting a `BaseGameManager` or `BaseGameController` will accept the heuristic-specific subclass.  
• **Interface Segregation** – `SnakeAgent` keeps the contract laser-focused (one method).  
• **Dependency Inversion** – high-level managers depend on the `SnakeAgent` protocol, not concretions (BFS, PPO, LLM).

--------------------------------------------------------------------
5.  Naming & import hygiene
--------------------------------------------------------------------
All `core/` modules respect the `game_*.py` rule, and every helper in `utils/` ends with `*_utils.py`.  That guarantees newcomers (or static analysis tools) can glob for patterns without hard-coding paths.

--------------------------------------------------------------------
6.  Conclusion
--------------------------------------------------------------------
The present refactor already delivers the “BaseClassBlabla” vision:

• The *bases* own every attribute/method that Task-1 → Task-5 need.  
• All LLM-only state lives exclusively in Task-0 subclasses.  
• Future tasks can extend the system by subclassing or implementing `SnakeAgent`, without touching first-citizen code.

Therefore no further changes are required in `core/` right now.  Any additional abstraction would be “premature” because every generic hook is already exercised by Task-0, keeping us compliant with the roadmap’s rule: **“Task-0 first, everything else second.”**


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

Both numbers should be identical, confirming the round counter increments and the buffer flush at game-over.

Therefore Task-0 round logic—and its serialisation for replays and metrics— works exactly as before while remaining ready for Tasks 1-5.


## Replay
Read file: replay/replay_engine.py

Read file: replay/replay_engine.py

Status of the `replay/` layer after the recent refactor
======================================================

1 . Two-tier class hierarchy  
• `BaseReplayEngine (BaseGameController)`  
 – Pure, LLM-agnostic skeleton.  
 – Holds *all* generic attributes you enumerated (grid, board, counters, GUI flags, etc.).  
 – Exposes helper methods that every task can reuse:  
  • `set_gui` (keeps pause state in sync)  
  • `load_next_game`  
  • `execute_replay_move` (handles sentinel moves & board update)  
  • `_build_state_base` (returns a JSON-serialisable snapshot).  
 – Leaves only four hooks abstract (`load_game_data`, `update`, `handle_events`, `run`).  
     A second-citizen task can implement these in < 100 LOC or simply copy the Task-0 versions.

• `ReplayEngine (BaseReplayEngine, GameController)`  
 – Task-0 concrete subclass.  
 – Adds the LLM extras (`primary_llm`, `secondary_llm`, `llm_response`, etc.), plugs the Task-0 JSON parser, and keeps the old PyGame controls exactly as before.

2 . Data container  
• `replay/replay_data.py`  
 – `BaseReplayData` → just `apple_positions`, `moves`, `game_end_reason`.  
 – `ReplayData` extends it with LLM-specific fields.

3 . Sentinel-move safety guard (only change I introduced)  
Inside `BaseReplayEngine.execute_replay_move` we now call the LLM-only helpers **only if they exist**:

```python
elif direction_key == "EMPTY":
    if hasattr(self.game_state, "record_empty_move"):
        self.game_state.record_empty_move()
elif direction_key == "SOMETHING_IS_WRONG":
    if hasattr(self.game_state, "record_something_is_wrong_move"):
        self.game_state.record_something_is_wrong_move()
```

This prevents an `AttributeError` if a future task inadvertently logs an `EMPTY` or `SOMETHING_IS_WRONG` step but uses the generic `BaseGameData`.

4 . No hidden LLM leakage in the base  
• `BaseReplayEngine` imports only `BaseGameController`, never `llm.*`.  
• All LLM-specific constants (`END_REASON_MAP`) are used **only** in the Task-0 subclass.  
• `_build_state_base` deliberately omits LLM fields; the Task-0 subclass adds them in its override.

5 . How Tasks 1-5 will reuse the code  
A heuristic replay engine, for instance, needs just:

```python
from replay.replay_engine import BaseReplayEngine
from replay.replay_utils import load_game_json   # can be re-used
from replay.replay_data import BaseReplayData    # or its own schema

class HeuristicReplayEngine(BaseReplayEngine):
    def load_game_data(self, game_number: int):
        file_path, data = load_game_json(self.log_dir, game_number)
        parsed: BaseReplayData = my_bfs_parser(data)            # implement once
        self.apple_positions = parsed.apple_positions
        self.moves = parsed.moves
        self.game_end_reason = parsed.game_end_reason
        self.move_index = self.apple_index = 0
        self.reset()
        self.set_apple_position(self.apple_positions[0])

    # The default execute_replay_move + sentinel handling already works.
    # Optionally override update
```

## GUI

Below is a full resend of the analysis that was cut off.  It reviews the `gui/` package in light of the “BaseClassBlabla” philosophy and explains why it is already future-proof for Tasks 1-5, plus a few tiny clean-up suggestions (none of them break Task-0).

────────────────────────────────────────────────
1.  Present structure of the GUI layer
────────────────────────────────────────────────
1.1  Generic backbone  
   • `gui/base_gui.py`  
     – Owns ALL shared attributes in your “should-be-generic” list:  
       • window/board geometry (`grid_size`, `width`, `height`, `pixel`)  
       • runtime state toggles (`show_grid`, `screen`, `font`, `extra_panels`)  
       • low-level drawing helpers (`draw_square`, `draw_snake_segment`, `draw_apple`)  
       • optional HUD plug-in mechanism (`InfoPanel` protocol + `register_panel`).  
     – Absolutely no mention of LLM or Task-0 specifics.

1.2  Task-0 concrete subclasses  
   • `gui/game_gui.py` – live PyGame window used during LLM play.  
   • `gui/replay_gui.py` – adds extra HUD fields for playback (LLM models, parsed response, etc.).  
   Both inherit every board-drawing routine from `BaseGUI` and *only* add the LLM/HUD bits.

────────────────────────────────────────────────
2.  Attribute & method mapping
────────────────────────────────────────────────
Your “should NOT be in BaseClassBlabla” list: nowhere inside `BaseGUI`.  
Your “should be in BaseClassBlabla” list: all that apply to rendering already live in `BaseGUI`.  
(Things like `round_counts`, `round_manager`, etc. belong to data/manager classes, not GUI.)

────────────────────────────────────────────────
3.  How future tasks plug in
────────────────────────────────────────────────
A heuristic or RL track can reuse the window exactly as is:

```python
from gui.base_gui import BaseGUI

class HeuristicGUI(BaseGUI):
    """Optional richer HUD for Task-1 (heuristics)."""
    def __init__(self):
        super().__init__()
        self.init_display("Heuristic Snake")   # different caption

    def draw_game_info(self, game_info):
        # Show generic stats
        score = game_info["score"]
        steps = game_info["steps"]
        # custom rendering here …

        super().draw_game_info(game_info)      # keeps plug-ins working
```

Even easier: if a task only wants an extra chart, it can register a plug-in once during start-up:

```python
from gui.base_gui import register_panel, InfoPanel

class QValuePanel(InfoPanel):
    def draw(self, surface, game):
        # draw onto surface using game’s observation
        pass

register_panel(QValuePanel())
```

Every future `GameGUI`/`ReplayGUI` instance will automatically show that widget—**no subclassing required**, honouring the Open/Closed principle.

────────────────────────────────────────────────
4.  SOLID checklist
────────────────────────────────────────────────
• **S**ingle-responsibility – `BaseGUI` does the low-level drawing only; specialised GUIs add presentation logic.  
• **O**pen/Closed – New tasks extend via subclass or `InfoPanel`; no edits to first-citizen files.  
• **L**iskov – Any subclass can be passed where a `BaseGUI` is expected.  
• **I**nterface segregation – HUD plug-in contract is one tiny `draw()` method.  
• **D**ependency inversion – Game logic depends only on the *abstraction* `BaseGUI`, not concrete PyGame details.

────────────────────────────────────────────────
5.  Minor polish ideas (optional, zero impact on Task-0)
────────────────────────────────────────────────
1. Rename the method stub `draw_game_info(self, game_info)` in `BaseGUI` to `draw_info_panel` to avoid clashing with domain wording; keep an alias for back-compat.  
2. Add a one-liner `toggle_fullscreen()` utility inside `BaseGUI`—handy for RL visualisation without code changes.  
3. In `base_gui.py` add a runtime check so `self.pixel` never falls below 2 px (1-px squares virtually disappear on hi-DPI screens).

Those tweaks are cosmetic; implement them only when you touch the file for another reason.

────────────────────────────────────────────────
6.  Conclusion
────────────────────────────────────────────────
The `gui/` folder is already fully refactored along the “BaseClassBlabla” vision:

• `BaseGUI` is clean, LLM-agnostic, and contains every attribute/method that Tasks 1-5 will need.  
• Task-0-specific UIs inherit from it and add extra fields without polluting the base.  
• Future tasks can either subclass `BaseGUI` or just drop HUD widgets via the `InfoPanel` plug-in registry—**no change to first-citizen code required**.

Therefore no mandatory action is needed for the GUI layer right now; it is SOLID-compliant and future-ready.



## Round, RoundManager, Rounds



## 


## BaseGameManager

## BaseGameController

## BaseGameData

## BaseGameLoop
