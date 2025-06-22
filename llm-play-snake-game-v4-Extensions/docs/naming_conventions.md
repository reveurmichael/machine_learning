# Naming Conventions – First-Citizen vs Second-Citizen

This document is the *single, authoritative* reference for how every class,
function, file and directory should be named in **Snake-GTP**.  It codifies
what has been discussed in design docs, `System-Prompt.txt`, and recent PRs.
The goal is to keep the codebase discoverable, avoid circular dependencies and
protect Task-0 as the first-citizen of the repository.

---

## 1 Guiding Principles

1. **Task-0 first, everything else second** – default implementations in the
   root packages exist *for Task-0*.  Extensions live in `extensions/`.
2. **Base-class prepared, but never abstract-only** – if you add a `Base…`
   class it **must** be instantiated by Task-0 code immediately so CI covers
   it.
3. **No pollution** – second-citizen tasks (heuristics, RL, …) import from
   core/root.  The reverse dependency is forbidden.

---

## 2 Classes

| Scope                      | Prefix & Example            | Rationale |
|----------------------------|-----------------------------|-----------|
| Generic, reusable by all tasks (0-5) | `Base` prefix  → `BaseGamePlayController`, `BaseGameData` | Signals a thin contract ready for extension. |
| Task-0 concrete            | *No prefix*  → `GamePlayController`, `GameManager` | Task-0 is the default, so terse names stay in the root. |
| Task-N concrete (N≥1)      | Task tag prefix inside its package → `HeuristicGameManager`, `RLGameManager` | Keeps second-citizen code out of root namespace. |
| Legacy / transitional      | Deprecated names stay as **aliases** only inside factories/tests; to be removed after a grace period. |

### 2.1 Why not suffix the base (`AbstractGameManager`)?
Because the base is *meant to be instantiated*.  `Base…` conveys "foundation"
without implying it is abstract-only.

> **Real-world example**   `BaseRoundManager` started as a thin aggregation of
> round counters. As the project evolved it gained `round_buffer` flushing
> logic that every task (heuristic, RL, etc.) can reuse. The class kept the
> `Base` prefix because Task-0 still instantiates it directly via
> `GameData`, ensuring CI covers the shared behaviour.

**Subclass naming tips**

* Use the *algorithm* or *domain* as the prefix: `BFSGameManager`,
  `PPOGameManager`, `VisionConvNet`, `HamiltonianCycleAgent`.
* Avoid redundant suffixes: prefer `ReplayEngine` over `ReplayEngineImpl`.
* When two concretes differ only by parameterisation, expose the parameter
  instead of inventing two names, e.g. `RLGameManager(algorithm="DQN")`.

---

## 3 Files

| Category          | Pattern                  | Examples |
|-------------------|--------------------------|----------|
| Core engine       | `core/game_*.py`         | `game_loop.py`, `game_stats.py` |
| Utilities         | `utils/*_utils.py`       | `board_utils.py`, `json_utils.py` |
| GUI modules       | `*_gui.py`               | `game_gui.py`, `replay_gui.py` |
| Extension module  | `extensions/<task>/…`    | `extensions/heuristics/manager.py` |
| Tests             | `tests/test_*.py`        | `tests/test_mvc_framework.py` |

**Rationale**

* Glob-friendly patterns let tooling (Ruff, PyTest, coverage) include the
  correct files without a custom manifest.
* The `game_*.py` prefix in *core* means a newcomer can `rg "class .*Game"`
  and immediately see the entire engine surface area.
* Mirroring the root layout inside `extensions/` keeps navigation muscle
  memory intact: pressing the *same* hotkeys in an IDE jumps to equivalent
  modules regardless of task.

---

## 4 Web MVC Layer (specific)

```
BaseWebController
├── BaseGamePlayController   – common gameplay behaviour
│   ├── GamePlayController    – Task-0 LLM game
│   └── HumanGameController   – Task-0 human game
└── BaseGameViewingController – common viewer behaviour
    └── ReplayController      – Task-0 replay viewer
```

Extensions will add e.g. `HeuristicGamePlayController` under their own
sub-package and inherit from the *Base* classes above.

### Why not merge the controllers?

Because gameplay vs. viewing differs fundamentally in **write-access** to the
game state. Conflating them would invite subtle bugs where a replay endpoint
mutates live state or an RL agent pauses the global Flask app. Separate bases
give each concern its own guard-rails.

---

## 5 Factory Registry Keys

The `ControllerFactory` maps simple slug strings to classes:

```python
_controller_registry = {
    'game': GamePlayController,      # Task-0 default
    'human_game': HumanGameController,
    'replay': ReplayController,
    # extensions register their own: 'heuristic_game', 'rl_game', …
}
```

Alias keys (`'llm_game'`) exist for a short transitional period after renames
so external scripts are not broken; they will be removed in the next major
release.

---

## 6 Deprecation Timeline

| Stage | Action |
|-------|--------|
| T      | Rename classes & keep import aliases/factory keys. |
| T+30d | Update documentation & examples (done). |
| T+60d | Remove aliases, bump **major** version. |

---

## 7 Constants (ALL_CAPS)

Python constants are **module-level**, written in screaming-snake case and –
critically – colocated with the *domain* they belong to.

| Domain               | Location / Example                               | Notes |
|----------------------|---------------------------------------------------|-------|
| Game-rule numbers    | `config/game_constants.py`  → `MAX_STEPS_ALLOWED` | Shared by every task. |
| UI / colour palette  | `config/ui_constants.py`    → `COLORS`, `GRID_SIZE` | First-citizen SSoT for both PyGame & Web. |
| LLM tuning knobs     | `config/llm_constants.py`   → `TEMPERATURE`       | **Task-0 only**; do **not** move to a base module. |
| HTTP / network       | `config/network_constants.py` → `DEFAULT_PORT`    | Used by scripts & tests. |

Golden rule : *If a constant is consumed by ≥2 tasks it belongs in `config/`*.

**Naming tips**

* Indicate units when ambiguous: `TIME_TICK_MS`, `SLEEP_AFTER_EMPTY_STEP_SEC`.
* Avoid pre-optimisation: keep logarithms, lookup tables or environment-
  specific tweaks **out of constants**; compute them lazily in code.
* Versioned constants belong in their own file (e.g. `schema_v2_constants.py`)
  so migrations can coexist temporarily.

---

## 8 Variables & Functions

1. **snake_case** for variables and functions.<br/>
2. **PascalCase** for classes & enums (already covered above).<br/>
3. Use **verb_noun** pairs for functions that *do* something: `reset_game()`,
   `update_board_array()`.
4. Leading underscore `_helper()` marks a private utility used *only* inside
   the module/class.  Avoid the Java-style `public`/`private` comments – the
   name says it all.

### 8.1 Boolean Flags

Prefix with **is_**, **has_**, **use_** or **allow_** so linters (and humans)
can infer the type: `is_paused`, `has_collision`, `use_gui`, `allow_reset`.

**Function docstring rule of thumb**

If the function signature cannot be understood by reading the *name + types*
alone, add a one-line docstring. If the body exceeds ~20 lines *or* the
function has side effects beyond its module, add a full NumPy-style docstring.

---

## 9 Package & Directory Names

| Layer           | Path pattern                         | Rationale |
|-----------------|---------------------------------------|-----------|
| Core engine     | `core/`                              | No task prefix because it is first-citizen. |
| Generic utils   | `utils/`                             | Pure functions, no heavy deps. |
| GUI             | `gui/`                               | PyGame windows & helpers. |
| Web MVC         | `web/`                               | Flask blueprints, JS, templates. |
| Extensions      | `extensions/<task_name>/…`           | Keeps second-citizen code quarantined. |
| Docs            | `docs/`                              | Markdown only – auto-rendered by GitHub. |

Inside an extension folder, mirror the root layout:

```text
extensions/heuristics/
│   algorithms/
│   gui_heuristics.py
│   web/
│   app.py
│   config.py
│   …
```

This symmetry lets devs jump between Task-0 and Task-N without context switch.

> **Hint for extension authors**  If your task needs an external dependency
> (e.g. Stable-Baselines3) list it in
> `extensions/<task>/requirements.txt` – the root requirements must stay
> lightweight for casual contributors.

---

## 10 Pull-Request Checklist for New Names

1. **Does the class belong in Task-0?**  → Use the short name.<br/>
2. **Is it a reusable contract?**         → Prefix with `Base` and ensure
   Task-0 instantiates it.<br/>
3. **Is it extension-specific?**          → Live under `extensions/<task>/` and
   prefix with the task tag (`Heuristic…`, `RL…`).<br/>
4. **Did you update `docs/naming_conventions.md`?**<br/>
5. **Did you add/adjust factory keys?**   (`extensions_controller_factory.py` or
   root `web/factories.py`).

Failing any of the above blocks the PR.

**CI naming lint**

A lightweight script `scripts/ci/check_names.py` (TODO) will scan PR diff and
flag violations automatically. Until then reviewers must enforce the list
above manually.

---

## 11 Common Pitfalls & Anti-Patterns

| ❌ Anti-pattern                         | ✅ Correct Approach |
|----------------------------------------|---------------------|
| Creating `LLMWhatever` in *root*       | Use `Whatever` (short) – Task-0 is implicit. |
| Importing Task-0 code **inside** an extension | Depend only on base classes & utils. |
| Adding unused abstraction `AbstractFoo` | Promote to `BaseFoo` **only if Task-0 uses it**; else keep it private. |
| Mixing GUI & headless logic            | Gate PyGame calls behind `if self.use_gui…`. |
| Saving extension logs under `logs/` root | Use `logs/<task_name>/…` to avoid clutter. |

Additional gotchas:

* **`import core.game_loop as gl`** – Alias imports obscure greps; import the
  symbol you need (`from core.game_loop import run_game_loop`).
* **Prefix-drift** – Once you commit to `BaseFoo` do *not* later rename it to
  `FooBase`; consistency beats perceived elegance.

---

## 12 Quick-Reference Cheatsheet

```python
# Good
class BaseReplayEngine: ...
class ReplayEngine(BaseReplayEngine):  # Task-0 concrete

class HeuristicReplayEngine(BaseReplayEngine):  # extension-specific
    pass

# Bad – violates naming & dependency rules
class RLReplayEngine(ReplayEngine):  # ❌ depends on Task-0 concrete
    pass

# Good constant location
from config.game_constants import MAX_STEPS_ALLOWED

# Bad – local magic number replicates constant
MAX_STEPS = 500  # ❌ duplicate single-source-of-truth
```

**Remember** – names are a *public API* for future you. Choose clarity over
brevity when the intent is not obvious.

---

*When in doubt, open this file and follow the examples.  Consistency beats
creativity in names!*

---

*Last updated: 2025-06-22*