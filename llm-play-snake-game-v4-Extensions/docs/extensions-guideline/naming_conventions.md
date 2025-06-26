> **Important — Authoritative Reference:** This guide is **supplementary** to the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`). **If any statement here conflicts with a Final Decision document, the latter always prevails.**

# Naming Conventions

## 🎯 **Core Philosophy: Names as Contracts**

In this project, names are not arbitrary labels; they are contracts. A well-chosen name tells a developer (or an AI assistant) what a component is, where it belongs, and how it should be used.

Our naming conventions are designed to enforce:
*   **Clarity:** The purpose of a file or class should be obvious from its name.
*   **Consistency:** The same patterns are used everywhere, making the codebase predictable.
*   **Automation:** Consistent names allow us to build reliable automated tools for code generation, analysis, and refactoring.

## 📋 **The Rules: A Quick Reference Guide**

This table provides the mandatory naming conventions for the entire project.

| Category               | Pattern                                        | Example(s)                                   |
| ---------------------- | ---------------------------------------------- | -------------------------------------------- |
| **File: Agents**       | `agent_{algorithm_name}.py`                    | `agent_bfs.py`, `agent_dqn.py`               |
| **File: Game Engine**  | `game_{component}.py`                          | `game_logic.py`, `game_manager.py`           |
| **File: Utilities**    | `{purpose}_utils.py`                           | `path_utils.py`, `dataset_utils.py`          |
| **Class: Base**        | `Base{ConceptName}`                            | `BaseGameManager`, `BaseAgent`               |
| **Class: Task-0**      | `{ConceptName}` (No prefix)                    | `GameManager`, `GameLogic`                   |
| **Class: Extension**   | `{ExtensionType}{ConceptName}`                 | `HeuristicGameManager`, `RLGameLogic`        |
| **Class: Agent**       | `{AlgorithmName}Agent`                         | `BFSAgent`, `DQNAgent`, `AStarAgent`         |
| **Variables/Functions**| `snake_case`                                   | `current_score`, `calculate_next_move()`     |
| **Constants**          | `UPPER_SNAKE_CASE`                             | `MAX_STEPS`, `GRID_SIZE`                     |
| **Booleans**           | `is_`, `has_`, `use_` + `snake_case`             | `is_game_over`, `has_path`, `use_gui`        |
| **Private/Internal**   | `_{leading_underscore}`                        | `_internal_state`, `_calculate_risk()`       |

## 🚫 **Common Anti-Patterns to Avoid**

Violating these patterns introduces confusion and inconsistency. The following are strictly forbidden.

#### **1. Confusing `Base` and `Task-0` Classes**
An extension must inherit from the `Base` class, never the concrete `Task-0` class.
```python
# ❌ INCORRECT: Inherits from the Task-0 implementation
class HeuristicGameManager(GameManager):
    pass

# ✅ CORRECT: Inherits from the abstract Base class
class HeuristicGameManager(BaseGameManager):
    pass
```


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

### 2.1 Rationale for Base Class Naming Convention
Because the base is *meant to be instantiated*.  `Base…` conveys "foundation"
without implying it is abstract-only.

> **Real-world example**   `BaseRoundManager` started as a thin aggregation of
> round counters. As the project evolved it gained `round_buffer` flushing
> logic that every task (heuristic, RL, etc.) can reuse. The class kept the
> `Base` prefix because Task-0 still instantiates it directly via
> `GameData`, ensuring CI covers the shared behaviour.

**Subclass naming tips**

* Use the *algorithm* or *domain* as the prefix: `BFSGameManager`,
  `PPOGameManager`, `HamiltonianCycleAgent`.
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



## 7 Constants (ALL_CAPS)

Python constants are **module-level**, written in screaming-snake case and –
critically – colocated with the *domain* they belong to.

| Domain               | Location / Example                               | Notes |
|----------------------|---------------------------------------------------|-------|
| Game-rule numbers    | `config/game_constants.py`  → `MAX_STEPS_ALLOWED` | Shared by every task. |
| UI / colour palette  | `config/ui_constants.py`    → `COLORS`, `GRID_SIZE` | First-citizen SSoT for both PyGame & Web. |
| LLM tuning knobs     | `config/llm_constants.py`   → `TEMPERATURE`       | **Task-0 & LLM-focused extensions**; other extensions must not import. |
| HTTP / network       | `config/network_constants.py` → `DEFAULT_PORT`    | Used by scripts & tests. |

Golden rule (for Task-0 only, because Task-1-5 will be standalone) : *If a constant is consumed by ≥2 tasks it belongs in `config/`*.

**Naming tips**

* Indicate units when ambiguous: `TIME_TICK_MS`, `SLEEP_AFTER_EMPTY_STEP_SEC`.

**Implementation Note**: Unit indicators should be consistently applied across the codebase. The validation utilities in `extensions/common/validation/` should enforce this naming standard for new constants.

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

---

## 9 Package & Directory Names

| Layer           | Path pattern                         | Rationale |
|-----------------|---------------------------------------|-----------|
| Core engine     | `core/`                              | No task prefix because it is first-citizen. |
| Generic utils   | `utils/`                             | Pure functions, no heavy deps. |
| GUI             | `gui/`                               | PyGame windows & helpers. |
| Web MVC         | `web/`                               | Flask blueprints, JS, templates (ROOT/web infrastructure for extensions) |
| Extensions      | `extensions/<task_name>/…`