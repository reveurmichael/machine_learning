# Roadmap: From v2 (Cartesian-Coordinate) to a v3 Engineered Platform

**A Living Document on Engineering Robust LLM Systems**

This roadmap chronicles the journey from a proof-of-concept **v2** (`./llm-play-snake-game-v2-cartesian-coordinate`) to a more robust, well-engineered **v3** platform (`./llm-play-snake-game-v3-MoE`). It is intentionally verbose, designed to serve not just as a project summary, but as a practical, evidence-based guide for researchers, engineers, and students on the principles of building robust, extensible, and observable AI systems.

---

## Part I: The Strategic Context


### 1. Introduction: From Prototype to Platform
The v2 codebase demonstrated that a general-purpose LLM can, with the right prompt, control a Snake agent from a plain-text board description. That proof-of-concept served its purpose but was difficult to extend. v3 focuses on turning the idea into a maintainable research tool rather than changing gameplay mechanics.

### 2. The Core Hypothesis of v2: Proving Feasibility
v2 asked a single question: "Can a language model play Snake from Cartesian coordinates alone?" The answer was "yes"—but with minimal engineering safeguards.

### 3. The Strategic Need for v3: Unlocking Research Velocity
Once feasibility was proven, the next bottleneck became iteration speed. Adding another provider, changing the prompt, or replaying a game in v2 all required manual work. v3's aim is therefore higher research velocity through modular code, structured logs, and multiple UI front-ends.

### 4. Executive Summary: The v2-to-v3 Comparison
Moving from v2 to v3 required refactoring most source files. The table below highlights the main differences.

| Topic                 | v2: The Prototype                               | v3: The Platform                                                              |
|-----------------------|-------------------------------------------------|-------------------------------------------------------------------------------|
| **Project Layout**    | 6 core Python files in a flat directory        | 8 domain-driven packages (`core/`, `gui/`, `llm/`, `utils/`, `dashboard/`, `replay/`, `config/`, `web/`) |
| **Architecture**      | Monolithic (`snake_game.py` ~600 LOC)           | Layered, package-based modules with clear interfaces                           |
| **LLM Abstraction**   | Hard-coded, single-provider logic               | Provider registry with optional secondary-LLM parser support                   |
| **Configuration**     | Single `config.py` with global constants        | Hierarchical `config/` package with dedicated modules for constants           |
| **Logging**           | Scattered `print()` statements & `.txt` files   | Structured `game_N.json` logs that capture token usage and basic timing metrics |
| **Observability**     | Manual log-scrolling                            | Structured JSON logs plus a replay engine and a Streamlit dashboard for monitoring sessions |
| **GUI**               | Pygame-only, tightly coupled to game logic      | Decoupled `gui/`, `dashboard/`, and `web/` layers supporting Pygame desktop, Streamlit configuration, and Flask web interfaces (live gameplay, human play, replay) |
| **Testability**       | GUI-bound, effectively untestable               | Headless-capable core (`GameController.make_move()`) designed to enable unit testing |
| **Error Handling**    | Silent failures (e.g., empty LLM response)      | Explicit error sentinels (`EMPTY`, `INVALID_REVERSAL`) with retry logic       |
| **Extensibility**     | High-friction; new features required deep edits   | Lower-friction; new providers/features can be added mostly by extension (minor registry edit) |

### 5. Architectural Analysis of the v2 Monolith (`snake_game.py`)
v2 revolved around one ~600-line file, `snake_game.py`, which combined rules, rendering, logging, and LLM communication. This "everything-in-one-place" design made local experiments easy but created maintenance friction once new features were considered.

## Part II: Where v2 Hurt in Day-to-Day Use

### 6. Pain Point 1: GUI Coupling
Because `SnakeGame` handled rendering, it imported `DrawWindow` directly from `gui.py`. This single-direction dependency tightly coupled game logic with the Pygame UI: the core class could not be instantiated, nor its methods unit-tested, without a valid graphical context. Although there was no circular import, the coupling still violated separation-of-concerns and limited reuse.

### 7. Pain Point 2: Minimal Test Coverage
Creating a `SnakeGame` instance required an active Pygame window. Mocking that environment was possible but cumbersome, so in practice almost no automated tests were written. All verification happened manually.

### 8. Pain Point 3: Scattered Configuration
Only a subset of tunables lived in `config.py`; colours and window sizes were embedded in `snake_game.py`, and prompt strings hid inside `llm_client.py`. Changing a single parameter required a multi-file search.

### 9. Pain Point 4: Limited Observability
Each v2 run produced raw text logs but no unified timeline. Without board snapshots or time stamps it was hard to diagnose why a game crashed at step 37.

### 10. Pain Point 5: Difficult Extensibility
Adding functionality beyond the demo (e.g., new provider, resume-game option, alternative GUI) required coordinated edits across `snake_game.py`, `main.py`, and `llm_client.py`. The likelihood of breaking existing behaviour discouraged experimentation.

---

## Part III: Design Philosophy and Guiding Principles

*This part establishes the foundational engineering principles that guided the v3 architecture.*

### 11. Guiding Principle 1: Separation of Concerns
In v3 the core rules (`core/game_controller.py`) know nothing about Pygame, and provider code (`llm/providers/*`) knows nothing about grid rules. Each layer can now be tested—or replaced—without touching the others.

### 12. Guiding Principle 2: Pure Core, Impure Shell
Core classes avoid high-latency operations; GUI updates, network calls, and disk writes live in outer layers. The core occasionally prints to stdout for debugging, which does not interfere with head-less use.

### 13. Guiding Principle 3: Configuration as Code (CaC)
All tunable parameters—from `GRID_SIZE` to default model names—now live in the `config/` package. Changing the board size is a one-line edit in `config/game_constants.py`.

### 14. Guiding Principle 4: Replay-Driven Debugging
Each game produces a `game_N.json` with the move list, apple positions, and token counts. `replay/replay_engine.py` can deterministically reconstruct the session. A bug that cannot be reproduced from this log is treated as a logging gap.

### 15. Guiding Principle 5: Fail Loud, Recover Gracefully
When an LLM returns no content, v3 records the `EMPTY` sentinel, then either retries or skips the step (see `core/game_logic.py::_handle_llm_failure`). Silent failures are avoided.

### 16. The Type-Hinted Codebase
v3 adds **type hints** to all public classes and most helper functions. These annotations improve IDE completion and allow tools like `mypy` to detect mismatched types (for example, passing `List[str]` where `str` is expected). Runtime behaviour remains unchanged.

#### 16a. V2's Type Wilderness: Dynamic Programming Without Guardrails
The v2 codebase suffered from a lack of type annotations, leading to several classes of bugs:

```python
# v2: llm_client.py - No type hints
class LLMClient:
    def __init__(self, provider: str = "hunyuan", model: str = None):  # model should be Optional[str]
        self.provider = provider
        self.model = model

    def generate_response(self, prompt: str, **kwargs) -> str:  # Return type unclear
        # Could return str, None, or throw exception
        if self.provider == "hunyuan":
            response = self._generate_hunyuan_response(prompt, **kwargs)
        # ... no indication of what **kwargs contains
```

#### 16b. V3's Type Fortress: Consistent Static Typing
V3 applies consistent type annotations using modern Python typing features:

```python
# v3: core/game_controller.py - Comprehensive type hints
from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from numpy.typing import NDArray

class GameController:
    def __init__(self, grid_size: int = GRID_SIZE, use_gui: bool = True) -> None:
        """Type-safe constructor with clear parameter contracts."""
        
    def make_move(self, direction_key: str) -> Tuple[bool, bool]:
        """Returns (game_active: bool, apple_eaten: bool)"""
        
    def _generate_apple(self) -> NDArray[np.int_]:
        """Returns numpy array with integer coordinates."""
```

### 17. The DRY Principle: From Code Duplication to Elegant Abstraction
Reducing repetition was a key driver of the refactor. The largest win came from centralising JSON extraction/parsing so that every module calls the same helper.

#### 17a. V2's Repetition Problem: Copy-Paste Programming
In v2, at least three files (`snake_game.py`, `json_utils.py`, `llm_client.py`) each carried a slightly different `parse_llm_response()` helper:

#### 17b. V3's DRY Implementation: Centralized Utilities
`utils/json_utils.py` is now the single location for JSON extraction/parsing, and higher layers simply import `extract_json_from_text()`.

### 18. Single Source of Truth: Eliminating Data Redundancy
V3 adopts a **Single Source of Truth (SSOT)** approach to reduce data inconsistencies and cognitive load.

#### 18a. V2's Data Duplication Problem
In v2, the same information often lived in multiple places:

```python
# v2: Multiple sources for game state
# Game state in SnakeGame class
# Display state in GUI
# Log state in separate files
# No authoritative source for "current game state"
```

#### 18b. V3's SSOT Implementation
V3 designates `GameData` as the single authoritative source:

```python
# v3: core/game_data.py - Single source of truth
class GameData:
    def __init__(self):
        self._score = 0  # Private backing field
        
    @property
    def score(self) -> int:
        """All score access goes through this property."""
        return self._score
```

### 19. Property vs. Attribute Design: Intelligent Data Access
V3 strategically uses Python's `@property` decorator to create intelligent interfaces to data, providing computed values, validation, and future extensibility points.

#### 19a. V2's Direct Attribute Access
```python
# v2: Direct attribute access everywhere
snake_length = len(self.snake_positions)  # Computed every time
```

#### 19b. V3's Property-Based Design
```python
# v3: Intelligent property access
@property
def snake_length(self) -> int:
    """Computed property - single implementation."""
    return len(self.snake_positions)
```

### 20. Continue Mode: Stateful Session Persistence
v3 introduces **Continue Mode** – the ability to resume interrupted game sessions by re-loading a saved JSON log. This required careful state-serialisation but avoids invasive changes to gameplay logic.

#### 20a. The Continue Mode Architecture
**State Persistence Layer**: V3 saves complete game state to structured JSON:

```python
# v3: core/game_data.py - Comprehensive state serialization
def generate_game_summary(self) -> dict:
    return {
        "game_number": self.game_number,
        "score": self.score,
        "steps": self.steps,
        "snake_positions": self.snake_positions,
        "is_continuation": getattr(self, "is_continuation", False),
        "continuation_count": getattr(self, "continuation_count", 0),
    }
```

---

## Part IV: Core Architecture - The Game Engine

*This part explores the heart of v3 - the core game logic and state management systems.*

### 21. Section Overview: The Domain-Driven Package Structure
V3 organizes code into packages based on their domain responsibility:
- `core/`: The game's heart. Rules, state, and headless logic.
- `llm/`: All LLM-related code: provider abstractions, clients, prompts.
- `gui/`: The Pygame desktop GUI.
- `dashboard/`: The Streamlit web GUI.
- `replay/`: The game replay engine (both CLI and web-based).
- `utils/`: Shared helper functions (file I/O, networking, etc.).
- `config/`: All configuration constants and templates.
- `web/`: Directory containing Flask-based web templates and static files.
- `logs/`: The output directory for all game data.

### 22. Deep Dive: `core/game_controller.py` - The Headless Rule Engine
`core/game_controller.py` contains the `GameController` class, responsible for basic rules such as movement, collision checks and apple placement. The class exposes methods that can run entirely without a GUI, which simplifies automated tests and server-side deployments.

### 23. Deep Dive: `core/game_data.py` - The Game Statistics Tracker
`core/game_data.py` defines `GameData`, a dataclass-like container that records each move, score increment, and apple spawn. The method `generate_game_summary()` emits a serialisable snapshot used by both the replay engine and the Streamlit dashboard.

### 24. Deep Dive: `core/game_logic.py` - The AI Bridge
`core/game_logic.py` subclasses `GameController`, augmenting it with the code needed to chat with an LLM. It formats prompts via `llm/prompt_utils.py`, forwards them through `LLMClient`, and converts model output back into validated moves using `llm/parsing_utils.py`.

### 25. Deep Dive: `core/game_loop.py` - The Main Loop
`core/game_loop.py` provides `run_game_loop()`, a wrapper around a standard `while` loop that advances the game one tick at a time:
```python
while game_manager.running:
    process_events(game_manager)
    if game_manager.game_active:
        _process_active_game(game_manager) # plan ↔ move ↔ GUI ↔ logging
```

### 26. Deep Dive: `core/game_manager.py` - Session Orchestrator
`core/game_manager.py` coordinates everything: it spins up the chosen GUI (if any), instantiates the `LLMClient`, allocates a fresh log directory, and maintains aggregate statistics across multiple games.

### 27. Deep Dive: `core/game_stats.py` - Game Metrics
`core/game_stats.py` groups the small dataclasses (`StepStats`, `TimeStats`, `TokenStats`) that hold per-step timing and token-usage data. Having a single module avoids name-drift and keeps statistical helpers in one place.

### 28. Deep Dive: `core/game_rounds.py` - Understanding Agent Planning Cycles
`core/game_rounds.py` introduces a "round" abstraction – the chunk of moves executed between consecutive LLM calls. `RoundManager` records the plan proposed by the model and the subset that actually gets executed, making discrepancies easy to diagnose.

---

## Part V: LLM Integration - The AI Layer

*This part examines how v3 abstracts and manages AI model interactions through a sophisticated provider system.*

### 29. Deep Dive: `llm/providers/base_provider.py` - The Provider Contract
`llm/providers/base_provider.py` contains `BaseProvider`, an abstract class that standardises the signature for `generate_response()` and exposes optional helpers such as `validate_model()`. Concrete providers subclass it so that `LLMClient` can swap them at runtime without special-case code.

### 30. Deep Dive: `llm/providers/__init__.py` - Provider Registry
`llm/providers/__init__.py` maintains `_PROVIDER_REGISTRY`, a simple `{name: class}` mapping. Helper functions such as `create_provider()` and `get_default_model()` give the rest of the codebase a single entry-point for creating provider instances.

### 31. Deep Dive: Concrete Providers
Files like `deepseek_provider.py` or `ollama_provider.py` wrap the HTTP or CLI calls needed for each service. They translate provider-specific responses into plain strings and optionally return token-usage dictionaries.

### 32. Deep Dive: `llm/client.py` - Request Gateway
`llm/client.py` hosts `LLMClient`, a thin wrapper that selects the active provider, forwards parameters, and surfaces token counts. This keeps higher-level modules free from provider-specific logic.

### 33. Deep Dive: `llm/communication_utils.py` - Prompt/Response I/O
`llm/communication_utils.py` glues the game layer to the LLM layer: it builds the prompt via `prompt_utils`, calls `LLMClient`, persists prompt/response pairs on disk, and hands parsed moves back to the caller.

### 34. Deep Dive: `llm/prompt_utils.py` - Prompt Templates
`llm/prompt_utils.py` converts internal board state into the plain-text grid that models have been trained to interpret (`prepare_snake_prompt`). Centralising template code helps keep the prompt in sync across different entry points (desktop, web, tests).

### 35. Deep Dive: `llm/parsing_utils.py` - Response Parsing
`llm/parsing_utils.py` extracts structured moves from free-form model output. It looks for JSON blocks, fallbacks to regex extraction, and validates each direction token against `DIRECTIONS`.

### 36. Deep Dive: `gui/base_gui.py` - Common GUI Helpers
`gui/base_gui.py` defines `BaseGUI`, which supplies shared Pygame helpers (colour constants, text rendering, board sizing). Actual front-ends such as `game_gui.py` or the Streamlit dashboard implement only the rendering specifics, not low-level drawing math.

---

## Part VI: User Interfaces - Multi-Modal Access

*This part covers the various interfaces that make the v3 platform accessible across different environments and use cases.*

### 37. Deep Dive: `gui/game_gui.py` - Pygame Desktop UI
`gui/game_gui.py` implements a classic Pygame window. It receives draw calls from `GameController`, converts logical board coordinates into pixels, and shows live planned-move overlays for debugging.

### 38. Deep Dive: `dashboard/` - Streamlit Web UI
`dashboard/` contains a Streamlit app that lets users configure parameters (grid size, provider, model), start games, and watch token/score charts update live. Because Streamlit runs in a browser it works even on machines without a local display.

### 39. Browser-Based Interfaces (Flask)
V3 ships several lightweight Flask apps:

* `main_web.py` – watch an LLM play in real time.
* `human_play_web.py` – play Snake directly in the browser.
* `replay_web.py` – step through a saved `game_N.json`.

### 40. Deep Dive: `replay/replay_engine.py` - Offline Replay
`replay/replay_engine.py` replays any JSON log. It reconstructs each frame in Pygame, optionally throttled for quick scrubbing, which is useful when an LLM makes an unexpected move.

### 41. Deep Dive: `utils/file_utils.py` - File Management
`utils/file_utils.py` abstracts path construction and atomic write helpers so that prompt/response pairs and logs end up in predictable locations.

### 42. Deep Dive: `utils/continuation_utils.py` - Resume Support
`utils/continuation_utils.py` inspects `summary.json` plus the most recent `game_N.json` to rebuild `GameManager` state, enabling an interrupted run to continue without data loss.

### 43. Deep Dive: `utils/web_utils.py` - Shared Web Helpers
Used by every Flask route, `utils/web_utils.py` handles common tasks such as RGB-to-hex conversion and serialising board arrays into JSON-friendly dictionaries.

### 44. Deep Dive: `utils/network_utils.py` - Free Port Discovery
`utils/network_utils.py` scans localhost ports to find an unused one before launching a Flask or Streamlit service, avoiding hard-coded port clashes.

### 45. Deep Dive: `config/` - Typed Constants and Templates
Instead of a single monolithic `config.py`, v3 splits constants across focused modules: `game_constants.py` (board dimensions, colours), `llm_constants.py` (model defaults), `ui_constants.py` (font sizes) and `prompt_templates.py` (string literals). All files carry type hints.

### 46. Design Principle: Single-Responsibility in `core`
Each core class addresses a single concern: `GameController` applies the rules, `GameData` records statistics, and `GameLogic` bridges the AI. Changes in one area rarely ripple into the others.

---

## Part VII: Supporting Infrastructure

*This part examines the utility systems, configuration management, and supporting infrastructure.*

### 47. Deep Dive: `utils/file_utils.py` - File Management
This module centralizes file I/O operations. It ensures that all prompts and responses are saved to the correct directories with consistent naming conventions.

### 48. Deep Dive: `utils/continuation_utils.py` - Session Persistence
This utility allows a user to stop a multi-game session and resume it later. It reads the `summary.json` and the last `game_N.json` from a log directory to restore the `GameManager`'s state.

### 49. Deep Dive: `utils/web_utils.py` - Shared Web Infrastructure
This module centralizes functionality required by all Flask apps, including color mapping for consistent visuals and game-state serialization utilities for JSON APIs.

### 50. Deep Dive: `utils/network_utils.py` - Network Port Management
Provides utilities for discovering free TCP ports so multiple web services can run concurrently without conflicts.

### 51. Deep Dive: `config/` - Configuration Management
V3 elevates configuration from a single file to a dedicated package. Files like `game_constants.py`, `llm_constants.py`, `ui_constants.py`, and `prompt_templates.py` each have a clear responsibility.

---

## Part VIII: Engineering Patterns and Design Principles

*This part provides an explicit breakdown of how classic software engineering principles were applied in the v3 codebase.*

### 52. SOLID Principle: Single-Responsibility in `core`
A class should have only one reason to change. This is exemplified by the split between `GameController` (core rules), `GameData` (statistics), and `GameLogic` (AI integration).

### 53. SOLID Principle: Open/Closed in `llm/providers`
Software entities should be open for extension, but closed for modification. The LLM provider system demonstrates this: one can add a new provider without touching existing code.

### 54. SOLID Principle: Liskov Substitution in `LLMClient`
Subtypes must be substitutable for their base types. Because all LLM providers implement the `BaseProvider` interface, the `LLMClient` can use any of them interchangeably.

### 55. SOLID Principle: Interface Segregation in `gui`
No client should be forced to depend on methods it does not use. The `GameController` uses optional GUI dependency injection with a minimal interface.

### 56. SOLID Principle: Dependency Inversion in `GameManager`
High-level modules should not depend on low-level modules; both should depend on abstractions. The `GameManager` depends on abstract `BaseProvider`, not concrete implementations.

### 57. Design Pattern: The Factory/Registry for Providers
The `create_provider()` function acts as a factory. It takes a string ("ollama") and returns a fully-formed `OllamaProvider` object.

### 58. Design Pattern: The Strategy Pattern for LLMs and GUIs
The ability to select an LLM provider or a GUI at runtime is a classic example of the Strategy pattern.

### 59. Design Pattern: The Facade Pattern of `GameManager`
The `GameManager` class serves as a Facade. It provides a simple, unified interface (`run()`) to a complex subsystem.

### 60. Design Pattern: Utility Base Class in `BaseGUI`
The `BaseGUI` class provides common functionality for GUI implementations, promoting code reuse and consistent behavior.

---

## Part IX: Outcomes and Lessons Learned

*This part reflects on the results of the transformation and extracts actionable insights.*

### 61. Advantages of v3 over v2
- **Modularity**: Code is organized by domain, making it easier to navigate and maintain.
- **Extensibility**: New functionality can be added primarily by extension.
- **Testability**: A modular, headless core makes it easier to add unit tests.
- **Observability**: Structured JSON logs plus a replay engine enable deterministic debugging.
- **Configuration Management**: All tunable parameters live in version-controlled modules.
- **Error Handling**: Explicit sentinels and retry logic make failures transparent.
- **Multi-Platform Access**: The core supports desktop, web dashboard, and Flask interfaces.

### 62. Key Lesson for Programmers: Abstraction is Leverage
The core lesson from this refactor is that good abstractions (like `BaseProvider` and `BaseGUI`) provide significant benefits. They require an initial investment of thought, but they pay dividends by making future changes easier.

### 63. Key Lesson for AI Practitioners: Engineering is the Foundation of Research
Strong AI models or prompts are not sufficient to create a useful research tool. The v3 project demonstrates that solid software engineering provides an important foundation for meaningful and reproducible AI research.

### 64. Guidelines for Migrating to a Modular Architecture
1. **Carve Out a Modular Core** – Isolate domain rules to minimize I/O and reduce external dependencies.
2. **Define Stable Interfaces** – Introduce abstract base classes to decouple high-level policy from low-level detail.
3. **Invert Dependencies** – Make orchestrators depend on interfaces, then inject concrete implementations at runtime.
4. **Centralize Configuration & Logging** – Move constants into a dedicated package and route all file I/O through helper modules.
5. **Add Observability Early** – Log every interaction in machine-readable form.
6. **Refactor Incrementally** – Migrate one domain at a time, shipping tests with each slice.

---

## Part X: Future Directions and Practical Guidance

*This part looks forward to potential enhancements and provides practical guidance for contributors.*

### 65. A Guide for Future Contributors: How to Add a New LLM Provider
Thanks to the v3 architecture, adding a new provider requires a straightforward process:
1. Create `new_provider.py` in `llm/providers/`, subclassing `BaseProvider`.
2. Implement the required `generate_response` and `get_default_model` methods.
3. Import the new class in `llm/providers/__init__.py`.
4. Add the new provider's name and class to the `_PROVIDER_REGISTRY` dictionary.
5. The provider will automatically appear in the dashboard.

### 66. The Road to v4: Potential Future Directions
The v3 platform provides a foundation for future enhancements:
- **Self-Play Fine-Tuning:** Use the JSON logs from successful games as a dataset to fine-tune a specialized Snake-playing model.
- **RLHF Integration:** Add a "good move" / "bad move" button to collect human feedback.
- **Dynamic Provider Selection:** Implement a router system that dispatches requests based on game state complexity.
- **Curriculum Learning:** Create a system that automatically adjusts difficulty based on agent performance.

### 67. Final Words & Acknowledgements
The journey from v2 to v3 demonstrates the value of disciplined software engineering for research tools. By investing in a modular and observable platform, we have improved our ability to ask and answer questions about the capabilities of large language models.

The comprehensive principles explored throughout this roadmap—from architectural patterns and modular design to **Type Hints**, **DRY**, **Single Source of Truth**, **Property vs. Attribute Design**, and **Continue Mode**—represent the complete foundation that makes v3 not just functional, but truly robust and maintainable.

Together, these principles create a codebase that is not only more reliable than v2, but fundamentally more extensible and maintainable. This work drew inspiration from the broader software engineering community and lessons learned from building AI systems.

As you apply these patterns to your own projects, remember that good software engineering is not about following rules—it's about building systems that **scale**, **adapt**, and **endure**. Happy hacking! 🎮🐍🤖 