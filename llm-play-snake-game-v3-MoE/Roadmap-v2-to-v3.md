# Roadmap: From v2 (Cartesian-Coordinate) to a v3 Engineered Platform

**A Living Document on Engineering Robust LLM Systems**

This roadmap chronicles the journey from a proof-of-concept **v2** (`./llm-play-snake-game-v2-cartesian-coordinate`) to a more robust, well-engineered **v3** platform (`./llm-play-snake-game-v3-MoE`). It is intentionally verbose, designed to serve not just as a project summary, but as a practical, evidence-based guide for researchers, engineers, and students on the principles of building robust, extensible, and observable AI systems.

---

## Part I: The Strategic Context


### 1. Introduction: From Prototype to Platform
The v2 codebase successfully proved a core hypothesis: a sufficiently advanced LLM *can* play Snake from a text-only representation of the game board. This was a critical first step. However, a successful experiment is not a useful tool. The v3 migration represents the deliberate engineering effort to transform that initial prototype into a stable, reusable, and extensible platform for research and development.

### 2. The Core Hypothesis of v2: Proving Feasibility
The singular goal of the v2 codebase was to answer one question: "Is it possible for a large language model to play a game like Snake given only a text-based, Cartesian coordinate representation of the world?" The code was written with this direct purpose in mind, and it succeeded, proving that the concept was viable.

### 3. The Strategic Need for v3: Unlocking Research Velocity
A successful project attracts new questions. The limitations of v2 made answering these questions slow and painful. V3 was created to accelerate the research lifecycle. It was not about changing the core game, but about building a more maintainable and extensible research tool around it.

### 4. Executive Summary: The v2-to-v3 Transformation Matrix
The transition from v2 to v3 was a significant architectural refactor, touching nearly every layer of the stack. The table below captures the magnitude of the change across key dimensions.

| Topic                 | v2: The Prototype                               | v3: The Platform                                                              |
|-----------------------|-------------------------------------------------|-------------------------------------------------------------------------------|
| **Project Layout**    | 6 core Python files in a flat directory        | 8 domain-driven packages (`core/`, `gui/`, `llm/`, `utils/`, `dashboard/`, `replay/`, `config/`, `web/`) |
| **Architecture**      | Monolithic (`snake_game.py` ≈ 586 LOC)          | Multi-layered (SOLID, OOP, Design Patterns)                                   |
| **LLM Abstraction**   | Hard-coded, single-provider logic               | Provider registry with optional dual-LLM fallback support                       |
| **Configuration**     | Single `config.py` with global constants        | Hierarchical `config/` package with dedicated modules for constants           |
| **Logging**           | Scattered `print()` statements & `.txt` files   | Structured `game_N.json` logs that capture token usage and basic timing metrics |
| **Observability**     | Manual log-scrolling                            | Structured JSON logs plus a replay engine and a Streamlit dashboard for monitoring sessions |
| **GUI**               | Pygame-only, tightly coupled to game logic      | Decoupled `gui/`, `dashboard/`, and `web/` layers supporting Pygame desktop, Streamlit configuration, and Flask web interfaces (live gameplay, human play, replay) |
| **Testability**       | GUI-bound, effectively untestable               | Headless-capable core (`GameController.make_move()`) designed to enable unit testing |
| **Error Handling**    | Silent failures (e.g., empty LLM response)      | Explicit error sentinels (`EMPTY`, `INVALID_REVERSAL`) with retry logic       |
| **Extensibility**     | High-friction; new features required deep edits   | Lower-friction; new providers/features can be added mostly by extension (minor registry edit) |

---

## Part II: The Problems We Solved

*This part provides a frank analysis of v2's limitations and the specific pain points that drove the architectural changes.*

### 5. Architectural Analysis of the v2 Monolith (`snake_game.py`)
The centerpiece of v2 was `snake_game.py`, a 585-line file containing a single class, `SnakeGame`. This class was a monolith: it directly handled game state (snake position, apple position), game rules (collision detection), statistics (score, steps), rendering logic (calling Pygame drawing functions), and LLM response parsing. This concentration of responsibility was the root cause of most subsequent pain points.

### 6. Pain Point 1: The Circular Dependency Trap
Because `SnakeGame` handled rendering, it needed to import `gui.py`. The `gui.py` file had optional conditional imports of `snake_game.py` for standalone testing. While this created some coupling, it was managed through conditional imports rather than direct circular dependencies.

### 7. Pain Point 2: The Untestable Abyss
Unit testing `SnakeGame` was practically impossible. To even instantiate the class, one needed to have a Pygame display initialized. To test a single method like `make_move()`, the entire GUI and LLM client apparatus had to be mocked or instantiated. This high barrier meant that, in practice, no unit tests were written, and all testing was manual and tedious.

### 8. Pain Point 3: The Chaos of Scattered Configuration
In v2, `config.py` held some constants like `GRID_SIZE`. However, many other configuration values were hard-coded directly where they were used. For example, Pygame window dimensions and colors were hard-coded directly in `snake_game.py`, while LLM prompt templates were defined in `config.py` as a large string template. This made tuning the system more difficult than necessary.

### 9. Pain Point 4: The Black Hole of Observability (No Replay)
When a v2 game session ended, the only artifacts were a folder of raw `.txt` files (one for each prompt and response) and a simple `info.txt` with the final score. There was no way to reconstruct the game state at a specific step. Debugging a failure involved manually reading through dozens of text files and trying to mentally simulate the game, which was error-prone and inefficient.

### 10. Pain Point 5: The Friction of Extensibility
The most significant long-term problem was the difficulty of adding new features. Adding a new LLM provider required editing `llm_client.py` and `main.py`. Adding a "continue from last game" feature would have required significant refactoring of state management. The monolithic design meant that new ideas required substantial changes to existing code.

---

## Part III: Design Philosophy and Guiding Principles

*This part establishes the foundational engineering principles that guided the v3 architecture.*

### 11. Guiding Principle 1: Rigorous Separation of Concerns (SoC)
Minimize mixed responsibilities. Game logic should not know about GUI rendering. LLM provider logic should not know about game rules. This principle guides the v3 package structure and addresses the v2 monolith issues.

### 12. Guiding Principle 2: The Pure Core, Impure Shell Model
Wherever possible, core logic is designed to minimize side effects. While the core contains numerous I/O operations (like print statements for debugging and user feedback), the goal is to push most I/O operations (file writing, network calls, GUI updates) to the outermost layers of the application (the "shell"). This makes the core logic more testable and predictable.

### 13. Guiding Principle 3: Configuration as Code (CaC)
All tunable parameters—from `GRID_SIZE` to LLM model names—should reside in dedicated, version-controlled configuration files, not be hard-coded in application logic. This makes the system transparent and easy to tune.

### 14. Guiding Principle 4: Designing for Replay and Reproducibility
The primary artifact of any game run is a structured log file that allows for deterministic reconstruction of **board state** (real-time timing may vary across replays). This principle suggests that if a bug cannot be reproduced from the logs, the logging may need improvement.

### 15. Guiding Principle 5: The Fail-Loud, Recover-Gracefully Doctrine
Errors should not pass silently. An empty response from an LLM is not a "nothing happened" event; it's an "EMPTY" move that must be explicitly logged and handled. V3 distinguishes between different failure modes and has strategies for each, ensuring that the system is resilient and its behavior is transparent.

### 16. The Type System Revolution: From Dynamic Chaos to Static Certainty
One of the most significant but understated improvements in v3 is the comprehensive adoption of **type hints**. While Python is dynamically typed, v3 leverages static type annotations to provide compile-time safety, better IDE support, and enhanced code documentation.

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

#### 16b. V3's Type Fortress: Comprehensive Static Typing
V3 introduces systematic type annotations using modern Python typing features:

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
**Don't Repeat Yourself (DRY)** was a major driver in the v2-to-v3 refactor. V2 suffered from substantial code duplication that made maintenance error-prone and changes expensive.

#### 17a. V2's Repetition Problem: Copy-Paste Programming
**Duplicate JSON Parsing Logic**: V2 had similar JSON parsing scattered across multiple files:

```python
# v2: snake_game.py - JSON parsing logic #1
def parse_llm_response(self, response):
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return parsed.get("moves", [])
        except:
            pass
    return []
```

#### 17b. V3's DRY Implementation: Centralized Utilities
**Unified JSON Processing System**: V3 centralizes all JSON parsing in `utils/json_utils.py`:

```python
# v3: utils/json_utils.py - Single source for JSON logic
def extract_json_from_text(response: str) -> Optional[Dict[str, List[str]]]:
    """Single, robust JSON extraction used throughout the codebase."""
```

### 18. Single Source of Truth: Eliminating Data Redundancy
V3 implements rigorous **Single Source of Truth (SSOT)** patterns to eliminate data inconsistencies and reduce cognitive load on developers.

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
One of v3's most sophisticated features is **Continue Mode** - the ability to resume interrupted game sessions. This represents a masterclass in state management, persistence, and system recovery.

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

### 22. Deep Dive: `core/game_controller.py` - The Pure Rules Engine
This file defines the `GameController` class. It is the lowest-level component, responsible primarily for the fundamental rules of Snake: moving the snake (`make_move`), detecting collisions, and placing apples. It has minimal dependencies on LLMs or GUIs, making it well-suited for headless operation.

### 23. Deep Dive: `core/game_data.py` - The Game Statistics Tracker
This file defines the `GameData` class, a mutable container for all statistics and historical data for a single game. It tracks every move, every apple position, and every timestamp. Its key method, `generate_game_summary()`, serializes the entire game state into a structured dictionary, which becomes the `game_N.json` log.

### 24. Deep Dive: `core/game_logic.py` - The AI-Rules Bridge
The `GameLogic` class inherits from `GameController` and adds the layer of AI-specific logic. It knows how to take the current game state, format it into a prompt using `llm/prompt_utils.py`, and parse the LLM's response using `llm/parsing_utils.py`.

### 25. Deep Dive: `core/game_loop.py` - The Main Game Loop
This module contains the `run_game_loop` function, the `while` loop that drives the game forward:
```python
while game_manager.running:
    process_events(game_manager)
    if game_manager.game_active:
        _process_active_game(game_manager) # plan ↔ move ↔ GUI ↔ logging
```

### 26. Deep Dive: `core/game_manager.py` - The Session Orchestrator
The `GameManager` class is the top-level orchestrator. It manages multi-game sessions, initializes all components (LLM clients, log directories), tracks aggregate statistics across games, and handles the overall application lifecycle.

### 27. Deep Dive: `core/game_stats.py` - Game Metrics
This file provides the data structures for game metrics. It defines typed dataclasses for `StepStats`, `TimeStats`, and `TokenStats`. By centralizing these definitions, it ensures that metrics are collected and named consistently across the entire application.

### 28. Deep Dive: `core/game_rounds.py` - Understanding Agent Planning Cycles
This module introduces the concept of a "round"—the set of moves executed between LLM calls. The `RoundManager` class buffers the moves planned by the LLM versus the moves actually executed.

---

## Part V: LLM Integration - The AI Layer

*This part examines how v3 abstracts and manages AI model interactions through a sophisticated provider system.*

### 29. Deep Dive: `llm/providers/base_provider.py` - The Provider Contract
The `BaseProvider` abstract base class defines the contract that all LLM providers must adhere to. It requires methods like `generate_response()` and `get_default_model()`, ensuring that the `LLMClient` can treat any provider interchangeably.

### 30. Deep Dive: `llm/providers/__init__.py` - The Factory and Registry
This file implements the Factory Method and Registry patterns. It maintains a dictionary (`_PROVIDER_REGISTRY`) mapping provider names (e.g., "deepseek") to their concrete classes. The `create_provider()` function allows the rest of the application to request a provider instance by its string name.

### 31. Deep Dive: The Concrete Providers (`deepseek_provider.py`, etc.)
Each file like `deepseek_provider.py`, `ollama_provider.py`, etc., provides a concrete implementation of the `BaseProvider` interface. It handles the specific API calls, authentication, and error handling for that particular LLM service.

### 32. Deep Dive: `llm/client.py` - The Central Gateway to LLMs
The `LLMClient` class acts as the single point of contact for any part of the application that needs to communicate with a language model. It holds an instance of a concrete provider and exposes a simple `generate_response()` method.

### 33. Deep Dive: `llm/communication_utils.py` - Orchestrating an LLM Call
This module's `get_llm_response()` function handles the communication between the game state and the LLM. It takes the `GameManager`, formats the prompt, sends it to the `LLMClient`, saves the prompt and response to files, and initiates the parsing process.

### 34. Deep Dive: `llm/prompt_utils.py` - Prompt Generation
This utility centralizes the logic for creating the prompts sent to the LLM. `prepare_snake_prompt` takes the game state (head, body, apple positions) and formats it into the text representation the model expects.

### 35. Deep Dive: `llm/parsing_utils.py` - Response Parsing
This utility handles parsing the LLM's (often unstructured) text output into a clean list of move strings (e.g., `["UP", "RIGHT", "RIGHT"]`). It contains logic to handle JSON formatting errors and other inconsistencies.

---

## Part VI: User Interfaces - Multi-Modal Access

*This part covers the various interfaces that make the v3 platform accessible across different environments and use cases.*

### 36. Deep Dive: `gui/base_gui.py` - The GUI Base Class
`BaseGUI` provides a base class with common functionality for GUI implementations, including helper methods for display setup, drawing, and text rendering. The `GameController` uses optional GUI dependency injection.

### 37. Deep Dive: `gui/game_gui.py` - The Pygame Desktop Experience
`GameGUI` inherits from `BaseGUI` and provides the concrete implementation using the Pygame library. This is the traditional, low-latency desktop GUI for local play.

### 38. Deep Dive: `dashboard/` - The Streamlit Web Experience
The files in the `dashboard/` package use the Streamlit library to create a rich, interactive web application. This dashboard allows users to configure and launch game sessions, monitor progress in real-time, and view results.

### 39. Flask-Based Interactive Web Interfaces
V3 provides **three** distinct Flask-based web applications:

**Live LLM Mode (`main_web.py`)** – A real-time web interface for watching LLM-controlled gameplay.

**Human Play Mode (`human_play_web.py`)** – A browser-based version of Snake for human players.

**Web Replay (`replay_web.py`)** – A replay viewer that reconstructs any `game_N.json` file in the browser.

### 40. Deep Dive: `replay/replay_engine.py` - Debugging Support
This file contains the logic for the replay engine. The replay script can be run from the command line, taking a `game_N.json` file as input and reconstructing the game visually using Pygame, move by move.

---

## Part VII: Supporting Infrastructure

*This part examines the utility systems, configuration management, and supporting infrastructure.*

### 41. Deep Dive: `utils/file_utils.py` - File Management
This module centralizes file I/O operations. It ensures that all prompts and responses are saved to the correct directories with consistent naming conventions.

### 42. Deep Dive: `utils/continuation_utils.py` - Session Persistence
This utility allows a user to stop a multi-game session and resume it later. It reads the `summary.json` and the last `game_N.json` from a log directory to restore the `GameManager`'s state.

### 43. Deep Dive: `utils/web_utils.py` - Shared Web Infrastructure
This module centralizes functionality required by all Flask apps, including color mapping for consistent visuals and game-state serialization utilities for JSON APIs.

### 44. Deep Dive: `utils/network_utils.py` - Network Port Management
Provides utilities for discovering free TCP ports so multiple web services can run concurrently without conflicts.

### 45. Deep Dive: `config/` - Configuration Management
V3 elevates configuration from a single file to a dedicated package. Files like `game_constants.py`, `llm_constants.py`, `ui_constants.py`, and `prompt_templates.py` each have a clear responsibility.

---

## Part VIII: Engineering Patterns and Design Principles

*This part provides an explicit breakdown of how classic software engineering principles were applied in the v3 codebase.*

### 46. SOLID Principle: Single-Responsibility in `core`
A class should have only one reason to change. This is exemplified by the split between `GameController` (core rules), `GameData` (statistics), and `GameLogic` (AI integration).

### 47. SOLID Principle: Open/Closed in `llm/providers`
Software entities should be open for extension, but closed for modification. The LLM provider system demonstrates this: one can add a new provider without touching existing code.

### 48. SOLID Principle: Liskov Substitution in `LLMClient`
Subtypes must be substitutable for their base types. Because all LLM providers implement the `BaseProvider` interface, the `LLMClient` can use any of them interchangeably.

### 49. SOLID Principle: Interface Segregation in `gui`
No client should be forced to depend on methods it does not use. The `GameController` uses optional GUI dependency injection with a minimal interface.

### 50. SOLID Principle: Dependency Inversion in `GameManager`
High-level modules should not depend on low-level modules; both should depend on abstractions. The `GameManager` depends on abstract `BaseProvider`, not concrete implementations.

### 51. Design Pattern: The Factory/Registry for Providers
The `create_provider()` function acts as a factory. It takes a string ("ollama") and returns a fully-formed `OllamaProvider` object.

### 52. Design Pattern: The Strategy Pattern for LLMs and GUIs
The ability to select an LLM provider or a GUI at runtime is a classic example of the Strategy pattern.

### 53. Design Pattern: The Facade Pattern of `GameManager`
The `GameManager` class serves as a Facade. It provides a simple, unified interface (`run()`) to a complex subsystem.

### 54. Design Pattern: Utility Base Class in `BaseGUI`
The `BaseGUI` class provides common functionality for GUI implementations, promoting code reuse and consistent behavior.

---

## Part IX: Outcomes and Lessons Learned

*This part reflects on the results of the transformation and extracts actionable insights.*

### 55. Advantages of v3 over v2
- **Modularity**: Code is organized by domain, making it easier to navigate and maintain.
- **Extensibility**: New functionality can be added primarily by extension.
- **Testability**: A modular, headless core makes it easier to add unit tests.
- **Observability**: Structured JSON logs plus a replay engine enable deterministic debugging.
- **Configuration Management**: All tunable parameters live in version-controlled modules.
- **Error Handling**: Explicit sentinels and retry logic make failures transparent.
- **Multi-Platform Access**: The core supports desktop, web dashboard, and Flask interfaces.

### 56. Key Lesson for Programmers: Abstraction is Leverage
The core lesson from this refactor is that good abstractions (like `BaseProvider` and `BaseGUI`) provide significant benefits. They require an initial investment of thought, but they pay dividends by making future changes easier.

### 57. Key Lesson for AI Practitioners: Engineering is the Foundation of Research
Strong AI models or prompts are not sufficient to create a useful research tool. The v3 project demonstrates that solid software engineering provides an important foundation for meaningful and reproducible AI research.

### 58. Guidelines for Migrating to a Modular Architecture
1. **Carve Out a Modular Core** – Isolate domain rules to minimize I/O and reduce external dependencies.
2. **Define Stable Interfaces** – Introduce abstract base classes to decouple high-level policy from low-level detail.
3. **Invert Dependencies** – Make orchestrators depend on interfaces, then inject concrete implementations at runtime.
4. **Centralize Configuration & Logging** – Move constants into a dedicated package and route all file I/O through helper modules.
5. **Add Observability Early** – Log every interaction in machine-readable form.
6. **Refactor Incrementally** – Migrate one domain at a time, shipping tests with each slice.

---

## Part X: Future Directions and Practical Guidance

*This part looks forward to potential enhancements and provides practical guidance for contributors.*

### 59. A Guide for Future Contributors: How to Add a New LLM Provider
Thanks to the v3 architecture, adding a new provider requires a straightforward process:
1. Create `new_provider.py` in `llm/providers/`, subclassing `BaseProvider`.
2. Implement the required `generate_response` and `get_default_model` methods.
3. Import the new class in `llm/providers/__init__.py`.
4. Add the new provider's name and class to the `_PROVIDER_REGISTRY` dictionary.
5. The provider will automatically appear in the dashboard.

### 60. The Road to v4: Potential Future Directions
The v3 platform provides a foundation for future enhancements:
- **Self-Play Fine-Tuning:** Use the JSON logs from successful games as a dataset to fine-tune a specialized Snake-playing model.
- **RLHF Integration:** Add a "good move" / "bad move" button to collect human feedback.
- **Dynamic Provider Selection:** Implement a router system that dispatches requests based on game state complexity.
- **Curriculum Learning:** Create a system that automatically adjusts difficulty based on agent performance.

### 61. Final Words & Acknowledgements
The journey from v2 to v3 demonstrates the value of disciplined software engineering for research tools. By investing in a modular and observable platform, we have improved our ability to ask and answer questions about the capabilities of large language models.

The comprehensive principles explored throughout this roadmap—from architectural patterns and modular design to **Type Hints**, **DRY**, **Single Source of Truth**, **Property vs. Attribute Design**, and **Continue Mode**—represent the complete foundation that makes v3 not just functional, but truly robust and maintainable.

Together, these principles create a codebase that is not only more reliable than v2, but fundamentally more extensible and maintainable. This work drew inspiration from the broader software engineering community and lessons learned from building AI systems.

As you apply these patterns to your own projects, remember that good software engineering is not about following rules—it's about building systems that **scale**, **adapt**, and **endure**. Happy hacking! 🎮🐍🤖 