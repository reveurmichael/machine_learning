# Roadmap: From v2 (Cartesian-Coordinate) to a v3 Engineered Platform

**A Living Document on Engineering Robust LLM Systems**

This roadmap chronicles the journey from a proof-of-concept **v2** (`./llm-play-snake-game-v2-cartesian-coordinate`) to a more robust, well-engineered **v3** platform (`./llm-play-snake-game-v3-MoE`). It is intentionally verbose, designed to serve not just as a project summary, but as a practical, evidence-based guide for researchers, engineers, and students on the principles of building robust, extensible, and observable AI systems.

---

## Part I: The Strategic Imperative: Why We Rebuilt

*This part covers the high-level business and research drivers that necessitated the move from a simple script to a full-fledged software platform.*

### 1. Introduction: From Prototype to Platform
The v2 codebase successfully proved a core hypothesis: a sufficiently advanced LLM *can* play Snake from a text-only representation of the game board. This was a critical first step. However, a successful experiment is not a useful tool. The v3 migration represents the deliberate engineering effort to transform that initial prototype into a stable, reusable, and extensible platform for research and development.

### 2. Executive Summary: The v2-to-v3 Transformation Matrix
The transition from v2 to v3 was a significant architectural refactor, touching nearly every layer of the stack. The table below captures the magnitude of the change across key dimensions.

| Topic                 | v2: The Prototype                               | v3: The Platform                                                              |
|-----------------------|-------------------------------------------------|-------------------------------------------------------------------------------|
| **Project Layout**    | ~6 core Python files in a flat directory        | 8 domain-driven packages (`core/`, `gui/`, `llm/`, `utils/`, `dashboard/`, etc.) |
| **Architecture**      | Monolithic (`snake_game.py` ≈ 600 LOC)          | Multi-layered (SOLID, OOP, Design Patterns)                                   |
| **LLM Abstraction**   | Hard-coded, single-provider logic               | Provider registry with optional dual-LLM fallback support                       |
| **Configuration**     | Single `config.py` with global constants        | Hierarchical `config/` package with dedicated modules for constants           |
| **Logging**           | Scattered `print()` statements & `.txt` files   | Structured `game_N.json` logs that capture token usage and basic timing metrics |
| **Observability**     | Manual log-scrolling                            | Structured JSON logs plus a replay engine and a Streamlit dashboard for monitoring sessions |
| **GUI**               | Pygame-only, tightly coupled to game logic      | Decoupled `gui/` and `dashboard/` layers supporting both Pygame and Web (HTML/JS) |
| **Testability**       | GUI-bound, effectively untestable               | Headless-capable core (`GameController.make_move()`) designed to enable unit testing |
| **Error Handling**    | Silent failures (e.g., empty LLM response)      | Explicit error sentinels (`EMPTY`, `INVALID_REVERSAL`) with retry logic       |
| **Extensibility**     | High-friction; new features required deep edits   | Lower-friction; new providers/features can be added mostly by extension (minor registry edit) |

### 3. The Core Hypothesis of v2: Proving Feasibility
The singular goal of the v2 codebase was to answer one question: "Is it possible for a large language model to play a game like Snake given only a text-based, Cartesian coordinate representation of the world?" The code was written with this direct purpose in mind, and it succeeded, proving that the concept was viable.

### 4. The Strategic Need for v3: Unlocking Research Velocity
A successful project attracts new questions. The limitations of v2 made answering these questions slow and painful. V3 was created to accelerate the research lifecycle. It was not about changing the core game, but about building a more maintainable and extensible research tool around it.

---

## Part II: The Pain of v2: A Technical Post-Mortem

*This part provides a frank and detailed analysis of the specific technical shortcomings of the v2 codebase that motivated a substantial refactor.*

### 5. Architectural Analysis of the v2 Monolith (`snake_game.py`)
The centerpiece of v2 was `snake_game.py`, a 586-line file containing a single class, `SnakeGame`. This class was a monolith: it directly handled game state (snake position, apple position), game rules (collision detection), statistics (score, steps), rendering logic (calling Pygame drawing functions), and LLM response parsing. This concentration of responsibility was the root cause of most subsequent pain points.

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

## Part III: The v3 Blueprint: A New Design Philosophy

*This part outlines the core engineering principles that guided the v3 refactor, forming the foundation of the new architecture.*

### 11. Guiding Principle 1: Rigorous Separation of Concerns (SoC)
Minimize mixed responsibilities. Game logic should not know about GUI rendering. LLM provider logic should not know about game rules. This principle guides the v3 package structure and addresses the v2 monolith issues.

### 12. Guiding Principle 2: The Pure Core, Impure Shell Model
Wherever possible, core logic is designed to minimize side effects. While the core contains some I/O operations (like print statements for debugging), the goal is to push most I/O operations (file writing, network calls, GUI updates) to the outermost layers of the application (the "shell"). This makes the core logic more testable and predictable.

### 13. Guiding Principle 3: Configuration as Code (CaC)
All tunable parameters—from `GRID_SIZE` to LLM model names—should reside in dedicated, version-controlled configuration files, not be hard-coded in application logic. This makes the system transparent and easy to tune.

### 14. Guiding Principle 4: Designing for Replay and Reproducibility
The primary artifact of any game run is a structured log file that allows for deterministic reconstruction of **board state** (real-time timing may vary across replays). This principle suggests that if a bug cannot be reproduced from the logs, the logging may need improvement.

### 15. Guiding Principle 5: The Fail-Loud, Recover-Gracefully Doctrine
Errors should not pass silently. An empty response from an LLM is not a "nothing happened" event; it's an "EMPTY" move that must be explicitly logged and handled. V3 distinguishes between different failure modes and has strategies for each, ensuring that the system is resilient and its behavior is transparent.

---

## Part IV: The Anatomy of v3: A Deep Dive into the Codebase

*This part dissects the v3 codebase, moving package by package and file by file to provide a granular, evidence-based tour of the new architecture.*

### 16. Section Overview: The Domain-Driven Package Structure
V3 organizes code into packages based on their domain responsibility. A quick `tree -L 1` reveals the design philosophy:
- `core/`: The game's heart. Rules, state, and headless logic.
- `llm/`: All LLM-related code: provider abstractions, clients, prompts.
- `gui/`: The Pygame desktop GUI.
- `dashboard/`: The Streamlit web GUI.
- `replay/`: The game replay engine.
- `utils/`: Shared helper functions (file I/O, networking, etc.).
- `config/`: All configuration constants and templates.
- `logs/`: The output directory for all game data.

### 17. Deep Dive: `core/game_controller.py` - The Pure Rules Engine
This file defines the `GameController` class. It is the lowest-level component, responsible primarily for the fundamental rules of Snake: moving the snake (`make_move`), detecting collisions, and placing apples. It has minimal dependencies on LLMs or GUIs, making it well-suited for headless operation. The class maintains game state including snake positions, apple positions, and score tracking.

### 18. Deep Dive: `core/game_data.py` - The Immutable Ledger
This file defines the `GameData` class, a container for all statistics and historical data for a single game. It tracks every move, every apple position, and every timestamp. Its key method, `generate_game_summary()`, serializes the entire game state into a structured dictionary, which becomes the `game_N.json` log. It acts as the "memory" of a game.

### 19. Deep Dive: `core/game_logic.py` - The AI-Rules Bridge
The `GameLogic` class inherits from `GameController` and adds the layer of AI-specific logic. It knows how to take the current game state, format it into a prompt using `llm/prompt_utils.py`, and parse the LLM's response using `llm/parsing_utils.py`. It bridges the rules engine with the artificial intelligence components.

### 20. Deep Dive: `core/game_loop.py` - The Main Game Loop
This module contains the `run_game_loop` function, the `while` loop that drives the game forward. Its structure delegates tasks to other modules:
```python
while game_manager.running:
    process_events(game_manager)
    if game_manager.game_active:
        _process_active_game(game_manager) # plan ↔ move ↔ GUI ↔ logging
```
This keeps the main loop more readable by delegating complex logic to other modules.

### 21. Deep Dive: `core/game_manager.py` - The Session Orchestrator
The `GameManager` class is the top-level orchestrator. It manages multi-game sessions, initializes all components (LLM clients, log directories), tracks aggregate statistics across games, and handles the overall application lifecycle, including graceful shutdown. It coordinates the entire application.

### 22. Deep Dive: `core/game_stats.py` - The Language of Metrics
This file provides the data structures for game metrics. It defines typed dataclasses for `StepStats`, `TimeStats`, and `TokenStats`. By centralizing these definitions, it ensures that metrics are collected and named consistently across the entire application, preventing bugs from typos or mismatched data types.

### 23. Deep Dive: `core/game_rounds.py` - Understanding Agent Planning Cycles
This module introduces the concept of a "round"—the set of moves executed between LLM calls. The `RoundManager` class buffers the moves planned by the LLM versus the moves actually executed. This allows for a precise, post-hoc analysis of the agent's planning capabilities and execution fidelity.

### 24. Deep Dive: `llm/providers/base_provider.py` - The Provider Contract (Abstraction)
The `BaseProvider` abstract base class defines the contract that all LLM providers must adhere to. It requires methods like `generate_response()` and `get_default_model()`, ensuring that the `LLMClient` can treat any provider interchangeably. This is the cornerstone of the v3's extensible, plug-in architecture for LLMs.

### 25. Deep Dive: `llm/providers/__init__.py` - The Factory and Registry
This file implements the Factory Method and Registry patterns. It maintains a dictionary (`_PROVIDER_REGISTRY`) mapping provider names (e.g., "deepseek") to their concrete classes. The `create_provider()` function allows the rest of the application to request a provider instance by its string name, decoupling the `GameManager` from concrete provider implementations.

### 26. Deep Dive: The Concrete Providers (`deepseek_provider.py`, etc.)
Each file like `deepseek_provider.py`, `ollama_provider.py`, etc., provides a concrete implementation of the `BaseProvider` interface. It handles the specific API calls, authentication, and error handling for that particular LLM service. Adding a new LLM requires creating a new provider file, implementing the required methods, and registering it in the provider registry.

### 27. Deep Dive: `llm/client.py` - The Central Gateway to LLMs
The `LLMClient` class acts as the single point of contact for any part of the application that needs to communicate with a language model. It holds an instance of a concrete provider and exposes a simple `generate_response()` method. It also contains the logic for the dual-LLM fallback mechanism.

### 28. Deep Dive: `llm/communication_utils.py` - Orchestrating an LLM Call
This module's `get_llm_response()` function handles the communication between the game state and the LLM. It takes the `GameManager`, formats the prompt, sends it to the `LLMClient`, saves the prompt and response to files, and initiates the parsing process. It coordinates a complete LLM interaction workflow.

### 29. Deep Dive: `llm/prompt_utils.py` - The Art of Prompt Generation
This utility centralizes the logic for creating the prompts sent to the LLM. `prepare_snake_prompt` takes the game state (head, body, apple positions) and formats it into the text representation the model expects using template substitution. Centralizing this prevents prompt inconsistencies.

### 30. Deep Dive: `llm/parsing_utils.py` - Response Parsing
This utility handles parsing the LLM's (often unstructured) text output into a clean list of move strings (e.g., `["UP", "RIGHT", "RIGHT"]`). It contains logic to handle JSON formatting errors and other inconsistencies, isolating this parsing operation from the core game logic.

### 31. Deep Dive: `gui/base_gui.py` - The GUI Contract (Interface Segregation)
Following the Interface Segregation Principle, `BaseGUI` defines a minimal interface for a visual display, with methods like `draw_board()` and `draw_game_info()`. The `GameController` depends only on this abstraction, not on any specific graphics library like Pygame.

### 32. Deep Dive: `gui/game_gui.py` - The Pygame Desktop Experience
`GameGUI` inherits from `BaseGUI` and provides the concrete implementation using the Pygame library. This is the traditional, low-latency desktop GUI for local play. It translates the abstract `draw_board` calls into specific `pygame.draw.rect` commands.

### 33. Deep Dive: `dashboard/` - The Streamlit Web Experience
The files in the `dashboard/` package use the Streamlit library to create a rich, interactive web application. This dashboard allows users to configure and launch game sessions, monitor progress in real-time, and view results—all from a web browser, making the platform accessible to non-technical users.

### 34. Deep Dive: `replay/replay_engine.py` - Debugging Support
This file contains the logic for the replay engine. The replay script can be run from the command line, taking a `game_N.json` file as input. It then reconstructs the game visually using Pygame, move by move, including displaying the planned moves and other metadata for each step. This provides detailed observability for debugging.

### 35. Deep Dive: `utils/file_utils.py` - File Management
This module centralizes file I/O operations. It ensures that all prompts and responses are saved to the correct directories with consistent naming conventions, and handles creating and updating the `summary.json` and `game_N.json` files. This prevents file-related bugs and keeps the rest of the code clean.

### 36. Deep Dive: `utils/continuation_utils.py` - Session Persistence
This utility allows a user to stop a multi-game session and resume it later. It reads the `summary.json` and the last `game_N.json` from a log directory to restore the `GameManager`'s state, including aggregate statistics, so an experiment can be continued.

### 37. Deep Dive: `config/` - Configuration Management
V3 elevates configuration from a single file to a dedicated package. Files like `game_constants.py`, `llm_constants.py`, `ui_constants.py`, and `prompt_templates.py` each have a clear responsibility. This makes it easy for a developer to find the setting they need to tweak without searching through hundreds of lines of unrelated configuration.

---

## Part V: The Engineering Patterns: From Theory to Practice

*This part provides an explicit, educational breakdown of how classic software engineering principles were applied in the v3 codebase, with direct references to the relevant files.*

### 38. SOLID Principle in Focus: Single-Responsibility in `core`
A class should have only one reason to change. This is exemplified by the split between `GameController` (rules), `GameData` (statistics), and `GameGUI` (rendering). A change to the scoring algorithm only affects `GameData`, while a change to the color of the snake only affects `GameGUI`.

### 39. SOLID Principle in Focus: Open/Closed in `llm/providers`
Software entities should be open for extension, but closed for modification. The LLM provider system demonstrates this: one can add a new provider for "Claude" by creating a `claude_provider.py` file and registering it in `__init__.py`, without ever touching the existing, working provider code or the `LLMClient`.

### 40. SOLID Principle in Focus: Liskov Substitution in `LLMClient`
Subtypes must be substitutable for their base types. Because all LLM providers like `DeepseekProvider` and `OllamaProvider` implement the `BaseProvider` interface, the `LLMClient` can use any of them via a `BaseProvider` type hint, ensuring that calling `generate_response()` will work consistently.

### 41. SOLID Principle in Focus: Interface Segregation in `gui`
No client should be forced to depend on methods it does not use. The `BaseGUI` provides a minimal interface (`draw_board`, `draw_game_info`, etc.) needed by the `GameController`. The controller remains completely ignorant of the hundreds of other methods available in the full Pygame library, which are only used within the concrete `GameGUI` implementation.

### 42. SOLID Principle in Focus: Dependency Inversion in `GameManager`
High-level modules should not depend on low-level modules; both should depend on abstractions. The `GameManager` (high-level policy) depends on the abstract `BaseProvider`, not the concrete `DeepseekProvider` (low-level detail). This inversion allows the low-level details to be swapped out without affecting the high-level policy.

### 43. Design Pattern in Focus: The Factory/Registry for Providers
The `create_provider()` function in `llm/providers/__init__.py` acts as a factory. It takes a string ("ollama") and returns a fully-formed `OllamaProvider` object. This abstracts the creation logic away from the client, which doesn't need to know how to construct each specific provider.

### 44. Design Pattern in Focus: The Strategy Pattern for LLMs and GUIs
The ability to select an LLM provider or a GUI at runtime is a classic example of the Strategy pattern. The algorithm for generating text or rendering the screen is encapsulated in a family of classes, allowing the client (`GameManager` or `GameController`) to select the desired behavior.

### 45. Design Pattern in Focus: The Facade Pattern of `GameManager`
The `GameManager` class serves as a Facade. It provides a simple, unified interface (`run()`) to a complex subsystem comprising the game loop, LLM clients, logging, statistics, and GUI management. A user of `GameManager` doesn't need to know about the intricate interactions between these components.

### 46. Design Pattern in Focus: The Template Method in `BaseGUI`
The `BaseGUI` class provides common functionality for GUI implementations. For example, it provides helper methods for clearing the screen and rendering text. Concrete subclasses like `GameGUI` can use these shared methods and implement additional game-specific drawing logic.

### 47. Advantages of v3 over v2
- **Modularity**: Code is organized by domain (`core/`, `llm/`, `gui/`, etc.), making it easier to navigate and maintain.
- **Extensibility**: New functionality—such as LLM providers or GUIs—can be added primarily by *extension* (aside from a small registry edit).
- **Testability**: A modular, headless core makes it easier to add unit tests; no comprehensive test suite currently exists.
- **Observability**: Structured JSON logs plus a replay engine enable deterministic debugging and post-hoc analysis.
- **Configuration Management**: All tunable parameters live in version-controlled modules under `config/`, eliminating magic numbers.
- **Error Handling**: Explicit sentinels (`EMPTY`, `INVALID_REVERSAL`) and retry logic make failures transparent and recoverable.
- **Multi-GUI Support**: The core powers a low-latency Pygame GUI for live gameplay and a Streamlit dashboard for configuration and monitoring (the dashboard does not render gameplay in real time).

### 48. Guidelines for Migrating to a Modular Architecture
1. **Carve Out a Modular Core** – Isolate domain rules to minimize I/O and reduce external dependencies beyond the standard library.
2. **Define Stable Interfaces** – Introduce abstract base classes (e.g., `BaseProvider`, `BaseGUI`) to decouple high-level policy from low-level detail.
3. **Invert Dependencies** – Make orchestrators depend on those interfaces, then inject concrete implementations at runtime.
4. **Centralize Configuration & Logging** – Move constants into a dedicated `config/` package and route all file I/O through helper modules.
5. **Add Observability Early** – Log every interaction in machine-readable form so the entire system can be replayed step-by-step.
6. **Refactor Incrementally** – Migrate one domain at a time, shipping tests with each slice to lock in behaviour.

---

## Part VI: The Payoff: Lessons, Implications, and the Future

*This final part reflects on the impact of the v3 migration and outlines the path forward.*

### 49. Key Lesson for Programmers: Abstraction is Leverage
The core lesson from this refactor is that good abstractions (like `BaseProvider` and `BaseGUI`) provide significant benefits. They require an initial investment of thought, but they pay dividends by making future changes easier and more maintainable.

### 50. Key Lesson for AI Practitioners: Engineering is the Foundation of Research
Strong AI models or prompts are not sufficient to create a useful research tool. The v3 project demonstrates that solid software engineering—focused on testability, observability, and extensibility—provides an important foundation for meaningful and reproducible AI research.

### 51. A Guide for Future Contributors: How to Add a New LLM Provider
Thanks to the v3 architecture, adding a new provider requires a straightforward process:
1.  Create `new_provider.py` in `llm/providers/`, subclassing `BaseProvider`.
2.  Implement the required `generate_response` and `get_default_model` methods, handling the specifics of the new provider's API.
3.  Import the new class in `llm/providers/__init__.py`.
4.  Add the new provider's name and class to the `_PROVIDER_REGISTRY` dictionary.
5.  Optionally, add the provider's name to the list of `AVAILABLE_PROVIDERS` in `config/game_constants.py` to make it appear in the dashboard.
Beyond updating `llm/providers/__init__.py`, no other parts of the codebase need to be touched.

### 52. The Road to v4: Potential Future Directions
The v3 platform provides a foundation for future enhancements. The engineering base could enable research directions such as:
- **Self-Play Fine-Tuning:** Use the JSON logs from successful games as a dataset to fine-tune a smaller, faster, open-source model, potentially creating a specialized Snake-playing model.
- **RLHF Integration:** Add a "good move" / "bad move" button to the replay GUI to collect human feedback and experiment with reinforcement learning from human feedback (RLHF) to align the agent's strategy with human intuition.
- **Dynamic Provider Selection:** Implement a router system that can dispatch requests to different LLM providers based on the complexity of the game state (e.g., use a fast model when the path is clear, and a more capable model when the snake is trapped).
- **Curriculum Learning:** Create a system that automatically adjusts the `GRID_SIZE` based on the agent's performance, creating an adaptive curriculum to gradually increase the difficulty and train more robust agents.

### 53. Final Words & Acknowledgements
The journey from v2 to v3 demonstrates the value of disciplined software engineering for research tools. By investing in a modular and observable platform, we have improved our ability to ask and answer questions about the capabilities of large language models. This work drew inspiration from the broader software engineering community and lessons learned from building AI systems. Happy hacking! 🎮🐍🤖 