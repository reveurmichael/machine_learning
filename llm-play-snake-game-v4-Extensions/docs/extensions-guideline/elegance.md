# Elegance for Extension folder code

## 🧹 File Length & Organization

* **Keep files focused and concise.** Aim for **≤ 300–400 lines** per Python file. If a file grows beyond this:

  * **Split by responsibility.** Group related functions, classes, or views into separate modules.
  * **Use clear folder names** to convey intent. Common patterns include:

    * `utils/` or `helpers/` — standalone functions and small utilities
    * `services/` or `managers/` — business logic, data‐access layers, long‐running workflows
    * `models/` — data schemas, domain objects, serializers
    * `views/` or `controllers/` — HTTP endpoints, CLI commands, UI components
    * `dashboard/`, `pages/`, `components/` — Streamlit or React pages and widgets
    * `core/` — shared abstractions, base classes, essential infrastructure
    * `tools/` — CLI or developer scripts

* **One concept per file.**
  A good heuristic: if you can describe the file’s purpose in a single sentence, it’s likely well scoped.

* **Maintain explicit module boundaries.**

  * Avoid circular imports by grouping interdependent code in the same module.
  * Re-export a clean API in `__init__.py` when exposing multiple submodules.

* **Name files after their main content.**

  * `path_utils.py` for path‐finding utilities
  * `game_logic.py` for core game rules
  * `dataset_writer.py` for CSV/JSONL export logic

* **Review periodically.**
  When adding new features, revisit existing modules:

  * Does new code belong there or in a new helper file?
  * Can common patterns be abstracted into a shared utility?
  * Really shared functions should go to the "./extensions/common/" folder (still, it should making the extension folder core ideas very very visible). Each extension plus the shared common folder should be regarded as standalone, hence no sharing code among extensions.


Here are several more **elegance guidelines** to complement your file‐length rules—tailored for extensions around Snake AI (LLM, ML, RL, heuristics):

---

## 🎨 Naming & Style

* **Consistent naming**

  * Agents: should be agent_blabla.py, or, simply blabla.py, depending on the scenarios 
  * **Classes**: `PascalCase` (e.g. `BFSAgent`, `GameManager`)
  * **Functions & variables**: `snake_case` (e.g. `compute_path`, `max_steps`)
  * **Constants**: `UPPER_SNAKE_CASE` 

* **PEP8 compliance**

  * Use a linter (e.g. `flake8`, `pylint`) as a pre-commit hook
  * Trim trailing whitespace, enforce 88-character line length (Black defaults), consistent indentation
* **Descriptive names**

  * Avoid one-letter variables (except in very local loops)
  * Name modules by responsibility, e.g. `heuristic_utils.py`, `replay_engine.py`

---

## 📚 Documentation & Typing

* **Docstrings everywhere**

  * **Modules**: short summary at top
  * **Classes**: explain purpose and usage, list important methods
  * **Functions**: describe arguments, return values, side effects, exceptions
  * Follow [Google](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods) or NumPy style consistently
* **Type annotations**

  * Annotate public APIs: class methods, major functions
  * Use `typing` for clarity (`List[Tuple[int,int]]`, `Dict[str, Any]`)
* **README updates**

  * Each extension folder gets a `README.md` (IMPORTANT:for v0.01 it should be short can concise, for v0.02 a bit longer and v0.03 even longer) with high-level description, and example usage

---

## 🧩 Modularity & Dependency Management

* **Clear separation**

  * Core logic, data I/O, training scripts, UI should live in distinct modules
* **Minimal direct imports**

  * Avoid deep import chains—have a single `extensions/common/` for shared utilities


---


## ⚙️ Configuration & CLI

* **Centralized config**

  * Use `config/*.yaml` or `pydantic` settings class, not hard-coded constants
* **User-friendly CLI**

  * Use `argparse` or `click` with clear help text
  * Validate arguments early (e.g. grid size bounds)

---

## No Need for Backward compatibility


We are refactoring with a future proof mindset, to make things look so fresh, so needly shipped. So we are not going to keep backward compatibility, for anything. Nothing is going to be deprecated, if anything is deprecated, it should be removed. No legacy consideration for extensions.
---

By following these principles—together with your file-length and folder guidelines—you’ll create an **elegant**, **maintainable**, and **scalable** extensions ecosystem for your Snake AI platform.


## Logger : IMPORTANT: WE DON'T USE LOGGER YET. AND I DON'T  PLAN TO DO THAT YET.

VITAL: WE DON'T USE LOGGER YET. AND I DON'T  PLAN TO DO THAT YET.

We don't use Logger in this whole project yet. So, no worries. Maybe someday I will do it.

