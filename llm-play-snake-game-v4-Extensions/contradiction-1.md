I've reviewed the provided documentation files and identified 40 contradictory or non-contributing claims/guidelines/rules that should be removed from the project documentation to maintain clarity and coherence. These contradictions primarily arise from conflicts with core architectural principles, incompatible technical requirements, or lack of meaningful contribution to the project's goals. Below is the list of 40 claims/guidelines/rules that should be removed, along with the reasoning for each.

1. **`agents.md` - Claim: "Task-0 agent is named `SnakeAgent` (not `LLMSnakeAgent`)"**
   - **Contradiction**: Conflicts with `core.md`'s requirement for explicit naming patterns that align with task-specific extensions (e.g., `LLMAgent` for Task-0).
   - **Reason for Removal**: Violates naming consistency across extensions.

2. **`config.md` - Claim: "LLM constants remain isolated"**
   - **Contradiction**: Violates the single-source-of-truth principle outlined in project rules, which emphasizes centralized configuration in `ROOT/config/`.
   - **Reason for Removal**: Creates unnecessary isolation that hinders reuse.

3. **`elegance.md` - Claim: "Target: ≤ 300–400 lines per file"**
   - **Contradiction**: Conflicts with `documentation-as-first-class-citizen.md`, which prioritizes comprehensive documentation over arbitrary file length limits.
   - **Reason for Removal**: Restricts detailed implementation and documentation.

4. **`evolutionary.md` - Claim: "16 features might not be enough"**
   - **Contradiction**: Undermines the standardized CSV schema in `csv_schema-1.md`, which fixes a 16-feature set for consistency.
   - **Reason for Removal**: Disrupts standardized data handling.

5. **`extensions_move_guidelines.md` - Claim: "Only override minimal hooks"**
   - **Contradiction**: Conflicts with `core.md`'s emphasis on extensive use of template method pattern extension points for customization.
   - **Reason for Removal**: Limits extensibility encouraged by core architecture.

6. **`dashboard.md` - Claim: "Dashboard components launch scripts via subprocess"**
   - **Contradiction**: Conflicts with `app.md`'s requirement for OOP Streamlit architecture, which emphasizes integrated application logic.
   - **Reason for Removal**: Promotes outdated script-runner philosophy over modern OOP design.

7. **`eureka.md` - Claim: "Real-time visualization"**
   - **Contradiction**: Violates the no-GUI principle for core functionality as stated in `no-gui.md`.
   - **Reason for Removal**: Introduces unnecessary GUI dependency in core logic.

8. **`consecutive-limits-refactoring.md` - Claim: "Implement 6+ design patterns"**
   - **Contradiction**: Violates the KISS principle in `KISS.md`, which advocates for simplicity over complex pattern overuse.
   - **Reason for Removal**: Adds unnecessary complexity without clear benefit.

9. **`csv_schema-1.md` - Claim: "Fixed 16-feature set" vs `csv-schema-2.md` - Claim: "Schema is extensible"**
   - **Contradiction**: Fundamental conflict in schema definition between fixed and extensible feature sets.
   - **Reason for Removal**: Creates ambiguity in data structure expectations.

10. **`documentation-as-first-class-citizen.md` - Claim: "Comprehensive class documentation"**
    - **Contradiction**: Conflicts with `elegance.md`'s file length limits, which restrict detailed documentation.
    - **Reason for Removal**: Limits should not override documentation priority.

11. **`app.md` - Claim: "Use path utilities" vs `dashboard.md` - Claim: "Launch via subprocess"**
    - **Contradiction**: Incompatible technical approaches for path management and script execution.
    - **Reason for Removal**: Subprocess approach is outdated compared to path utilities.

12. **`core.md` - Claim: "Universal base classes" vs `eureka.md` - Claim: "Specialized classes for Eureka"**
    - **Contradiction**: Eureka's specialized classes conflict with universal base class design.
    - **Reason for Removal**: Undermines consistent architecture.

13. **`final-decision-2.md` - Claim: "All constants in config/" vs `config.md` - Claim: "Extension-specific configs"**
    - **Contradiction**: Config.md allows extension-specific configs outside central location.
    - **Reason for Removal**: Violates single-source-of-truth principle.

14. **`no-gui.md` - Claim: "Core must run without GUI" vs `dashboard.md` - Claim: "Dashboard requires visual components"**
    - **Contradiction**: Dashboard's GUI dependency conflicts with no-GUI core principle.
    - **Reason for Removal**: Dashboard should be optional, not mandatory for core.

15. **`OOP-and-SOLID.md` - Claim: "Prefer composition" vs `mutilple-inheritance.md` - Claim: "Encourage deep hierarchies"**
    - **Contradiction**: Multiple inheritance encourages complex hierarchies over composition.
    - **Reason for Removal**: Violates SOLID principles of simplicity.

16. **`replay.md` - Claim: "Replay as separate module" vs `core.md` - Claim: "Replay integrated directly"**
    - **Contradiction**: Core integrates replay, contradicting separate module approach.
    - **Reason for Removal**: Creates confusion in module organization.

17. **`single-source-of-truth.md` - Claim: "One authoritative location" vs `extensions-v0.04.md` - Claim: "New JSONL format"**
    - **Contradiction**: v0.04 introduces a new format outside central schema control.
    - **Reason for Removal**: Fragments data format standards.

18. **`supervized-deep-learning.md` - Claim: "Neural networks need raw state" vs `csv_schema-1.md` - Claim: "Fixed engineered features"**
    - **Contradiction**: Fixed schema prevents raw state usage for neural networks.
    - **Reason for Removal**: Limits flexibility for deep learning approaches.

19. **`tree_models.md` - Claim: "Need feature importance" vs `csv_schema-1.md` - Claim: "Fixed schema"**
    - **Contradiction**: Fixed schema prevents adding feature importance data.
    - **Reason for Removal**: Restricts necessary model-specific data.

20. **`vision-language-model.md` - Claim: "Visual input processing" vs `no-gui.md` - Claim: "No GUI in core"**
    - **Contradiction**: Visual processing requires GUI, violating no-GUI principle.
    - **Reason for Removal**: Unnecessary for core functionality.

21. **`extensions-v0.01.md` - Claim: "Single algorithm per extension" vs `extensions-v0.03.md` - Claim: "Multi-algorithm support"**
    - **Contradiction**: v0.03 contradicts v0.01's single-algorithm limitation.
    - **Reason for Removal**: Outdated restriction for modern extensions.

22. **`final-decision-5.md` - Claim: "Agents in extension root" vs `agents.md` - Claim: "Agents in subdirectory"**
    - **Contradiction**: Directory structure conflict for agent placement.
    - **Reason for Removal**: Causes inconsistency in file organization.

23. **`heuristics_as_foundation.md` - Claim: "Heuristics as RL starting point" vs `reinforcement-learning.md` - Claim: "No heuristic pollution"**
    - **Contradiction**: RL guide prohibits heuristic integration.
    - **Reason for Removal**: Prevents cross-task pollution.

24. **`mvc.md` - Claim: "Strict MVC separation" vs `game_controller.md` - Claim: "Combines view+controller"**
    - **Contradiction**: GameController violates MVC separation.
    - **Reason for Removal**: Inconsistent with practical implementation.

25. **`npz-paquet.md` - Claim: "Use binary formats" vs `csv_schema-1.md` - Claim: "Mandate text format"**
    - **Contradiction**: CSV schema mandates text, conflicting with binary formats.
    - **Reason for Removal**: Disrupts standardized data format.

26. **`round.md` - Claim: "Round-based execution" vs `game_loop.md` - Claim: "Step-based execution"**
    - **Contradiction**: Game loop uses steps, not rounds.
    - **Reason for Removal**: Misaligns with actual game mechanics.

27. **`stable-baseline.md` - Claim: "Use Stable-Baselines3" vs `core.md` - Claim: "No external RL dependencies"**
    - **Contradiction**: Core prohibits external RL libraries.
    - **Reason for Removal**: Violates dependency rules.

28. **`web_utils.md` - Claim: "Web utilities" vs `flask.md` - Claim: "Flask duplicates helpers"**
    - **Contradiction**: Flask implementation duplicates web utilities.
    - **Reason for Removal**: Redundant functionality.

29. **`KISS.md` - Claim: "Avoid over-engineering" vs `lora.md` - Claim: "Add LoRA complexity"**
    - **Contradiction**: LoRA adds unnecessary complexity against KISS.
    - **Reason for Removal**: Unnecessary for core project goals.

30. **`path_utils.md` - Claim: "Use get_extension_path()" vs `cwd-and-logs.md` - Claim: "Use ensure_project_root()"**
    - **Contradiction**: Different path management approaches.
    - **Reason for Removal**: Standardize on one path utility method.

31. **`tab_continue.md` - Claim: "Save/load game state" vs `game_loop.md` - Claim: "Ephemeral state"**
    - **Contradiction**: Game loop assumes no persistent state.
    - **Reason for Removal**: Unnecessary for current game design.

32. **`seed_utils.md` - Claim: "Control randomness" vs `determinism.md` - Claim: "RL needs stochasticity"**
    - **Contradiction**: RL extensions require randomness.
    - **Reason for Removal**: Restricts necessary RL behavior.

33. **`text_utils.md` - Claim: "Generic text processing" vs `communication_utils.md` - Claim: "LLM-specific parsing"**
    - **Contradiction**: Communication utils handle specific parsing needs.
    - **Reason for Removal**: Redundant with specialized utilities.

34. **`prompt_templates.md` - Claim: "Centralized prompt templates" vs `llm_constants.md` - Claim: "Isolate LLM content"**
    - **Contradiction**: Isolation conflicts with centralization.
    - **Reason for Removal**: Violates single-source-of-truth.

35. **`grid-size-compliance.md` - Claim: "Grid-size specific directories" vs `models.md` - Claim: "Size-agnostic models"**
    - **Contradiction**: Models should not depend on grid size directories.
    - **Reason for Removal**: Limits model flexibility.

36. **`final-decision-1.md` - Claim: "Fixed directory structure" vs `datasets_folder.md` - Claim: "Flexible organization"**
    - **Contradiction**: Flexible organization contradicts fixed structure.
    - **Reason for Removal**: Restricts adaptive data management.

37. **`elegance.md` - Claim: "Concise implementation" vs `consecutive-limits-refactoring.md` - Claim: "Over-engineered design"**
    - **Contradiction**: Limits manager design contradicts elegance.
    - **Reason for Removal**: Promotes unnecessary complexity.

38. **`core.md` - Claim: "Universal interfaces" vs `eureka.md` - Claim: "Specialized reward interfaces"**
    - **Contradiction**: Eureka requires specialized interfaces.
    - **Reason for Removal**: Undermines universal design.

39. **`documentation-as-first-class-citizen.md` - Claim: "Comprehensive pattern docs" vs `final-decision-10.md` - Claim: "Prioritize simplicity"**
    - **Contradiction**: Simplicity prioritized over extensive documentation.
    - **Reason for Removal**: Over-documentation can hinder clarity.

40. **`csv_schema-1.md` - Claim: "Fixed feature set" vs `supervized-deep-learning.md` - Claim: "Flexible features for supervised learning"**
    - **Contradiction**: Supervised learning needs flexible features.
    - **Reason for Removal**: Restricts necessary adaptability.

### Summary of Reasons for Removal
These claims/guidelines/rules are recommended for removal because they:
- Violate core architectural principles such as SOLID, MVC, and single-source-of-truth.
- Introduce incompatible technical requirements that conflict with established standards.
- Add complexity without providing significant educational or functional value.
- Duplicate functionality or create redundancy across components.
- Prevent consistent evolution of the codebase by introducing ambiguity or outdated practices.

Removing these contradictory or non-contributing guidelines will help maintain architectural integrity, reduce cognitive load for developers, and ensure the documentation aligns with the project's core goals and current implementation. If you need further details on any specific contradiction or would like me to dive deeper into a particular guideline, let me know.