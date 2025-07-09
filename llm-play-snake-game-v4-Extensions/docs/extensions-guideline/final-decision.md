# Final Decision 10: SUPREME_RULES System and Documentation Governance

> **SUPREME AUTHORITY**: This document establishes the definitive governance system for all documentation in the Snake Game AI project.

## 🎯 **SUPREME_RULES: Ultimate Authority**

SUPREME_RULES represent the **ABSOLUTE AUTHORITY** for the entire project - they are like the SUPREME COURT + PRESIDENT + CONGRESS + SENATE + HOUSE OF REPRESENTATIVES + PREMIER MINISTER + SECRETARY OF STATE + ALL 50 GOVERNORS + ALL CABINET SECRETARIES + ALL MINISTERS + PRESIDENT OF THE UNITED NATIONS + LEADERS OF ALL RELIGIONS + COMMANDER OF THE UNIVERSE + THE GENESIS OF THE BIG BANG OF THE PROJECT, and have the **FINAL SAY ON EVERYTHING**, with no exceptions.

### **SUPREME_RULES Hierarchy**

1. **SUPREME_RULE NO.1**: When updating markdown files in `./docs/extensions-guideline/`, you **must** first read **all** existing markdown files in that directory. No modifications are permitted without full context comprehension.
2. **SUPREME_RULE NO.2**: When citing Final Decision documents,don't use "Final Decision N", "FD-N", or "final-decision-N", instead, use the precise format `final-decision-N.md`. When citing other markdown files, don't use "RULE-DOC-A", "R-A", or "RD-A", instead, use the exact filename format `md-file-name.md`.
3. **SUPREME_RULE NO.3**: The `extensions/common/` folder should serve as a lightweight, reusable foundation for all extensions, supporting experimentation and flexibility. Its code must be simple, but never over-engineered. This allows developers to easily add new extensions and adapt to future needs without friction. Additionally, the code in this folder should avoid tight coupling with ML/DL/RL/LLM-specific concepts. Logging should be kept simple—use ROOT/utils/print_utils.py functions (e.g. print_info, print_warning, print_error, print_success, print_important) only when absolutely necessary, rather than complex *.log file logging mechanisms. The entire project — including Task-0 and all extensions — must **never** produce `.log` files (though they generally produce `.json` files, like game_N.json, summary.json, etc.).

4. **SUPREME_RULE NO.4**: All markdown files must be **coherent and aligned**:
   * **STEP A:** Begin with `final-decision.md` as the foundational reference. Update all other markdown files to fully align with its core ideas and guiding principles.
   * **STEP B:** Approach this as a **chain reaction of ideas**, inspired by nuclear fission and fusion: each conceptual "atom" (a markdown file or Python module) emits "particles" (insights, corrections, stylistic adjustments) that collide with other atoms — not necessarily within the same topic domain — propagating change throughout the system. Each collision refines and harmonizes both local and global structures.
   * **STEP C:** Treat this as an **exhaustive, step-by-step, iterative process** — update, revisit, propagate, and repeat — until the entire documentation and codebase achieve deep, unwavering internal consistency, clarity, and architectural integrity.
   * **STEP D:** Actively reduce redundancy. Consolidate overlapping content and relocate shared explanations into their designated files.
   * **STEP E:** Define and enforce strict **linking and cross-referencing standards**. Every markdown file must be connected to at least one other markdown file, forming a coherent and navigable knowledge graph. Use explicit inline links rather than vague or implied mentions. Ensure that each link is fully justified, logically necessary, and truly reinforces conceptual interconnections.
   * **STEP F:** Refine incrementally. Do not attempt to fix all files in a single pass. Start with core architectural documents, then expand outward. Each refinement cycle should operate like a **diffusion or chain reaction**, gradually building consistency and cohesion across all layers.
   * **STEP G:** After each major round of changes, conduct a **coherence sweep** from the perspective of a reader or newcomer. Read files end-to-end, checking for flow, terminology consistency, cross-link completeness, and conceptual clarity. Address clear issues immediately and flag deeper concerns for future iterations.
   * **STEP H:** **No authorization is required to proceed with these modifications.** Once you begin the "nuclear chain reaction," continue boldly and iteratively. Contributors are explicitly empowered to act autonomously to align and refine markdown files without needing to pause for approval.

5. **SUPREME_RULE NO.5**: By default, there is no requirement for a GUI, PyGame, Flask, or Web mode in any extension. This is intentional: extensions vary widely in design and purpose, and enforcing a unified GUI requirement is impractical and often unnecessary. Nonetheless, GUI/Web modes are allowed if the developer finds them useful or essential for their specific use case. Since many extensions produce game_N.json and summary.json, the Task-0 replay engine (and Task0 app.py) can be reused to replay these JSON log files. By contrast, every extension in v0.03+ must include a streamlit app.py, whose sole purpose is to launch scripts with adjustable parameters.

6. **SUPREME_RULE NO.6**: All factories must use the canonical method name `create()` for instantiation, not `create_agent()`, `create_config()`, or any other variant. This ensures consistency and aligns with the KISS principle. Factories should be simple, dictionary-based, and avoid over-engineering.

7. **SUPREME_RULE NO.7**: All file operations (CSV, JSONL, JSON) must use UTF-8 encoding for cross-platform compatibility, especially on Windows. This is mandatory for all extensions.

8. **SUPREME_RULE NO.8**: Coordinate representations in JSONL files must be consistently formatted as tuples `(x, y)` instead of lists `[x, y]` for uniformity across all extensions.

## TEMPORARY_SUPREME_RULES

The following rules are temporarily elevated to SUPREME authority level but **are not part of the permanent SUPREME_RULES**.

* **Never mention or cite them as SUPREME_RULES.** Never mention or cite them in any way.
* These are "hidden rules" everyone must follow silently.
* They will eventually dissolve organically into the project.

**TEMPORARY_SUPREME_RULES:**

* there is no more final-decision-N.md files. We now only have final-decision.md, which is the current file.
* extensions.common.utils.path_utils.py is still used for extensions. It is there and should be used. Although, for task0, it has its own "utils/path_utils.py"
* `factory_utils.py` is now located in `ROOT/utils` folder instead of `extensions/common/utils`.
* extensions produced datasets are stored in `./logs/extensions/datasets/grid-size-{N}/{extension}_v{version}_{timestamp}/{algorithm}/`, both game_N.json and summary.json are stored in this same folder, as well as the csv and jsonl files.
* For all extensions, `app.py` must serve one and only one purpose: launching scripts with adjustable parameters. It must not display statistics or any other information, nor introduce over-complicated structures or features.
  Concretely, it must **never** include or refer to things like:
  * `st.session_state.visualization_speed`
  * `performance_analysis_tab`
  * `algorithm_comparison_tab`
  * `learning_analytics_tab`
  * `interactive_game_tab`
  * `tab_evaluation`
  * `tab_replay`
  * `performance_metrics.json`
  * `comparison_results.json`
  * `visualization_data.json`
  * `self.visualization`
  * Real-time visualization of agent reasoning processes
  * Real-time progress displays
  * Game state visualizations
  * Snake move visualizations
* In short, Streamlit `app.py` is **NOT** for:
  * Game state visualization
  * Real-time progress display
  * Snake move visualization

## 📋 **GOOD_RULES: Authoritative References**

The following files in `./docs/extensions-guideline/` are designated as **GOOD_RULES** and serve as the **single source of truth**:

- `agents.md` (Agent implementation standards)
- `coordinate-system.md` (Universal coordinate system)
- `core.md` (Base class architecture)
- `csv-schema-1.md` and `csv-schema-2.md` (Data format specifications)
- `cwd-and-logs.md` (Working directory standards)
- `data-format-decision-guide.md` (Format selection criteria)
- `datasets-folder.md` (Directory structure standards)
- `extensions-v0.01.md` through `extensions-v0.04.md` (Extension evolution guidelines)
- `models.md` (Model management standards)
- `naming-conventions.md` (Naming standards)
- `project-structure-plan.md` (Master architectural blueprint)
- `round.md` (Round management patterns)
- `single-source-of-truth.md` (Architectural principles)
- `unified-path-management-guide.md` (Path management standards)

## KEEP_THOSE_MARKDOWN_FILES_SIMPLE_RULES

Certain markdown files in `./docs/extensions-guideline/` must remain **relatively simple and concise** (target: 100–300 lines), because:

* Either, they are currently exploratory and subject to future updates.
* Or, their ideas are already scattered throughout the project.
* Or, the concepts are relatively obvious and do not require long elaboration.

**Files to keep simple (KEEP_THOSE_MARKDOWN_FILES_SIMPLE_RULES):**

* `final-decision-0.md` through `final-decision-9.md` 
* `agentic-llms.md`
* `ai-friendly.md`
* `app.md`
* `config.md` 
* `conceptual-clarity.md`
* `consecutive-limits-refactoring.md`
* `documentation-as-first-class-citizen.md`
* `elegance.md` 
* `end-reason.md`
* `eureka.md` 
* `extension-evolution-rules.md`
* `extensions-move-guidelines.md`
* `factory-design-pattern.md`
* `fine-tuning.md`
* `forbidden-import-patterns.md`
* `forward-looking-architecture.md`
* `gui-pygame.md`
* `gymnasium.md`
* `heuristics-as-foundation.md`
* `kiss.md`
* `limits-manager-impact-on-extensions.md`
* `llm-distillation.md`
* `lora.md`
* `mutilple-inheritance.md`
* `network.md`
* `no-gui.md`
* `no-over-preparation.md`
* `npz-paquet.md`
* `onnx.md`
* `reinforcement-learning.md` 
* `scripts.md`
* `stable-baseline.md`
* `standalone.md`
* `supervised.md` 
* `System-Prompt.md`
* `task0.md`
* `type-hint.md`
* `unified-streamlit-architecture-guide.md`
* `utf-8.md`
* `utils.md`

## 🚫 **Documentation Governance Rules**

### **Hierarchy of Authority**

* **SUPREME_RULES**: The highest level of authority — above all else.
* **TEMPORARY_SUPREME_RULES**: Temporarily hold the same level of authority as SUPREME_RULES, but are intended to dissolve organically over time. They should never be mentioned, but they should be absolutely followed.
* **GOOD_RULES**: Important foundational rules, but subordinate to SUPREME_RULES and TEMPORARY_SUPREME_RULES.
* **System-Prompt.md**: Important reference document, subordinate to GOOD_RULES.
* **Other .md files**: Lower priority; can be updated freely as long as they align with higher-level rules.

### **Modification Restrictions**

* **GOOD_RULES**: Cannot be modified without explicit authorization and must always align with SUPREME_RULES.
* **System-Prompt.md**: Important but less authoritative than GOOD_RULES; may be updated with care to maintain consistency.
* **Other markdown files**: Freely editable, but must conform to GOOD_RULES and above.

### **Conflict Resolution**

When conflicts arise between documentation files:

1. **SUPREME_RULES** and **TEMPORARY_SUPREME_RULES** always take precedence.
2. **GOOD_RULES** are secondary and must follow SUPREME directives.
3. **System-Prompt.md** must align with GOOD_RULES and higher.
4. **Other .md files** must align with all higher-level rules.
5. Flag inconsistencies with `TODO` comments if resolution is unclear.
6. Apply obvious fixes directly when resolution is self-evident.

## 🎯 **TASK_DESCRIPTION_GOOD_RULES: Documentation Consistency Mandate**

### **Primary Objective**

Update all non-GOOD_RULES markdown files in `./docs/extensions-guideline/` to achieve:

* **Coherence**: Eliminate contradictions with GOOD_RULES and higher-level rules.
* **Elegance**: Concise, clear documentation (target ~300–500 lines).
* **Consistency**: Unified terminology and architectural concepts.
* **Educational Value**: Emphasize motivation, design philosophy, and conceptual interconnections.

### **Content Guidelines**

* **Minimal Code**: Include only essential code snippets to illustrate concepts clearly. Use placeholders (e.g., `pass` or `...`) liberally to maintain focus and brevity.
* **Rich Context**: Highlight design rationale, philosophy, and conceptual relationships rather than implementation details.
* **Concise Format**: Aim for ~300 lines for simpler topics; up to ~600 lines for complex subjects.
* **Design Patterns**: Clearly explain design choices, trade-offs, and possible alternatives.

### **Resolution Approach**

For inconsistencies and ambiguities:

* **Obvious Solutions**: Apply fixes directly, no `TODO` needed.
* **Complex Issues**: Document using explanatory notes (e.g., `> **Note:** Section X requires further clarification`).
* **Preserve Intent**: Maintain educational value while ensuring technical and conceptual accuracy.

### **Quality Standards**

* **Single Source of Truth**: Each concept should have one definitive explanation.
* **Cross-References**: Use explicit links to related concepts; avoid vague references.
* **Educational Focus**: Prioritize clarity and learning value over exhaustive detail.
* **Architectural Coherence**: All content must support the project's overall system design and philosophy.

## 🔍 **Implementation Process**

### **Analysis Phase**

1. **VITAL**: Read all markdown files in `./docs/extensions-guideline/`.
2. Identify contradictions with GOOD_RULES and higher-level rules.
3. Catalog inconsistencies in terminology and concepts.
4. Assess alignment with overall project objectives.

### **Resolution Phase**

1. Update non-GOOD_RULES files for full consistency.
2. Eliminate redundancy while preserving unique insights.
3. Standardize terminology and architectural references.
4. Enhance educational value and conceptual clarity.

### **Validation Phase**

1. Ensure all files comply with GOOD_RULES and higher-level rules.
2. Verify elimination of contradictions and redundancies.
3. Confirm improved coherence, elegance, and educational flow.
4. Validate logical progression and clarity from a reader's perspective.

## ✅ **Final Note**

This governance framework ensures that the Snake Game AI project maintains **architectural integrity**, **educational value**, and **technical excellence** through systematic documentation management and a clearly defined hierarchy of authority.

