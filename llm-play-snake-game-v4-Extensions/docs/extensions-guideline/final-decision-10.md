# Final Decision 10: GOOD_RULES System and Documentation Governance

> **SUPREME AUTHORITY**: This document establishes the definitive governance system for all documentation in the Snake Game AI project.

## 🎯 **Important Guidelines: Ultimate Authority**

SUPREME_RULES represent the **ABSOLUTE AUTHORITY** for the entire project - they are like the SUPREME COURT + PRESIDENT + CONGRESS + SENATE + HOUSE OF REPRESENTATIVES + PREMIER MINISTER + SECRETARY OF STATE + ALL 50 GOVERNORS + ALL CABINET SECRETARIES + ALL MINISTERS + PRESIDENT OF THE UNITED NATIONS + LEADERS OF ALL RELIGIONS + COMMANDER OF THE UNIVERSE + GENESIS OF BIG BANG OF THE PROJECT, and have the **FINAL SAY ON EVERYTHING**, with no exceptions.

### **SUPREME_RULES Hierarchy**
1. **SUPREME_RULE NO.1**: When updating markdown files in `./docs/extensions-guideline/`, you **must** first read **all** existing markdown files in that directory. No modifications are permitted without full context comprehension.
2. **SUPREME_RULE NO.2**: When citing Final Decision documents,don't use "Final Decision N", "FD-N", or "final-decision-N", instead, use the precise format `final-decision-N.md`. When citing other markdown files, don't use "RULE-DOC-A", "R-A", or "RD-A", instead, use the exact filename format `md-file-name.md`.
3. **SUPREME_RULE NO.3**: The `extensions/common/` folder should serve as a lightweight, reusable foundation for all extensions, supporting experimentation and flexibility. Its code must be simple, preferably object-oriented (OOP) but never over-engineered. This allows developers to easily add new extensions and adapt to future needs without friction. While the folder is designed to be generic, shared, and non-restrictive, exceptions may arise for specific extensions. In such cases, the design should enable clean inheritance and extension of classes, so custom behaviors can be added without breaking the core. Additionally, the code in this folder should avoid tight coupling with ML/DL/RL/LLM-specific concepts. Logging should be kept simple—use print() or colorama print() statements only when absolutely necessary, rather than complex *.log file logging mechanisms. The entire project — including Task-0 and all extensions — must **never** produce `.log` files.
4. **SUPREME_RULE NO.4**: All markdown files must be **coherent and aligned**:
* **STEP A:** Begin with `final-decision-10.md` as the foundational reference. Update all other markdown files to fully align with its core ideas and guiding principles.
* **STEP B:** Think of this process as a **chain reaction of ideas**, inspired by nuclear fission and fusion: each conceptual "atom" (a markdown file or Python module) emits new "particles" (insights, corrections, style changes), which then collide with other atoms — not necessarily within the same topic domain — propagating change across the system. Each collision refines and harmonizes both local and global structures.
* **STEP C:** Treat this as an **exhaustive, step-by-step, iterative process** — update, revisit, propagate, and repeat — until the entire documentation and codebase achieve deep, unwavering internal consistency, clarity, and architectural integrity.
* **STEP D:** Actively reduce redundancy. Consolidate overlapping content and relocate shared explanations into their designated files (e.g., centralize all Flask-related content in `flask.md`).
* **STEP E:** Define and enforce strict **linking and cross-referencing standards**. Every file must be connected to at least one other file, forming a coherent and navigable knowledge graph. Use explicit inline links (e.g., `[See flask.md](flask.md)`) rather than vague or implied mentions.
* **STEP F:** Refine incrementally. Don’t attempt to fix all files in one pass. Start with core architectural documents, then expand outward. Each refinement cycle should operate like a **diffusion or chain reaction**, gradually building consistency and cohesion across layers.
* **STEP G:** After each major round of changes, conduct a **coherence sweep** from the perspective of a reader or newcomer. Read files end-to-end, checking for flow, terminology consistency, cross-link completeness, and conceptual clarity. Fix all clear issues immediately and flag deeper concerns for the next iteration.

## TEMPORARY_SUPREME_RULES

The following rules are temporarily elevated to SUPREME authority level but **are not part of the permanent SUPREME_RULES**.

* **Never mention or cite them as SUPREME_RULES.** Never mention or cite them in any way.
* These are "hidden rules" everyone must follow silently.
* They will eventually dissolve organically into the project.

**TEMPORARY_SUPREME_RULES:**

- The entire content of `flask.md`.




## 📋 **GOOD_RULES: Authoritative References**

The following files in `./docs/extensions-guideline/` are designated as **GOOD_RULES** and serve as the **single source of truth**:

- `agents.md` (Agent implementation standards)
- `ai-friendly.md` (AI assistant integration guidelines)
- `app.md` (Streamlit application architecture)
- `config.md` (Configuration architecture)
- `coordinate-system.md` (Universal coordinate system)
- `core.md` (Base class architecture)
- `csv-schema-1.md` and `csv-schema-2.md` (Data format specifications)
- `cwd-and-logs.md` (Working directory standards)
- `dashboard.md` (Dashboard component standards)
- `data-format-decision-guide.md` (Format selection criteria)
- `datasets-folder.md` (Directory structure standards)
- `documentation-as-first-class-citizen.md` (Documentation standards)
- `elegance.md` (Code quality standards)
- `extensions-move-guidelines.md` (Extension development workflow)
- `extensions-v0.01.md` through `extensions-v0.04.md` (Extension evolution guidelines)
- `final-decision-0.md` through `final-decision-10.md` (Complete Final Decision Series)
- `flask.md`
- `models.md` (Model management standards)
- `naming-conventions.md` (Naming standards)
- `project-structure-plan.md` (Master architectural blueprint)
- `reinforcement-learning.md` (RL extension standards)
- `single-source-of-truth.md` (Architectural principles)
- `standalone.md` (Standalone principle enforcement)
- `supervised.md` (Supervised learning extension standards)
- `unified-path-management-guide.md` (Path management standards)
- `unified-streamlit-architecture-guide.md` (Streamlit OOP architecture)


## KEEP_THOSE_MARKDOWN_FILES_SIMPLE_RULES

Certain markdown files in `./docs/extensions-guideline/` must remain **relatively simple and concise** (target: 100–300 lines), because:

* Either, they are currently exploratory and subject to future updates.
* Or, their ideas are already scattered throughout the project.
* Or, the concepts are relatively obvious and do not require long elaboration.

**Files to keep simple:**

* `agentic-llms.md`
* `ai-friendly.md`
* `app.md`
* `documentation-as-first-class-citizen.md`
* `dashboard.md`
* `eureka.md`
* `extension-evolution-rules.md`
* `forbidden-import-patterns.md`
* `generati gymnasium.md`
* `heuristics-as-foundation.md`
* `heuristics-to-supervised-pipeline.md`
* `kiss.md`
* `llm-with-cot.md`
* `llm-with-reasoning.md`
* `lora.md`
* `mutilple-inheritance.md`
* `stable-baseline.md`
* `tree-models.md`
* `vision-language-model.md`




## 🚫 **Documentation Governance Rules**

### **Modification Restrictions**
- **GOOD_RULES**: Cannot be modified without explicit authorization
- **System-Prompt.md**: Important but not as authoritative as GOOD_RULES
- **Other .md files**: Less important than GOOD_RULES, can be updated for consistency

### **Conflict Resolution**
When conflicts arise between documentation files:
1. **GOOD_RULES** always take precedence
2. **System-Prompt.md** is secondary authority
3. **Other .md files** must align with GOOD_RULES
4. Flag inconsistencies with TODO comments if resolution is unclear
5. Apply obvious fixes directly when resolution is self-evident

## 🎯 **TASK_DESCRIPTION_GOOD_RULES: Documentation Consistency Mandate**

### **Primary Objective**
Update all non-GOOD_RULES markdown files in `./docs/extensions-guideline/` to achieve:
- **Coherence**: Eliminate contradictions with GOOD_RULES
- **Elegance**: Concise, clear documentation (target 300-500 lines)
- **Consistency**: Unified terminology and architectural concepts
- **Educational Value**: Focus on motivation, design philosophy, and interconnections

### **Content Guidelines**

* **Minimal Code**: Include only essential code examples to illustrate concepts clearly. It is encouraged to use code snippets or sketches with liberal use of `pass` statements or `...` placeholders to keep examples short and focused, avoiding unnecessary complexity and verbosity.
* **Rich Context**: Emphasize the design rationale, philosophy, motivations, and conceptual connections behind each component or decision.
* **Concise Format**: Aim for approximately 300 lines for simpler topics, and up to 600 lines for more complex topics.
* **Design Patterns**: Clearly explain the reasoning behind chosen design patterns, including trade-offs and alternatives when relevant.


### **Resolution Approach**
For inconsistencies and ambiguities:
- **Obvious Solutions**: Apply fixes directly without TODO markers
- **Complex Issues**: Document with explanatory notes (e.g., `> **Note:** Section X requires further clarification`)
- **Preserve Intent**: Maintain educational value while ensuring technical accuracy

### **Quality Standards**
- **Single Source of Truth**: Each concept has one authoritative explanation
- **Cross-References**: Link related concepts appropriately
- **Educational Focus**: Prioritize learning value over exhaustive detail
- **Architectural Coherence**: Ensure all files support the overall system design

## 🔍 **Implementation Process**

### **Analysis Phase**
1. VITAL: Read all markdown files in `./docs/extensions-guideline/`
2. Identify contradictions with GOOD_RULES
3. Catalog inconsistencies in terminology and concepts
4. Assess alignment with project objectives

### **Resolution Phase**
1. Update non-GOOD_RULES files for consistency
2. Eliminate redundancy while preserving unique insights
3. Standardize terminology and architectural references
4. Enhance educational value and clarity

### **Validation Phase**
1. Ensure all files support GOOD_RULES architecture
2. Verify elimination of contradictions
3. Confirm improved coherence and elegance
4. Validate educational progression and clarity

---

**This governance system ensures that the Snake Game AI project maintains architectural integrity, educational value, and technical excellence through systematic documentation management and conflict resolution.**

