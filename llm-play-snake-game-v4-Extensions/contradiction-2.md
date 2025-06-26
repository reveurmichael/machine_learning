# CONTRADICTION-2: Inconsistencies in docs/extensions-guideline

This document captures the most obvious contradictions and formatting errors in the Markdown files under `docs/extensions-guideline`, along with proposed solutions to restore coherence and maintain the single source of truth.

## 1. Path Reference Mismatch
- **Issue:** `final-decision-10.md` (SUPREME_RULES NO.1) refers to `./docs/extensions-guideline/` (with an extra "s"). The actual folder is `docs/extensions-guideline/`.
- **Solution:** Amend all references to use the correct path `docs/extensions-guideline/`.

## 2. GOOD_RULES Filename Typos and Formatting
- **Issue:** In the bullet list inside `final-decision-10.md`:
  - `heuristicsto_supervised_pipeline.md` does not match `heuristics-to-supervised-pipeline.md`.
  - `csv_schema-2.md` should be `csv-schema-2.md`.
  - The entries `elegance.md`, `extensions-v0.01.md`, etc., lack leading hyphens, breaking the list format.
- **Solution:**
  1. Correct the filenames in the Good Rules list to match actual files:
     - `heuristics-to-supervised-pipeline.md`
     - `csv-schema-2.md`
  2. Ensure every entry in the Good Rules list is prefixed with a hyphen (`-`).

## 3. Spelling Errors
- **Issue:** Filename `mutilple-inheritance.md` is misspelled; should be `multiple-inheritance.md`.
- **Solution:** Rename the file and update all references (including Good Rules and cross-links).

## 4. Inconsistent Capitalization and Naming Style
- **Issue:** `Grid-Size-Directory-Structure-Compliance-Report.md` uses uppercase letters and PascalCase, while most other files use lowercase hyphen-case.
- **Solution:** Rename to `grid-size-directory-structure-compliance-report.md` to match the repository's naming conventions.

## 5. Next Steps & Notes
- Apply the above renames and list formatting fixes in `final-decision-10.md`.
- Validate that the Good Rules list exactly mirrors the filenames under `docs/extensions-guideline/`.
- After renaming, run a path validation (e.g., `validate_path_structure`) to catch any residual mismatches.

> **Note:** Other non-Good Rules guideline files may contain outdated or overlapping content; those will be addressed in subsequent, focused reviews as needed.
