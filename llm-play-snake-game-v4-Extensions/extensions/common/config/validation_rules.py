"""
Very-Lightweight Validation Rules
================================
This module intentionally keeps the amount of *enforced* validation to an
absolute minimum.  Its sole purpose is to provide a **single source of truth**
for a few symbolic constants that some of the helpers inside
`extensions/common/` import.  The rules are *suggestions* rather than hard
constraints – extensions are free to ignore or extend them.

If you need stricter checks inside your own extension simply *extend* these
values in your package-level code instead of trying to modify the common one.
"""

from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# CSV dataset suggestions
# ---------------------------------------------------------------------------

CSV_VALIDATION_RULES: Dict[str, Any] = {
    # Avoid rows with missing values in the 16-feature tabular format.
    "no_missing_values": True,
    # Expected minimum number of rows in a dataset – mostly to catch empty files.
    "min_rows": 1,
}

# ---------------------------------------------------------------------------
# Generic data quality thresholds
# ---------------------------------------------------------------------------

DATA_QUALITY_THRESHOLDS: Dict[str, Any] = {
    "min_unique_games": 1,
    "min_average_score": 0,  # Extensions can tighten this if they want.
}

# ---------------------------------------------------------------------------
# Very loose naming convention hints (non-enforced)
# ---------------------------------------------------------------------------

NAMING_CONVENTIONS: Dict[str, List[str]] = {
    # We **prefer** agent implementation files to end with _agent.py because this
    # makes grepping for agents a lot easier, but we do NOT enforce it here.
    "agent_file_suffix": ["_agent.py"],
} 