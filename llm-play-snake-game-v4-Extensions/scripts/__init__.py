"""Command-line entry scripts.

The folder collects thin wrappers around the original top-level files
(`main.py`, `replay.py`, …) so external tools can invoke them via
`python scripts/<name>.py` without relying on the project root as the working
directory.  Each wrapper first changes the process CWD to the repository root
and then delegates execution to the real implementation, guaranteeing that
relative file paths continue to resolve exactly as in Task-0.
"""

from __future__ import annotations

__all__: list[str] = [] 