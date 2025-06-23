"""Heuristics-Supervised Integration v0.01
--------------------

Proof-of-concept pipeline that demonstrates **end-to-end automation**:
1. Generate a heuristic dataset (CSV) via the common dataset generator.
2. Train a tiny supervised model (MLP-Classifier) on that dataset.
3. Save the trained model using `extensions.common.model_utils`.

The package depends only on *core* + `extensions/common`, therefore
`integration-v0.01 + common` remains standalone.
"""

from __future__ import annotations

from extensions.common.path_utils import ensure_project_root_on_path, setup_extension_paths

ensure_project_root_on_path()
setup_extension_paths()

__all__: list[str] = [] 