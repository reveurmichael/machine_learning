"""Utility helpers to make experiments fully reproducible.

Only Task-0 needs it today, but future research tracks (RL, supervised
learning, fine-tuning) will rely heavily on deterministic behaviour across
`random`, `numpy`, and optional deep-learning back-ends.  Importing this module
has **no side-effects**; call :func:`seed_everything` explicitly at program
start-up.
"""

from __future__ import annotations

import os
import random
import types
from typing import Final, Protocol

__all__: Final = ["seed_everything"]


class _Seeder(Protocol):
    def seed(self, seed: int) -> None:  # noqa: D401 – simple seed method
        ...


def _maybe_seed(module: types.ModuleType | None, seed: int, attr: str = "seed") -> None:
    """Call *module.seed(seed)* if *module* is not None and exposes *attr*."""

    if module is None:
        return

    fn = getattr(module, attr, None)
    if callable(fn):
        try:
            fn(seed)  # type: ignore[misc] – duck typed
        except Exception:  # pragma: no cover – defensive; keep silent in prod
            # Silently ignore failures (e.g. CuPy without GPU)
            pass


def seed_everything(seed: int = 42) -> int:
    """Seed Python `random`, NumPy and *optionally* Torch/JAX/TensorFlow.

    The function is intentionally *best-effort*: if an optional library is not
    installed it will simply be skipped without raising an error.  It always
    returns the seed so that call-sites can log the value.
    """

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    # NumPy – plentiful across tasks
    try:
        import numpy as _np  # type: ignore
    except ModuleNotFoundError:
        _np = None  # type: ignore
    _maybe_seed(_np, seed)

    # Torch – RL / supervised tasks
    try:
        import torch as _torch  # type: ignore
    except ModuleNotFoundError:
        _torch = None  # type: ignore
    _maybe_seed(_torch, seed, attr="manual_seed")
    if _torch is not None and hasattr(_torch.cuda, "manual_seed_all"):
        _torch.cuda.manual_seed_all(seed)  # type: ignore[arg-type]

    # JAX – potential future research
    try:
        import jax as _jax  # type: ignore
    except ModuleNotFoundError:
        _jax = None  # type: ignore
    if _jax is not None:
        _jax.random.PRNGKey(seed)  # type: ignore[attr-defined]

    # TensorFlow – distillation or fine-tuning track
    try:
        import tensorflow as _tf  # type: ignore
    except ModuleNotFoundError:
        _tf = None  # type: ignore
    if _tf is not None and hasattr(_tf.random, "set_seed"):
        _tf.random.set_seed(seed)

    return seed 