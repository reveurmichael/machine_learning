"""
Reproducibility Utilities

This module provides a robust, centralized function to seed all major sources
of randomness used in the project, ensuring experimental reproducibility.

It is designed to be a best-effort utility:
- It seeds standard libraries like `random` and `numpy`.
- It attempts to seed optional deep learning libraries (PyTorch, JAX)
  if they are installed, but does not fail if they are not.

Importing this module has no side effects. The `seed_everything` function
must be called explicitly at program startup.
"""

from __future__ import annotations

import os
import random
from typing import Final, Protocol

__all__: Final = ["seed_everything"]


class _Seeder(Protocol):
    """A protocol for modules that have a seeding function."""

    def seed(self, seed: int) -> None:
        ...


def _maybe_seed(
    module_name: str, seed: int, verbose: bool, seed_func_name: str = "seed"
) -> None:
    """
    Attempts to import and seed a module if it exists.

    Args:
        module_name: The name of the module to import (e.g., 'numpy').
        seed: The integer seed value.
        verbose: If True, prints which libraries are seeded or skipped.
        seed_func_name: The name of the seeding function in the module.
    """
    try:
        module = __import__(module_name)
        seed_func = getattr(module, seed_func_name, None)
        if callable(seed_func):
            seed_func(seed)
            if verbose:
                print(f"🌱 Seeded `{module_name}` with value {seed}.")
        # Special handling for more complex libraries
        elif module_name == "torch" and hasattr(module.cuda, "manual_seed_all"):
            module.cuda.manual_seed_all(seed)
            if verbose:
                print(f"🌱 Seeded `torch.cuda` with value {seed}.")
        elif module_name == "jax":
            module.random.PRNGKey(seed)
            if verbose:
                print(f"🌱 Seeded `jax` with value {seed}.")
    except ImportError:
        if verbose:
            print(f"⚪️ `{module_name}` not found, skipping seeding.")
    except Exception as e:
        if verbose:
            print(f"⚠️ Failed to seed `{module_name}`: {e}")


def seed_everything(seed: int = 42, verbose: bool = True) -> int:
    """
    Seeds Python's `random`, `numpy`, and optionally PyTorch and JAX
    to ensure reproducibility.

    Args:
        seed: The integer seed value. Defaults to 42.
        verbose: If True, prints messages about which libraries are being
                 seeded. Defaults to True.

    Returns:
        The seed value used, allowing call sites to log it.
    """
    # 1. Set Python's hash seed environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Seed Python's built-in random module
    random.seed(seed)
    if verbose:
        print(f"🌱 Seeded `random` with value {seed}.")

    # 3. Seed major libraries if they are installed
    _maybe_seed("numpy", seed, verbose)
    _maybe_seed("torch", seed, verbose, seed_func_name="manual_seed")
    _maybe_seed("jax", seed, verbose)

    return seed 