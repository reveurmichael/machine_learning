"""Tiny set of helpers for choosing TCP ports.

All functions bind *locally* only; they never attempt to connect or listen,
so they are safe to call in restricted environments (CI, Streamlit Cloud, …).

This whole module is NOT Task0 specific.
"""

from __future__ import annotations

import random
import socket

__all__ = [
    "find_free_port",
    "is_port_free",
    "ensure_free_port",
    "random_free_port",
]


DEFAULT_MIN_PORT: int = 8000  # lowest port considered when picking at random


def find_free_port(start: int = DEFAULT_MIN_PORT, max_port: int = 65_535) -> int:
    """Return the first **free** TCP port ``>= start``.

    Parameters
    ----------
    start
        Minimum candidate port (inclusive).
    max_port
        Upper bound before we give up and raise.
    """

    for port in range(start, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                # Bind to *localhost* to avoid conflicts with services bound to
                # 0.0.0.0 but not 127.0.0.1.
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue

    raise RuntimeError("No free port found in the specified range")


def is_port_free(port: int) -> bool:
    """Return *True* iff the given TCP ``port`` is currently unused."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
            return True
        except OSError:
            return False


def ensure_free_port(port: int) -> int:
    """Return ``port`` if free, otherwise the next available one."""

    return port if is_port_free(port) else find_free_port(max(port + 1, 1_024))


def random_free_port(min_port: int = DEFAULT_MIN_PORT, max_port: int = 9_000) -> int:
    """Return a *random* free port within ``[min_port, max_port]``.

    Useful for dashboard defaults so multiple widgets don't all suggest 8000.
    Falls back to :func:`find_free_port` if unlucky after 1000 attempts.
    """

    for _ in range(1_000):
        candidate = random.randint(min_port, max_port)
        if is_port_free(candidate):
            return candidate

    # Fallback – unlikely
    return find_free_port(min_port) 