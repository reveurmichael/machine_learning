"""Tiny set of helpers for choosing TCP ports.

All functions bind *locally* only; they never attempt to connect or listen,
so they are safe to call in restricted environments (CI, Streamlit Cloud, …).

This whole module is NOT Task0 specific.
"""

from __future__ import annotations

import random
import socket
import os as _os

from config.network_constants import (
    DEFAULT_HOST,
    DEFAULT_PORT_RANGE_START,
    DEFAULT_PORT_RANGE_END,
)

__all__ = [
    "find_free_port",
    "is_port_free",
    "ensure_free_port",
    "random_free_port",
    "get_server_host_port",
]


def find_free_port(
    start: int = DEFAULT_PORT_RANGE_START, max_port: int = DEFAULT_PORT_RANGE_END
) -> int:
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
                sock.bind((DEFAULT_HOST, port))
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

    return port if is_port_free(port) else find_free_port(max(port + 1, 1024))


def random_free_port(min_port: int = DEFAULT_PORT_RANGE_START, max_port: int = DEFAULT_PORT_RANGE_END) -> int:
    """Return a *random* free port within ``[min_port, max_port]``.

    Useful for dashboard defaults so multiple widgets don't all suggest 8000.
    Falls back to :func:`find_free_port` if unlucky after 1000 attempts.
    """

    for _ in range(1000):
        candidate = random.randint(min_port, max_port)
        if is_port_free(candidate):
            return candidate

    # Fallback – unlikely
    return find_free_port(min_port)


def get_server_host_port(default_host: str = DEFAULT_HOST, default_port: int | None = None) -> tuple[str, int]:
    """Return a tuple *(host, port)* suitable for server binding.

    Parameters
    ----------
    default_host
        Bind address to use when ``HOST`` environment variable is absent.
    default_port
        Port to bind when ``PORT`` env-var is absent.  If *None* we will
        pick a free one starting from :pydata:`DEFAULT_PORT_RANGE_START`.

    The helper centralises the "pick a free port" logic so every script does
    not have to duplicate it.  It first honours explicit environment
    variables because that is convenient for Docker / CI pipelines.
    """

    host = _os.getenv("HOST", default_host)
    port_env = _os.getenv("PORT")

    # 1. Determine the *candidate* port ---------------------
    if port_env is not None and port_env.isdigit():
        candidate = int(port_env)
    else:
        candidate = default_port or DEFAULT_PORT_RANGE_START

    # 2. Check if the (host, candidate) tuple is free.  We bind *exactly* to
    #    that host so that a process listening on 127.0.0.1 does not get
    #    missed when we probe "0.0.0.0" (which would otherwise succeed and
    #    later raise the very same OSError we are trying to avoid).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _sock:
        _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            _sock.bind((host, candidate))
            # Success → port is genuinely free for **that** host.
            port = candidate
        except OSError:
            # Busy → fallback to the generic helper starting *above* the
            # original request so we don't re-test the same port.
            port = ensure_free_port(max(candidate + 1, 1024))

    return host, port 
