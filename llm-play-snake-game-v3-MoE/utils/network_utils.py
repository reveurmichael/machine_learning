"""Network-related helper functions."""
from __future__ import annotations
import socket
import random

def find_free_port(start: int = 8000, max_port: int = 65535) -> int:
    """Return first TCP port >= *start* that is free on localhost.

    Raises RuntimeError if none found before *max_port*.
    """
    for port in range(start, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                # Bind specifically to localhost so we don't get false positives
                # when another process is already listening on 127.0.0.1 but not on 0.0.0.0.
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found in range")

def is_port_free(port: int) -> bool:
    """Check quickly if *port* is available for listening on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
            return True
        except OSError:
            return False

def ensure_free_port(port: int) -> int:
    """Return *port* if available, otherwise the next free one.

    Example
    -------
    >>> ensure_free_port(8000)
    8000  # if 8000 free, else 8001 or next available
    """
    if is_port_free(port):
        return port
    return find_free_port(max(port + 1, 1024))

def random_free_port(min_port: int = 8000, max_port: int = 9000) -> int:
    """Return a random free port within the given range.

    This is useful for UI defaults where multiple widgets each need a
    *different* free port suggestion so the dashboard doesn't keep showing
    8000 everywhere.
    """
    attempt = 0
    while attempt < 1000:
        port = random.randint(min_port, max_port)
        if is_port_free(port):
            return port
        attempt += 1
    # Fallback to sequential search if unlucky
    return find_free_port(min_port) 