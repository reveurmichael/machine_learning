"""Network-related helper functions."""
from __future__ import annotations
import socket

def find_free_port(start: int = 8000, max_port: int = 65535) -> int:
    """Return first TCP port >= *start* that is free on localhost.

    Raises RuntimeError if none found before *max_port*.
    """
    for port in range(start, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found in range") 