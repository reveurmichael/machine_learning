"""
Network configuration constants.

This module centralizes network-related configuration that is shared
across all network operations (Flask servers, port allocation, host binding).

Following DRY principles, these constants ensure consistent network
setup and make maintenance easier across all network components.

=== SINGLE SOURCE OF TRUTH ===
This module serves as the SINGLE SOURCE OF TRUTH for all network-related constants.
Any constant defined here is immediately available to ALL network operations.

=== NETWORK ARCHITECTURE ===
These constants support the random port allocation strategy and dynamic
host/port resolution used throughout the Snake Game AI project.

=== USAGE PATTERNS ===
- Flask server configuration: Host and port settings
- Port allocation: Range definitions for random port selection
- Network utilities: Default values for socket operations
- Environment variables: Fallback values for deployment scenarios
"""

from typing import Final

# Host configuration
DEFAULT_HOST: Final[str] = "127.0.0.1"
HOST_CHOICES: Final[list[str]] = ["localhost", "0.0.0.0", "127.0.0.1"]

# Port allocation ranges
DEFAULT_PORT_RANGE_START: Final[int] = 8000
DEFAULT_PORT_RANGE_END: Final[int] = 16000

# Environment variable names for network configuration
HOST_ENV_VAR: Final[str] = "HOST"
PORT_ENV_VAR: Final[str] = "PORT"

# Network timeout settings
HTTP_TIMEOUT: Final[int] = 30  # seconds
SOCKET_TIMEOUT: Final[int] = 5  # seconds

# Socket configuration
SOCKET_REUSE_ADDR: Final[int] = 1
MIN_SAFE_PORT: Final[int] = 1024  # Minimum safe port number
MAX_PORT_ATTEMPTS: Final[int] = 1000  # Maximum attempts for random port selection 