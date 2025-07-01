from typing import Final

# Host configuration
DEFAULT_HOST: Final[str] = "127.0.0.1"
HOST_CHOICES: Final[list[str]] = ["localhost", "0.0.0.0", "127.0.0.1"]

# Port allocation ranges
DEFAULT_PORT_RANGE_START: Final[int] = 8000
DEFAULT_PORT_RANGE_END: Final[int] = 16000 