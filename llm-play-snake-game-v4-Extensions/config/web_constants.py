"""
Web server configuration constants.

This module centralizes Flask application configuration that is shared
across all web mode scripts (main_web.py, replay_web.py, human_play_web.py).

Following DRY principles, these constants ensure consistent web server
setup and make maintenance easier across the web interface components.

=== SINGLE SOURCE OF TRUTH ===
This module serves as the SINGLE SOURCE OF TRUTH for all web-related constants.
Any constant defined here is immediately available to ALL web applications.

=== DEBUG MODE INTEGRATION ===
FLASK_DEBUG_MODE controls both server-side Flask debug mode and client-side
JavaScript debug output. This ensures consistent debugging behavior across
the entire web stack.

=== USAGE PATTERNS ===
- Server-side: Flask debug mode for development
- Client-side: JavaScript console logging and error reporting
- Template rendering: Conditional debug information display
- API responses: Enhanced error details in debug mode
"""

from pathlib import Path
from typing import Final

# Repository root path discovery
REPO_ROOT: Final[Path] = Path(__file__).parent.parent

# Flask application configuration
FLASK_STATIC_FOLDER: Final[str] = str(REPO_ROOT / "web" / "static")
FLASK_TEMPLATE_FOLDER: Final[str] = str(REPO_ROOT / "web" / "templates")

# Default web server settings
DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT_RANGE_START: Final[int] = 8000
DEFAULT_PORT_RANGE_END: Final[int] = 20000

# Debug mode configuration (SINGLE SOURCE OF TRUTH)
FLASK_DEBUG_MODE: Final[bool] = True  # Controls both server and client debug behavior

# Debug mode constants for client-side JavaScript
DEBUG_MODE_ENABLED: Final[bool] = FLASK_DEBUG_MODE
DEBUG_MODE_DISABLED: Final[bool] = not FLASK_DEBUG_MODE

# Debug mode string representation for template rendering
DEBUG_MODE_STRING: Final[str] = str(FLASK_DEBUG_MODE).lower()  # "true" or "false"
