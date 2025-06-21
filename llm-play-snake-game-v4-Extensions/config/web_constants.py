"""
Web server configuration constants.

This module centralizes Flask application configuration that is shared
across all web mode scripts (main_web.py, replay_web.py, human_play_web.py).

Following DRY principles, these constants ensure consistent web server
setup and make maintenance easier across the web interface components.
"""

from pathlib import Path
from typing import Final

# Repository root path discovery
REPO_ROOT: Final[Path] = Path(__file__).parent.parent #TODO: this one is not correct. 

# Flask application configuration
FLASK_STATIC_FOLDER: Final[str] = str(REPO_ROOT / "web" / "static")
FLASK_TEMPLATE_FOLDER: Final[str] = str(REPO_ROOT / "web" / "templates")

# Default web server settings
DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT_RANGE_START: Final[int] = 8000
DEFAULT_PORT_RANGE_END: Final[int] = 20000
