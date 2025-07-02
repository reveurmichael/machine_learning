from pathlib import Path
from typing import Final

# Repository root path discovery
REPO_ROOT: Final[Path] = Path(__file__).parent.parent

# Flask application configuration
FLASK_STATIC_FOLDER: Final[str] = str(REPO_ROOT / "web" / "static")
FLASK_TEMPLATE_FOLDER: Final[str] = str(REPO_ROOT / "web" / "templates")

# Debug mode configuration (SINGLE SOURCE OF TRUTH)
FLASK_DEBUG_MODE: Final[bool] = False  # Controls both server and client debug behavior
