"""launch_web.py - Web Interface Launcher Script

CLI script to launch the Streamlit web interface for v0.03.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.chdir(str(project_root))

import subprocess
import argparse


def main():
    """Launch the web interface."""
    parser = argparse.ArgumentParser(description="Launch Heuristics-Supervised Integration v0.03 Web Interface")
    parser.add_argument("--port", type=int, default=8501, help="Port to run on")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    
    args = parser.parse_args()
    
    # Get app path
    app_path = current_file.parent.parent / "app.py"
    
    # Launch Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    
    print(f"🚀 Launching web interface on http://{args.host}:{args.port}")
    print(f"Command: {' '.join(cmd)}")
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
