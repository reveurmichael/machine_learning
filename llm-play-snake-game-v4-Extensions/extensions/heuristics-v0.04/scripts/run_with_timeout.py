import subprocess
import sys

if __name__ == "__main__":
    cmd = [
        sys.executable,
        "extensions/heuristics-v0.04/scripts/main.py",
        "--algorithm", "BFS",
        "--max-games", "2",
    ]
    try:
        result = subprocess.run(cmd, timeout=5)
        sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        print("[FAIL-FAST] Process killed after 5 seconds (timeout reached)") # Excpetionally, here we don't use print_error, because we want to keep this file as simple as possible.
        sys.exit(124) 

