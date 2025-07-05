import subprocess
import sys

if __name__ == "__main__":
    cmd = [
        sys.executable,
        "extensions/heuristics-v0.04/scripts/main_dataset_generator.py",
        "--algorithm", "BFS",
        "--max-games", "2",
        "--output-dir", "test_output"
    ]
    try:
        result = subprocess.run(cmd, timeout=5)
        sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        print("[FAIL-FAST] Process killed after 5 seconds (timeout reached)")
        sys.exit(124) 