"""
Convenience launcher to run both the FastAPI backend (port 8000)
and the React frontend (port 5173) concurrently.
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
PYTHON_EXE = sys.executable


def main():
    print("=" * 60)
    print("Starting EMP-26 Full-Stack Application")
    print("=" * 60)
    print("  - Backend API: http://127.0.0.1:8000")
    print("  - Frontend UI:  http://localhost:5173")
    print("=" * 60)

    # 1. Start FastAPI Backend
    backend_cmd = [PYTHON_EXE, "-m", "uvicorn", "server.api:app", "--host", "127.0.0.1", "--port", "8000"]
    print("\n[Launcher] Starting FastAPI server...")
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(PROJECT_ROOT))

    time.sleep(1.5)

    # 2. Start Vite React Frontend
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]
    print("[Launcher] Starting Vite React dev server...")
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(FRONTEND_DIR))

    print("\nBoth services are running! Press Ctrl+C to terminate both.\n")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
