from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
REQUIREMENTS_PATH = ROOT_DIR / "requirements.txt"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guardian_of_truth.utils import load_local_env


def ensure_runtime_dependencies() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)])


def main() -> None:
    default_host = os.environ.get(
        "HOST",
        "0.0.0.0" if ("PORT" in os.environ or "RENDER" in os.environ) else "127.0.0.1",
    )
    default_port = int(os.environ.get("PORT", "8501"))

    parser = argparse.ArgumentParser(description="Launcher for the Guardian of Truth Streamlit interface.")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--inbrowser", action="store_true")
    args = parser.parse_args()

    load_local_env()
    ensure_runtime_dependencies()

    from streamlit.web import cli as stcli

    app_path = str(ROOT_DIR / "streamlit_app.py")
    cli_args = [
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    if not args.inbrowser:
        cli_args.extend(["--server.headless", "true"])
    sys.argv = cli_args
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
