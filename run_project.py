from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
REQUIREMENTS_PATH = ROOT_DIR / "requirements.txt"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def ensure_runtime_dependencies() -> None:
    try:
        import gradio  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)])


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click launcher for the Guardian of Truth frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--share", dest="share", action="store_true")
    parser.add_argument("--no-share", dest="share", action="store_false")
    parser.add_argument("--inbrowser", action="store_true")
    parser.set_defaults(share=True)
    args = parser.parse_args()

    ensure_runtime_dependencies()

    from guardian_of_truth.gradio_app import launch_demo

    launch_demo(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
        share=args.share,
        inbrowser=args.inbrowser,
    )


if __name__ == "__main__":
    main()
