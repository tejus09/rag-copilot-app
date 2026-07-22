"""CLI launcher.

Run as `uv run python app/cli.py <command>` rather than the installed
`rag-copilot` console-script: that script's `from rag_copilot_app.cli import
main` depends on the project's editable install correctly registering src/ on
sys.path, which some environments don't do reliably. This file sidesteps that
by bootstrapping src/ itself before importing anything from the package, the
same approach app/streamlit_app.py uses.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_copilot_app.cli import app

if __name__ == "__main__":
    app()
