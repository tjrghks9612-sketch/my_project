from __future__ import annotations

import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fib_sem_measurement_tool.app import run_app


if __name__ == "__main__":
    run_app()

