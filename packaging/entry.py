"""PyInstaller entry point of VNDub.exe (windowed, no console)."""

from __future__ import annotations

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from vndub_app.shell import main

    sys.exit(main())
