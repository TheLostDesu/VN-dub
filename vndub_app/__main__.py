"""``python -m vndub_app`` == ``vndub-app``."""

from __future__ import annotations

import sys

from vndub_app.shell import main

if __name__ == "__main__":
    sys.exit(main())
