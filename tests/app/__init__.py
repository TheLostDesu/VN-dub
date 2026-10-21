"""Desktop-app tests.

This file makes ``tests/app/conftest.py`` import as ``app.conftest`` instead of shadowing ``tests/conftest.py``
(both would be the top-level module ``conftest``). The folder is also put on ``sys.path`` so the helper modules
next to the tests (``app_helpers``, ``o_helpers``) keep importing by their bare names.
"""

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
