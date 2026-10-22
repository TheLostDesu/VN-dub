"""``python -m vndub_app`` == ``vndub-app``.

``--reaper`` is the helper process ``shell`` starts to clean up after the app (see ``vndub_app.winproc.reaper_main``): it is
decided here, before ``vndub_app.shell`` is imported, so the helper stays a few MB (no server, no GUI, no instance lock).
"""

from __future__ import annotations

import sys


def _run() -> int:
    if "--reaper" in sys.argv[1:]:
        from vndub_app.winproc import reaper_main

        return reaper_main(sys.argv[1:])
    from vndub_app.shell import main

    return main()


if __name__ == "__main__":
    sys.exit(_run())
