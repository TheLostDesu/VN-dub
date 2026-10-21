"""Tiny adapter used by tests/test_runner_protocol.py (runs with the test interpreter).

Adapter keys: ``echo`` (normal), ``nowts`` (weights missing on load),
``unavail`` (backend unavailable on load).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from adapters.common.rpc import ModelImpl, UnavailableError, WeightsMissingError, main


class Echo(ModelImpl):
    selftest_op = "echo"

    def load(self) -> dict[str, Any]:
        print("noise on stdout must not break the protocol")  # redirected to stderr by serve()
        self.loads = int(os.environ.get("FAKE_LOADS", "0")) + 1
        return {"loaded": True, "pid": os.getpid()}

    def op_echo(self, item: dict[str, Any], **kw: Any) -> dict[str, Any]:
        return {"echo": item.get("text"), "kw": kw}

    def op_fail(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
        if item.get("text") == "bad":
            raise ValueError("bad item")
        return {"ok": True}

    def op_sleep(self, item: dict[str, Any], seconds: float = 5.0, **_: Any) -> dict[str, Any]:
        time.sleep(seconds)
        return {"slept": seconds}

    def op_crash_once(self, item: dict[str, Any], marker: str = "", **_: Any) -> dict[str, Any]:
        m = Path(marker)
        if not m.exists():
            m.write_text("crashed", encoding="utf-8")
            os._exit(3)
        return {"pid": os.getpid(), "recovered": True}

    def op_pid(self, **_: Any) -> dict[str, Any]:
        return {"pid": os.getpid()}


class NoWeights(ModelImpl):
    def load(self) -> dict[str, Any]:
        raise WeightsMissingError(self.name, "fake")


class Unavailable(ModelImpl):
    def load(self) -> dict[str, Any]:
        raise UnavailableError("fake backend needs CUDA", "use a GPU")


TABLE = {"echo": Echo, "nowts": NoWeights, "unavail": Unavailable}

if __name__ == "__main__":
    main(TABLE, "fake")
