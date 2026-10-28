"""Every training call of the voicebank stage passes ``timeout_s``.

``RemoteBackend.call`` defaults to ``120 + timeout_item * n`` seconds. The route-C VoxCPM LoRA training runs for hours, so a
call without ``timeout_s`` killed the adapter mid-training (and left its training subprocess holding the VRAM); the RVC call
(route B) always passed ``timeout_train_s``.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "vndub" / "stages" / "voicebank.py"


def _training_calls() -> list[ast.Call]:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr == "train":
            out.append(node)  # backend.train(job, ...)
        elif node.func.attr == "call" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in ("train", "train_lora"):
            out.append(node)
    return out


def test_every_training_call_sets_a_timeout() -> None:
    calls = _training_calls()
    assert len(calls) >= 2, "expected the RVC (train) and the route-C (train_lora) calls"
    for c in calls:
        assert "timeout_s" in {k.arg for k in c.keywords}, f"{SRC.name}:{c.lineno}: training call without timeout_s"


def test_the_lora_timeout_is_hours_by_default() -> None:
    text = SRC.read_text(encoding="utf-8")
    assert text.count('.get("timeout_train_s", 6 * 3600)') >= 2  # RVC and route C read the same option with the same default
