"""The text patch against a real Ren'Py: lint and (optionally) a scripted play-through of the SDK's sample game.

Skipped unless ``VNDUB_RENPY_SDK`` points at a Ren'Py SDK folder (``renpy.py`` and its ``lib/py3-*`` Python). The play-through opens a
window for a few seconds, so it also needs ``VNDUB_RENPY_SDK_WALK=1``. The SDK folder is never written to: ``the_question`` is
copied to ``tmp_path`` first.

The pipeline runs on the test profile (``configs/pipeline.test.yaml``, mock backends): extract .. translate, then every translation
gets a Russian text with the characters Ren'Py treats specially (quotes, ``%``, brackets, braces, backslashes), the character
names get a Russian spelling, and pack writes the patch.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SDK = Path(os.environ["VNDUB_RENPY_SDK"]) if os.environ.get("VNDUB_RENPY_SDK") else None
REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(SDK is None or not (SDK / "renpy.py").exists(), reason="VNDUB_RENPY_SDK does not point at a Ren'Py SDK")

NASTY = [
    'Привет, "мир"! Это реплика номер {n}.',
    "Я уверен на 100%, что это реплика {n}.",
    "Тут [не переменная] и {{не тег}}, реплика {n}.",
    "Обратная \\ косая, реплика {n}, и ещё \\n внутри.",
    "Строка с 'апострофами', «ёлочками» — тире… реплика {n}.",
    "  Пробелы   внутри    и по краям, реплика {n}.  ",
    "Одинокая { скобка и одинокая ] скобка, а ещё %(name)s и %s, реплика {n}.",
    "Жирный {{b}}текст{{/b}} от переводчика, реплика {n}.",
    "Эмодзи \U0001F600 и 日本語, реплика {n}.",
    "Обычная реплика {n} без сюрпризов.",
]
WALK = """testcase vndub_walk:
    click "start"
    advance until screen "choice"
    click "Выбор"
    advance until screen "choice"
    click "Выбор"
    advance until screen "main_menu"
"""


def _sdk_python() -> Path:
    cands = sorted((SDK / "lib").glob("py3-*/python*"))
    exe = [c for c in cands if c.name == ("python.exe" if os.name == "nt" else "python")]
    if not exe:
        pytest.skip("no Python inside the SDK's lib folder")
    return exe[0]


def _lint(game: Path, report: Path) -> list[str]:
    r = subprocess.run([str(_sdk_python()), "renpy.py", str(game), "lint", str(report)], cwd=SDK, capture_output=True, text=True, timeout=600)
    assert report.exists(), f"lint did not write a report: {r.stdout[-500:]} {r.stderr[-500:]}"
    return [l.strip() for l in report.read_text(encoding="utf-8-sig").splitlines() if re.match(r"\s*game/", l)]


def _clean_copy(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)
    shutil.rmtree(dst / "game" / "tl" / "russian", ignore_errors=True)  # the game's own Russian translation would collide with ours
    shutil.rmtree(dst / "game" / "cache", ignore_errors=True)
    for c in dst.rglob("*.rpyc"):
        c.unlink()


def _vndub(*args: str) -> None:
    r = subprocess.run([sys.executable, "-m", "vndub", *args], cwd=REPO, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, f"vndub {' '.join(args[:3])} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"


@pytest.fixture(scope="module")
def patched(tmp_path_factory) -> dict[str, Path]:
    from vndub.manifest.io import load_characters, save_characters

    root = tmp_path_factory.mktemp("sdk")
    pristine = root / "pristine"
    _clean_copy(SDK / "the_question", pristine)
    proj, work, out = root / "proj", root / "work", root / "out"
    shutil.copytree(pristine, proj)
    base = ["run", "--config", "configs/pipeline.test.yaml", "--project", str(proj), "--work-dir", str(work), "--out-dir", str(out), "--set", "synth.voice_unvoiced=true"]
    _vndub(*base, "--to", "translate")
    tl = work / "translation" / "lines.ru.jsonl"
    rows = []
    for n, line in enumerate([l for l in tl.read_text(encoding="utf-8").splitlines() if l.strip()], 1):
        r = json.loads(line)
        t = NASTY[n % len(NASTY)].replace("{{", "{").replace("}}", "}").replace("{n}", str(n))
        r.update(text_tgt=t, text_tts=t, text_plain=t)
        r["flags"] = [f for f in r.get("flags") or [] if f != "check_failed"]
        if isinstance(r.get("choices_tgt"), list):
            r["choices_tgt"] = [f'Выбор "{i}" на 100% [{n}]' for i, _ in enumerate(r["choices_tgt"])]
        rows.append(json.dumps(r, ensure_ascii=False))
    tl.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    chars_path = work / "manifest" / "characters.yaml"
    chars = load_characters(chars_path)
    for sid, c in chars.items():
        if sid not in ("narrator", "voice_over") and c.names.get("src"):
            c.names["tgt"] = "Имя " + str(c.names["src"]).replace('"', "")
    save_characters(chars_path, chars)
    (work / "final_index.jsonl").write_text("", encoding="utf-8")  # nothing was dubbed: only the text patch is under test
    _vndub(*base, "--from", "pack", "--to", "pack")
    base_game = root / "base"
    shutil.copytree(pristine, base_game)
    patched_game = root / "patched"
    shutil.copytree(pristine, patched_game)
    shutil.copytree(out / "renpy" / "game", patched_game / "game", dirs_exist_ok=True)
    return {"root": root, "base": base_game, "patched": patched_game, "out": out / "renpy" / "game"}


def test_lint_reports_nothing_new_and_ids_are_the_games_own(patched: dict[str, Path]) -> None:
    root = patched["root"]
    before = set(_lint(patched["base"], root / "lint_base.txt"))
    after = _lint(patched["patched"], root / "lint_patched.txt")
    new = [l for l in after if l not in before]
    # the generated scripts run at init priority 1000: lint notes it, nothing else may be new
    bad = [l for l in new if not (re.match(r"game/vndub_\w+\.rpy:\d+ The init priority", l))]
    assert not bad, bad
    gen = (patched["out"] / "tl" / "russian" / "vndub_dialogue.rpy").read_text(encoding="utf-8")
    ours = re.findall(r"^translate russian (\w+):", gen, re.M)
    real = set(re.findall(r"^translate russian (\w+):", (SDK / "the_question" / "game" / "tl" / "russian" / "script.rpy").read_text(encoding="utf-8-sig"), re.M))
    assert len(ours) > 50
    assert [i for i in ours if i not in ("strings", "python", "style") and i not in real] == []
    assert 'old "Имя' not in gen or 'translate russian strings:' in gen  # the character names are strings entries
    assert re.search(r'old "[A-Za-z]+"\n    new "Имя ', gen)


@pytest.mark.skipif(os.environ.get("VNDUB_RENPY_SDK_WALK") != "1", reason="the play-through opens a window: set VNDUB_RENPY_SDK_WALK=1")
def test_scripted_play_through_passes(patched: dict[str, Path], tmp_path: Path) -> None:
    game = patched["patched"]
    (game / "game" / "vndub_walk.rpy").write_text(WALK, encoding="utf-8")
    env = {**os.environ, "SDL_AUDIODRIVER": "dummy"}
    r = subprocess.run([str(_sdk_python()), "renpy.py", "--savedir", str(tmp_path / "saves"), str(game), "test", "vndub_walk"], cwd=SDK, env=env, capture_output=True, text=True, timeout=600)
    assert "Status: PASSED" in r.stdout + r.stderr, (r.stdout + r.stderr)[-2000:]
