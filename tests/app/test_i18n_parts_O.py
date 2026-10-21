"""The i18n texts of the doctor / installer / models / patch-install / shell modules: complete, in step,
and matching what the code emits.

They were written as the part ``app/i18n/parts/O.{ru,en}.json`` and folded into the main tables by
``scripts/merge_i18n.py --delete``; while a part file exists it is checked on its own, afterwards the
same checks run on the keys of those modules in the main tables.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
I18N = ROOT / "app" / "i18n"
MODULES = ("doctor", "installer", "models_api", "patch_install", "downloads", "shell")
PLACEHOLDER = re.compile(r"\{(\w+)\}")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")


def load(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


O_PREFIXES = ("doctor.", "install.", "installer.", "shell.", "models.", "tools.", "envs.")


@pytest.fixture(scope="module")
def parts() -> dict[str, dict[str, str]]:
    out = {"ru": load(I18N / "parts" / "O.ru.json"), "en": load(I18N / "parts" / "O.en.json")}
    if not out["ru"] and not out["en"]:  # merged: the module texts now live in the main tables
        out = {lang: {k: v for k, v in load(I18N / f"{lang}.json").items() if k.startswith(O_PREFIXES)} for lang in ("ru", "en")}
    return out


@pytest.fixture(scope="module")
def tables(parts: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Main table (when present) overlaid on the parts, as the merge script does."""
    out = {}
    for lang in ("ru", "en"):
        out[lang] = {**parts[lang], **load(I18N / f"{lang}.json")}
    return out


def source(name: str) -> ast.Module:
    return ast.parse((ROOT / "vndub_app" / f"{name}.py").read_text(encoding="utf-8"))


def have(tables: dict[str, dict[str, str]], key: str) -> bool:
    return key in tables["ru"] and key in tables["en"]


def test_parts_are_flat_string_maps_in_step(parts: dict[str, dict[str, str]]) -> None:
    ru, en = parts["ru"], parts["en"]
    assert ru and en
    assert all(isinstance(v, str) and v.strip() for t in (ru, en) for v in t.values())
    assert set(ru) == set(en), f"ru only {sorted(set(ru) - set(en))}; en only {sorted(set(en) - set(ru))}"
    assert all("." in k for k in ru), "flat dotted keys only"
    bad = [k for k in ru if sorted(PLACEHOLDER.findall(ru[k])) != sorted(PLACEHOLDER.findall(en[k]))]
    assert not bad, f"placeholders differ: {bad}"
    emoji = [k for t in (ru, en) for k, v in t.items() if EMOJI.search(v)]
    assert not emoji, f"emoji in {emoji}"
    html = [k for t in (ru, en) for k, v in t.items() if re.search(r"</?[a-zA-Z]", v)]
    assert not html, f"markup in {html}"


def test_russian_texts_are_russian(parts: dict[str, dict[str, str]]) -> None:
    latin = [k for k, v in parts["ru"].items() if not k.startswith(("tools.", "models.", "envs.")) and not re.search("[А-Яа-яЁё]", v)]
    assert not latin, f"Russian texts without Cyrillic: {latin}"
    # model names of the form "Голос: ..." / "Перевод: ..." (the spec's examples)
    assert parts["ru"]["models.qwen3_tts.name"] == "Голос: Qwen3-TTS 1.7B — основной"
    assert parts["ru"]["models.gemma4_12b.name"] == "Перевод: Gemma 4 12B"


def test_every_registered_backend_has_a_name_and_a_purpose(tables: dict[str, dict[str, str]]) -> None:
    from vndub.backends import registry

    missing = []
    for name in registry.all_specs():
        for field in ("name", "purpose"):
            if not have(tables, f"models.{name}.{field}"):
                missing.append(f"models.{name}.{field}")
    assert not missing, missing
    purposes = [tables["ru"][f"models.{n}.purpose"] for n in registry.all_specs()]
    assert all(len(p) < 220 for p in purposes), "one sentence of purpose, not a paragraph"


def test_no_orphan_model_keys(parts: dict[str, dict[str, str]]) -> None:
    from vndub.backends import registry

    names = set(registry.all_specs())
    # per-backend texts are models.<backend>.name / .purpose; other models.* keys are screen texts
    orphans = sorted(k for k in parts["ru"] if k.startswith("models.") and k.count(".") == 2 and k.endswith((".name", ".purpose")) and k.split(".")[1] not in names and k.split(".")[1] not in ("col", "preset", "role", "status"))
    assert not orphans, orphans


def test_tools_and_envs_are_named(tables: dict[str, dict[str, str]]) -> None:
    from vndub_app import installer

    manifest = json.loads((ROOT / "app" / "tools_manifest.json").read_text(encoding="utf-8"))
    miss = [f"tools.{e['name']}.{f}" for e in manifest["tools"] for f in ("name", "purpose") if not have(tables, f"tools.{e['name']}.{f}")]
    miss += [f"envs.{n}.name" for n in installer.env_names() if not have(tables, f"envs.{n}.name")]
    assert not miss, miss


def test_doctor_rows_have_texts_for_every_state_the_code_can_produce(tables: dict[str, dict[str, str]]) -> None:
    from vndub_app import doctor

    for rid in doctor.ROW_IDS:
        for key in (f"doctor.{rid}.title", f"doctor.{rid}.green", f"doctor.{rid}.yellow", f"doctor.{rid}.red"):
            assert have(tables, key), key
    for action in doctor.FIX_ACTIONS:
        assert have(tables, f"doctor.fix.{action}"), action
    for s in ("ok", "warn", "fail"):
        assert have(tables, f"doctor.summary.{s}")
    assert have(tables, "doctor.defender.settings_path")
    # every (row, status, reason) the row builders emit has its variant text (or falls back to the generic one)
    gpus = [None] + [doctor.GpuInfo("N", v, "1", c) for v in (None, 2.0, 6.0, 12.0) for c in (None, "11.8", "12.4", "12.8")]
    emitted: set[tuple[str, str, str]] = set()
    for g in gpus:
        for row in (doctor.row_gpu(g), doctor.row_driver(g), doctor.row_vram(g)):
            if row["params"].get("reason"):
                emitted.add((row["id"], row["status"], row["params"]["reason"]))
    assert {("vram", "red", "no_nvidia"), ("vram", "red", "low_vram"), ("driver", "red", "too_old"), ("driver", "yellow", "old_driver"), ("vram", "yellow", "tight")} <= emitted
    for rid, status, reason in sorted(emitted):
        assert have(tables, f"doctor.{rid}.{status}.{reason}") or have(tables, f"doctor.{rid}.{status}"), (rid, status, reason)


def placeholders_used(tables: dict[str, dict[str, str]], key: str) -> set[str]:
    return set(PLACEHOLDER.findall(tables["ru"][key]))


def message_calls(module: str) -> list[tuple[str, set[str]]]:
    out = []
    for node in ast.walk(source(module)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "message" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                out.append((a0.value, {k.arg for k in node.keywords if k.arg}))
            elif isinstance(a0, ast.IfExp):  # ctx.message("a" if x else "b", count=n)
                for branch in (a0.body, a0.orelse):
                    if isinstance(branch, ast.Constant):
                        out.append((branch.value, {k.arg for k in node.keywords if k.arg}))
    return out


def test_job_messages_exist_and_their_placeholders_match_the_params(tables: dict[str, dict[str, str]]) -> None:
    calls = [c for m in MODULES for c in message_calls(m)]
    assert len(calls) >= 12
    for key, kwargs in calls:
        assert have(tables, key), f"missing job message {key}"
        assert placeholders_used(tables, key) <= kwargs, f"{key}: text uses {placeholders_used(tables, key)}, call passes {kwargs}"


def test_error_codes_raised_by_the_modules_have_titles_and_hints(tables: dict[str, dict[str, str]]) -> None:
    codes: set[str] = set()
    for m in MODULES:
        codes |= set(re.findall(r"AppError\(\s*\"([a-z][a-z_]+)\"", (ROOT / "vndub_app" / f"{m}.py").read_text(encoding="utf-8")))
    assert {"download_unreachable", "checksum_mismatch", "install_manual_only", "tool_install_failed", "env_install_failed", "git_missing", "models_download_failed", "disk_full"} <= codes
    for c in sorted(codes):
        assert have(tables, f"errors.{c}.title") and have(tables, f"errors.{c}.hint"), c


def test_error_titles_only_use_params_the_code_provides(tables: dict[str, dict[str, str]]) -> None:
    """e.g. errors.download_unreachable.title uses {filename}; every AppError of that code passes it."""
    provided: dict[str, list[set[str]]] = {}
    for m in MODULES:
        for node in ast.walk(source(m)):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "AppError" and node.args and isinstance(node.args[0], ast.Constant):
                for kw in node.keywords:
                    if kw.arg == "params" and isinstance(kw.value, ast.Dict):
                        provided.setdefault(node.args[0].value, []).append({k.value for k in kw.value.keys if isinstance(k, ast.Constant)})
    for code in ("download_unreachable", "checksum_mismatch", "tool_install_failed", "env_install_failed", "models_download_failed", "disk_full", "git_missing"):
        used = placeholders_used(tables, f"errors.{code}.title") | placeholders_used(tables, f"errors.{code}.hint")
        assert provided[code], code
        for params in provided[code]:
            assert used <= params, f"{code}: the text needs {used}, an AppError passes {params}"


def test_install_texts_cover_every_reason_key_the_plan_can_return(tables: dict[str, dict[str, str]]) -> None:
    text = (ROOT / "vndub_app" / "patch_install.py").read_text(encoding="utf-8")
    keys = set(re.findall(r"\"(install\.(?:manual|note)\.[a-z_]+)\"", text))
    assert {"install.manual.unity", "install.manual.repack", "install.manual.encrypted_kirikiri", "install.manual.loose_unverified", "install.manual.no_patch", "install.manual.game_missing", "install.manual.tyrano_packed", "install.manual.layout_mismatch", "install.note.unvoiced_manual"} <= keys
    for k in sorted(keys):
        assert have(tables, k), k


def test_shell_texts(tables: dict[str, dict[str, str]]) -> None:
    for k in re.findall(r"_t\(\"(shell\.[a-z.]+)\"", (ROOT / "vndub_app" / "shell.py").read_text(encoding="utf-8")):
        assert have(tables, k), k


def test_reason_codes_of_the_model_table_have_texts(tables: dict[str, dict[str, str]]) -> None:
    from vndub_app import models_api

    codes = {c for _, c in models_api._REASON_PREFIXES} | {"unknown"}
    for c in codes:
        assert have(tables, f"models.reason.{c}"), c
    for c in ("weights_missing", "env_missing", "license", "check_failed"):
        assert have(tables, f"models.selftest.code.{c}"), c
