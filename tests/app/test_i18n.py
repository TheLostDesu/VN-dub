"""i18n completeness of the desktop app (no fixtures from conftest: pure file checks).

The UI reads two flat tables, ``app/i18n/ru.json`` (default) and ``app/i18n/en.json``. These tests
check that they stay in step with each other and with everything that needs a key: the SPA sources,
the error codes / actions / job kinds / stages / doctor rows of the contract (``vndub_app/API.md``) and
of the backend code. Other tasks keep their strings in ``app/i18n/parts/*`` until
``scripts/merge_i18n.py`` folds them in, so a failure that lists keys which exist only in a part
file is expected until that merge.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
I18N = ROOT / "app" / "i18n"
WEB_SRC = ROOT / "vndub_app" / "web" / "src"
API_MD = ROOT / "vndub_app" / "API.md"
PLACEHOLDER = re.compile(r"\{(\w+)\}")
# pictographs / emoji (the UI must stay plain text)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")


def load(lang: str) -> dict[str, str]:
    return json.loads((I18N / f"{lang}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ru() -> dict[str, str]:
    return load("ru")


@pytest.fixture(scope="module")
def en() -> dict[str, str]:
    return load("en")


def missing_report(keys: set[str] | list[str], ru: dict[str, str], en: dict[str, str], what: str) -> None:
    """Fail with the list of keys that are missing in either table."""
    miss_ru = sorted(k for k in keys if k not in ru)
    miss_en = sorted(k for k in keys if k not in en)
    if miss_ru or miss_en:
        lines = [f"{what}: keys missing from the i18n tables"]
        if miss_ru:
            lines.append(f"  ru.json ({len(miss_ru)}):")
            lines += [f"    {k}" for k in miss_ru]
        if miss_en:
            lines.append(f"  en.json ({len(miss_en)}):")
            lines += [f"    {k}" for k in miss_en]
        lines.append("  (keys that only exist in app/i18n/parts/* need `python scripts/merge_i18n.py`)")
        pytest.fail("\n".join(lines))


# ---- tables -------------------------------------------------------------------------------------


def test_tables_are_flat_string_maps(ru: dict[str, str], en: dict[str, str]) -> None:
    for name, table in (("ru", ru), ("en", en)):
        assert table, f"{name}.json is empty"
        bad = [k for k, v in table.items() if not isinstance(v, str)]
        assert not bad, f"{name}.json has non-string values: {bad[:10]}"


def test_same_key_set(ru: dict[str, str], en: dict[str, str]) -> None:
    only_ru = sorted(set(ru) - set(en))
    only_en = sorted(set(en) - set(ru))
    assert not only_ru and not only_en, f"keys only in ru.json: {only_ru}\nkeys only in en.json: {only_en}"


def test_no_empty_values(ru: dict[str, str], en: dict[str, str]) -> None:
    empty = {name: sorted(k for k, v in table.items() if not v.strip()) for name, table in (("ru", ru), ("en", en))}
    assert not empty["ru"] and not empty["en"], f"empty values: {empty}"


def test_placeholders_match(ru: dict[str, str], en: dict[str, str]) -> None:
    bad = []
    for k in sorted(set(ru) & set(en)):
        a, b = sorted(PLACEHOLDER.findall(ru[k])), sorted(PLACEHOLDER.findall(en[k]))
        if a != b:
            bad.append(f"{k}: ru {a} vs en {b}")
    assert not bad, "placeholders differ between languages:\n  " + "\n  ".join(bad)


def test_no_emoji_or_stray_markup(ru: dict[str, str], en: dict[str, str]) -> None:
    bad = []
    for name, table in (("ru", ru), ("en", en)):
        for k, v in table.items():
            if EMOJI.search(v):
                bad.append(f"{name}:{k} contains an emoji/pictograph")
            if "<" in v and ">" in v and re.search(r"</?[a-zA-Z]", v):
                bad.append(f"{name}:{k} contains HTML-like markup")
    assert not bad, "\n".join(bad)


def test_russian_table_is_russian(ru: dict[str, str]) -> None:
    """Every Russian text has Cyrillic letters, except the few that are names / file names only."""
    allowed_latin_only = {"app.name", "lang.en", "review.delivery.layer.llm"}
    latin_only = [k for k, v in ru.items() if k not in allowed_latin_only and not re.search("[А-Яа-яЁё]", v) and not re.fullmatch(r"[\W\d_]*", v)]
    # names of products and engines are legitimately Latin
    latin_only = [k for k in latin_only if not k.startswith(("engines.", "tools.", "models.", "envs."))]
    assert not latin_only, f"Russian texts without a single Cyrillic letter: {latin_only}"


# ---- every key the SPA asks for exists -----------------------------------------------------------

LITERAL_KEY = re.compile(r"""(?<![\w.$])(?:t|tk)\(\s*(["'])([^"'\n]+?)\1""")


def literal_keys_in_web_src() -> dict[str, list[str]]:
    used: dict[str, list[str]] = {}
    if not WEB_SRC.is_dir():
        return used
    for f in sorted(WEB_SRC.rglob("*")):
        if f.suffix not in (".ts", ".tsx") or f.name.endswith(".d.ts"):
            continue
        text = f.read_text(encoding="utf-8")
        for m in LITERAL_KEY.finditer(text):
            key = m.group(2)
            if key.endswith(".") or " " in key or "/" in key:
                continue  # a prefix of a dynamic key, or not a key at all
            used.setdefault(key, []).append(f.relative_to(WEB_SRC).as_posix())
    return used


def test_web_src_exists() -> None:
    assert WEB_SRC.is_dir(), "vndub_app/web/src is missing"
    assert literal_keys_in_web_src(), "no t('...') calls found in the SPA sources"


def test_every_literal_key_used_in_web_src_exists(ru: dict[str, str], en: dict[str, str]) -> None:
    used = literal_keys_in_web_src()
    missing_report(set(used), ru, en, "keys used by t('...') in vndub_app/web/src")


# ---- keys derived from the contract and the backend ---------------------------------------------


def api_text() -> str:
    return API_MD.read_text(encoding="utf-8")


def api_error_codes() -> list[str]:
    m = re.search(r"Codes used so far:(.*?)(?:\n\s*\n|\Z)", api_text(), re.S)
    assert m, "API.md section 5 no longer lists the error codes"
    codes = re.findall(r"[a-z][a-z_]+", m.group(1))
    assert len(codes) >= 20, f"suspiciously few error codes parsed: {codes}"
    return sorted(set(codes))


def backend_error_codes() -> set[str]:
    codes: set[str] = set()
    for f in (ROOT / "vndub_app").rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        codes.update(re.findall(r"AppError\(\s*\"([a-z][a-z_]+)\"", text))
    return codes


def test_error_codes_of_the_contract_have_keys(ru: dict[str, str], en: dict[str, str]) -> None:
    keys: set[str] = set()
    for code in api_error_codes():
        keys |= {f"errors.{code}.title", f"errors.{code}.hint"}
    keys |= {"errors.show_details", "errors.server_unreachable.title", "errors.server_unreachable.hint"}
    missing_report(keys, ru, en, "error codes of API.md section 5")


def test_error_codes_raised_by_the_backend_have_keys(ru: dict[str, str], en: dict[str, str]) -> None:
    keys: set[str] = set()
    for code in backend_error_codes():
        keys |= {f"errors.{code}.title", f"errors.{code}.hint"}
    missing_report(keys, ru, en, "AppError codes used in vndub_app/*.py")


def errors_actions() -> list[str]:
    tree = ast.parse((ROOT / "vndub_app" / "errors.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ACTIONS" for t in node.targets):
            return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]  # type: ignore[attr-defined]
    raise AssertionError("ACTIONS not found in vndub_app/errors.py")


def test_every_error_action_has_a_label(ru: dict[str, str], en: dict[str, str]) -> None:
    acts = errors_actions()
    assert "retry" in acts and "none" in acts
    missing_report({f"actions.{a}" for a in acts}, ru, en, "ACTIONS of vndub_app/errors.py")


def api_job_kinds() -> list[str]:
    m = re.search(r"Job kinds \(i18n `jobs\.<kind>`\):\s*(.*?)\.\s*\n", api_text(), re.S)
    assert m, "API.md section 4 no longer lists the job kinds"
    kinds = re.findall(r"[a-z][a-z_]+", m.group(1))
    assert len(kinds) >= 15, f"suspiciously few job kinds parsed: {kinds}"
    return kinds


def backend_job_kinds() -> set[str]:
    kinds: set[str] = set()
    for f in (ROOT / "vndub_app").rglob("*.py"):
        kinds.update(re.findall(r"\.submit\(\s*\"([a-z][a-z_]+)\"", f.read_text(encoding="utf-8")))
    return kinds


def test_every_job_kind_has_a_title(ru: dict[str, str], en: dict[str, str]) -> None:
    keys = {f"jobs.{k}" for k in api_job_kinds()} | {f"jobs.{k}" for k in backend_job_kinds()}
    missing_report(keys, ru, en, "job kinds (API.md section 4 and runner().submit calls)")


def pipeline_stages() -> list[str]:
    stages: set[str] = {"text_emotion"}
    tree = ast.parse((ROOT / "vndub" / "stages" / "common.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STAGES" for t in node.targets):
            stages.update(e.value for e in node.value.elts if isinstance(e, ast.Constant))  # type: ignore[attr-defined]
    m = re.search(r"STAGES order becomes:\s*`([^`]+)`", api_text())
    if m:
        stages.update(m.group(1).split())
    return sorted(stages)


def test_every_pipeline_stage_has_a_label(ru: dict[str, str], en: dict[str, str]) -> None:
    stages = pipeline_stages()
    assert "text_emotion" in stages and "synth" in stages and len(stages) >= 12
    keys = {f"stages.{s}" for s in stages} | {f"stage_noun.{s}" for s in stages}
    missing_report(keys, ru, en, "pipeline stages")


def test_doctor_rows_have_keys(ru: dict[str, str], en: dict[str, str]) -> None:
    m = re.search(r"row ids: ((?:`\w+`,?\s*)+)", api_text())
    assert m, "API.md no longer lists the doctor row ids"
    ids = re.findall(r"`(\w+)`", m.group(1))
    assert {"gpu", "vram", "ffmpeg", "models"} <= set(ids), ids
    keys = set()
    for i in ids:
        keys |= {f"doctor.{i}.title", f"doctor.{i}.green", f"doctor.{i}.yellow", f"doctor.{i}.red"}
    keys |= {f"doctor.fix.{a}" for a in ("download_ffmpeg", "install_vcredist", "open_data_dir", "open_driver_page", "open_vcredist_page", "add_defender_exclusion_help")}
    assert {"vcredist", "git"} <= set(ids), ids  # the rows for what the installer carries (Visual C++ runtime, Git)
    missing_report(keys, ru, en, "doctor rows")


def test_verdict_and_why_keys(ru: dict[str, str], en: dict[str, str]) -> None:
    text = api_text()
    m = re.search(r"Keys \(i18n `verdict\.<metric>\.<key>`\):(.*?)Mapping lives", text, re.S)
    assert m, "API.md no longer lists the verdict keys"
    keys: set[str] = set()
    metrics = re.findall(r"(\w+) `([a-z_|]+)`", m.group(1))
    assert len(metrics) >= 5, metrics
    for metric, variants in metrics:
        keys.add(f"verdict.tooltip.{metric}")
        keys |= {f"verdict.{metric}.{v}" for v in variants.split("|")}
    w = re.search(r"`why\[\]\.key` \(i18n `why\.<key>`\):(.*?)\.\s*\n", text, re.S)
    assert w, "API.md no longer lists the why keys"
    why = re.findall(r"`(\w+)`", w.group(1))
    assert {"cer", "spk_sim", "no_candidates"} <= set(why), why
    keys |= {f"why.{k}" for k in why}
    keys |= {"why.translation_failed", "why.pending_change", "why.unknown"}  # emitted by vndub_app/verdicts.py
    missing_report(keys, ru, en, "verdicts and why")


def test_wizard_steps_and_emotions_have_keys(ru: dict[str, str], en: dict[str, str]) -> None:
    tree = ast.parse((ROOT / "vndub_app" / "appdirs.py").read_text(encoding="utf-8"))
    steps: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WIZARD_STEPS" for t in node.targets):
            steps = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]  # type: ignore[attr-defined]
    assert len(steps) == 8, steps
    e2v = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
    keys = {f"steps.{s}" for s in steps} | {f"emotion.{e}" for e in [*e2v, "embarrassed"]}
    keys |= {f"route.{r}.{p}" for r in "ABC" for p in ("name", "hint")}
    keys |= {f"gender.{g}" for g in ("m", "f", "unknown")}
    keys |= {f"casting.kind.{k}" for k in ("voice_design", "reference", "clone")}
    keys |= {f"review.status.{s}" for s in ("ok", "check", "failed")}
    keys |= {f"review.delivery.layer.{s}" for s in ("rule", "cue", "llm", "neutral")}
    keys |= {f"models.preset.{p}" for p in ("min", "recommended", "all")}
    keys |= {f"install.manual.{k}" for k in ("game_missing", "no_patch", "unity", "repack", "loose_unverified", "encrypted_kirikiri")}
    keys |= {f"engines.{e}" for e in ("renpy", "kirikiri", "nscripter", "siglus", "bgi", "unity", "tyrano", "generic")}
    missing_report(keys, ru, en, "wizard steps, emotions, routes, engines")


def test_job_message_keys_used_by_the_backend_exist(ru: dict[str, str], en: dict[str, str]) -> None:
    """``ctx.message("<key>", ...)`` / ``job.message("<key>", ...)`` keys are shown by the progress panel as is."""
    keys: set[str] = set()
    for f in (ROOT / "vndub_app").rglob("*.py"):
        keys.update(re.findall(r"\.message\(\s*\"([a-z][\w.]+)\"", f.read_text(encoding="utf-8")))
    missing_report(keys, ru, en, "job messages (ctx.message calls in vndub_app/*.py)")


def test_engine_instructions_and_job_files_have_keys(ru: dict[str, str], en: dict[str, str]) -> None:
    """Step 7 explains every engine in plain words (INSTALL.md of the pack stage is English only);
    download jobs show per-file states and counters in megabytes / environments / files."""
    engines = ("renpy", "kirikiri", "nscripter", "siglus", "bgi", "catsystem2", "artemis", "tyrano", "unity", "yuris", "majiro", "generic")
    keys = {f"install.engine.{e}.steps" for e in engines} | {"install.engine.title", "install.md.title"}
    keys |= {f"job.file.{s}" for s in ("pending", "downloading", "verifying", "extracting", "installing", "done", "failed", "skipped", "verified")}
    keys |= {"job.counter_mb", "job.counter_envs", "job.counter_files"}
    keys |= {"jobs.casting_auto", "casting.auto_done", "manual.download.target", "manual.download.open_target"}
    keys |= {"models.env_needed.title", "models.env_needed.text", "models.env_needed.install", "models.env_needed.open"}
    missing_report(keys, ru, en, "engine instructions, download job rows, manual card, env_missing models")


def test_help_documents_have_titles(ru: dict[str, str], en: dict[str, str]) -> None:
    """GET /api/help answers title_key = help.<id>.title for every bundled document."""
    src = (ROOT / "vndub_app" / "routes" / "general.py").read_text(encoding="utf-8")
    ids = re.findall(r'\(\s*"(\w+)",\s*"(?:docs/\w+|README)\.md"\s*\)', src)
    assert len(ids) >= 5, ids
    missing_report({f"help.{i}.title" for i in ids}, ru, en, "bundled help documents")


# ---- the merge script ------------------------------------------------------------------------------


def load_merge_module():
    spec = importlib.util.spec_from_file_location("merge_i18n", ROOT / "scripts" / "merge_i18n.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses with postponed annotations look the module up
    spec.loader.exec_module(mod)
    return mod


def test_merge_script_adds_missing_keys_and_reports_conflicts(tmp_path: Path) -> None:
    mod = load_merge_module()
    d = tmp_path / "i18n"
    (d / "parts").mkdir(parents=True)
    (d / "ru.json").write_text(json.dumps({"a.one": "один", "a.two": "два"}, ensure_ascii=False), encoding="utf-8")
    (d / "en.json").write_text(json.dumps({"a.one": "one", "a.two": "two"}), encoding="utf-8")
    (d / "parts" / "o.ru.json").write_text(json.dumps({"a.one": "ДРУГОЙ", "b.new": "новый"}, ensure_ascii=False), encoding="utf-8")
    (d / "parts" / "o.en.json").write_text(json.dumps({"b": {"new": "new"}, "b.extra": "extra"}), encoding="utf-8")

    dry = mod.merge(d, dry_run=True)
    assert json.loads((d / "ru.json").read_text(encoding="utf-8")) == {"a.one": "один", "a.two": "два"}
    assert [(c[0], c[1]) for c in dry.conflicts] == [("ru", "a.one")]

    rep = mod.merge(d)
    ru = json.loads((d / "ru.json").read_text(encoding="utf-8"))
    en = json.loads((d / "en.json").read_text(encoding="utf-8"))
    assert ru["a.one"] == "один" and ru["b.new"] == "новый"  # main wins, new key added
    assert en["b.new"] == "new" and en["b.extra"] == "extra"  # nested part is flattened
    assert rep.only_in_one["ru"] == ["b.extra"] and rep.only_in_one["en"] == []
    assert (d / "parts" / "o.ru.json").exists()  # not deleted without --delete

    rep2 = mod.merge(d, delete=True)
    assert not (d / "parts").exists() and len(rep2.deleted) == 2


def test_merge_script_rejects_bad_files(tmp_path: Path) -> None:
    mod = load_merge_module()
    d = tmp_path / "i18n"
    (d / "parts").mkdir(parents=True)
    (d / "ru.json").write_text("{}", encoding="utf-8")
    (d / "en.json").write_text("{}", encoding="utf-8")
    (d / "parts" / "x.ru.json").write_text(json.dumps({"k": 5}), encoding="utf-8")
    with pytest.raises(mod.I18nFileError):
        mod.merge(d)
