"""Step 2 data: translation settings, glossary, delivery cues.

The choices live in ``app_state.json``; ``glossary.yaml`` (the project glossary the translate
stage reads, ``translate.glossary``) is *generated* from them and from ``characters.yaml``:

=============================== =========================================================
app choice                      pipeline
=============================== =========================================================
honorifics drop / keep / adapt  glossary ``honorifics`` drop / transliterate / map
address_default ты / вы         glossary ``address_matrix.<sid>.default`` for every character
player_gender m / f / unknown   glossary ``player.gender`` (the protagonist the characters address as "you"; unknown = not set)
n_candidates, routes, backends  ``config_overrides`` -> ``pipeline.yaml`` (synth.*)
license policy                  global ``Settings.license_policy`` -> ``pipeline.yaml``
source auto / import            ``translate.import`` (path) in ``pipeline.yaml``
tgt_lang                        ``project.tgt_lang`` and ``work/project.yaml`` (translate prefers it)
src_lang auto / code            ``project.src_lang`` and ``work/project.yaml``: auto = the language the
                                pipeline detected from the script text (``src_lang_detected`` record)
=============================== =========================================================
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml

from vndub_app import appdirs, session, viewdata
from vndub_app.errors import AppError

log = logging.getLogger("vndub_app.translation")

_TRANSLATION_LOCK = threading.Lock()  # read-modify-write of the translation file (sync_text_flags)

HONORIFICS_TO_GLOSSARY = {"drop": "drop", "keep": "transliterate", "adapt": "map"}
HONORIFICS_FROM_GLOSSARY = {v: k for k, v in HONORIFICS_TO_GLOSSARY.items()}
ADDRESS_FORMS = ("ты", "вы")
PLAYER_GENDERS = ("m", "f", "unknown")
TGT_LANGS = ("ru", "en")
SRC_LANGS = ("auto", "ja", "en", "zh", "ko", "ru", "uk", "de", "fr", "es", "it", "pt")
LICENSE_POLICIES = ("research", "commercial")
ROUTES = ("A", "B", "C")
DEFAULT_HONORIFIC_MAP = {"さん": "", "ちゃん": "", "くん": "", "様": "", "先輩": "семпай", "先生": "сэнсэй"}


def _bad(field: str, value: Any = None, **kw: Any) -> AppError:
    return AppError("bad_request", params={"field": field, "value": value, **kw}, status=422)


# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------


def llm_available(cfg: Any) -> bool:
    """Is the configured translation LLM usable (weights + llama-server tool), or a mock (app test profile)?"""
    from vndub.backends import registry
    from vndub.gpu import resolve_models_dir
    from vndub.tools import Tools

    models_dir = resolve_models_dir(cfg)
    tools = Tools.load(cfg.resolve(cfg.tools_file), cfg.tools)
    for name in [cfg.translate.llm.backend, *cfg.translate.llm.fallbacks]:
        try:
            spec = registry.get(name)
        except KeyError:
            continue
        if spec.name.startswith("mock"):
            return True
        if registry.usable(spec, policy=cfg.license_policy, models_dir=models_dir).ok and tools.resolve("llama_server") is not None:
            return True
    return False


def _backend_rows(cfg: Any) -> list[dict[str, Any]]:
    from vndub.backends import registry

    rows = []
    for name, opts in cfg.synth.backends.items():
        try:
            spec = registry.get(name)
        except KeyError:
            continue
        rows.append({
            "name": name, "enabled": bool(opts.enabled), "vram_gb": spec.vram_estimate_gb, "license": spec.license,
            "commercial_ok": bool(spec.commercial_ok), "role": spec.role,
        })
    return rows


def source_language(h: appdirs.ProjectHandle) -> dict[str, Any]:
    """``{"choice": "auto" | code, "effective": code | "", "detected": record | None}`` of the script text.

    With the choice ``auto`` the code comes from the detector (parse writes it into ``work/project.yaml`` with
    the ``src_lang_detected`` record). A project file that still holds a plain code without that record (made
    before the detector existed, or by a code chosen earlier) is detected again over the parsed lines, so an old
    project does not stay "Japanese" because that used to be the default. Before the import nothing is known.
    """
    from vndub.config import load_project_yaml, save_project_yaml
    from vndub.manifest.io import load_lines
    from vndub.text.lang import AUTO, detect_language

    st = appdirs.read_app_state(h)
    choice = str(st.get("translation", {}).get("src_lang") or AUTO)
    if choice != AUTO:
        return {"choice": choice, "effective": choice, "detected": None}
    py = h.work / "project.yaml"
    if not py.exists():
        return {"choice": choice, "effective": "", "detected": None}
    proj = load_project_yaml(py)
    rec = getattr(proj, "src_lang_detected", None)
    if rec and proj.src_lang and proj.src_lang != AUTO:
        return {"choice": choice, "effective": proj.src_lang, "detected": dict(rec)}
    paths = session.work_paths(h)
    if paths.lines.exists():
        guess = detect_language(l.text_src for l in load_lines(paths.lines))
        if guess.lang:
            proj.src_lang, proj.src_lang_detected = guess.lang, guess.as_dict()
            save_project_yaml(py, proj)
            log.info("project %s: source language %s detected (confidence %.2f)", h.slug, guess.lang, guess.confidence)
            return {"choice": choice, "effective": guess.lang, "detected": guess.as_dict()}
    return {"choice": choice, "effective": proj.src_lang if proj.src_lang != AUTO else "", "detected": None}


def get_settings(h: appdirs.ProjectHandle) -> dict[str, Any]:
    from vndub.config import load_config

    if not h.config_file.exists():
        appdirs.write_project_config(h)
    cfg = load_config(h.config_file)
    st = appdirs.read_app_state(h)
    tr = st["translation"]
    src = source_language(h)
    return {
        "tgt_lang": tr.get("tgt_lang", "ru"),
        "src_lang": src["choice"],
        "src_lang_effective": src["effective"],
        "src_lang_detected": src["detected"],
        "source": tr.get("source", "auto"),
        "import_path": tr.get("import_path", ""),
        "honorifics": tr.get("honorifics", "drop"),
        "address_default": tr.get("address_default", "ты"),
        "player_gender": tr.get("player_gender", "unknown") if tr.get("player_gender") in PLAYER_GENDERS else "unknown",  # an old app_state has no key
        "license_policy": appdirs.load_settings().license_policy,
        "llm_available": llm_available(cfg),
        "advanced": {
            "n_candidates": cfg.synth.n_candidates,
            "routes": {r: bool(cfg.synth.routes.get(r, False)) for r in ROUTES},
            "asr_check": bool((st.get("advanced") or {}).get("asr_check", False)),
            "backends": _backend_rows(cfg),
        },
    }


def put_settings(h: appdirs.ProjectHandle, body: dict[str, Any]) -> dict[str, Any]:
    with session.STATE_LOCK:  # read-modify-write of app_state.json
        return _put_settings(h, body)


def _put_settings(h: appdirs.ProjectHandle, body: dict[str, Any]) -> dict[str, Any]:
    from vndub.config import load_config

    st = appdirs.read_app_state(h)
    tr = dict(st["translation"])
    adv_state = dict(st.get("advanced") or {})
    ov = dict(st.get("config_overrides") or {})

    if "tgt_lang" in body:
        if body["tgt_lang"] not in TGT_LANGS:
            raise _bad("tgt_lang", body["tgt_lang"])
        tr["tgt_lang"] = body["tgt_lang"]
    if "src_lang" in body:
        if body["src_lang"] not in SRC_LANGS:
            raise _bad("src_lang", body["src_lang"])
        tr["src_lang"] = body["src_lang"]
    if "source" in body:
        if body["source"] not in ("auto", "import"):
            raise _bad("source", body["source"])
        tr["source"] = body["source"]
    if "import_path" in body:
        tr["import_path"] = str(body["import_path"] or "")
    if "honorifics" in body:
        if body["honorifics"] not in HONORIFICS_TO_GLOSSARY:
            raise _bad("honorifics", body["honorifics"])
        tr["honorifics"] = body["honorifics"]
    if "address_default" in body:
        if body["address_default"] not in ADDRESS_FORMS:
            raise _bad("address_default", body["address_default"])
        tr["address_default"] = body["address_default"]
    if "player_gender" in body:
        if body["player_gender"] not in PLAYER_GENDERS:
            raise _bad("player_gender", body["player_gender"])
        tr["player_gender"] = body["player_gender"]
    if "license_policy" in body:
        if body["license_policy"] not in LICENSE_POLICIES:
            raise _bad("license_policy", body["license_policy"])
        s = appdirs.load_settings()
        if s.license_policy != body["license_policy"]:
            s.license_policy = body["license_policy"]
            appdirs.save_settings(s)
        ov.pop("license_policy", None)  # one source of truth: the global setting

    adv = body.get("advanced") or {}
    if "asr_check" in adv:
        adv_state["asr_check"] = bool(adv["asr_check"])
    if "n_candidates" in adv:
        try:
            n = int(adv["n_candidates"])
        except (TypeError, ValueError):
            raise _bad("n_candidates", adv["n_candidates"]) from None
        if not 1 <= n <= 16:
            raise _bad("n_candidates", n, min=1, max=16)
        ov["synth.n_candidates"] = n
        ov["synth.n_candidates_short"] = n + 2  # keep the pipeline's default relation (4 / 6)
    if "routes" in adv:
        cur = {r: bool(load_config(h.config_file).synth.routes.get(r, False)) for r in ROUTES}
        for r, v in (adv["routes"] or {}).items():
            if r not in ROUTES:
                raise _bad("routes", r)
            cur[r] = bool(v)
        if not any(cur.values()):
            raise _bad("routes", cur, reason="none_enabled")
        for r, v in cur.items():
            ov[f"synth.routes.{r}"] = v
    if "backends" in adv:
        cfg0 = load_config(h.config_file)
        items = adv["backends"]
        pairs = items.items() if isinstance(items, dict) else [(b.get("name"), b.get("enabled")) for b in items]
        for name, enabled in pairs:
            if name not in cfg0.synth.backends:
                raise _bad("backends", name)
            ov[f"synth.backends.{name}.enabled"] = bool(enabled)

    st["translation"], st["advanced"], st["config_overrides"] = tr, adv_state, ov
    appdirs.write_app_state(h, st)
    appdirs.write_project_config(h)
    _sync_project_lang(h, tr["tgt_lang"], str(tr.get("src_lang") or "auto"))
    write_glossary(h)
    viewdata.clear_cache(h.slug)
    return get_settings(h)


def _sync_project_lang(h: appdirs.ProjectHandle, lang: str, src: str = "auto") -> None:
    """The stages prefer ``work/project.yaml`` (``tgt_lang``, ``src_lang``): keep it equal to the app choices.

    A source code chosen by the user replaces any detection; ``auto`` drops a code that was not detected so the
    next read (:func:`source_language`) or parse detects it again.
    """
    py = h.work / "project.yaml"
    if not py.exists():
        return
    from vndub.config import load_project_yaml, save_project_yaml

    proj = load_project_yaml(py)
    changed = False
    if proj.tgt_lang != lang:
        proj.tgt_lang = lang
        changed = True
    detected = getattr(proj, "src_lang_detected", None)
    if src != "auto" and (proj.src_lang != src or detected):
        proj.src_lang, proj.src_lang_detected, changed = src, None, True
    elif src == "auto" and not detected and proj.src_lang != "auto":
        proj.src_lang, changed = "auto", True
    if changed:
        save_project_yaml(py, proj)


# ----------------------------------------------------------------------------
# edited text -> flags
# ----------------------------------------------------------------------------

LAUGHTER_FLAGS = ("laughter", "interjection")


def retag_flags(flags: list[str], text: str) -> list[str]:
    """The translation record's ``flags`` after its dialogue text became ``text``: ``laughter`` / ``interjection`` follow the new
    text (``vndub.stages.translate.laughter_flags``), every other flag stays.

    ``interjection`` is kept when the old flags show it did not come from a laugh (a kana interjection of the source text carries
    it without ``laughter``): only a laughter-derived ``interjection`` is recomputed. Order and duplicates of the kept flags are
    unchanged; the new ones are appended.
    """
    from vndub.stages.translate import laughter_flags

    was_laugh = "laughter" in flags
    kept = [f for f in flags if f != "laughter" and not (f == "interjection" and was_laugh)]
    return kept + [f for f in laughter_flags(text) if f not in kept]


def sync_text_flags(h: appdirs.ProjectHandle, line_id: str, text: str) -> bool:
    """Make the ``laughter`` / ``interjection`` flags of a line's translation record follow the text the user wrote for it.

    The score gate (``score.interjection_cer_max``, no CER check of a laughter-only line) and the number of takes (``synth``) read
    these flags from the record, not from the override that carries the user's text, so a line the user turned into laughter (or out
    of it) would otherwise be judged by its old text. Returns whether the record changed. A line without a record is left alone.
    """
    from vndub.manifest.io import TranslationRecord, read_jsonl, write_jsonl

    ctx = session.read_ctx(h)
    path = ctx.paths.translation(ctx.cfg.project.tgt_lang)
    with _TRANSLATION_LOCK:
        if not path.exists():
            return False
        recs = read_jsonl(path, TranslationRecord)
        changed = False
        for r in recs:
            if r.line_id == line_id:
                new = retag_flags(list(r.flags or []), text)
                if new != list(r.flags or []):
                    r.flags, changed = new, True
                break
        if changed:
            write_jsonl(path, recs)  # atomic replace
            log.info("line %s: flags follow the edited text -> %s", line_id, ", ".join(f for f in new if f in LAUGHTER_FLAGS) or "no laughter")
    if changed:
        viewdata.clear_cache(h.slug)
    return changed


# ----------------------------------------------------------------------------
# glossary
# ----------------------------------------------------------------------------


def _read_glossary(h: appdirs.ProjectHandle) -> dict[str, Any]:
    if not h.glossary_file.exists():
        return {}
    try:
        d = yaml.safe_load(h.glossary_file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return d if isinstance(d, dict) else {}


def _example_honorific_map() -> dict[str, str]:
    p = appdirs.resource("configs", "glossary.example.yaml")
    try:
        return dict((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("honorific_map") or DEFAULT_HONORIFIC_MAP)
    except (OSError, yaml.YAMLError):
        return dict(DEFAULT_HONORIFIC_MAP)


def write_glossary(h: appdirs.ProjectHandle, terms: list[dict[str, Any]] | None = None) -> Path:
    """(Re)generate ``glossary.yaml`` from ``characters.yaml`` + the app choices (free terms are kept)."""
    from vndub.manifest.io import load_characters

    st = appdirs.read_app_state(h)
    tr = st["translation"]
    old = _read_glossary(h)
    g: dict[str, Any] = dict(old)
    g["target_lang"] = tr.get("tgt_lang", "ru")
    g["honorifics"] = HONORIFICS_TO_GLOSSARY.get(tr.get("honorifics", "drop"), "drop")
    g.setdefault("honorific_map", _example_honorific_map())
    default_addr = tr.get("address_default", "ты")
    chars_path = session.work_paths(h).characters
    chars = load_characters(chars_path) if chars_path.exists() else {}
    matrix = {k: dict(v or {}) for k, v in (old.get("address_matrix") or {}).items()}
    gchars = dict(old.get("characters") or {})
    for sid, c in chars.items():
        if sid in (viewdata.NARRATOR, viewdata.VOICE_OVER):  # no name to translate or address
            continue
        matrix.setdefault(sid, {})["default"] = default_addr
        prev = dict(gchars.get(sid) or {})
        gchars[sid] = {
            **prev, "src": c.names.get("src", "") or prev.get("src", sid), "tgt": c.names.get("tgt", ""),
            "gender": c.gender, "age_hint": c.age_hint, "notes": c.notes or c.description or prev.get("notes", ""),
        }
    g["address_matrix"] = matrix
    g["characters"] = gchars
    pg = tr.get("player_gender", "unknown")
    old_player = old.get("player") if isinstance(old.get("player"), dict) else {}
    g["player"] = {"name": "", "notes": "", **old_player, "gender": pg if pg in PLAYER_GENDERS else "unknown"}  # name / notes stay hand-editable in the file
    if terms is not None:
        g["terms"] = terms
    g.setdefault("terms", [])
    text = yaml.safe_dump(g, allow_unicode=True, sort_keys=False)
    if not h.glossary_file.exists() or h.glossary_file.read_text(encoding="utf-8") != text:
        h.glossary_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = h.glossary_file.with_suffix(".yaml.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(h.glossary_file)
    return h.glossary_file


def get_glossary(h: appdirs.ProjectHandle) -> dict[str, Any]:
    from vndub.manifest.io import load_characters

    chars_path = session.work_paths(h).characters
    chars = load_characters(chars_path) if chars_path.exists() else {}
    g = _read_glossary(h)
    gchars = g.get("characters") or {}
    out_chars = []
    for sid, c in chars.items():
        if sid in (viewdata.NARRATOR, viewdata.VOICE_OVER):
            continue
        out_chars.append({
            "speaker_id": sid, "src": c.names.get("src", "") or (gchars.get(sid) or {}).get("src", sid),
            "tgt": c.names.get("tgt", "") or (gchars.get(sid) or {}).get("tgt", ""), "gender": c.gender,
        })
    terms = [{"src": str(t.get("src", "")), "tgt": str(t.get("tgt", "")), "note": str(t.get("note") or t.get("notes") or "")} for t in (g.get("terms") or []) if isinstance(t, dict)]
    return {"characters": out_chars, "terms": terms}


def put_glossary(h: appdirs.ProjectHandle, body: dict[str, Any]) -> dict[str, Any]:
    from vndub_app import casting

    for c in body.get("characters") or []:
        if c.get("gender", "unknown") not in casting.GENDERS:
            raise _bad("gender", c.get("gender"))
    terms = []
    for t in body.get("terms") or []:
        src = str(t.get("src", "")).strip()
        if not src:
            continue
        terms.append({"src": src, "tgt": str(t.get("tgt", "")).strip(), **({"notes": str(t["note"])} if t.get("note") else {})})
    if body.get("characters"):
        with casting.edit_characters(h) as chars:
            for c in body["characters"]:
                sid = c.get("speaker_id")
                ch = chars.get(sid)
                if ch is None:
                    raise AppError("bad_request", params={"field": "speaker_id", "value": sid}, status=422)
                ch.names = {**ch.names, "tgt": str(c.get("tgt", ch.names.get("tgt", "")))}
                if "src" in c and str(c["src"]).strip():
                    ch.names["src"] = str(c["src"])
                ch.gender = c.get("gender", ch.gender)
    write_glossary(h, terms if "terms" in body else None)
    return get_glossary(h)


# ----------------------------------------------------------------------------
# delivery cues (text_emotion)
# ----------------------------------------------------------------------------


def cues_path(h: appdirs.ProjectHandle) -> Path:
    """The project's own copy if the user edited the table, else the bundled one."""
    own = h.root / "delivery_cues.yaml"
    if own.exists():
        return own
    from vndub.config import load_config

    return Path(load_config(h.config_file).text_emotion.cues) if h.config_file.exists() else appdirs.resource("configs", "delivery_cues.yaml")


def get_cues(h: appdirs.ProjectHandle) -> dict[str, Any]:
    p = cues_path(h)
    try:
        from vndub.stages import text_emotion

        fn = getattr(text_emotion, "load_cues", None)
        if fn is not None:
            return dict(fn(p))
    except ImportError:
        pass
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def put_cues(h: appdirs.ProjectHandle, data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise _bad("body", None)
    own = h.root / "delivery_cues.yaml"
    try:
        from vndub.stages import text_emotion

        fn = getattr(text_emotion, "save_cues", None)
    except ImportError:
        fn = None
    try:
        if fn is not None:
            fn(own, data)  # validates the table; the message names the offending entry
        else:
            own.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except ValueError as e:
        raise AppError("bad_request", params={"field": "cues", "reason": str(e)}, details=str(e), status=422) from e
    with session.STATE_LOCK:
        st = appdirs.read_app_state(h)
        st["config_overrides"] = {**(st.get("config_overrides") or {}), "text_emotion.cues": str(own)}
        appdirs.write_app_state(h, st)
        appdirs.write_project_config(h)
    return get_cues(h)
