"""Step 2 data: translation settings, glossary, delivery cues.

The choices live in ``app_state.json``; ``glossary.yaml`` (the project glossary the translate
stage reads, ``translate.glossary``) is *generated* from them and from ``characters.yaml``:

=============================== =========================================================
app choice                      pipeline
=============================== =========================================================
honorifics drop / keep / adapt  glossary ``honorifics`` drop / transliterate / map
address_default ты / вы         glossary ``address_matrix.<sid>.default`` for every character
n_candidates, routes, backends  ``config_overrides`` -> ``pipeline.yaml`` (synth.*)
license policy                  global ``Settings.license_policy`` -> ``pipeline.yaml``
source auto / import            ``translate.import`` (path) in ``pipeline.yaml``
tgt_lang                        ``project.tgt_lang`` and ``work/project.yaml`` (translate prefers it)
=============================== =========================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from vndub_app import appdirs, session, viewdata
from vndub_app.errors import AppError

log = logging.getLogger("vndub_app.translation")

HONORIFICS_TO_GLOSSARY = {"drop": "drop", "keep": "transliterate", "adapt": "map"}
HONORIFICS_FROM_GLOSSARY = {v: k for k, v in HONORIFICS_TO_GLOSSARY.items()}
ADDRESS_FORMS = ("ты", "вы")
TGT_LANGS = ("ru", "en")
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


def get_settings(h: appdirs.ProjectHandle) -> dict[str, Any]:
    from vndub.config import load_config

    if not h.config_file.exists():
        appdirs.write_project_config(h)
    cfg = load_config(h.config_file)
    st = appdirs.read_app_state(h)
    tr = st["translation"]
    return {
        "tgt_lang": tr.get("tgt_lang", "ru"),
        "source": tr.get("source", "auto"),
        "import_path": tr.get("import_path", ""),
        "honorifics": tr.get("honorifics", "drop"),
        "address_default": tr.get("address_default", "ты"),
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
    _sync_project_lang(h, tr["tgt_lang"])
    write_glossary(h)
    viewdata.clear_cache(h.slug)
    return get_settings(h)


def _sync_project_lang(h: appdirs.ProjectHandle, lang: str) -> None:
    """``translate`` prefers ``work/project.yaml: tgt_lang``; keep it equal to the app choice."""
    py = h.work / "project.yaml"
    if not py.exists():
        return
    from vndub.config import load_project_yaml, save_project_yaml

    proj = load_project_yaml(py)
    if proj.tgt_lang != lang:
        proj.tgt_lang = lang
        save_project_yaml(py, proj)


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
