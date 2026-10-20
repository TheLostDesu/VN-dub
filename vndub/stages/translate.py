"""translate: target-language text for every voiced line (+ sustain continuations).

Sources, in priority order:

1. ``translate.import`` -- an existing translation (VNTextPatch JSON/XLSX folder,
   or a JSONL/CSV/TSV mapping ``line_id -> text``); the LLM is not started.
2. Interjection lexicon (``configs/interjections.yaml``, target ru): short
   interjection lines get their natural equivalent directly.
3. Local LLM (llama.cpp ``llama-server``; test profile: MockTranslator): per
   scene, batches of <= ``batch_lines`` sent as JSON
   ``[{id, speaker, gender, emotion, text}]`` with a rolling window of the
   previous translated lines, a scene summary and the glossary in the system
   prompt; the answer must be ``{"lines": [{id, <tgt>, <tgt>_short, notes}]}``.
   Invalid / incomplete answers are retried at temperature 0, then the batch is
   halved; lines that still fail keep the source text and are flagged
   ``translate_failed``.

Post-processing (Russian): numbers -> words, abbreviations, ruaccent stress
(``text_tts``; ``text_tts_plus`` with '+' marks when a backend needs them) and
the stress-free CER reference ``text_plain``.

``--seed-glossary`` writes/extends ``configs/glossary.yaml`` from characters.yaml.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from vndub.backends import registry
from vndub.backends.base import BackendUnavailable
from vndub.manifest.io import Character, LineRecord, TranslationRecord, read_json, read_jsonl, write_jsonl, write_yaml
from vndub.stages._opcache import OpCache
from vndub.stages.common import StageContext, StageError
from vndub.text import ja, ru
from vndub.util import hash_obj

NAME = "translate"
VERSION = "1"
CONFIG_SECTIONS = ("translate",)

LANG_NAMES = {"ru": "Russian", "en": "English", "de": "German", "fr": "French", "es": "Spanish", "zh": "Chinese", "ko": "Korean", "it": "Italian", "pt": "Portuguese"}

DEFAULT_RULES = [
    "Keep the meaning, register and emotional tone of each line.",
    "Interjections, laughter and hesitations must sound natural in the target language, not literal.",
    "Do not add explanations, notes or quotation marks into the translated text.",
    "Keep character names exactly as listed in the glossary.",
]


# ----------------------------------------------------------------------------
# glossary
# ----------------------------------------------------------------------------


def glossary_path(ctx: StageContext) -> Path:
    p = Path(ctx.cfg.translate.glossary)
    return p if p.is_absolute() else ctx.cfg.base_dir() / p


def load_glossary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def seed_glossary(ctx: StageContext) -> dict[str, Any]:
    path = glossary_path(ctx)
    g = load_glossary(path)
    g.setdefault("target_lang", ctx.cfg.project.tgt_lang)
    g.setdefault("honorifics", "drop")
    g.setdefault("honorific_map", {"さん": "", "ちゃん": "", "くん": "", "様": "", "先輩": "семпай", "先生": "сэнсэй"})
    g.setdefault("address_matrix", {})
    chars_g = g.setdefault("characters", {}) or {}
    g["characters"] = chars_g
    g.setdefault("terms", [])
    g.setdefault("style_rules", list(DEFAULT_RULES))
    added = 0
    for sid, c in ctx.characters().items():
        if sid == "narrator" or sid in chars_g:
            continue
        chars_g[sid] = {
            "src": c.names.get("src", sid),
            "tgt": c.names.get("tgt", ""),
            "gender": c.gender,
            "age_hint": c.age_hint,
            "notes": c.notes,
        }
        added += 1
    write_yaml(path, g)
    return {"glossary": str(path), "characters_added": added, "characters_total": len(chars_g)}


def _gender_of(sid: str, chars: dict[str, Character], gloss: dict[str, Any]) -> str:
    gc = (gloss.get("characters") or {}).get(sid) or {}
    g = gc.get("gender") or (chars[sid].gender if sid in chars else "unknown")
    return g if g in ("m", "f") else "unknown"


def _display_name(sid: str, l: LineRecord, chars: dict[str, Character], gloss: dict[str, Any]) -> str:
    if sid == "narrator":
        return "narrator"
    gc = (gloss.get("characters") or {}).get(sid) or {}
    if gc.get("tgt"):
        return str(gc["tgt"])
    c = chars.get(sid)
    if c and c.names.get("tgt"):
        return c.names["tgt"]
    return (c.names.get("src") if c else None) or l.speaker_raw or sid


def glossary_hits(text: str, gloss: dict[str, Any], chars: dict[str, Character]) -> list[str]:
    hits: list[str] = []
    for sid, gc in (gloss.get("characters") or {}).items():
        src = str((gc or {}).get("src") or "")
        if src and src in text:
            hits.append(src)
    for sid, c in chars.items():
        src = c.names.get("src", "")
        if src and src in text and src not in hits:
            hits.append(src)
    for t in gloss.get("terms") or []:
        src = str((t or {}).get("src") or "")
        if src and src in text and src not in hits:
            hits.append(src)
    return hits


def system_prompt(tgt: str, batch: list[dict[str, Any]], speakers: list[str], gloss: dict[str, Any], chars: dict[str, Character], short_words: int) -> str:
    lang = LANG_NAMES.get(tgt, tgt)
    text_all = " ".join(b["text"] for b in batch)
    rules = list(gloss.get("style_rules") or DEFAULT_RULES)
    out = [
        f"You are a professional game localizer translating a Japanese visual novel into {lang}.",
        "Rules:",
        *[f"- {r}" for r in rules],
        "- The speaker's gender decides grammatical agreement (past tense, adjectives) when the speaker talks about themself.",
        f"- For each line give '{tgt}': the full translation; '{tgt}_short': a version at least 20% shorter when the translation has more than {short_words} words (otherwise repeat the full translation); 'notes': a delivery hint of at most 6 words in English (for example: teasing, rising at the end).",
        f"- Answer with JSON only: {{\"lines\": [{{\"id\": \"...\", \"{tgt}\": \"...\", \"{tgt}_short\": \"...\", \"notes\": \"...\"}}]}} containing exactly one entry for every input id.",
    ]
    hon = gloss.get("honorifics")
    if hon:
        policy = {"drop": "drop Japanese honorifics (-san, -chan, -kun)", "transliterate": "transliterate honorifics (-сан, -тян, -кун)", "map": "render honorifics with the mapping below"}.get(str(hon), str(hon))
        out.append(f"- Honorifics: {policy}.")
        hmap = gloss.get("honorific_map") or {}
        used = {k: v for k, v in hmap.items() if k in text_all and v}
        if used:
            out.append("  Honorific mapping: " + "; ".join(f"{k} -> {v}" for k, v in used.items()))
    matrix = gloss.get("address_matrix") or {}
    addr = [f"{sid}: " + ", ".join(f"to {k} '{v}'" for k, v in (matrix.get(sid) or {}).items()) for sid in speakers if matrix.get(sid)]
    if addr:
        out.append("- Address forms (speaker: listener -> form):")
        out += [f"  {a}" for a in addr]
    names = []
    gchars = gloss.get("characters") or {}
    for sid in sorted(set(speakers) | {sid for sid, gc in gchars.items() if str((gc or {}).get("src") or "") and str(gc["src"]) in text_all}):
        if sid == "narrator":
            continue
        gc = gchars.get(sid) or {}
        c = chars.get(sid)
        src = gc.get("src") or (c.names.get("src") if c else sid)
        tgt_name = gc.get("tgt") or (c.names.get("tgt") if c else "")
        gender = _gender_of(sid, chars, gloss)
        note = gc.get("notes") or (c.notes if c else "")
        names.append(f"  - {src} -> {tgt_name or '(transliterate)'} [id {sid}, gender {gender}]" + (f": {note}" if note else ""))
    if names:
        out.append("Characters:")
        out += names
    terms = [t for t in (gloss.get("terms") or []) if t and str(t.get("src", "")) and str(t["src"]) in text_all]
    if terms:
        out.append("Terms:")
        out += [f"  - {t['src']} -> {t.get('tgt', '')}" + (f" ({t['notes']})" if t.get("notes") else "") for t in terms]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# LLM protocol
# ----------------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def parse_response(content: str, batch_ids: list[str], tgt: str) -> dict[str, dict[str, str]]:
    """Valid entries of a model answer keyed by id (invalid/missing ids are omitted)."""
    if not isinstance(content, str) or not content.strip():
        return {}
    text = _FENCE.sub("", content.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    rows = data.get("lines") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return {}
    ids = set(batch_ids)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id", ""))
        full = r.get(tgt)
        if rid not in ids or not isinstance(full, str) or not full.strip():
            continue
        short = r.get(f"{tgt}_short")
        notes = r.get("notes")
        out[rid] = {
            "text_tgt": full.strip(),
            "text_tgt_short": short.strip() if isinstance(short, str) and short.strip() else full.strip(),
            "notes": " ".join(notes.split()[:6]) if isinstance(notes, str) else "",
        }
    return out


class LlmTranslator:
    def __init__(self, ctx: StageContext, backend: Any, backend_name: str, tgt: str, gloss: dict[str, Any], chars: dict[str, Character]) -> None:
        self.ctx = ctx
        self.backend = backend
        self.name = backend_name
        self.tgt = tgt
        self.gloss = gloss
        self.chars = chars
        self.t = ctx.cfg.translate
        self.calls = 0
        self.failures = 0

    def _extra(self) -> dict[str, Any]:
        # MockTranslator needs the target language code; llama-server derives it from the prompt
        return {"tgt": self.tgt} if self.name.startswith("mock") else {}

    def messages(self, batch: list[dict[str, Any]], context: list[dict[str, Any]], summary: str) -> list[dict[str, str]]:
        speakers = sorted({b["speaker_id"] for b in batch})
        visible = [{k: v for k, v in b.items() if k != "speaker_id"} for b in batch]
        msgs = [{"role": "system", "content": system_prompt(self.tgt, batch, speakers, self.gloss, self.chars, self.t.short_variant_min_words)}]
        if summary or context:
            ctx_text = (f"Scene summary: {summary}\n" if summary else "") + (
                "Previously translated lines, for context only:\n" + json.dumps(context, ensure_ascii=False) if context else ""
            )
            msgs.append({"role": "user", "content": ctx_text.strip()})
            msgs.append({"role": "assistant", "content": "Understood."})
        msgs.append({"role": "user", "content": "Translate these lines and answer with JSON only:\n" + json.dumps(visible, ensure_ascii=False)})
        return msgs

    def attempt(self, batch: list[dict[str, Any]], context: list[dict[str, Any]], summary: str, temperature: float) -> dict[str, dict[str, str]]:
        self.calls += 1
        try:
            content = self.backend.chat(
                self.messages(batch, context, summary),
                temperature=temperature,
                max_tokens=max(512, 160 * len(batch)),
                json_mode=True,
                seed=1234,
                **self._extra(),
            )
        except BackendUnavailable:
            raise
        except Exception as e:  # adapter/http errors count as an invalid answer
            self.ctx.log.warning("llm call failed (%s): %s", self.name, e)
            return {}
        got = parse_response(content, [b["id"] for b in batch], self.tgt)
        if len(got) < len(batch):
            self.failures += 1
            self.ctx.log.info("llm answer covered %d/%d lines (T=%.1f)", len(got), len(batch), temperature)
        return got

    def translate(self, batch: list[dict[str, Any]], context: list[dict[str, Any]], summary: str) -> dict[str, dict[str, str]]:
        got = self.attempt(batch, context, summary, self.t.llm.temperature)
        for _ in range(max(1, self.t.llm.max_retries - 1)):
            missing = [b for b in batch if b["id"] not in got]
            if not missing:
                return got
            got.update(self.attempt(missing, context, summary, 0.0))
        missing = [b for b in batch if b["id"] not in got]
        if missing and len(missing) > 1:
            half = len(missing) // 2
            for part in (missing[:half], missing[half:]):
                got.update(self.translate(part, context, summary))
        return got

    def summarize(self, scene: str, rows: list[dict[str, Any]]) -> str:
        if self.name.startswith("mock"):
            return derived_summary(scene, rows)
        msgs = [
            {"role": "system", "content": "Summarize a visual novel scene in two short English sentences. Answer with JSON only: {\"summary\": \"...\"}."},
            {"role": "user", "content": "\n".join(f"{r['speaker']}: {r['text']}" for r in rows[:80])},
        ]
        try:
            data = json.loads(_FENCE.sub("", self.backend.chat(msgs, temperature=0.0, max_tokens=200, json_mode=True, seed=1234, **self._extra())))
            s = data.get("summary") if isinstance(data, dict) else None
            if isinstance(s, str) and s.strip():
                return s.strip()
        except BackendUnavailable:
            raise
        except Exception as e:
            self.ctx.log.info("scene summary failed for %s: %s", scene, e)
        return derived_summary(scene, rows)


def derived_summary(scene: str, rows: list[dict[str, Any]]) -> str:
    speakers = list(dict.fromkeys(r["speaker"] for r in rows if r["speaker"] != "narrator"))
    first = rows[0]["text"] if rows else ""
    return f"Scene {scene}: dialogue between {', '.join(speakers) or 'unnamed speakers'} ({len(rows)} lines). It opens with: {first[:60]}"


# ----------------------------------------------------------------------------
# import
# ----------------------------------------------------------------------------

_SRC_KEYS = ("original", "source", "src", "ja", "text_src", "japanese")
_ID_KEYS = ("line_id", "id")


def _tgt_keys(tgt: str) -> tuple[str, ...]:
    return ("text_tgt", tgt, "translation", "translated", "target", "text", "message")


def _map_by_source(entries: list[tuple[str, str]], lines: list[LineRecord]) -> dict[str, str]:
    """(source, target) pairs -> line_id -> target, first unused line with equal normalized source."""
    pool: dict[str, list[str]] = {}
    for l in lines:
        pool.setdefault(ja.normalize_for_match(l.text_src), []).append(l.line_id)
    out: dict[str, str] = {}
    for src, tgt_text in entries:
        ids = pool.get(ja.normalize_for_match(src))
        if ids:
            out[ids.pop(0)] = tgt_text
    return out


def _import_json_file(p: Path, lines: list[LineRecord], tgt: str, log: Any) -> dict[str, str]:
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        if all(isinstance(v, str) for v in data.values()):
            return {str(k): v for k, v in data.items()}  # {line_id: text}
        data = data.get("lines") or data.get("messages") or []
    if not isinstance(data, list):
        return {}
    rows = [r for r in data if isinstance(r, dict)]
    tkeys = _tgt_keys(tgt)

    def tgt_of(r: dict[str, Any]) -> str | None:
        for k in tkeys:
            if isinstance(r.get(k), str):
                return r[k]
        return None

    if rows and any(k in rows[0] for k in _ID_KEYS):
        out = {}
        for r in rows:
            rid = next((str(r[k]) for k in _ID_KEYS if k in r), None)
            t = tgt_of(r)
            if rid and t is not None:
                out[rid] = t
        return out
    if rows and any(k in rows[0] for k in _SRC_KEYS):
        pairs = []
        for r in rows:
            s = next((r[k] for k in _SRC_KEYS if isinstance(r.get(k), str)), None)
            t = tgt_of({k: v for k, v in r.items() if k not in _SRC_KEYS})
            if s is not None and t is not None:
                pairs.append((s, t))
        return _map_by_source(pairs, lines)
    # VNTextPatch translated file: positional within the script file of the same stem
    stem = p.stem.lower()
    cands = [l for l in lines if (l.source_file and Path(l.source_file).stem.lower() == stem) or l.scene.lower() == stem]
    msgs = [r.get("message") for r in rows if isinstance(r.get("message"), str)]
    if len(cands) != len(msgs):
        log.warning("import %s: %d messages vs %d script lines; mapping the common prefix", p.name, len(msgs), len(cands))
    return {l.line_id: m for l, m in zip(cands, msgs)}


def _import_xlsx(p: Path, lines: list[LineRecord], tgt: str) -> dict[str, str]:
    import openpyxl

    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    out: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        header = [str(c or "").strip().lower() for c in rows[0]]

        def col(pred: Any) -> int | None:
            return next((i for i, h in enumerate(header) if pred(h)), None)

        ci = col(lambda h: h in _ID_KEYS)
        cs = col(lambda h: any(k in h for k in _SRC_KEYS))
        ct = col(lambda h: any(k in h for k in ("translation", "translated", "target", "text_tgt", "russian")) or h == tgt)
        if ct is None:
            continue
        for r in rows[1:]:
            t = r[ct] if ct < len(r) else None
            if t is None or str(t).strip() == "":
                continue
            if ci is not None and r[ci] is not None:
                out[str(r[ci])] = str(t)
            elif cs is not None and r[cs] is not None:
                pairs.append((str(r[cs]), str(t)))
    out.update(_map_by_source(pairs, lines))
    return out


def load_import(path: Path, lines: list[LineRecord], tgt: str, log: Any) -> dict[str, str]:
    if not path.exists():
        raise StageError(f"translate.import path not found: {path}")
    files = sorted(path.iterdir()) if path.is_dir() else [path]
    out: dict[str, str] = {}
    for p in files:
        suf = p.suffix.lower()
        if suf == ".jsonl":
            for r in read_jsonl(p):
                rid = next((str(r[k]) for k in _ID_KEYS if k in r), None)
                t = next((r[k] for k in _tgt_keys(tgt) if isinstance(r.get(k), str)), None)
                if rid and t is not None:
                    out[rid] = t
        elif suf == ".json":
            out.update(_import_json_file(p, lines, tgt, log))
        elif suf in (".csv", ".tsv"):
            delim = "\t" if suf == ".tsv" else ","
            with open(p, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f, delimiter=delim))
            if rows and rows[0] and rows[0][0].strip().lower() in _ID_KEYS:
                hdr = [h.strip().lower() for h in rows[0]]
                ti = next((i for i, h in enumerate(hdr) if h in _tgt_keys(tgt)), 1)
                rows = [[r[0], r[ti] if ti < len(r) else ""] for r in rows[1:]]
            for r in rows:
                if len(r) >= 2 and r[0].strip():
                    out[r[0].strip()] = r[1]
        elif suf == ".xlsx":
            out.update(_import_xlsx(p, lines, tgt))
    return out


# ----------------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------------


def needs_plus(ctx: StageContext) -> bool:
    if ctx.cfg.translate.stress.format == "plus":
        return True
    for name in ctx.cfg.enabled_backends():
        try:
            if registry.get(name).caps.get("stress_format") == "plus":
                return True
        except KeyError:
            continue
    return False


def make_stresser(ctx: StageContext, texts: list[str]) -> tuple[Any, str, str]:
    """(callable normalized->'+'-stressed, backend name, failure reason)."""
    st = ctx.cfg.translate.stress
    if st.tool in ("", "none"):
        return None, "", "stress disabled"
    try:
        spec, _ = ctx.models.resolve(st.tool)
    except BackendUnavailable as e:
        return None, "", str(e)
    store = OpCache(ctx, f"translate_stress_{spec.name}")
    memo: dict[str, str] = {}
    todo: list[tuple[str, str]] = []
    for t in dict.fromkeys(x for x in texts if x):
        key = store.key(spec.name, st.model, st.yo, t)
        hit = store.get(t, key)
        if hit is not None:
            memo[t] = hit
        else:
            todo.append((t, key))
    if todo:
        try:
            backend = ctx.models.acquire(st.tool, {"model": st.model})
            for i in range(0, len(todo), 64):
                chunk = todo[i : i + 64]
                res = backend.stress([t for t, _ in chunk], fmt="plus", yo=st.yo)
                for (t, key), s in zip(chunk, res):
                    memo[t] = s
                    store.put(t, key, s)
        except BackendUnavailable as e:
            store.save()
            return None, spec.name, str(e)
    store.save()
    return (lambda t: memo.get(t, t)), spec.name, ""


def finalize(rec: TranslationRecord, tgt: str, stresser: Any, fmt: str, plus: bool, yo: bool) -> None:
    if "translate_failed" in rec.flags or not rec.text_tgt:
        rec.text_tts = rec.text_tts_short = rec.text_plain = ""
        return
    if tgt != "ru":
        rec.text_tts = " ".join(rec.text_tgt.split())
        rec.text_tts_short = " ".join((rec.text_tgt_short or rec.text_tgt).split())
        rec.text_plain = rec.text_tts
        rec.syllables = ru.count_syllables(rec.text_tgt)
        return
    short = rec.text_tgt_short or rec.text_tgt
    rec.text_tts, rec.text_plain = ru.prepare_tts_text(rec.text_tgt, stresser, fmt, yo)
    rec.text_tts_short, plain_short = ru.prepare_tts_text(short, stresser, fmt, yo)
    extra: dict[str, Any] = {"text_plain_short": plain_short}
    if plus:
        extra["text_tts_plus"] = ru.prepare_tts_text(rec.text_tgt, stresser, "plus", yo)[0]
        extra["text_tts_short_plus"] = ru.prepare_tts_text(short, stresser, "plus", yo)[0]
    for k, v in extra.items():
        setattr(rec, k, v)
    rec.syllables = ru.count_syllables(rec.text_plain)
    if stresser is None and "no_stress" not in rec.flags:
        rec.flags.append("no_stress")


# ----------------------------------------------------------------------------
# stage
# ----------------------------------------------------------------------------


def lines_to_translate(ctx: StageContext) -> list[LineRecord]:
    voiced = {l.line_id for l in ctx.voiced_lines(apply_only=False)}
    all_lines = ctx.lines()
    out = [l for l in all_lines if l.line_id in voiced or "voice_sustain" in l.flags]
    return ctx.select(out, voiced_only=False)


def _acquire_llm(ctx: StageContext) -> tuple[Any, str]:
    llm = ctx.cfg.translate.llm
    primary = llm.backend
    if ctx.extra.get("real") and primary.startswith("mock"):
        primary = llm.real_backend
    options: dict[str, Any] = {"ctx": llm.ctx, "n_gpu_layers": llm.n_gpu_layers, "temperature": llm.temperature, "port": llm.port}
    if ctx.cfg.translate.mock_dictionary:
        options["dictionary"] = str(ctx.cfg.resolve(ctx.cfg.translate.mock_dictionary))
    reasons = []
    for name in [primary, *llm.fallbacks]:
        try:
            spec, _ = ctx.models.resolve(name)
            return ctx.models.acquire(name, options), spec.name
        except BackendUnavailable as e:
            reasons.append(str(e))
    raise StageError("no translation backend usable: " + "; ".join(reasons))


def run(ctx: StageContext) -> dict[str, Any]:
    t = ctx.cfg.translate
    if ctx.extra.get("seed_glossary"):
        return seed_glossary(ctx)
    tgt = (ctx.project.tgt_lang if ctx.project else None) or ctx.cfg.project.tgt_lang
    lines = lines_to_translate(ctx)
    chars = ctx.characters()
    gloss = load_glossary(glossary_path(ctx))
    lexicon = ja.load_interjections(ctx.cfg.resolve(t.interjections)) if tgt == "ru" else {}
    max_morae = ctx.cfg.synth.interjection_max_syllables

    entries: dict[str, dict[str, Any]] = {}
    flags: dict[str, list[str]] = {l.line_id: [] for l in lines}
    backend_name = ""

    # 1. interjections
    for l in lines:
        if ja.is_interjection(l.text_src, lexicon, max_morae):
            flags[l.line_id].append("interjection")
        mapped = ja.match_interjection(l.text_src, lexicon, max_morae) if lexicon else None
        if mapped:
            entries[l.line_id] = {"text_tgt": mapped, "text_tgt_short": mapped, "notes": "", "source": "interjection"}

    # 2. import or LLM
    if t.import_path:
        imported = load_import(ctx.cfg.resolve(t.import_path), ctx.lines(), tgt, ctx.log)
        for l in lines:
            if l.line_id in imported and str(imported[l.line_id]).strip():
                txt = str(imported[l.line_id]).strip()
                entries[l.line_id] = {"text_tgt": txt, "text_tgt_short": txt, "notes": "", "source": "import"}
        backend_name = "import"
    else:
        todo = [l for l in lines if l.line_id not in entries]
        if todo:
            backend_name = translate_llm(ctx, todo, entries, flags, tgt, gloss, chars)

    # 3. records + post-processing
    plus = needs_plus(ctx) and tgt == "ru"
    fmt = "plus" if t.stress.format == "plus" else "plain"
    recs: list[TranslationRecord] = []
    for l in lines:
        e = entries.get(l.line_id)
        fl = flags[l.line_id]
        if e is None:
            if t.import_path:
                e = {"text_tgt": "", "text_tgt_short": "", "notes": "", "source": "import"}
                fl.append("translate_missing")
            else:
                e = {"text_tgt": l.text_src, "text_tgt_short": l.text_src, "notes": "", "source": f"llm:{backend_name}"}
                fl.append("translate_failed")
                ctx.log.error("translation failed for %s: source text kept", l.line_id)
        recs.append(
            TranslationRecord(
                line_id=l.line_id,
                text_tgt=e["text_tgt"],
                text_tgt_short=e.get("text_tgt_short") or e["text_tgt"],
                notes=e.get("notes", ""),
                source=e.get("source", ""),
                glossary_hits=glossary_hits(l.text_src, gloss, chars),
                flags=list(dict.fromkeys(fl)),
            )
        )

    stresser, stress_backend, stress_reason = (None, "", "not russian") if tgt != "ru" else make_stresser(
        ctx, [ru.normalize_for_tts(x) for r in recs if "translate_failed" not in r.flags for x in (r.text_tgt, r.text_tgt_short) if x]
    )
    if tgt == "ru" and stresser is None:
        ctx.log.warning("no stress marks: %s", stress_reason)
    for r in recs:
        finalize(r, tgt, stresser, fmt, plus, t.stress.yo)

    out_path = ctx.paths.translation(tgt)
    existing = {r.line_id: r for r in read_jsonl(out_path, TranslationRecord)} if out_path.exists() else {}
    existing.update({r.line_id: r for r in recs})
    order = {l.line_id: i for i, l in enumerate(ctx.lines())}
    write_jsonl(out_path, sorted(existing.values(), key=lambda r: order.get(r.line_id, 1 << 30)))

    src_counts = Counter(r.source.split(":")[0] for r in recs)
    return {
        "lines": len(recs),
        "sources": dict(src_counts),
        "backend": backend_name or "-",
        "interjections": sum(1 for r in recs if "interjection" in r.flags),
        "failed": sum(1 for r in recs if "translate_failed" in r.flags),
        "missing": sum(1 for r in recs if "translate_missing" in r.flags),
        "stress": stress_backend or f"none ({stress_reason})",
        "plus_variant": plus,
        "glossary": str(glossary_path(ctx)) if gloss else "none",
        "out": str(out_path),
    }


def translate_llm(
    ctx: StageContext,
    todo: list[LineRecord],
    entries: dict[str, dict[str, Any]],
    flags: dict[str, list[str]],
    tgt: str,
    gloss: dict[str, Any],
    chars: dict[str, Character],
) -> str:
    t = ctx.cfg.translate
    store = OpCache(ctx, "translate_llm")
    gloss_hash = hash_obj(gloss)
    scenes: dict[str, list[LineRecord]] = {}
    for l in todo:
        scenes.setdefault(l.scene, []).append(l)

    def row(l: LineRecord) -> dict[str, Any]:
        style = (read_json(ctx.paths.passport(l.line_id)) or {}).get("style") or {}
        return {
            "id": l.line_id,
            "speaker": _display_name(l.speaker_id, l, chars, gloss),
            "speaker_id": l.speaker_id,
            "gender": _gender_of(l.speaker_id, chars, gloss),
            "emotion": style.get("emotion", ""),
            "text": l.text_src,
        }

    translator: LlmTranslator | None = None
    backend_name = ""

    def get_translator() -> LlmTranslator:
        nonlocal translator, backend_name
        if translator is None:
            backend, backend_name = _acquire_llm(ctx)
            translator = LlmTranslator(ctx, backend, backend_name, tgt, gloss, chars)
        return translator

    # the backend name decides the cache key: resolve without loading
    llm = t.llm
    primary = llm.real_backend if (ctx.extra.get("real") and llm.backend.startswith("mock")) else llm.backend
    try:
        key_backend = ctx.models.resolve(primary)[0].name
    except BackendUnavailable:
        key_backend = primary

    for scene, scene_lines in scenes.items():
        rows = [row(l) for l in scene_lines]
        summary_key = store.key("summary", key_backend, scene, [r["text"] for r in rows])
        summary = store.get(f"summary:{scene}", summary_key)
        history: list[dict[str, Any]] = []
        for i in range(0, len(rows), max(1, t.batch_lines)):
            batch = rows[i : i + t.batch_lines]
            context = history[-t.context_window_lines :] if t.context_window_lines else []
            bkey = store.key(key_backend, tgt, gloss_hash, batch, context, llm.temperature)
            item_id = f"batch:{scene}:{batch[0]['id']}:{len(batch)}"
            got = store.get(item_id, bkey)
            if got is None:
                tr = get_translator()
                if summary is None:
                    summary = tr.summarize(scene, rows)
                    store.put(f"summary:{scene}", summary_key, summary)
                got = tr.translate(batch, context, summary)
                if len(got) == len(batch):
                    store.put(item_id, bkey, got)
                store.save()
            for b in batch:
                e = got.get(b["id"])
                if e is None:
                    continue
                entries[b["id"]] = {**e, "source": f"llm:{backend_name or key_backend}"}
                history.append({"speaker": b["speaker"], "text": b["text"], tgt: e["text_tgt"]})
    store.save()
    if translator is not None:
        ctx.log.info("llm calls: %d, incomplete answers: %d", translator.calls, translator.failures)
    return backend_name or key_backend
