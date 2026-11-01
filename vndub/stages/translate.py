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

Delivery (mode B, speaker lines without original audio -- translated like the voiced ones unless
``translate.include_unvoiced`` is ``off`` (``auto`` follows ``synth.voice_unvoiced``); narration never): for those lines the same request carries
``"want_delivery": true`` and the same answer entry gets one more field, ``delivery``
``{emotion, intensity, tempo, loudness, events, confidence}``, produced with the neighbouring script lines
(+-``text_emotion.llm_context_lines``, narration included) and the character sheet. Invalid delivery objects are
dropped (the stage ``text_emotion`` then falls back to its rule/cue layers); they never fail or re-run a batch.
The result is stored as the extra field ``TranslationRecord.delivery``. These lines are sent in their own
batches after the scene's regular batches, so the cached batches of voiced lines are never invalidated.
``text_emotion.llm: off`` disables the request; with an imported translation (``translate.import``) no LLM is
loaded for delivery unless ``text_emotion.llm: on`` (then a delivery-only request runs through the same client).

``preview_import(ctx, path, n)`` shows how an import file would match the lines (no writes).
"""

from __future__ import annotations

import csv
import json
import re
import threading
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

from vndub.backends import registry
from vndub.backends.base import BackendUnavailable
from vndub.engines.base import VOICE_OVER
from vndub.manifest.io import Character, LineRecord, TranslationRecord, read_json, read_jsonl, write_jsonl, write_yaml
from vndub.stages import text_emotion as te_stage
from vndub.stages._opcache import OpCache
from vndub.stages.common import StageContext, StageError
from vndub.text import ja, laughter, ru
from vndub.text.delivery import EMOTIONS, EVENTS, LOUDNESSES, TEMPOS, validate_delivery
from vndub.text.lang import LANG_NAMES, language_name
from vndub.util import hash_obj, now_iso

NAME = "translate"
VERSION = "1"
CONFIG_SECTIONS = ("translate", "text_emotion")
DELIVERY_PROTOCOL = "1"  # part of the cache key of delivery batches: bump when the delivery request changes

#: raw model answers that did not cover their batch are appended here (one JSON object per line) for diagnosis
ANSWER_LOG = "llm_answers.jsonl"

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
        if sid in ("narrator", VOICE_OVER) or sid in chars_g:
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


_NEUTRAL_ANSWER = '{"emotion": "neutral", "intensity": 0.3, "tempo": "normal", "loudness": "normal", "events": [], "confidence": 0.2}'


def _delivery_object_doc() -> str:
    return (
        f'{{"emotion": one of {"|".join(EMOTIONS)}, "intensity": 0..1, "tempo": {"|".join(TEMPOS)}, '
        f'"loudness": {"|".join(LOUDNESSES)}, "events": a list drawn from {"|".join(EVENTS)} (usually empty), "confidence": 0..1}}'
    )


def delivery_rules() -> list[str]:
    """Prompt lines that ask for the ``delivery`` object (only added to batches that contain target lines)."""
    return [
        '- Lines with "want_delivery": true are spoken by a character nobody voices yet. For those lines also give '
        f"'delivery': an object describing HOW the line is spoken: {_delivery_object_doc()}. 'other' means expressive without a specific emotion.",
        "- Decide the delivery from the line itself, from the neighbouring lines in 'scene_lines' (narration often says how a "
        "line is spoken: shouted, whispered, through tears ...) and from the character sheet.",
        f"- When nothing in the text or the surrounding narration indicates a delivery, answer {_NEUTRAL_ANSWER} -- low confidence. "
        "Do not invent drama.",
    ]


def _character_lines(speakers: list[str], text_all: str, gloss: dict[str, Any], chars: dict[str, Character], delivery: bool = False) -> list[str]:
    names = []
    gchars = gloss.get("characters") or {}
    for sid in sorted(set(speakers) | {sid for sid, gc in gchars.items() if str((gc or {}).get("src") or "") and str(gc["src"]) in text_all}):
        if sid in ("narrator", VOICE_OVER):  # the voice-over has no name in the text and no character sheet
            continue
        gc = gchars.get(sid) or {}
        c = chars.get(sid)
        src = gc.get("src") or (c.names.get("src") if c else sid)
        tgt_name = gc.get("tgt") or (c.names.get("tgt") if c else "")
        gender = _gender_of(sid, chars, gloss)
        note = gc.get("notes") or (c.notes if c else "")
        if delivery and c is not None:
            note = "; ".join(x for x in (note, f"age {c.age_hint}" if c.age_hint else "", c.description) if x)
        names.append(f"  - {src} -> {tgt_name or '(transliterate)'} [id {sid}, gender {gender}]" + (f": {note}" if note else ""))
    return names


def delivery_system_prompt(batch: list[dict[str, Any]], speakers: list[str], gloss: dict[str, Any], chars: dict[str, Character], src: str = "ja") -> str:
    """System prompt of the delivery-only request (imported translations + ``text_emotion.llm: on``)."""
    src_name = language_name(src)
    out = [
        f"You are an assistant for the voice-acting team of a {src_name} visual novel.",
        f"For each input line decide HOW it is spoken. The 'text' is the {src_name} line, 'translation' its (already final) translation.",
        *delivery_rules()[1:],
        f"- 'delivery' is an object {_delivery_object_doc()}.",
        '- Answer with JSON only: {"lines": [{"id": "...", "delivery": {...}}]} containing exactly one entry for every input id.',
    ]
    names = _character_lines(speakers, " ".join(b["text"] for b in batch), gloss, chars, delivery=True)
    if names:
        out.append("Characters:")
        out += names
    return "\n".join(out)


def system_prompt(tgt: str, batch: list[dict[str, Any]], speakers: list[str], gloss: dict[str, Any], chars: dict[str, Character], short_words: int, delivery: bool = False, src: str = "ja") -> str:
    lang = LANG_NAMES.get(tgt, tgt)
    text_all = " ".join(b["text"] for b in batch)
    rules = list(gloss.get("style_rules") or DEFAULT_RULES)
    out = [
        f"You are a professional game localizer translating a visual novel from {language_name(src)} into {lang}.",
        "Rules:",
        *[f"- {r}" for r in rules],
        "- The speaker's gender decides grammatical agreement (past tense, adjectives) when the speaker talks about themself.",
        f"- For each line give '{tgt}': the full translation; '{tgt}_short': a version at least 20% shorter when the translation has more than {short_words} words (otherwise repeat the full translation); 'notes': a delivery hint of at most 6 words in English (for example: teasing, rising at the end).",
        f"- Answer with JSON only: {{\"lines\": [{{\"id\": \"...\", \"{tgt}\": \"...\", \"{tgt}_short\": \"...\", \"notes\": \"...\"}}]}} containing exactly one entry for every input id.",
    ]
    if delivery:
        out[-1] = (
            f'- Answer with JSON only: {{"lines": [{{"id": "...", "{tgt}": "...", "{tgt}_short": "...", "notes": "...", '
            '"delivery": {...} (only for lines with want_delivery)}]} containing exactly one entry for every input id.'
        )
        out += delivery_rules()
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
    names = _character_lines(speakers, text_all, gloss, chars, delivery)
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


def _json_object(content: str) -> Any:
    """The JSON value of a model answer (fences / chatter around it tolerated), None when unusable."""
    if not isinstance(content, str) or not content.strip():
        return None
    text = _FENCE.sub("", content.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"[\{\[].*[\}\]]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _salvage_rows(content: str) -> list[Any]:
    """The complete objects of a ``lines`` array (or a bare array) that was cut off mid-way (token budget hit).

    Objects are decoded one by one from the first ``[`` that follows ``"lines"``; the first object that does
    not decode (the truncated one) ends the list. Nothing is guessed about the cut-off object.
    """
    if not isinstance(content, str):
        return []
    text = _FENCE.sub("", content.strip())
    m = re.search(r'"lines"\s*:\s*\[', text)
    start = m.end() if m else (text.find("[") + 1 if text.lstrip().startswith("[") else -1)
    if start <= 0:
        return []
    dec = json.JSONDecoder()
    out: list[Any] = []
    i = start
    while True:
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text) or text[i] != "{":
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        out.append(obj)
        i = end
    return out


def _json_rows(content: str) -> list[Any] | None:
    """The ``lines`` list of a model answer (fences / chatter around the JSON tolerated), None when unusable.

    A truncated answer still yields the objects that were complete (:func:`_salvage_rows`).
    """
    data = _json_object(content)
    rows = data.get("lines") if isinstance(data, dict) else data
    if isinstance(rows, list):
        return rows
    salvaged = _salvage_rows(content)
    return salvaged or None


def parse_delivery_response(content: str, batch_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Valid ``delivery`` objects of a delivery-only answer keyed by id (invalid ones are omitted)."""
    ids = set(batch_ids)
    out: dict[str, dict[str, Any]] = {}
    for r in _json_rows(content) or []:
        if isinstance(r, dict) and str(r.get("id", "")) in ids:
            dv = validate_delivery(r.get("delivery"), str(r["id"]))
            if dv:
                out[str(r["id"])] = dv
    return out


def parse_response(content: str, batch_ids: list[str], tgt: str, want_delivery: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Valid entries of a model answer keyed by id (invalid/missing ids are omitted).

    For ids in ``want_delivery`` a valid ``delivery`` object is added as ``entry["delivery"]``; an invalid or
    missing one is simply left out (it never invalidates the translation).
    """
    rows = _json_rows(content)
    if rows is None:
        return {}
    ids = set(batch_ids)
    out: dict[str, dict[str, Any]] = {}
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
        if want_delivery and rid in want_delivery:
            dv = validate_delivery(r.get("delivery"), rid)
            if dv:
                out[rid]["delivery"] = dv
    return out


class LlmTranslator:
    def __init__(self, ctx: StageContext, backend: Any, backend_name: str, tgt: str, gloss: dict[str, Any], chars: dict[str, Character], src: str = "ja") -> None:
        self.ctx = ctx
        self.backend = backend
        self.name = backend_name
        self.tgt = tgt
        self.src = src
        self.gloss = gloss
        self.chars = chars
        self.t = ctx.cfg.translate
        self.calls = 0
        self.failures = 0
        self.truncated = 0
        self.delivery_errors = 0  # delivery requests that raised (not: answers without a valid delivery)
        self._lock = threading.Lock()  # the counters are shared by the scene threads

    def _count(self, name: str) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def _last(self) -> dict[str, Any]:
        """What the backend recorded about its latest answer (llama-server: finish_reason, reasoning chars)."""
        last = getattr(self.backend, "last", None)
        return last if isinstance(last, dict) else {}

    def _log_answer(self, kind: str, ids: list[str], got: int, temperature: float, content: Any) -> None:
        """Incomplete answers go to ``work/logs/llm_answers.jsonl`` so a bad model answer can be looked at."""
        last = self._last()
        reason = last.get("finish_reason")
        if reason == "length":
            self._count("truncated")
        self.ctx.log.info(
            "llm answer covered %d/%d lines (T=%.1f%s%s)", got, len(ids), temperature,
            f", finish={reason}" if reason else "", f", thought {last['reasoning_chars']} chars" if last.get("reasoning_chars") else "",
        )
        try:
            path = self.ctx.paths.logs / ANSWER_LOG
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {"time": now_iso(), "kind": kind, "backend": self.name, "ids": ids, "covered": got, "temperature": temperature, **last,
                   "content": content if isinstance(content, str) else repr(content)}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as e:  # the log is a diagnostic aid, never a failure
            self.ctx.log.debug("answer log not written: %s", e)

    def _extra(self) -> dict[str, Any]:
        # MockTranslator needs the target language code; llama-server derives it from the prompt
        return {"tgt": self.tgt} if self.name.startswith("mock") else {}

    def _context_messages(self, summary: str, context: list[dict[str, Any]], scene_context: list[dict[str, Any]] | None) -> list[dict[str, str]]:
        ctx_text = (f"Scene summary: {summary}\n" if summary else "") + (
            "Previously translated lines, for context only:\n" + json.dumps(context, ensure_ascii=False) if context else ""
        )
        if scene_context:
            ctx_text += ("\n" if ctx_text else "") + (
                "scene_lines: the script around the lines to deliver (source text, narration included; the lines you answer for "
                "are marked deliver=true), for context only:\n" + json.dumps(scene_context, ensure_ascii=False)
            )
        if not ctx_text.strip():
            return []
        return [{"role": "user", "content": ctx_text.strip()}, {"role": "assistant", "content": "Understood."}]

    def messages(self, batch: list[dict[str, Any]], context: list[dict[str, Any]], summary: str, scene_context: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
        speakers = sorted({b["speaker_id"] for b in batch})
        visible = [{k: v for k, v in b.items() if k != "speaker_id"} for b in batch]
        delivery = any(b.get("want_delivery") for b in batch)
        msgs = [{"role": "system", "content": system_prompt(self.tgt, batch, speakers, self.gloss, self.chars, self.t.short_variant_min_words, delivery, self.src)}]
        msgs += self._context_messages(summary, context, scene_context)
        msgs.append({"role": "user", "content": "Translate these lines and answer with JSON only:\n" + json.dumps(visible, ensure_ascii=False)})
        return msgs

    def _dead_reason(self, exc: BaseException) -> str:
        """Why the model server cannot answer any more ("" while it may still answer the next request).

        A request that failed because the server is gone (llama-server exited, was killed by a Cancel or by the
        driver, refuses connections) fails the same way for every retry and every half of the halving below: the
        stage stops instead of making dozens of calls that cannot succeed. An HTTP error status or a timeout is
        an answer of a living server and keeps the usual retry path.
        """
        if isinstance(exc, urllib.error.HTTPError):
            return ""
        poll = getattr(getattr(self.backend, "proc", None), "poll", None)
        if callable(poll):
            try:
                code = poll()
            except Exception:  # noqa: BLE001 - a probe of the process must never be the error
                code = None
            if code is not None:
                return f"the server process exited with code {code}"
        if isinstance(exc, ConnectionError):
            return f"connection lost ({exc})"
        if isinstance(exc, urllib.error.URLError) and not isinstance(exc.reason, TimeoutError):
            return f"connection failed ({exc.reason})"
        return ""

    def attempt(
        self, batch: list[dict[str, Any]], context: list[dict[str, Any]], summary: str, temperature: float, scene_context: list[dict[str, Any]] | None = None
    ) -> dict[str, dict[str, Any]]:
        self._count("calls")
        want = {b["id"] for b in batch if b.get("want_delivery")}
        try:
            content = self.backend.chat(
                self.messages(batch, context, summary, scene_context),
                temperature=temperature,
                # a line costs ~120 tokens of answer (translation, short variant, notes, keys); long lines more
                max_tokens=max(1024, 220 * len(batch) + 100 * len(want)),
                json_mode=True,
                seed=1234,
                **self._extra(),
            )
        except BackendUnavailable:
            self.ctx.check_cancel()
            raise
        except Exception as e:  # adapter/http errors count as an invalid answer ...
            # ... unless the job was cancelled: Cancel kills the server, so the failure that follows is not an answer
            # and must not start the retry / halving below (that is dozens of calls to a dead server)
            self.ctx.check_cancel()
            self.ctx.log.warning("llm call failed (%s): %s", self.name, e)
            dead = self._dead_reason(e)
            if dead:
                raise StageError(f"the translation model {self.name} stopped answering: {dead}. See work/logs/llama_server.log; the lines already translated are kept") from e
            return {}
        ids = [b["id"] for b in batch]
        got = parse_response(content, ids, self.tgt, want)
        for entry in got.values():  # every entry says which model wrote it, whatever happens to the batch later
            entry["source"] = f"llm:{self.name}"
        if len(got) < len(batch):
            self._count("failures")
            self._log_answer("translate", ids, len(got), temperature, content)
        return got

    def translate(self, batch: list[dict[str, Any]], context: list[dict[str, Any]], summary: str, scene_context: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        got = self.attempt(batch, context, summary, self.t.llm.temperature, scene_context)
        for _ in range(max(1, self.t.llm.max_retries - 1)):
            missing = [b for b in batch if b["id"] not in got]
            if not missing:
                return got
            got.update(self.attempt(missing, context, summary, 0.0, scene_context))
        missing = [b for b in batch if b["id"] not in got]
        if missing and len(missing) > 1:
            half = len(missing) // 2
            for part in (missing[:half], missing[half:]):
                got.update(self.translate(part, context, summary, scene_context))
        return got

    def deliver(self, batch: list[dict[str, Any]], scene_context: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        """Delivery-only request (imported translations, ``text_emotion.llm: on``): ``{id: validated delivery}``.

        One attempt plus one at temperature 0 for the lines still missing; failures only mean "no llm delivery".
        """
        got: dict[str, dict[str, Any]] = {}
        for _ in range(2):
            todo = [b for b in batch if b["id"] not in got]
            if not todo:
                break
            self._count("calls")
            speakers = sorted({b["speaker_id"] for b in todo})
            visible = [{k: v for k, v in b.items() if k != "speaker_id"} for b in todo]
            msgs = [{"role": "system", "content": delivery_system_prompt(todo, speakers, self.gloss, self.chars, self.src)}]
            msgs += self._context_messages("", [], scene_context)
            msgs.append({"role": "user", "content": "Describe how these lines are spoken and answer with JSON only:\n" + json.dumps(visible, ensure_ascii=False)})
            try:
                content = self.backend.chat(msgs, temperature=0.0, max_tokens=max(1024, 160 * len(todo)), json_mode=True, seed=1234, **self._extra())
            except BackendUnavailable:
                self.ctx.check_cancel()
                raise
            except Exception as e:
                self.ctx.check_cancel()  # a Cancel is not "no llm delivery"
                self._count("delivery_errors")
                self.ctx.log.warning("llm delivery call failed (%s): %s", self.name, e)
                break
            ids = [b["id"] for b in todo]
            part = parse_delivery_response(content, ids)
            if len(part) < len(todo):
                self._log_answer("delivery", ids, len(part), 0.0, content)
            got.update(part)
        return got

    def summarize_checked(self, scene: str, rows: list[dict[str, Any]]) -> tuple[str, bool]:
        """``(summary, from_model)``: the model's scene summary, or the derived fallback with ``from_model`` False.

        Only a summary the model wrote is worth keeping: the fallback is built from the lines in a moment, and a
        failed call (a server that is down, a Cancel) must not be remembered as "the summary of this scene".
        The mock translator's summary is derived by definition (nothing to keep).
        """
        if self.name.startswith("mock"):
            return derived_summary(scene, rows), False
        msgs = [
            {"role": "system", "content": "Summarize a visual novel scene in two short English sentences. Answer with JSON only: {\"summary\": \"...\"}."},
            {"role": "user", "content": "\n".join(f"{r['speaker']}: {r['text']}" for r in rows[:80])},
        ]
        content: Any = None
        try:
            content = self.backend.chat(msgs, temperature=0.0, max_tokens=400, json_mode=True, seed=1234, **self._extra())
            data = _json_object(content)
            s = data.get("summary") if isinstance(data, dict) else None
            if isinstance(s, str) and s.strip():
                return s.strip(), True
            self.ctx.log.info("scene summary unusable for %s", scene)
        except BackendUnavailable:
            self.ctx.check_cancel()
            raise
        except Exception as e:
            self.ctx.check_cancel()  # a Cancel lands here as a failed call; it must not turn into a cached fallback
            self.ctx.log.info("scene summary failed for %s: %s", scene, e)
        self._log_answer("summary", [scene], 0, 0.0, content)
        return derived_summary(scene, rows), False

    def summarize(self, scene: str, rows: list[dict[str, Any]]) -> str:
        return self.summarize_checked(scene, rows)[0]


def derived_summary(scene: str, rows: list[dict[str, Any]]) -> str:
    speakers = list(dict.fromkeys(r["speaker"] for r in rows if r["speaker"] != "narrator"))
    first = rows[0]["text"] if rows else ""
    return f"Scene {scene}: dialogue between {', '.join(speakers) or 'unnamed speakers'} ({len(rows)} lines). It opens with: {first[:60]}"


# ----------------------------------------------------------------------------
# import
# ----------------------------------------------------------------------------

_SRC_KEYS = ("original", "source", "src", "ja", "text_src", "japanese")
_ID_KEYS = ("line_id", "id")


def _src_keys(src: str = "ja") -> tuple[str, ...]:
    """Column / field names that hold the source text (the language code and name of the source added)."""
    extra = tuple(k for k in (src, language_name(src).lower()) if k and k not in _SRC_KEYS)
    return _SRC_KEYS + extra


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


def _import_json_file(p: Path, lines: list[LineRecord], tgt: str, log: Any, src: str = "ja") -> dict[str, str]:
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        if all(isinstance(v, str) for v in data.values()):
            return {str(k): v for k, v in data.items()}  # {line_id: text}
        data = data.get("lines") or data.get("messages") or []
    if not isinstance(data, list):
        return {}
    rows = [r for r in data if isinstance(r, dict)]
    tkeys = _tgt_keys(tgt)
    skeys = _src_keys(src)

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
    if rows and any(k in rows[0] for k in skeys):
        pairs = []
        for r in rows:
            s = next((r[k] for k in skeys if isinstance(r.get(k), str)), None)
            t = tgt_of({k: v for k, v in r.items() if k not in skeys})
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


def _import_xlsx(p: Path, lines: list[LineRecord], tgt: str, src: str = "ja") -> dict[str, str]:
    import openpyxl

    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    out: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    skeys = _src_keys(src)
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        header = [str(c or "").strip().lower() for c in rows[0]]

        def col(pred: Any) -> int | None:
            return next((i for i, h in enumerate(header) if pred(h)), None)

        ci = col(lambda h: h in _ID_KEYS)
        cs = col(lambda h: any(k in h for k in skeys))
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


def load_import(path: Path, lines: list[LineRecord], tgt: str, log: Any, src: str = "ja") -> dict[str, str]:
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
            out.update(_import_json_file(p, lines, tgt, log, src))
        elif suf in (".csv", ".tsv"):
            delim = "\t" if suf == ".tsv" else ","
            with open(p, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f, delimiter=delim))
            if rows and rows[0] and rows[0][0].strip().lower() in _ID_KEYS:
                hdr = [h.strip().lower() for h in rows[0]]
                ti = next((hdr.index(k) for k in _tgt_keys(tgt) if k in hdr), 1)  # the keys' priority decides, not the column order
                rows = [[r[0], r[ti] if ti < len(r) else ""] for r in rows[1:]]
            for r in rows:
                if len(r) >= 2 and r[0].strip():
                    out[r[0].strip()] = r[1]
        elif suf == ".xlsx":
            out.update(_import_xlsx(p, lines, tgt, src))
    return out


# ----------------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------------


def needs_plus(ctx: StageContext) -> bool:
    if ctx.cfg.translate.stress.format == "plus":
        return True
    for name in ctx.cfg.enabled_backends():
        try:
            if registry.get(name).caps.get("stress_format") in ("plus", "acute"):  # acute is made from the plus text at synth
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


def laughter_flags(text: str | None) -> list[str]:
    """Flags a piece of dialogue text earns for its laughter: ``laughter`` when it contains any laugh ("Haha, no way!"),
    plus ``interjection`` when it is nothing but laughter ("Haha!", "Хе-хе.", "(laughs)"). The interjection flag lowers the
    CER gate (``score.interjection_cer_max``), gives the line the short candidate count and keeps the original audio as a
    candidate, exactly like a Japanese kana interjection; the CER of a laughter-only line is not checked at all."""
    t = text or ""
    laughs = laughter.find_laughs(t)
    if not laughs:
        return []
    return ["laughter", "interjection"] if laughter.is_laughter_only(t, "auto") else ["laughter"]


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


def include_unvoiced(ctx: StageContext) -> bool:
    """Translate speaker lines without original audio too (mode B)? ``auto`` follows ``synth.voice_unvoiced``."""
    mode = ctx.cfg.translate.include_unvoiced
    if mode == "auto":
        return bool(getattr(ctx.cfg.synth, "voice_unvoiced", False))
    return mode == "on"


def lines_to_translate(ctx: StageContext, apply_select: bool = True) -> list[LineRecord]:
    voiced = {l.line_id for l in ctx.voiced_lines(apply_only=False)}
    all_lines = ctx.lines()
    extra = {l.line_id for l in te_stage.unvoiced_speaker_lines(all_lines, voiced)} if include_unvoiced(ctx) else set()
    out = [l for l in all_lines if l.line_id in voiced or "voice_sustain" in l.flags or l.line_id in extra]
    return ctx.select(out, voiced_only=False) if apply_select else out


def delivery_target_ids(ctx: StageContext, lines: list[LineRecord]) -> set[str]:
    """Lines of this run whose delivery is wanted from the LLM: unvoiced speaker lines, when text_emotion uses the LLM."""
    te = ctx.cfg.text_emotion
    if not te.enabled or te.llm == "off":
        return set()
    voiced = {l.line_id for l in ctx.voiced_lines(apply_only=False)}
    unvoiced = {l.line_id for l in te_stage.unvoiced_speaker_lines(ctx.lines(), voiced)}
    return {l.line_id for l in lines if l.line_id in unvoiced}


@dataclass
class _LlmPlan:
    primary: str  # the model the user asked for (the real one when ``--real`` replaces the test profile's mock)
    chain: list[str]  # primary first, then ``translate.llm.fallbacks``
    options: dict[str, Any]  # what ``acquire`` is given; the VRAM estimate is costed for exactly these


def _llm_plan(ctx: StageContext) -> _LlmPlan:
    llm = ctx.cfg.translate.llm
    primary = llm.backend
    if ctx.extra.get("real") and primary.startswith("mock"):
        primary = llm.real_backend
    options: dict[str, Any] = {"ctx": llm.ctx, "n_gpu_layers": llm.n_gpu_layers, "temperature": llm.temperature, "port": llm.port, "parallel": max(1, int(llm.parallel))}
    if getattr(llm, "extra_args", None):
        options["extra_args"] = [str(a) for a in llm.extra_args]
    if ctx.cfg.translate.mock_dictionary:
        options["dictionary"] = str(ctx.cfg.resolve(ctx.cfg.translate.mock_dictionary))
    return _LlmPlan(primary, list(dict.fromkeys([primary, *llm.fallbacks])), options)


def _llm_candidates(ctx: StageContext, plan: _LlmPlan, reasons: list[str]) -> Iterator[tuple[str, Any]]:
    """The models of the plan that would load right now, best first: ``(name, resolved spec)``.

    The one place that decides which model the stage works with. :func:`_acquire_llm` loads the first of them (the
    next one when a load fails) and :func:`_llm_key_backend` names the first one for the cache key, so the key
    can never name a model other than the one that runs. Why a model was skipped is appended to ``reasons``.
    """
    for name in plan.chain:
        spec, why = ctx.models.would_fit(name, plan.options)
        if why is None:
            yield name, spec
        else:
            reasons.append(why)


def _acquire_llm(ctx: StageContext) -> tuple[Any, str]:
    plan = _llm_plan(ctx)
    reasons: list[str] = []
    for name, spec in _llm_candidates(ctx, plan, reasons):
        reason = "; ".join(reasons)
        if name != plan.primary:
            # loud, and recorded as a substitution (report, gpu_events, the app's job warnings): before, the default
            # model could be skipped on every run (VRAM budget, an orphan holding the card) without a trace
            ctx.log.warning("translation LLM %s is not usable (%s): using %s", plan.primary, reason, spec.name)
        try:
            return ctx.models.acquire(name, plan.options, substituted_for=plan.primary if name != plan.primary else None, reason=reason), spec.name
        except BackendUnavailable as e:
            ctx.check_cancel()  # a Cancel that killed the server during the load is not "this model failed": no next model
            reasons.append(str(e))
    raise StageError("no translation backend usable: " + "; ".join(reasons))


def run(ctx: StageContext) -> dict[str, Any]:
    t = ctx.cfg.translate
    if ctx.extra.get("seed_glossary"):
        return seed_glossary(ctx)
    tgt = (ctx.project.tgt_lang if ctx.project else None) or ctx.cfg.project.tgt_lang
    src = ctx.src_lang
    lines = lines_to_translate(ctx)
    delivery_ids = delivery_target_ids(ctx, lines)
    chars = ctx.characters()
    gloss = load_glossary(glossary_path(ctx))
    # the interjection lexicon maps Japanese interjections to Russian ones; other pairs go through the model
    lexicon = ja.load_interjections(ctx.cfg.resolve(t.interjections)) if (src == "ja" and tgt == "ru") else {}
    max_morae = ctx.cfg.synth.interjection_max_syllables
    ctx.log.info("translating %s -> %s (%d lines)", src, tgt, len(lines))

    entries: dict[str, dict[str, Any]] = {}
    flags: dict[str, list[str]] = {l.line_id: [] for l in lines}
    backend_name = ""

    # 1. interjections (kana-only short lines are Japanese by definition) and laughter (any language)
    for l in lines:
        if src == "ja" and ja.is_interjection(l.text_src, lexicon, max_morae):
            flags[l.line_id].append("interjection")
        flags[l.line_id] += laughter_flags(l.text_src)
        mapped = ja.match_interjection(l.text_src, lexicon, max_morae) if lexicon else None
        if mapped:
            entries[l.line_id] = {"text_tgt": mapped, "text_tgt_short": mapped, "notes": "", "source": "interjection"}

    # 2. import or LLM (or nothing: the text already is in the target language)
    if t.import_path:
        imported = load_import(ctx.cfg.resolve(t.import_path), ctx.lines(), tgt, ctx.log, src)
        for l in lines:
            if l.line_id in imported and str(imported[l.line_id]).strip():
                txt = str(imported[l.line_id]).strip()
                entries[l.line_id] = {"text_tgt": txt, "text_tgt_short": txt, "notes": "", "source": "import"}
        backend_name = "import"
        # no LLM is loaded just for delivery unless text_emotion.llm == "on"
        if delivery_ids and ctx.cfg.text_emotion.llm == "on":
            targets = [l for l in lines if l.line_id in delivery_ids and (entries.get(l.line_id) or {}).get("text_tgt")]
            if targets:
                try:
                    translate_deliveries(ctx, targets, entries, tgt, gloss, chars, src)
                except (BackendUnavailable, StageError) as e:
                    ctx.log.warning("llm delivery skipped: %s", e)
    elif src == tgt:
        ctx.log.warning("the script text already is %s (the target language): lines are kept as they are, no model is started", language_name(tgt))
        for l in lines:
            if l.line_id not in entries:
                txt = " ".join((l.text_src or "").split())
                entries[l.line_id] = {"text_tgt": txt, "text_tgt_short": txt, "notes": "", "source": "same_lang"}
        backend_name = "same_lang"
    else:
        todo = [l for l in lines if l.line_id not in entries]
        if todo:
            backend_name = translate_llm(ctx, todo, entries, flags, tgt, gloss, chars, delivery_ids & {l.line_id for l in todo}, src)

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
        if "translate_failed" not in fl and "translate_missing" not in fl:
            fl += laughter_flags(e["text_tgt"])  # the model may turn "Haha!" into "Ха-ха!" (or a laugh into words)
        recs.append(
            TranslationRecord(
                line_id=l.line_id,
                text_tgt=e["text_tgt"],
                text_tgt_short=e.get("text_tgt_short") or e["text_tgt"],
                notes=e.get("notes", ""),
                source=e.get("source", ""),
                glossary_hits=glossary_hits(l.text_src, gloss, chars),
                flags=list(dict.fromkeys(fl)),
                **({"delivery": e["delivery"]} if e.get("delivery") else {}),
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
        "src_lang": src,
        "tgt_lang": tgt,
        "sources": dict(src_counts),
        "backend": backend_name or "-",
        "models_used": models_used(recs),
        "interjections": sum(1 for r in recs if "interjection" in r.flags),
        "laughter": sum(1 for r in recs if "laughter" in r.flags),
        "failed": sum(1 for r in recs if "translate_failed" in r.flags),
        "missing": sum(1 for r in recs if "translate_missing" in r.flags),
        "deliveries": sum(1 for r in recs if (r.model_extra or {}).get("delivery")),
        "stress": stress_backend or f"none ({stress_reason})",
        "plus_variant": plus,
        "glossary": str(glossary_path(ctx)) if gloss else "none",
        "out": str(out_path),
    }



def models_used(recs: list[TranslationRecord]) -> dict[str, int]:
    """``{model: lines}`` of this run's records, whatever their origin (translated now or read from the cache).

    A model line counts under the model named in the record's ``source`` (``llm:<model>``: the model that wrote it,
    stored with the cache entry), an imported translation under ``"import"``, a script that already was in the
    target language under ``"same_lang"``. Lexicon interjections are not model output and failed or missing lines
    have no translation: neither is counted. More than one entry means the project was translated by more than one
    model (a fallback after the first one did not fit, a resume with a different model). Largest first.
    """
    n: Counter[str] = Counter()
    for r in recs:
        if "translate_failed" in r.flags or "translate_missing" in r.flags:
            continue
        kind, _, model = r.source.partition(":")
        if kind == "llm":
            n[model or "unknown"] += 1
        elif kind in ("import", "same_lang"):
            n[kind] += 1
    return dict(sorted(n.items(), key=lambda kv: (-kv[1], kv[0])))


def _scene_lines(ctx: StageContext) -> dict[str, list[LineRecord]]:
    out: dict[str, list[LineRecord]] = {}
    for l in ctx.lines():
        out.setdefault(l.scene, []).append(l)
    return out


def _delivery_scene_context(
    scene_all: list[LineRecord], ids: list[str], k: int, chars: dict[str, Character], gloss: dict[str, Any]
) -> list[dict[str, Any]]:
    """The script around ``ids`` (+-k lines, narration included) for the delivery request; ``deliver`` marks the targets."""
    pos = {l.line_id: i for i, l in enumerate(scene_all)}
    keep: set[int] = set()
    for lid in ids:
        i = pos.get(lid)
        if i is not None:
            keep.update(range(max(0, i - k), min(len(scene_all), i + k + 1)))
    want = set(ids)
    out: list[dict[str, Any]] = []
    for i in sorted(keep):
        l = scene_all[i]
        if not (l.text_src or "").strip():
            continue
        r: dict[str, Any] = {"id": l.line_id, "speaker": _display_name(l.speaker_id, l, chars, gloss), "text": l.text_src}
        if l.line_id in want:
            r["deliver"] = True
        out.append(r)
    return out


def _llm_key_backend(ctx: StageContext) -> str:
    """The model name that decides the cache key: the first model of the plan that fits now, i.e. the one
    :func:`_acquire_llm` will load (nothing is loaded here). When none fits it is the primary: the stage raises
    when it needs a model, and a run that is served from the cache alone keeps the primary's entries."""
    plan = _llm_plan(ctx)
    first = next(_llm_candidates(ctx, plan, []), None)
    if first is not None:
        return first[1].name
    try:
        return ctx.models.resolve(plan.primary)[0].name
    except BackendUnavailable:
        return plan.primary


def _llm_preferred(ctx: StageContext, key_backend: str) -> list[str]:
    """Models the user prefers over ``key_backend`` (they come before it in the plan but do not fit now).

    Work they did earlier is still read from the cache, labelled with their name: a restart with the card
    briefly full must not throw a good translation away for a smaller model's. The cache is written under the
    model that really runs, so once the preferred model fits again its own entries win.
    """
    names: list[str] = []
    for n in _llm_plan(ctx).chain:
        try:
            names.append(ctx.models.resolve(n)[0].name)
        except BackendUnavailable:
            names.append(n)
    return names[: names.index(key_backend)] if key_backend in names else []


def translate_llm(
    ctx: StageContext,
    todo: list[LineRecord],
    entries: dict[str, dict[str, Any]],
    flags: dict[str, list[str]],
    tgt: str,
    gloss: dict[str, Any],
    chars: dict[str, Character],
    delivery_ids: set[str] | None = None,
    src: str = "ja",
) -> str:
    """Translate ``todo`` per scene. Lines in ``delivery_ids`` (unvoiced speaker lines) go in their own batches after
    the scene's regular ones and also return a ``delivery`` object (see the module docstring)."""
    t = ctx.cfg.translate
    delivery_ids = delivery_ids or set()
    store = OpCache(ctx, "translate_llm")
    gloss_hash = hash_obj(gloss)
    scenes: dict[str, list[LineRecord]] = {}
    for l in todo:
        scenes.setdefault(l.scene, []).append(l)
    scene_all = _scene_lines(ctx) if delivery_ids else {}

    def row(l: LineRecord, want: bool = False) -> dict[str, Any]:
        # the emotion hint comes from an audio passport; delivery targets have none (a text passport would feed back)
        style = {} if want else ((read_json(ctx.paths.passport(l.line_id)) or {}).get("style") or {})
        r = {
            "id": l.line_id,
            "speaker": _display_name(l.speaker_id, l, chars, gloss),
            "speaker_id": l.speaker_id,
            "gender": _gender_of(l.speaker_id, chars, gloss),
            "emotion": style.get("emotion", ""),
            "text": l.text_src,
        }
        if want:
            r["want_delivery"] = True
        return r

    translator: LlmTranslator | None = None
    load_error: BaseException | None = None
    backend_name = ""
    tr_lock = threading.Lock()

    def get_translator() -> LlmTranslator:
        nonlocal translator, backend_name, load_error
        with tr_lock:  # the model is loaded once, whichever scene thread misses the cache first
            if load_error is not None:
                raise load_error  # the walk over the whole model chain failed once: the other scenes do not walk (and spawn a server) again
            if translator is None:
                try:
                    backend, backend_name = _acquire_llm(ctx)
                except BaseException as e:
                    load_error = e
                    raise
                if backend_name != key_backend:
                    # the model the key walk named did not load after all (or the card changed in between): what this
                    # run writes goes under the key of the model that really answers, never under the other's
                    ctx.log.warning("the cache was keyed for %s but %s was loaded", key_backend, backend_name)
                translator = LlmTranslator(ctx, backend, backend_name, tgt, gloss, chars, src)
            return translator

    def context_before(scene: str, line_id: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for l in scene_all.get(scene, []):
            if l.line_id == line_id:
                break
            e = entries.get(l.line_id)
            if e is not None and e.get("text_tgt"):
                out.append({"speaker": _display_name(l.speaker_id, l, chars, gloss), "text": l.text_src, tgt: e["text_tgt"]})
        return out[-t.context_window_lines :] if t.context_window_lines else []

    llm = t.llm
    key_backend = _llm_key_backend(ctx)
    # cache entries of a model the user prefers over the one that runs now stay readable (labelled with their own model)
    readable = [key_backend, *_llm_preferred(ctx, key_backend)]
    lock = threading.Lock()  # ``entries`` is shared by the scene threads when llm.parallel > 1
    total = len(todo)
    n_done = 0

    def tick(n: int, item: str) -> None:
        # progress (and the job's pause / cancel checkpoint) after every batch, cached or not
        nonlocal n_done
        with lock:
            n_done += n
            ctx.progress(min(n_done, total), total, item)

    def cached(item_id: str, key_of: Any) -> Any:
        """The stored value of ``item_id`` under the key of the model that runs, else of a preferred one."""
        for model in readable:
            hit = store.get(item_id, key_of(model))
            if hit is not None:
                return hit
        return None

    def do_scene(scene: str, scene_lines: list[LineRecord]) -> None:
        rows = [row(l) for l in scene_lines if l.line_id not in delivery_ids]
        t_rows = [row(l, True) for l in scene_lines if l.line_id in delivery_ids]
        summary_rows = rows or t_rows

        def summary_key(model: str) -> str:
            return store.key("summary", model, scene, [r["text"] for r in summary_rows])

        summary = cached(f"summary:{scene}", summary_key)
        history: list[dict[str, Any]] = []

        def ensure_summary(tr: LlmTranslator) -> str:
            nonlocal summary
            if summary is None:
                text, from_model = tr.summarize_checked(scene, summary_rows)
                summary = text
                if from_model:  # the derived fallback serves this scene's batches now and is never stored
                    store.put(f"summary:{scene}", summary_key(tr.name), text)  # under the key of the model that wrote it
            return summary

        for i in range(0, len(rows), max(1, t.batch_lines)):
            batch = rows[i : i + t.batch_lines]
            context = history[-t.context_window_lines :] if t.context_window_lines else []
            def bkey(model: str, batch: list[dict[str, Any]] = batch, context: list[dict[str, Any]] = context) -> str:
                return store.key(model, src, tgt, gloss_hash, batch, context, llm.temperature)

            item_id = f"batch:{scene}:{batch[0]['id']}:{len(batch)}"
            got = cached(item_id, bkey)
            if got is None:
                tr = get_translator()
                got = tr.translate(batch, context, ensure_summary(tr))
                if len(got) == len(batch):
                    store.put(item_id, bkey(tr.name), got)  # under the key of the model that wrote it
                store.save()
            with lock:
                for b in batch:
                    e = got.get(b["id"])
                    if e is None:
                        continue
                    # the entry names the model that wrote it (cache entries of older runs: the model of their key)
                    entries[b["id"]] = {**e, "source": e.get("source") or f"llm:{key_backend}"}
                    history.append({"speaker": b["speaker"], "text": b["text"], tgt: e["text_tgt"]})
            tick(len(batch), batch[-1]["id"])
        # delivery stream: lines nobody voices, translated with the surrounding script as context
        for i in range(0, len(t_rows), max(1, t.batch_lines)):
            batch = t_rows[i : i + t.batch_lines]
            with lock:
                context = context_before(scene, batch[0]["id"])
            sc = _delivery_scene_context(scene_all.get(scene, []), [b["id"] for b in batch], ctx.cfg.text_emotion.llm_context_lines, chars, gloss)
            def dkey(model: str, batch: list[dict[str, Any]] = batch, context: list[dict[str, Any]] = context, sc: list[dict[str, Any]] = sc) -> str:
                return store.key(DELIVERY_PROTOCOL, model, src, tgt, gloss_hash, batch, context, sc, llm.temperature)

            item_id = f"batch:{scene}:{batch[0]['id']}:{len(batch)}:delivery"
            got = cached(item_id, dkey)
            if got is None:
                tr = get_translator()
                got = tr.translate(batch, context, ensure_summary(tr), sc)
                if len(got) == len(batch):
                    store.put(item_id, dkey(tr.name), got)
                store.save()
            with lock:
                for b in batch:
                    e = got.get(b["id"])
                    if e is not None:
                        entries[b["id"]] = {**e, "source": e.get("source") or f"llm:{key_backend}"}
            tick(len(batch), batch[-1]["id"])

    ctx.progress(0, total, None)
    workers = max(1, int(llm.parallel))
    if workers > 1 and len(scenes) > 1:
        # scenes are independent (the rolling context never crosses a scene): each worker drives one llama-server
        # slot, so the server decodes several answers in one batch
        ctx.log.info("translating %d scenes on %d llama-server slots", len(scenes), workers)
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="translate")
        try:
            for fut in as_completed([pool.submit(do_scene, scene, scene_lines) for scene, scene_lines in scenes.items()]):
                fut.result()  # the first failure (backend unavailable, a cancel ...) is raised here
        finally:
            pool.shutdown(wait=True, cancel_futures=True)  # the scenes that did not start yet are dropped, those running finish
    else:
        for scene, scene_lines in scenes.items():
            do_scene(scene, scene_lines)
    store.save()
    if translator is not None:
        ctx.log.info("llm calls: %d, incomplete answers: %d, cut off by the token budget: %d", translator.calls, translator.failures, translator.truncated)
    return backend_name or key_backend


def translate_deliveries(
    ctx: StageContext,
    targets: list[LineRecord],
    entries: dict[str, dict[str, Any]],
    tgt: str,
    gloss: dict[str, Any],
    chars: dict[str, Character],
    src: str = "ja",
) -> None:
    """Delivery-only LLM pass for lines whose translation is already known (``text_emotion.llm: on`` with an import).

    Adds ``entry["delivery"]`` where the model gave a valid object; the same client, batching and cache as
    :func:`translate_llm`, one model load shared with nothing else in this run.
    """
    t = ctx.cfg.translate
    store = OpCache(ctx, "translate_delivery")
    gloss_hash = hash_obj(gloss)
    key_backend = _llm_key_backend(ctx)
    readable = [key_backend, *_llm_preferred(ctx, key_backend)]
    scene_all = _scene_lines(ctx)
    translator: LlmTranslator | None = None
    scenes: dict[str, list[LineRecord]] = {}
    for l in targets:
        scenes.setdefault(l.scene, []).append(l)
    n_done = 0
    ctx.progress(0, len(targets), None)
    for scene, scene_lines in scenes.items():
        rows = [
            {
                "id": l.line_id, "speaker": _display_name(l.speaker_id, l, chars, gloss), "speaker_id": l.speaker_id,
                "gender": _gender_of(l.speaker_id, chars, gloss), "text": l.text_src,
                "translation": entries[l.line_id]["text_tgt"], "want_delivery": True,
            }
            for l in scene_lines
        ]
        for i in range(0, len(rows), max(1, t.batch_lines)):
            batch = rows[i : i + t.batch_lines]
            sc = _delivery_scene_context(scene_all.get(scene, []), [b["id"] for b in batch], ctx.cfg.text_emotion.llm_context_lines, chars, gloss)
            def dkey(model: str, batch: list[dict[str, Any]] = batch, sc: list[dict[str, Any]] = sc) -> str:
                return store.key(DELIVERY_PROTOCOL, "delivery_only", model, src, tgt, gloss_hash, batch, sc)

            item_id = f"batch:{scene}:{batch[0]['id']}:{len(batch)}"
            got = next((hit for model in readable if (hit := store.get(item_id, dkey(model))) is not None), None)
            if got is None:
                if translator is None:
                    backend, name = _acquire_llm(ctx)
                    translator = LlmTranslator(ctx, backend, name, tgt, gloss, chars, src)
                errors = translator.delivery_errors
                got = translator.deliver(batch, sc)
                if translator.delivery_errors == errors:  # a request that failed is retried next run, not remembered as "no delivery"
                    store.put(item_id, dkey(translator.name), got)
                store.save()
            for b in batch:
                if b["id"] in got:
                    entries[b["id"]]["delivery"] = got[b["id"]]
            n_done += len(batch)
            ctx.progress(min(n_done, len(targets)), len(targets), batch[-1]["id"])
    store.save()


# ----------------------------------------------------------------------------
# import preview (the app's "I have a translation" step)
# ----------------------------------------------------------------------------

IMPORT_SUFFIXES = (".jsonl", ".json", ".csv", ".tsv", ".xlsx")


def detect_import_format(path: Path, lines: list[LineRecord] | None = None) -> str:
    """Name of the import format ``load_import`` will use: jsonl | json_map | json_rows | json_pairs |
    vntextpatch | csv | tsv | xlsx | folder:<suffixes> (a folder holding several of them)."""
    if path.is_dir():
        kinds = sorted({p.suffix.lower().lstrip(".") for p in path.iterdir() if p.suffix.lower() in IMPORT_SUFFIXES})
        return "folder:" + "+".join(kinds) if kinds else "folder"
    suf = path.suffix.lower()
    if suf == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return "json"
        if isinstance(data, dict):
            if all(isinstance(v, str) for v in data.values()):
                return "json_map"
            data = data.get("lines") or data.get("messages") or []
        rows = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
        if rows and any(k in rows[0] for k in _ID_KEYS):
            return "json_rows"
        if rows and any(k in rows[0] for k in _SRC_KEYS):
            return "json_pairs"
        return "vntextpatch"
    return {".jsonl": "jsonl", ".csv": "csv", ".tsv": "tsv", ".xlsx": "xlsx"}.get(suf, "unknown")


def preview_import(ctx: StageContext, path: str, n: int = 5) -> dict[str, Any]:
    """How a translation file/folder would match the lines, with the importers ``translate.import`` uses. No writes.

    Returns ``{"format", "matched", "total", "preview": [{"line_id", "text_src", "text_tgt"}], "files", "in_file"}``:
    ``total`` = lines translate would handle (voiced + sustain, + unvoiced speaker lines when included), ``matched``
    = those the file gives a non-empty text for, ``preview`` = the first ``n`` of them in script order.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ctx.cfg.resolve(path)
    tgt = (ctx.project.tgt_lang if ctx.project else None) or ctx.cfg.project.tgt_lang
    all_lines = ctx.lines()
    cand = lines_to_translate(ctx, apply_select=False)
    try:
        imported = load_import(p, all_lines, tgt, ctx.log, ctx.src_lang)
        fmt = detect_import_format(p)
    except StageError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError, KeyError) as e:
        raise StageError(f"cannot read the translation {p.name}: {e}") from e
    matched = [l for l in cand if str(imported.get(l.line_id, "")).strip()]
    files = len([q for q in p.iterdir() if q.suffix.lower() in IMPORT_SUFFIXES]) if p.is_dir() else 1
    return {
        "format": fmt,
        "matched": len(matched),
        "total": len(cand),
        "preview": [{"line_id": l.line_id, "text_src": l.text_src, "text_tgt": str(imported[l.line_id]).strip()} for l in matched[: max(0, int(n))]],
        "files": files,
        "in_file": len(imported),
    }
