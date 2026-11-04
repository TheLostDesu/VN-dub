"""Non-speech sounds in dialogue text (snoring, kisses, yawns, sneezes ...): where they are, what kind, how to render them.

The sibling of :mod:`vndub.text.laughter` for the onomatopoeia a text-to-speech model cannot say: a line like ``ZZZ...`` or
``Muah!`` is read letter by letter ("зэ-зэ-зэ") unless it is spelled the way the target language writes the sound, replaced
by a backend's own event tag, replaced by a recorded sample, or left silent. Laughs are NOT handled here (the laughter
module owns them).

* :func:`find_sounds` returns every sound span of a line with its ``kind`` and ``lang``; the three scripts (kana,
  Cyrillic, Latin) do not overlap, so ``lang="auto"`` runs all three. Bracketed stage directions (``*yawns*``,
  ``(храпит)``, ``（あくび）``) are sounds too (``stage=True``): they describe the delivery but must not be read aloud.
* :func:`is_sound_only` tells whether a line is nothing but such a sound (plus punctuation): a line whose transcript cannot
  be checked word by word and that the voice models cannot really perform.
* :func:`strip_sounds` removes them (CER reference, "what else does the line say").
* :func:`spell_sounds` rewrites each sound through a renderer; :func:`table_renderer` builds one from a ``{kind: spelling}``
  table (``configs/sounds.yaml``, :func:`load_table`).
* :func:`summarize` is the ``{kind, n, only, stage}`` record the translate / text_emotion side stores and synth reads.

Kinds (:data:`KINDS`): ``snore`` (zzz, хр-р-р, ぐーぐー), ``kiss`` (muah, mwah, chu, чмок, ちゅっ), ``yawn`` (yaaawn,
ふぁ〜あ), ``gasp`` (gasp!, はっ), ``sneeze`` (achoo, апчхи, ハクション), ``cough`` (ahem, cough, кхе-кхе, ごほごほ),
``hum`` (mmm, hmm, м-м-м), ``hmph`` (hmph, хмф), ``shiver`` (brr, бр-р, ぶるぶる), ``eat`` (nom nom, slurp, gulp, ням-ням,
もぐもぐ). The translate LLM may name a kind too (``sound`` in its answer); this module is the deterministic fallback and
the renderer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from vndub.text import laughter as lg
from vndub.util import repo_root

KINDS = ("snore", "kiss", "yawn", "gasp", "sneeze", "cough", "hum", "hmph", "shiver", "eat")

# how a kind is described to a voice model (instruct phrase) and in a delivery note
KIND_NOTE = {
    "snore": "snoring",
    "kiss": "a kiss",
    "yawn": "yawning",
    "gasp": "a gasp",
    "sneeze": "a sneeze",
    "cough": "coughing",
    "hum": "humming",
    "hmph": "a huff",
    "shiver": "shivering",
    "eat": "eating",
}

# ----------------------------------------------------------------------------
# patterns
# ----------------------------------------------------------------------------

_LAT_B = r"(?<![A-Za-z])"
_LAT_E = r"(?![A-Za-z])"
_SEP = r"(?:[\s\-~]*)"
_REP = r"(?:[\s\-~.…]*)"  # between repeats of a snore ("Zzz... zzz...", "Хр-р-р... хр-р-р..."): the stand-in is one sound, so re-rendering it is idempotent
# order matters: a later alternative must not be a prefix of an earlier one ("hmph" before "hm")
_LAT_SPOKEN = re.compile(
    _LAT_B
    + r"(?:"
    + r"(?P<snore>z{2,}(?:" + _REP + r"z{2,})*)"
    + r"|(?P<kiss>m*w+a+h*|mu+a+h*|mwa+|chu+|smooch|mmch)"
    + r"|(?P<yawn>ya+w+n+|yaw+n+)"
    + r"|(?P<sneeze>a+h?[\-]?t?ch+o+o*|hatch+i+|atish+o+o*)"
    + r"|(?P<cough>a+h?hem|c+ough(?:" + _SEP + r"cough)*|khe+(?:" + _SEP + r"khe+)*)"
    + r"|(?P<hmph>h+m+p+h*|hu+mph|humf)"
    + r"|(?P<hum>m+[\-~]h+m+|m{2,}h?m*|h+m{1,})"
    + r"|(?P<shiver>b+r{2,})"
    + r"|(?P<eat>nom(?:" + _SEP + r"nom)+|sl+u+r+p+|gulp|munch(?:" + _SEP + r"munch)*|om+(?:" + _SEP + r"nom)+)"
    + r"|(?P<gasp>gasp)"
    + r")"
    + _LAT_E,
    re.I,
)
_LAT_STAGE = re.compile(
    r"[\(\*\[]\s*(?P<word>yawn\w*|snor\w*|sneez\w*|cough\w*|clears? (?:his|her|their) throat|ahem|gasp\w*|hum\w*|kiss\w*|smooch\w*"
    r"|shiver\w*|gulp\w*|slurp\w*|munch\w*|chew\w*|eats?|eating)[^)*\]\n]{0,24}[\)\*\]]",
    re.I,
)
_LAT_STAGE_KIND = {
    "yawn": "yawn", "snor": "snore", "sneez": "sneeze", "cough": "cough", "clears": "cough", "ahem": "cough", "gasp": "gasp",
    "hum": "hum", "kiss": "kiss", "smooch": "kiss", "shiver": "shiver", "gulp": "eat", "slurp": "eat", "munch": "eat",
    "chew": "eat", "eat": "eat",
}

_RU_B = r"(?<![а-яёА-ЯЁ])"
_RU_E = r"(?![а-яёА-ЯЁ])"
_RU_SPOKEN = re.compile(
    _RU_B
    + r"(?:"
    + r"(?P<snore>хр+(?:" + _SEP + r"р+)*(?:" + _REP + r"хр+(?:" + _SEP + r"р+)*)*|з(?:" + _SEP + r"з){2,})"
    + r"|(?P<kiss>чмок+(?:" + _SEP + r"чмок+)*|м*му+[\-~]*а+х*)"
    + r"|(?P<sneeze>а+п(?:" + _SEP + r"ч+х+и+)|апчх+и+)"
    + r"|(?P<cough>к[хэ]е+(?:" + _SEP + r"к[хэ]е+)*|кха+(?:" + _SEP + r"кха+)*|кхм+)"
    + r"|(?P<hmph>хм+ф+|хмпф+)"
    + r"|(?P<hum>м+(?:[\-~]м+)+|м{2,}|хм+)"
    + r"|(?P<shiver>бр+(?:" + _SEP + r"р+)+|бр{2,})"
    + r"|(?P<eat>ням+(?:" + _SEP + r"ням+)*|хрум+(?:" + _SEP + r"хрум+)*|хлюп+(?:" + _SEP + r"хлюп+)*|чавк+(?:" + _SEP + r"чавк+)*)"
    + r")"
    + _RU_E,
    re.I,
)
_RU_STAGE = re.compile(
    r"[\(\*\[]\s*(?P<word>зева\w*|храп\w*|чиха\w*|чихну\w*|кашл\w*|откашл\w*|целу\w*|поцелу\w*|чмока\w*|ахну\w*|охну\w*|мыч\w*|"
    r"напева\w*|ёжится|ежится|дрож\w*|глота\w*|сглатыва\w*|чавка\w*|жуёт|жует|ест|хлюпа\w*)[^)*\]\n]{0,24}[\)\*\]]",
    re.I,
)
_RU_STAGE_KIND = {
    "зева": "yawn", "храп": "snore", "чиха": "sneeze", "чихну": "sneeze", "кашл": "cough", "откашл": "cough", "целу": "kiss",
    "поцелу": "kiss", "чмока": "kiss", "ахну": "gasp", "охну": "gasp", "мыч": "hum", "напева": "hum", "ёжится": "shiver",
    "ежится": "shiver", "дрож": "shiver", "глота": "eat", "сглатыва": "eat", "чавка": "eat", "жуёт": "eat", "жует": "eat",
    "ест": "eat", "хлюпа": "eat",
}

_JA_WC = "ぁ-んァ-ヶ一-龥々A-Za-z0-9"
_JA_B = rf"(?<![{_JA_WC}ー])"
_JA_E = rf"(?![{_JA_WC}])"
_JA_TAIL = "[っッー〜~ぁぃぅぇぉ]*"
_JA_SPOKEN = re.compile(
    _JA_B
    + r"(?:"
    + r"(?P<snore>(?:ぐー|グー|ぐう|グウ|すー|スー|すぴー|スピー)(?:ぐー|グー|ぐう|グウ|すー|スー|すぴー|スピー)*)"
    + r"|(?P<kiss>ん?ちゅ[っゅうー〜]*|ン?チュ[ッュウー〜]*)"
    + r"|(?P<yawn>ふぁ[ぁあー〜]*|ファ[ァアー〜]*|ふわ[ぁあー〜]+|フワ[ァアー〜]+)"
    + r"|(?P<sneeze>はくしょん|ハクション|はっくしょん|ハックション|くしゅん|クシュン|へっくし|ヘックシ)"
    + r"|(?P<cough>(?:ごほ|ゴホ|けほ|ケホ)+|こほん|コホン|えへん|エヘン)"
    + r"|(?P<shiver>(?:ぶる|ブル){2,})"
    + r"|(?P<eat>(?:もぐ|モグ){2,}|ごくん|ゴクン|ごくごく|ゴクゴク|ずず+|ズズ+|(?:ぱく|パク){2,}|(?:むしゃ|ムシャ){2,})"
    + r")"
    + _JA_TAIL
    + _JA_E
)
_JA_STAGE = re.compile(r"[（(]\s*(?P<word>あくび|欠伸|いびき|寝息|くしゃみ|咳|咳払い|キス|口づけ|震え|身震い|食べる|飲む|息を呑む)\s*[)）]")
_JA_STAGE_KIND = {
    "あくび": "yawn", "欠伸": "yawn", "いびき": "snore", "寝息": "snore", "くしゃみ": "sneeze", "咳": "cough", "咳払い": "cough",
    "キス": "kiss", "口づけ": "kiss", "震え": "shiver", "身震い": "shiver", "食べる": "eat", "飲む": "eat", "息を呑む": "gasp",
}


@dataclass(frozen=True)
class Sound:
    """One non-speech sound in a line: ``text[start:end]`` is the written sound (or the bracketed stage direction)."""

    start: int
    end: int
    text: str
    kind: str
    lang: str  # ja | ru | en
    stage: bool = False


def _kind_of(m: re.Match[str]) -> str:
    return next(k for k, v in m.groupdict().items() if v is not None)


def _stage_kind(word: str, table: Mapping[str, str]) -> str:
    w = word.lower()
    for key, kind in table.items():
        if w.startswith(key):
            return kind
    return "gasp"


def _find(text: str, spoken: re.Pattern[str], stage: re.Pattern[str] | None, stage_kind: Mapping[str, str], lang: str) -> list[Sound]:
    out: list[Sound] = []
    for m in spoken.finditer(text):
        out.append(Sound(m.start(), m.end(), m.group(0), _kind_of(m), lang))
    if stage is not None:
        for m in stage.finditer(text):
            out.append(Sound(m.start(), m.end(), m.group(0), _stage_kind(m.group("word"), stage_kind), lang, stage=True))
    return out


def _dedupe(sounds: Iterable[Sound]) -> list[Sound]:
    """Drop spans inside another span (a spoken sound inside a stage direction); keep script order."""
    items = sorted(sounds, key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Sound] = []
    for s in items:
        if out and s.start < out[-1].end:
            continue
        out.append(s)
    return out


def find_sounds(text: str, lang: str = "auto") -> list[Sound]:
    """Every non-speech sound of ``text`` in script order; ``lang`` = ja | ru | en | auto (all three scripts).

    A laugh is never a sound (``Hehe``, ``Хм-хм`` are left to the laughter module), so laughs are masked first.
    """
    t = text or ""
    if not t.strip():
        return []
    masked = list(t)
    for l in lg.find_laughs(t, "auto"):
        for i in range(l.start, l.end):
            masked[i] = "\u0001"
    mt = "".join(masked)
    out: list[Sound] = []
    if lang in ("auto", "en", "ru", "ja"):
        if lang in ("auto", "en"):
            out += _find(mt, _LAT_SPOKEN, _LAT_STAGE, _LAT_STAGE_KIND, "en")
        if lang in ("auto", "ru"):
            out += _find(mt, _RU_SPOKEN, _RU_STAGE, _RU_STAGE_KIND, "ru")
        if lang in ("auto", "ja"):
            out += _find(mt, _JA_SPOKEN, _JA_STAGE, _JA_STAGE_KIND, "ja")
    else:
        out += _find(mt, _LAT_SPOKEN, _LAT_STAGE, _LAT_STAGE_KIND, "en")
    # the original text of every span (the mask only hid the laughs)
    return [Sound(s.start, s.end, t[s.start : s.end], s.kind, s.lang, s.stage) for s in _dedupe(out)]


def has_sound(text: str, lang: str = "auto") -> bool:
    return bool(find_sounds(text, lang))


def strip_sounds(text: str, lang: str = "auto", sounds: Iterable[Sound] | None = None) -> str:
    """``text`` without its sounds (and their stage directions); the rest is untouched."""
    ss = list(sounds) if sounds is not None else find_sounds(text, lang)
    return lg.spell_laughter(text, lambda _s: "", laughs=ss)  # type: ignore[arg-type]


def is_sound_only(text: str, lang: str = "auto") -> bool:
    """True when the line says nothing but a non-speech sound (stage directions and punctuation allowed)."""
    ss = find_sounds(text, lang)
    if not ss:
        return False
    rest = strip_sounds(text, lang, ss)
    return not lg._has_letters(rest) and not lg.find_laughs(text, "auto")


def dominant_kind(sounds: Iterable[Sound]) -> str | None:
    ss = list(sounds)
    if not ss:
        return None
    spoken = [s for s in ss if not s.stage]
    return (spoken or ss)[0].kind


def sound_kind(text: str, lang: str = "auto") -> str | None:
    return dominant_kind(find_sounds(text, lang))


def summarize(text: str, lang: str = "auto", sounds: Iterable[Sound] | None = None) -> dict[str, Any] | None:
    """``{kind, n, only, stage}`` of a line, ``None`` without a sound (what translate / text_emotion store)."""
    ss = list(sounds) if sounds is not None else find_sounds(text, lang)
    if not ss:
        return None
    spoken = [s for s in ss if not s.stage]
    kind = dominant_kind(ss) or "gasp"
    only = not lg._has_letters(strip_sounds(text, lang, ss)) and not lg.find_laughs(text, "auto")
    return {"kind": kind, "n": len(ss), "only": only, "stage": not spoken}


Renderer = Callable[[Sound], "str | None"]


def spell_sounds(text: str, render: Renderer, lang: str = "auto", sounds: Iterable[Sound] | None = None) -> str:
    """Replace every sound by ``render(sound)``; ``None`` keeps the original spelling, ``""`` removes it (stage directions).

    Same rules as :func:`vndub.text.laughter.spell_laughter`: the line's own punctuation wins over the spelling's, the
    capitalisation of a sound that opened the sentence is kept.
    """
    ss = list(sounds) if sounds is not None else find_sounds(text, lang)
    return lg.spell_laughter(text, render, laughs=ss)  # type: ignore[arg-type]


def table_renderer(table: Mapping[str, str], stage: str | None = "") -> Renderer:
    """Renderer from a ``{kind: spelling}`` table (``"any"`` = every kind without an entry).

    ``stage`` is what a bracketed stage direction becomes (``""`` = removed, ``None`` = left as written). A kind with no
    entry and no ``any`` keeps its spelling.
    """

    def render(s: Sound) -> str | None:
        if s.stage:
            return stage
        for key in (s.kind, "any"):
            v = table.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return None

    return render


# ----------------------------------------------------------------------------
# the spelling table (configs/sounds.yaml)
# ----------------------------------------------------------------------------

MODES = ("inline", "tag", "respell", "keep")
STAGE_MODES = ("remove", "respell")
_BACKEND_KEYS = ("mode", "token", "tokens", "spelling", "stage")
_LANG_KEY = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z]{2,4})?$|^default$")


def default_table_path() -> Path:
    return repo_root() / "configs" / "sounds.yaml"


def _check_lang_block(where: str, block: Any) -> dict[str, str]:
    if not isinstance(block, Mapping):
        raise ValueError(f"{where}: expected a mapping {{kind: spelling}}, got {type(block).__name__}")
    out: dict[str, str] = {}
    for kind, entry in block.items():
        if kind not in KINDS and kind != "any":
            raise ValueError(f"{where}: unknown sound kind {kind!r} (allowed: {', '.join(KINDS)}, any)")
        if entry is None:
            continue
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"{where}.{kind}: expected a non-empty spelling string")
        out[kind] = entry
    return out


def _check_backend(where: str, sec: Any) -> dict[str, Any]:
    if not isinstance(sec, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(sec).__name__}")
    unknown = sorted(set(sec) - set(_BACKEND_KEYS))
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {unknown} (allowed: {', '.join(_BACKEND_KEYS)})")
    out: dict[str, Any] = {}
    mode = sec.get("mode", "respell")
    if mode not in MODES:
        raise ValueError(f"{where}.mode: {mode!r} is not one of {', '.join(MODES)}")
    out["mode"] = mode
    if "token" in sec and sec["token"] is not None:
        if not isinstance(sec["token"], str) or not sec["token"].strip():
            raise ValueError(f"{where}.token: expected a non-empty string")
        out["token"] = sec["token"]
    if "tokens" in sec and sec["tokens"] is not None:
        toks = sec["tokens"]
        if not isinstance(toks, Mapping):
            raise ValueError(f"{where}.tokens: expected a mapping {{kind: token}}")
        for k, v in toks.items():
            if k not in KINDS:
                raise ValueError(f"{where}.tokens: unknown sound kind {k!r}")
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"{where}.tokens.{k}: expected a non-empty string")
        out["tokens"] = dict(toks)
    if mode in ("inline", "tag") and not out.get("token") and not out.get("tokens"):
        raise ValueError(f"{where}: mode {mode!r} needs 'token' or 'tokens'")
    if "spelling" in sec and sec["spelling"] is not None:
        if not isinstance(sec["spelling"], str) or not _LANG_KEY.match(sec["spelling"]):
            raise ValueError(f"{where}.spelling: expected a language code")
        out["spelling"] = sec["spelling"]
    if "stage" in sec and sec["stage"] is not None:
        if sec["stage"] not in STAGE_MODES:
            raise ValueError(f"{where}.stage: {sec['stage']!r} is not one of {', '.join(STAGE_MODES)}")
        out["stage"] = sec["stage"]
    return out


def validate_table(data: Any) -> dict[str, Any]:
    """The checked table: ``{"langs": {lang: {kind: spelling}}, "stage": remove|respell, "backends": {name: {...}}}``."""
    if not isinstance(data, Mapping):
        raise ValueError("sounds table: expected a mapping at the top level")
    langs: dict[str, dict[str, str]] = {}
    backends: dict[str, dict[str, Any]] = {}
    stage = "remove"
    for key, val in data.items():
        if key == "backends":
            if val is None:
                continue
            if not isinstance(val, Mapping):
                raise ValueError("sounds table: 'backends' must be a mapping")
            for name, sec in val.items():
                backends[str(name)] = _check_backend(f"backends.{name}", sec)
        elif key == "stage":
            if val not in STAGE_MODES:
                raise ValueError(f"sounds table: stage {val!r} is not one of {', '.join(STAGE_MODES)}")
            stage = val
        elif isinstance(key, str) and _LANG_KEY.match(key):
            langs[key] = _check_lang_block(key, val)
        else:
            raise ValueError(f"sounds table: unknown top-level key {key!r} (a language code, 'default', 'stage' or 'backends')")
    if not langs:
        raise ValueError("sounds table: no language block")
    return {"langs": langs, "stage": stage, "backends": backends}


def load_table(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else default_table_path()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return validate_table(data)


def lang_block(table: Mapping[str, Any], lang: str) -> dict[str, str]:
    """The spellings for ``lang`` (``ru-RU`` falls back to ``ru``, then ``default``, then ``en``)."""
    langs = table.get("langs") or {}
    code = (lang or "").lower()
    for key in (code, code.split("-")[0], "default", "en"):
        if key in langs:
            return dict(langs[key])
    return {}


def render_for_backend(
    text: str,
    lang: str,
    backend: str,
    table: Mapping[str, Any],
    sounds: Iterable[Sound] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """The TTS text with its sounds written for ``backend`` and the sidecar record ``{kind, n, mode, only, stage}``.

    ``respell`` (default) = the table spelling of the target language; ``inline`` = the backend token replaces the written
    sound; ``tag`` = the token goes before the spelling; ``keep`` = left as written (stage directions still follow
    ``stage``). A kind without a token under ``inline`` / ``tag`` is respelled.
    """
    ss = list(sounds) if sounds is not None else find_sounds(text, "auto")
    if not ss:
        return text, None
    sec = (table.get("backends") or {}).get(backend) or {}
    mode = sec.get("mode", "respell")
    spell = lang_block(table, sec.get("spelling") or lang)
    stage_mode = sec.get("stage") or table.get("stage") or "remove"
    tokens = dict(sec.get("tokens") or {})
    if sec.get("token"):
        for k in KINDS:
            tokens.setdefault(k, sec["token"])
    base = table_renderer(spell, stage="" if stage_mode == "remove" else None)

    def render(s: Sound) -> str | None:
        if s.stage:
            return "" if stage_mode == "remove" else base(Sound(s.start, s.end, s.text, s.kind, s.lang, False))
        if mode == "keep":
            return None
        tok = tokens.get(s.kind)
        if mode == "inline" and tok:
            return tok
        spelled = base(s)
        if mode == "tag" and tok:
            return tok + (spelled or s.text)
        return spelled

    out = spell_sounds(text, render, sounds=ss)
    rec = summarize(text, "auto", ss) or {}
    used = mode if (mode == "keep" or mode == "respell" or any(tokens.get(s.kind) for s in ss if not s.stage)) else "respell"
    rec["mode"] = used if used != "inline" and used != "tag" else f"{used}:{next((tokens[s.kind] for s in ss if not s.stage and tokens.get(s.kind)), '')}"
    return out, rec
