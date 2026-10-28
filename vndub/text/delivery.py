"""Delivery (how a line is *spoken*) derived from text, for lines without original audio.

Three evidence layers produce partial :class:`Layer` objects that :func:`merge_layers`
combines into one :class:`Delivery`:

``rule``  punctuation and markers of the line itself (``！`` ``？`` ``……`` ``♪`` elongation, laughter and
          sobbing tokens, whisper brackets, interjections, stammering), in the source text and, when
          available, the target text (``!`` ``?!`` ``...`` "ха-ха" ...). No model.
``cue``   speech verbs and manner adverbs in the narration lines around the dialogue line
          (``configs/delivery_cues.yaml``), see :func:`scan_cues` and ``stages/text_emotion.attribute_cues``.
``llm``   the ``delivery`` object the translation LLM returned for the line (validated by
          :func:`validate_delivery`).

Laughter (every source language, ``vndub.text.laughter``): a laugh in the source text or in the translation gives the
``laugh`` event, an emotion and a note by kind (hearty / giggle / chuckle -> happy, nervous -> embarrassed, evil and snort
-> other; "a hearty laugh", "giggling", "chuckling", "a nervous laugh", "a sinister laugh", "a snort"), a bigger intensity
bump for a long laugh (3+ syllables), and ``Delivery.laughter`` = ``{kind, units, stage, only}`` which reaches synth as
``StyleSpec.laughter`` (kind-specific instruct phrase; ``only`` = the line is nothing but a laugh). A bracketed stage
direction (``*laughs*``, ``(смеётся)``, ``(笑)``) is evidence too, with less weight than a spoken laugh.

A layer takes effect only when its confidence is at least ``min_confidence``; a line with no
accepted layer is *generic*: neutral emotion, intensity 0.3, normal tempo and loudness (the
performer's default delivery). Nothing here invents drama: absence of evidence means neutral.

The result is rendered through :class:`vndub.text.style_prompt.StyleSpec` by :func:`to_style`,
so every existing style renderer works unchanged.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from vndub.text import ja
from vndub.text.laughter import KIND_NOTE, KINDS, Laugh, find_laughs, is_laughter_only
from vndub.text.style_prompt import ALL_EMOTIONS, StyleSpec
from vndub.util import hash_obj, repo_root

EMOTIONS: tuple[str, ...] = tuple(ALL_EMOTIONS)
TEMPOS = ("slow", "normal", "fast")
LOUDNESSES = ("soft", "normal", "loud")
EVENTS = ("laugh", "sigh", "cry", "breath", "whisper")
ARCS = ("flat", "rising_end", "falling_end")
POSITIONS = ("any", "after", "before")
LAYERS = ("rule", "cue", "llm", "neutral")
CJK_LANGS = frozenset({"ja", "zh", "ko"})

# delivery.events (laugh ...) -> StyleSpec.events vocabulary used by the style renderers
STYLE_EVENT = {"laugh": "laughter", "sigh": "sigh", "cry": "cry", "breath": "breath", "whisper": "whisper"}

NEUTRAL_INTENSITY = 0.3
NEUTRAL_CONFIDENCE = 0.2
DEFAULT_CUE_CONFIDENCE = 0.75
MAX_CONFIDENCE = 0.95
_POS_CAP = 0.55  # cap on the summed positive intensity contributions of the rule pass


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------


@dataclass
class Layer:
    """Partial delivery from one evidence source. ``None`` fields mean "no opinion"."""

    name: str
    confidence: float = 0.0
    emotion: str | None = None
    intensity: float | None = None
    tempo: str | None = None
    loudness: str | None = None
    events: list[str] = field(default_factory=list)
    arc: str | None = None
    notes: list[str] = field(default_factory=list)
    cues: list[str] = field(default_factory=list)
    laughter: dict[str, Any] | None = None  # {kind, units, stage, only}: the laugh this layer found / was told about

    def summary(self) -> dict[str, Any]:
        return {
            "emotion": self.emotion, "intensity": self.intensity, "tempo": self.tempo, "loudness": self.loudness,
            "events": list(self.events), "confidence": round(self.confidence, 3),
        }


@dataclass
class Delivery:
    emotion: str = "neutral"
    intensity: float = NEUTRAL_INTENSITY
    tempo: str = "normal"
    loudness: str = "normal"
    events: list[str] = field(default_factory=list)
    confidence: float = NEUTRAL_CONFIDENCE
    layer: str = "neutral"
    cues: list[str] = field(default_factory=list)
    arc: str = "flat"
    notes: str = ""
    layers: dict[str, Any] = field(default_factory=dict)
    laughter: dict[str, Any] = field(default_factory=dict)  # {kind, units, stage, only}; empty = no laugh known

    @property
    def is_neutral(self) -> bool:
        return self.layer == "neutral"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "emotion": self.emotion,
            "intensity": round(float(self.intensity), 3),
            "tempo": self.tempo,
            "loudness": self.loudness,
            "events": list(self.events),
            "confidence": round(float(self.confidence), 3),
            "layer": self.layer,
            "cues": list(self.cues),
            "arc": self.arc,
            "notes": self.notes,
            "layers": self.layers,
        }
        if self.laughter:  # absent without a laugh: passports of earlier versions read the same
            d["laughter"] = dict(self.laughter)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Delivery":
        return cls(
            emotion=str(d.get("emotion", "neutral")), intensity=float(d.get("intensity", NEUTRAL_INTENSITY)),
            tempo=str(d.get("tempo", "normal")), loudness=str(d.get("loudness", "normal")),
            events=list(d.get("events") or []), confidence=float(d.get("confidence", NEUTRAL_CONFIDENCE)),
            layer=str(d.get("layer", "neutral")), cues=list(d.get("cues") or []), arc=str(d.get("arc", "flat")),
            notes=str(d.get("notes", "")), layers=dict(d.get("layers") or {}), laughter=dict(d.get("laughter") or {}),
        )


# ----------------------------------------------------------------------------
# rule pass
# ----------------------------------------------------------------------------


@dataclass
class _Sig:
    kind: str
    w: float
    emotion: str | None = None
    d_int: float = 0.0
    tempo: str | None = None
    loud: str | None = None
    events: tuple[str, ...] = ()
    arc: str | None = None
    note: str = ""
    match: str = ""
    laugh: dict[str, Any] | None = None  # the laugh signal only: {kind, units, stage, only}


def _add(sigs: dict[str, _Sig], sig: _Sig) -> None:
    """One signal per kind: the stronger evidence wins (the same mark in ja and ru is not counted twice)."""
    cur = sigs.get(sig.kind)
    if cur is None or sig.w > cur.w:
        sigs[sig.kind] = sig


def _prep(text: str) -> str:
    t = ja.strip_markup(ja.strip_ruby(text or ""))
    return unicodedata.normalize("NFKC", t).strip()


_BRACKET_TAG = re.compile(r"\[[^\]\n]*\]")
_BRACE_TAG = re.compile(r"\{[^}\n]*\}")


def _prep_spoken(text: str) -> str:
    """:func:`_prep` for the laughter finder: a bracketed stage direction (``[laughs]``, ``[giggles]``) is kept, because
    stripping ``[...]`` as a KAG tag made the laugh invisible in a Latin-script source (the translation was never stripped)."""
    t = ja.strip_ruby(text or "")
    t = _BRACKET_TAG.sub(lambda m: m.group(0) if any(l.stage for l in find_laughs(m.group(0))) else "", t)
    return unicodedata.normalize("NFKC", _BRACE_TAG.sub("", t)).strip()


_WC = "ぁ-んァ-ヶ一-龥々A-Za-z0-9"
_B = rf"(?<![{_WC}ー])"
_E = rf"(?![{_WC}])"
_TAIL = "[っッー〜~ぁぃぅぇぉ]*"

# Laughter is read by vndub.text.laughter (kana, Cyrillic and Latin; stage directions) in rule_pass, not by regexes here.
# kind -> (emotion, intensity bump of a short laugh, of a long one: 3+ syllables). "hearty / giggle / chuckle" are cheerful,
# a nervous laugh is embarrassed, an evil laugh and a snort have no better label than "other" (= expressive)
_LAUGH_KIND: dict[str, tuple[str, float, float]] = {
    "hearty": ("happy", 0.20, 0.30),
    "chuckle": ("happy", 0.12, 0.20),
    "giggle": ("happy", 0.15, 0.22),
    "nervous": ("embarrassed", 0.12, 0.20),
    "evil": ("other", 0.25, 0.35),
    "snort": ("other", 0.10, 0.15),
}
_LAUGH_W = 0.75  # weight of a spoken laugh
_LAUGH_W_STAGE = 0.65  # a bracketed stage direction ("*laughs*", "(笑)") is weaker evidence: it describes, it is not heard
_LAUGH_STAGE_FACTOR = 0.5  # ... and raises the intensity half as much
_OTHER_VOTE = 0.5  # "other" names no direction: it only wins when nothing more specific is named

_SOB_JA = re.compile(
    _B
    + "(?:"
    + "|".join(
        [
            "ぐすっ|ぐすん|ぐすぐす|ぐしゅ|ひっく|ひくっ|ひぐっ|ひっぐ|えぐっ|えぐえぐ|しくしく|めそめそ",
            "うわーん|うわあん|わーん|わあん|えーん|えええん|びえーん|ぴえん|うええん|ふええん",
            "ずびっ|ずびび|すんすん|うっうっ|う[うぅ]{2,}",
        ]
    )
    + ")"
    + _TAIL
    + _E
)
_WHIMPER_JA = re.compile(_B + "[うぅ]{2}" + _TAIL + _E)
_SOB_RU = re.compile(r"(?<![а-я])(?:хнык|всхлип|буэ+)|(?<![а-я])(?:ы-?){2,}(?![а-я])")

_WHISPER_MARK = re.compile(
    r"[(\[](?:小声|ひそひそ|ささやき|囁き?|ぼそっ|ぼそぼそ|こそこそ|内緒話|ないしょ話|whisper[^)\]]*|шепотом|шёпотом|тихо)[^)\]]{0,12}[)\]]",
    re.I,
)
_WHOLE_PAREN = re.compile(r"^\((?P<in>.+)\)$", re.S)
_WHOLE_ANGLE = re.compile(r"^[<≪〈《](?P<in>.+)[>≫〉》]$", re.S)

_ELLIPSIS = re.compile(r"\.{2,}|。{2,}|・{3,}")
_MUSIC = re.compile(r"[♪♫♬♩]")
_HEART = re.compile(r"[♡♥❤❣]")
_STAR = re.compile(r"[☆★]")
_CAPS = re.compile(r"(?<![A-Za-zА-ЯЁа-яё])[A-ZА-ЯЁ]{3,}(?![A-Za-zА-ЯЁа-яё])")
_STUTTER_JA = re.compile(r"^[「『(]?([ぁ-ん])[、,]\1")
_STUTTER_RU = re.compile(r"^[«\"(]?(?![эмаоуы])([а-я])-\1")

_LEAD_END = r"[っッー〜~ぁぃぅぇぉ]*(?=[\s、。,.!?…~〜♪♡」』)]|$)"
# (token regex, weight, d_int, tempo, loudness, events, note) per emotion; tokens are hiragana-normalized
_LEAD_JA: list[tuple[str, str | None, float, float, str | None, str | None, tuple[str, ...], str]] = [
    ("えっ|ええっ|えええ|えぇ|え(?=[?!])|うそっ?|うっそ|まさか|なっ|なんだって|ひゃっ|ひゃあ", "surprised", 0.6, 0.25, None, None, (), ""),
    ("あっ|あれ|はっ|わっ|わあ|わぁ|おおっ?|へえ|へぇ|ほう|ほお", "surprised", 0.5, 0.15, None, None, (), ""),
    ("ひっ|ひい|ひぃ|きゃっ|きゃあ|きゃー|たすけて|助けて", "fearful", 0.6, 0.3, None, None, (), ""),
    ("ちくしょう|畜生|くそっ?|ふざけるな|ふざけんな|うるさい|だまれ|黙れ|このやろう|この野郎|きさま|貴様|ゆるさない|許さない", "angry", 0.6, 0.3, None, None, (), ""),
    ("ちっ|ばかっ?|馬鹿|はあ(?=[?])|はぁ(?=[?])", "angry", 0.5, 0.2, None, None, (), ""),
    ("うげっ|うえっ|げっ|おえっ|きもっ|きもい|気持ち悪い", "disgusted", 0.6, 0.25, None, None, (), ""),
    ("やった[あぁー]*|わーい|わあい|やっほー?|ばんざい|万歳|よっしゃ|いえーい|うれしい|嬉しい|たのしい|楽しい|だいすき|大好き", "happy", 0.6, 0.2, None, None, (), ""),
    ("あうう?|はうう?|ひゃうう?|ふえ+|ふぇ+", "embarrassed", 0.6, 0.15, None, None, (), "flustered"),
    ("はあはあ|はぁはぁ|ぜえぜえ|ぜぇぜぇ|ふうふう|ふーふー|ふぅふぅ", None, 0.7, 0.15, None, None, ("breath",), "out of breath"),
    ("はあ(?![?])|はぁ(?![?])|ふう|ふぅ|やれやれ", None, 0.65, -0.05, "slow", "soft", ("sigh",), "sighing"),
    ("えーと|えっと|ええと|えと|あのー?|そのー?|うーん|んー|ううむ", None, 0.5, 0.0, "slow", None, (), "thinking aloud"),
    ("しーっ|しっ|しー", None, 0.6, -0.1, None, "soft", ("whisper",), "hushing"),
]
_LEAD_JA_RE = [(re.compile(r"^[「『(…\.\s―—-]*(?:" + tok + ")" + _LEAD_END), *rest) for tok, *rest in _LEAD_JA]

_LEAD_RU: list[tuple[str, str | None, float, float, str | None, str | None, tuple[str, ...], str]] = [
    ("ого+|ух ты|ничего себе|вау|ой|ай|ах|неужели", "surprised", 0.55, 0.25, None, None, (), ""),
    ("эх|ох|фух|уф|увы", None, 0.55, -0.05, "slow", "soft", ("sigh",), "sighing"),
    ("фу|тьфу|бр+", "disgusted", 0.55, 0.2, None, None, (), ""),
    ("ура+|йей", "happy", 0.55, 0.3, None, None, (), ""),
    ("черт|блин|проклятье|заткнись|идиот", "angry", 0.55, 0.3, None, None, (), ""),
    ("помогите|спасите|а-а-а|ааа+", "fearful", 0.55, 0.35, None, None, (), ""),
    ("э-?э+|м-?м+|хм+|эм+", None, 0.5, 0.0, "slow", None, (), "thinking aloud"),
    ("тс+|тише", None, 0.6, -0.1, None, "soft", ("whisper",), "hushing"),
]
_LEAD_RU_RE = [(re.compile(r"^[«\"(…\.\s—-]*(?:" + tok + r")(?=[\s,.!?…~-]|$)"), *rest) for tok, *rest in _LEAD_RU]


def _strip_quotes(t: str) -> str:
    t = t.strip()
    pairs = {"「": "」", "『": "』", "“": "”", "«": "»", '"': '"'}
    while len(t) >= 2 and t[0] in pairs and t[-1] == pairs[t[0]]:
        t = t[1:-1].strip()
    return t


def _signals_common(t: str, sigs: dict[str, _Sig]) -> None:
    """Marks shared by every language; ``t`` is NFKC text (full-width ！？ are already ASCII)."""
    inner = _strip_quotes(t)
    # exclamation marks
    total = t.count("!")
    if total:
        run = max(len(m) for m in re.findall(r"!+", t))
        if run >= 3:
            _add(sigs, _Sig("excl", 0.8, d_int=0.5, loud="loud", match="！！！"))
        elif run == 2:
            _add(sigs, _Sig("excl", 0.75, d_int=0.4, loud="loud", match="！！"))
        elif total >= 2:
            _add(sigs, _Sig("excl", 0.7, d_int=0.3, match="！"))
        else:
            _add(sigs, _Sig("excl", 0.65, d_int=0.25, match="！"))
    if re.search(r"!\?|\?!", t):
        _add(sigs, _Sig("interrobang", 0.7, emotion="surprised", d_int=0.15, arc="rising_end", match="？！"))
    tail = re.sub(r"[\s」』)）\"'”»~〜♪♡☆★.…]+$", "", t)
    if tail.endswith("?") or (tail.endswith("!") and re.search(r"\?!+$", tail)):
        _add(sigs, _Sig("question", 0.62, d_int=0.05, arc="rising_end", match="？"))
    # ellipsis
    runs = _ELLIPSIS.findall(t)
    if runs:
        longest = max(len(r) for r in runs)
        w = 0.65 if (longest >= 6 or len(runs) >= 2) else 0.55 if longest >= 3 else 0.45
        _add(sigs, _Sig("ellipsis", w, d_int=-0.05, tempo="slow", note="hesitant, with pauses", match="……" if longest >= 6 else "…"))
    # elongation: hiragana + ー, wave dash, repeated vowels / small kana / sokuon
    th = ja.kata_to_hira(t)
    kinds = 0
    found = ""
    t_el = re.sub(r"うーん|んー|えーと|あのー|そのー|ふーん", "", t)
    for rx, src in ((re.compile(r"[ぁ-ん]ー+"), t_el), (re.compile(r"[〜~]+"), t), (re.compile(r"([ぁぃぅぇぉあいうえお])\1{2,}"), th), (re.compile(r"[ぁぃぅぇぉ]{2,}"), th), (re.compile(r"っ{2,}"), th)):
        m = rx.search(src)
        if m:
            kinds += 1
            found = found or m.group(0)
    if kinds:
        _add(sigs, _Sig("elong", 0.4 if kinds == 1 else 0.5, d_int=0.1, note="drawn-out vowels", match="ー" if "ー" in found else "〜" if re.search(r"[〜~]", found) else found))
    # music / hearts / stars
    if (m := _MUSIC.search(t)):
        _add(sigs, _Sig("music", 0.7, emotion="happy", d_int=0.15, note="sing-song, cheerful", match=m.group(0)))
    if (m := _HEART.search(t)):
        _add(sigs, _Sig("heart", 0.55, emotion="happy", d_int=0.1, note="sweet, affectionate", match=m.group(0)))
    if (m := _STAR.search(t)):
        _add(sigs, _Sig("star", 0.45, emotion="happy", d_int=0.1, note="playful", match=m.group(0)))
    # whisper / inner voice
    if (m := _WHISPER_MARK.search(t)):
        _add(sigs, _Sig("whisper", 0.85, d_int=-0.1, loud="soft", events=("whisper",), note="whispered", match=m.group(0)))
    else:
        pm = _WHOLE_PAREN.match(inner) or _WHOLE_PAREN.match(t)
        if pm:
            _add(sigs, _Sig("inner", 0.62, d_int=-0.05, loud="soft", note="inner voice, soft", match="（）"))
        elif _WHOLE_ANGLE.match(inner):
            _add(sigs, _Sig("inner", 0.5, d_int=-0.05, loud="soft", note="hushed", match="＜＞"))
    # shouting in capitals (Latin / Cyrillic)
    if (m := _CAPS.search(t)) and (len(m.group(0)) >= 4 or "!" in t or len(_CAPS.findall(t)) >= 2):
        _add(sigs, _Sig("caps", 0.6, d_int=0.3, loud="loud", match=m.group(0)))


def _signals_ja(text: str, sigs: dict[str, _Sig]) -> None:
    t = _prep(text)
    if not t:
        return
    _signals_common(t, sigs)
    th = ja.kata_to_hira(t)
    if (m := _SOB_JA.search(th)):
        _add(sigs, _Sig("sob", 0.75, emotion="sad", d_int=0.2, events=("cry",), note="tearful", match=m.group(0)))
    elif (m := _WHIMPER_JA.search(th)):
        _add(sigs, _Sig("sob", 0.5, emotion="sad", d_int=0.1, events=("cry",), note="tearful", match=m.group(0)))
    if (m := _STUTTER_JA.match(th)):
        _add(sigs, _Sig("stutter", 0.6, emotion="embarrassed", d_int=0.1, note="stammering, flustered", match=m.group(0)))
    for rx, emo, w, d, tempo, loud, ev, note in _LEAD_JA_RE:
        if (m := rx.match(th)):
            _add(sigs, _Sig(f"lead:{emo or ev or note}", w, emotion=emo, d_int=d, tempo=tempo, loud=loud, events=ev, note=note, match=m.group(0).strip("「『(…. 　―—-")))


def _ru_tokens(t: str, sigs: dict[str, _Sig]) -> None:
    """Sobbing tokens of a lowercased, ё-free Russian text (laughter: :func:`_laugh_signal`)."""
    if (m := _SOB_RU.search(t)):
        _add(sigs, _Sig("sob", 0.7, emotion="sad", d_int=0.2, events=("cry",), note="tearful", match=m.group(0)))


def _signals_ru(text: str, sigs: dict[str, _Sig]) -> None:
    raw = unicodedata.normalize("NFKC", text or "").strip()
    if not raw:
        return
    _signals_common(raw, sigs)
    t = raw.lower().replace("ё", "е")
    _ru_tokens(t, sigs)
    if (m := _STUTTER_RU.match(t)):
        _add(sigs, _Sig("stutter", 0.6, emotion="embarrassed", d_int=0.1, note="stammering, flustered", match=m.group(0)))
    for rx, emo, w, d, tempo, loud, ev, note in _LEAD_RU_RE:
        if (m := rx.match(t)):
            _add(sigs, _Sig(f"lead:{emo or ev or note}", w, emotion=emo, d_int=d, tempo=tempo, loud=loud, events=ev, note=note, match=m.group(0).strip("«\"(…. —-")))


def _laugh_signal(laughs: Sequence[Laugh], only: bool) -> _Sig | None:
    """The one laugh signal of a line from every laugh found in its texts (source first): the longest spoken laugh sets the
    kind; a line whose laughs are all stage directions gets the weaker evidence."""
    if not laughs:
        return None
    spoken = [l for l in laughs if not l.stage]
    best = max(spoken or laughs, key=lambda l: l.units)  # the first of equally long laughs: the source text
    stage = not spoken
    emotion, short, long_ = _LAUGH_KIND[best.kind]
    d_int = (long_ if best.long else short) * (_LAUGH_STAGE_FACTOR if stage else 1.0)
    info = {"kind": best.kind, "units": best.units, "stage": stage, "only": bool(only)}
    return _Sig(
        "laugh", _LAUGH_W_STAGE if stage else _LAUGH_W, emotion=emotion, d_int=round(d_int, 3), events=("laugh",),
        note=KIND_NOTE[best.kind], match=best.text.strip(), laugh=info,
    )


def _is_bare_stage(text: str) -> bool:
    """The whole text (quotes aside) is one bracketed stage direction."""
    ls = find_laughs(text)
    return len(ls) == 1 and ls[0].stage and ls[0].text.strip() == _strip_quotes(text)


def rule_pass(text_src: str, text_tgt: str = "", lexicon: Mapping[str, str] | None = None, src_lang: str = "auto") -> Layer:
    """Rule layer of one line: punctuation, markers, laughter/sobbing tokens, whisper brackets, interjections.

    ``text_tgt`` (the translation) adds evidence the source does not carry (``ха-ха``, ``?!``); the same
    mark found in both languages counts once. ``lexicon`` (``configs/interjections.yaml``) supplies the
    Russian equivalent of an interjection line when there is no translation yet. ``src_lang`` (``ja`` | ``en`` |
    ``ru`` | any other code = Latin script | ``auto`` = every script) says which laughter spellings the source
    text can contain.
    """
    sigs: dict[str, _Sig] = {}
    _signals_ja(text_src, sigs)
    tgt = (text_tgt or "").strip()
    spoken_text = _prep_spoken(text_src)
    laughs = find_laughs(spoken_text, src_lang or "auto")
    if tgt:
        _signals_ru(tgt, sigs)
        tgt_norm = unicodedata.normalize("NFKC", tgt)
        laughs += find_laughs(tgt_norm)
        spoken_text = tgt_norm  # the dub speaks the translation
    elif lexicon:
        # no translation yet: the lexicon's Russian equivalent of an interjection line only adds its onomatopoeia
        # (хи-хи, ахаха); everything else it would say is already read from the Japanese
        mapped = ja.match_interjection(text_src, lexicon) or ""
        mapped = unicodedata.normalize("NFKC", mapped)
        _ru_tokens(mapped.lower().replace("ё", "е"), sigs)
        laughs += find_laughs(mapped, "ru")
    lsig = _laugh_signal(laughs, bool(laughs) and is_laughter_only(spoken_text))
    if lsig is not None:
        if lsig.laugh and lsig.laugh["stage"] and any(_is_bare_stage(t) for t in (_prep_spoken(text_src), spoken_text)):
            sigs.pop("inner", None)  # a line that is only "(笑)" / "(giggles)" is a stage direction, not an inner voice
        _add(sigs, lsig)
    if not sigs:
        return Layer("rule", confidence=NEUTRAL_CONFIDENCE, emotion="neutral", intensity=NEUTRAL_INTENSITY)

    ordered = sorted(sigs.values(), key=lambda s: -s.w)
    # emotion: weighted vote of the signals that name one
    votes: dict[str, float] = {}
    for s in ordered:
        if s.emotion:
            votes[s.emotion] = votes.get(s.emotion, 0.0) + s.w * (_OTHER_VOTE if s.emotion == "other" else 1.0)
    pos = min(_POS_CAP, sum(s.d_int for s in ordered if s.d_int > 0))
    neg = sum(-s.d_int for s in ordered if s.d_int < 0)
    intensity = round(min(0.9, max(0.1, NEUTRAL_INTENSITY + pos - neg)), 3)
    emotion: str | None
    if votes:
        emotion = max(votes, key=lambda e: (votes[e], e != "other"))
    else:
        emotion = "other" if intensity >= 0.5 else "neutral"
    # confidence: noisy-OR of the evidence, reduced when the named emotions disagree
    miss = 1.0
    for s in ordered:
        miss *= 1.0 - s.w
    conf = 1.0 - miss
    if votes:
        conf *= votes[emotion] / sum(votes.values())
    conf = round(min(MAX_CONFIDENCE, conf), 3)

    def first(attr: str) -> Any:
        return next((getattr(s, attr) for s in ordered if getattr(s, attr)), None)

    events: list[str] = []
    for s in ordered:
        for e in s.events:
            if e not in events:
                events.append(e)
    notes: list[str] = []
    for s in ordered:
        if s.note and s.note not in notes:
            notes.append(s.note)
    cues = list(dict.fromkeys(s.match for s in ordered if s.match))
    tempos = {s.tempo for s in ordered if s.tempo}
    return Layer(
        "rule", confidence=conf, emotion=emotion, intensity=intensity,
        tempo=first("tempo") if len(tempos) == 1 else None, loudness=first("loud"),
        events=events, arc=first("arc"), notes=notes, cues=cues, laughter=dict(sigs["laugh"].laugh) if "laugh" in sigs and sigs["laugh"].laugh else None,
    )


# ----------------------------------------------------------------------------
# cue table
# ----------------------------------------------------------------------------

_CUE_KEYS = ("cue", "emotion", "intensity", "tempo", "loudness", "events", "confidence", "position", "notes")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_cues(data: Any) -> dict[str, list[dict[str, Any]]]:
    """Validate and normalize a cue table; raises ``ValueError`` with a message that names the entry.

    Shape: ``{lang: [{cue, emotion?, intensity?, tempo?, loudness?, events?, confidence?, position?, notes?}]}``.
    Normalized entries carry every key (``None`` = "no opinion"; ``events`` ``[]``; ``confidence`` 0.75;
    ``position`` ``any``; ``notes`` ``""``).
    """
    if not isinstance(data, Mapping):
        raise ValueError("cue table must be a mapping {language: [entries]}")
    out: dict[str, list[dict[str, Any]]] = {}
    for lang, entries in data.items():
        lang_s = str(lang)
        if not isinstance(entries, list):
            raise ValueError(f"cues.{lang_s}: expected a list of entries, got {type(entries).__name__}")
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for i, e in enumerate(entries):
            where = f"cues.{lang_s}[{i}]"
            if not isinstance(e, Mapping):
                raise ValueError(f"{where}: entry must be a mapping, got {type(e).__name__}")
            unknown = sorted(set(e) - set(_CUE_KEYS))
            if unknown:
                raise ValueError(f"{where}: unknown field(s) {unknown} (allowed: {', '.join(_CUE_KEYS)})")
            cue = e.get("cue")
            if not isinstance(cue, str) or not cue.strip():
                raise ValueError(f"{where}: 'cue' must be a non-empty string")
            cue = cue.strip()
            where = f"{where} ({cue!r})"
            emotion = e.get("emotion")
            if emotion is not None and emotion not in EMOTIONS:
                raise ValueError(f"{where}: unknown emotion {emotion!r} (allowed: {', '.join(EMOTIONS)})")
            intensity = e.get("intensity")
            if intensity is not None and (not _is_number(intensity) or not 0.0 <= float(intensity) <= 1.0):
                raise ValueError(f"{where}: intensity must be a number in 0..1, got {intensity!r}")
            tempo = e.get("tempo")
            if tempo is not None and tempo not in TEMPOS:
                raise ValueError(f"{where}: tempo must be one of {', '.join(TEMPOS)}, got {tempo!r}")
            loudness = e.get("loudness")
            if loudness is not None and loudness not in LOUDNESSES:
                raise ValueError(f"{where}: loudness must be one of {', '.join(LOUDNESSES)}, got {loudness!r}")
            events = e.get("events") or []
            if not isinstance(events, list) or any(ev not in EVENTS for ev in events):
                raise ValueError(f"{where}: events must be a list drawn from {', '.join(EVENTS)}, got {events!r}")
            conf = e.get("confidence", DEFAULT_CUE_CONFIDENCE)
            if conf is None:
                conf = DEFAULT_CUE_CONFIDENCE
            if not _is_number(conf) or not 0.0 <= float(conf) <= 1.0:
                raise ValueError(f"{where}: confidence must be a number in 0..1, got {conf!r}")
            position = e.get("position") or "any"
            if position not in POSITIONS:
                raise ValueError(f"{where}: position must be one of {', '.join(POSITIONS)}, got {position!r}")
            notes = e.get("notes") or ""
            if not isinstance(notes, str):
                raise ValueError(f"{where}: notes must be a string")
            if emotion is None and intensity is None and tempo is None and loudness is None and not events:
                raise ValueError(f"{where}: the entry sets nothing (give emotion, intensity, tempo, loudness or events)")
            key = _norm_cue(lang_s, cue)
            if key in seen:
                raise ValueError(f"{where}: duplicate cue in cues.{lang_s}")
            seen.add(key)
            rows.append({
                "cue": cue, "emotion": emotion, "intensity": None if intensity is None else float(intensity),
                "tempo": tempo, "loudness": loudness, "events": list(events), "confidence": float(conf),
                "position": position, "notes": notes.strip(),
            })
        out[lang_s] = rows
    return out


_CUES_HEADER = """\
# Delivery cues for the text_emotion stage (narration -> how the neighbouring dialogue line is spoken).
# Edited from the VNDub app or by hand. Per language: a list of entries
#   {cue, emotion, intensity, tempo, loudness, events, confidence, position, notes}
# cue         ja: substring of the narration line (kana/katakana are compared as hiragana, spaces ignored).
#             other languages: word STEMS, matched as word prefixes (воскликнул matches воскликнула / воскликнули);
#             several words must follow each other; "ё" = "е"; a reflexive ending (-ся/-сь) is ignored, so
#             усмехнулся matches усмехнулась.
# emotion     angry disgusted fearful happy neutral other sad surprised unknown embarrassed (omit = no opinion)
# intensity   0..1     tempo  slow|normal|fast     loudness  soft|normal|loud     events  laugh sigh cry breath whisper
# confidence  0..1, default 0.75; a cue below text_emotion.min_confidence (default 0.6) after attribution is ignored
# position    after  = tag that follows the speech (...と叫んだ, "воскликнул"): describes the PRECEDING line
#             before = lead-in that precedes the speech: describes the FOLLOWING line
#             any    = nearest line; the speaker's name in the narration decides first
# Attribution: a cue applies to the dialogue line it describes: the line whose speaker name appears in the
# narration; otherwise the nearest line on the side given by position (the preceding line on a tie).
"""


def _entry_yaml(e: Mapping[str, Any]) -> str:
    row: dict[str, Any] = {"cue": e["cue"]}
    for k in ("emotion", "intensity", "tempo", "loudness"):
        if e.get(k) is not None:
            row[k] = e[k]
    if e.get("events"):
        row["events"] = list(e["events"])
    if e.get("confidence") is not None and abs(float(e["confidence"]) - DEFAULT_CUE_CONFIDENCE) > 1e-9:
        row["confidence"] = round(float(e["confidence"]), 3)
    if e.get("position") and e["position"] != "any":
        row["position"] = e["position"]
    if e.get("notes"):
        row["notes"] = e["notes"]
    return yaml.safe_dump(row, default_flow_style=True, allow_unicode=True, width=10_000, sort_keys=False).strip()


def load_cues(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Read and validate ``configs/delivery_cues.yaml``. ``FileNotFoundError`` / ``ValueError`` on problems."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"delivery cue table not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{p.name}: not valid YAML: {e}") from e
    try:
        return validate_cues(data)
    except ValueError as e:
        raise ValueError(f"{p.name}: {e}") from e


def save_cues(path: str | Path, data: Mapping[str, Any]) -> None:
    """Validate ``data`` and write it atomically as the cue table (``ValueError`` before touching the file)."""
    from vndub.manifest.io import _atomic_write_bytes

    table = validate_cues(data)
    lines = [_CUES_HEADER.rstrip("\n"), ""]
    for lang, rows in table.items():
        lines.append(f"{lang}:")
        lines += [f"  - {_entry_yaml(r)}" for r in rows]
        lines.append("")
    _atomic_write_bytes(Path(path), ("\n".join(lines)).encode("utf-8"))


def cue_table_hash(table: Mapping[str, Any]) -> str:
    return hash_obj({k: table[k] for k in sorted(table)})


def default_cues_path() -> Path:
    return repo_root() / "configs" / "delivery_cues.yaml"


@functools.lru_cache(maxsize=1)
def default_cues() -> "CompiledCues":
    try:
        return compile_cues(load_cues(default_cues_path()))
    except (FileNotFoundError, ValueError):
        return compile_cues({})


def _ja_norm(s: str) -> str:
    t = unicodedata.normalize("NFKC", ja.strip_markup(ja.strip_ruby(s or "")))
    return ja.kata_to_hira(t).replace(" ", "").replace("　", "").lower()


def _alpha_norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").lower().replace("ё", "е")


def _norm_cue(lang: str, cue: str) -> str:
    return _ja_norm(cue) if lang in CJK_LANGS else " ".join(_alpha_norm(cue).split())


@dataclass
class CueHit:
    """A cue found in a narration line (``confidence`` is adjusted by the attribution step)."""

    lang: str
    cue: str
    entry: dict[str, Any]
    line_id: str = ""
    confidence: float = DEFAULT_CUE_CONFIDENCE
    distance: int = 0
    by_name: bool = False


@dataclass
class CompiledCues:
    table: dict[str, list[dict[str, Any]]]
    hash: str
    _matchers: dict[str, list[tuple[dict[str, Any], str, Any]]] = field(default_factory=dict)

    def langs(self) -> list[str]:
        return list(self._matchers)


def compile_cues(table: Mapping[str, Any]) -> CompiledCues:
    norm = validate_cues(table)
    matchers: dict[str, list[tuple[dict[str, Any], str, Any]]] = {}
    for lang, rows in norm.items():
        ms = []
        for e in rows:
            key = _norm_cue(lang, e["cue"])
            if lang in CJK_LANGS:
                ms.append((e, key, None))
            else:
                # reflexive endings are dropped from the stem: усмехнулся -> усмехнул matches усмехнулась too
                toks = [tk[:-2] if len(tk) > 5 and tk.endswith(("ся", "сь")) else tk for tk in key.split()]
                rx = re.compile(r"(?<!\w)" + r"\s+".join(re.escape(tk) + r"\w*" for tk in toks))
                ms.append((e, key, rx))
        matchers[lang] = ms
    return CompiledCues(table=norm, hash=cue_table_hash(norm), _matchers=matchers)


def scan_cues(text: str, lang: str, cues: CompiledCues) -> list[CueHit]:
    """Cue entries of ``lang`` found in ``text``; a cue contained in a longer matching cue is dropped."""
    ms = cues._matchers.get(lang)
    if not ms or not (text or "").strip():
        return []
    if lang in CJK_LANGS:
        hay = _ja_norm(text)
        found = [(e, key) for e, key, _ in ms if key and key in hay]
    else:
        hay = _alpha_norm(text)
        found = [(e, key) for e, key, rx in ms if rx.search(hay)]
    keep: list[CueHit] = []
    for e, key in found:
        if any(key != k2 and key in k2 for _, k2 in found):
            continue
        keep.append(CueHit(lang=lang, cue=e["cue"], entry=e, confidence=float(e["confidence"])))
    return keep


def cue_layer(hits: Sequence[CueHit]) -> Layer | None:
    """Combine the cue hits attributed to one dialogue line into a cue layer (``None`` without hits)."""
    if not hits:
        return None
    ordered = sorted(hits, key=lambda h: (-h.confidence, -len(h.cue)))
    best = ordered[0]
    specific = [h for h in ordered if h.entry.get("emotion") not in (None, "neutral", "other", "unknown")]
    emo_src = specific[0] if specific else next((h for h in ordered if h.entry.get("emotion")), None)
    emotion = emo_src.entry["emotion"] if emo_src else None
    # hits that name a different specific emotion than the winner do not contribute their other fields
    compat = [h for h in ordered if not emotion or h.entry.get("emotion") in (emotion, None, "other", "neutral")]
    intensities = [h.entry["intensity"] for h in compat if h.entry.get("intensity") is not None]
    events: list[str] = []
    notes: list[str] = []
    for h in compat:
        for ev in h.entry.get("events") or []:
            if ev not in events:
                events.append(ev)
        if h.entry.get("notes") and h.entry["notes"] not in notes:
            notes.append(h.entry["notes"])
    tempo = next((h.entry["tempo"] for h in compat if h.entry.get("tempo")), None)
    loud = next((h.entry["loudness"] for h in compat if h.entry.get("loudness")), None)
    conf = best.confidence
    agree = [h for h in compat if h is not best]
    conf += 0.05 * len(agree)
    rival = {h.entry["emotion"] for h in specific if abs(h.confidence - best.confidence) <= 0.1}
    if len(rival) > 1:
        conf -= 0.15
    return Layer(
        "cue", confidence=round(max(0.0, min(MAX_CONFIDENCE, conf)), 3), emotion=emotion,
        intensity=max(intensities) if intensities else None, tempo=tempo, loudness=loud,
        events=events, notes=notes, cues=list(dict.fromkeys(h.cue for h in ordered)),
    )


# ----------------------------------------------------------------------------
# LLM layer (validation) and the mock LLM
# ----------------------------------------------------------------------------

_EVENT_ALIAS = {"laughter": "laugh", "laughing": "laugh", "crying": "cry", "sob": "cry", "sighing": "sigh", "whispering": "whisper"}
# an event word that names the kind of laugh: the event is ``laugh`` and the kind is remembered
_LAUGH_EVENT_KIND = {
    "giggle": "giggle", "giggling": "giggle", "chuckle": "chuckle", "chuckling": "chuckle", "snort": "snort", "snorting": "snort",
    "cackle": "evil", "cackling": "evil", "evil_laugh": "evil", "nervous_laugh": "nervous", "hearty_laugh": "hearty", "guffaw": "hearty",
}
_LAUGH_KIND_ALIAS = {**_LAUGH_EVENT_KIND, "laugh": "hearty", "laughing": "hearty", "laughter": "hearty", "sinister": "evil", "nervous_laughter": "nervous"}


def _laugh_kind_of(value: Any) -> str | None:
    """A laugh kind from an LLM word (``giggling``, ``Evil laugh``, ``nervous``); ``None`` when it names none."""
    if not isinstance(value, str):
        return None
    w = value.strip().lower().replace(" ", "_").replace("-", "_")
    return w if w in KINDS else _LAUGH_KIND_ALIAS.get(w)


def validate_delivery(obj: Any, line_id: str | None = None) -> dict[str, Any] | None:
    """Strict check of an LLM ``delivery`` object. Returns the normalized dict or ``None`` (field dropped).

    Required: emotion (one of EMOTIONS), intensity and confidence (numbers in 0..1), tempo, loudness;
    ``events`` optional (subset of EVENTS, ``laughter`` is accepted as ``laugh``; ``giggling`` / ``chuckling`` /
    ``snort`` ... are accepted as ``laugh`` of that kind). Optional ``laugh_kind`` (also read from a string
    ``laughter``): one of hearty chuckle giggle evil nervous snort or a word for it; an unknown word is ignored, it
    does not invalidate the object, and a kind implies the ``laugh`` event. Out-of-range numbers, unknown enum
    values and a foreign ``id`` invalidate the whole object.
    """
    if not isinstance(obj, Mapping):
        return None
    if line_id is not None and "id" in obj and str(obj["id"]) != str(line_id):
        return None
    emotion = obj.get("emotion")
    emotion = emotion.strip().lower() if isinstance(emotion, str) else None
    if emotion not in EMOTIONS:
        return None
    inten, conf = obj.get("intensity"), obj.get("confidence")
    if not (_is_number(inten) and _is_number(conf)) or not (0.0 <= float(inten) <= 1.0 and 0.0 <= float(conf) <= 1.0):
        return None
    tempo, loud = obj.get("tempo"), obj.get("loudness")
    tempo = tempo.strip().lower() if isinstance(tempo, str) else None
    loud = loud.strip().lower() if isinstance(loud, str) else None
    if tempo not in TEMPOS or loud not in LOUDNESSES:
        return None
    raw_events = obj.get("events", [])
    if raw_events is None:
        raw_events = []
    if not isinstance(raw_events, list):
        return None
    events: list[str] = []
    kind = _laugh_kind_of(obj.get("laugh_kind")) or _laugh_kind_of(obj.get("laughter"))
    for ev in raw_events:
        if not isinstance(ev, str):
            return None
        word = ev.strip().lower().replace(" ", "_")
        if word in _LAUGH_EVENT_KIND:
            kind = kind or _LAUGH_EVENT_KIND[word]
            word = "laugh"
        ev = _EVENT_ALIAS.get(word, word)
        if ev not in EVENTS:
            return None
        if ev not in events:
            events.append(ev)
    out = {
        "emotion": emotion, "intensity": round(float(inten), 3), "tempo": tempo, "loudness": loud,
        "events": events, "confidence": round(float(conf), 3),
    }
    if kind:
        out["laugh_kind"] = kind
        if "laugh" not in events:
            events.append("laugh")
    return out


def llm_layer(d: Mapping[str, Any] | None) -> Layer | None:
    v = validate_delivery(d) if d else None
    if v is None:
        return None
    return Layer(
        "llm", confidence=v["confidence"], emotion=v["emotion"], intensity=v["intensity"], tempo=v["tempo"],
        loudness=v["loudness"], events=list(v["events"]), laughter={"kind": v["laugh_kind"]} if v.get("laugh_kind") else None,
    )


_MOCK_KEYWORDS: list[tuple[str, float, tuple[str, ...]]] = [
    ("happy", 0.8, ("ありがとう", "嬉しい", "うれしい", "やった", "楽しい", "大好き", "だいすき")),
    ("sad", 0.75, ("ごめん", "悲しい", "かなしい", "寂しい", "さびしい", "つらい")),
    ("angry", 0.8, ("ふざけ", "許さ", "ゆるさ", "うるさい", "バカ", "ばか")),
    ("fearful", 0.75, ("怖い", "こわい", "助けて", "たすけて")),
    ("embarrassed", 0.7, ("恥ずかしい", "はずかしい", "照れ")),
]


def mock_delivery(text: str, narration: Sequence[str] = (), src_lang: str = "auto") -> dict[str, Any]:
    """Deterministic, plausible stand-in for the LLM delivery answer (used by MockTranslator).

    Takes the strongest of: rule pass of the line, cue scan of the neighbouring narration lines, a tiny
    keyword lexicon. Without evidence it answers what the real prompt asks for: neutral, 0.3, low confidence.
    A laugh in the text comes back as the ``laugh`` event with its ``laugh_kind``, like the real prompt may answer.
    """
    cands: list[tuple[float, dict[str, Any]]] = []
    r = rule_pass(text, src_lang=src_lang)
    if r.confidence >= 0.5 and r.cues:
        cand = {"emotion": r.emotion, "intensity": r.intensity, "tempo": r.tempo or "normal", "loudness": r.loudness or "normal", "events": r.events}
        if r.laughter and r.laughter.get("kind"):
            cand["laugh_kind"] = r.laughter["kind"]
        cands.append((r.confidence + 0.1, cand))
    hits: list[CueHit] = []
    cc = default_cues()
    for n in narration:
        hits += scan_cues(n, "ja", cc)
    cl = cue_layer(hits)
    if cl is not None and cl.confidence >= 0.5:
        cands.append((cl.confidence + 0.1, {"emotion": cl.emotion or "neutral", "intensity": cl.intensity if cl.intensity is not None else 0.6, "tempo": cl.tempo or "normal", "loudness": cl.loudness or "normal", "events": cl.events}))
    for emo, conf, words in _MOCK_KEYWORDS:
        if any(w in text for w in words):
            cands.append((conf, {"emotion": emo, "intensity": 0.6, "tempo": "normal", "loudness": "normal", "events": []}))
    if not cands:
        return {"emotion": "neutral", "intensity": 0.3, "tempo": "normal", "loudness": "normal", "events": [], "confidence": 0.3}
    conf, d = max(cands, key=lambda c: c[0])
    d = dict(d)
    d["confidence"] = round(min(MAX_CONFIDENCE, conf), 3)
    return validate_delivery(d) or {"emotion": "neutral", "intensity": 0.3, "tempo": "normal", "loudness": "normal", "events": [], "confidence": 0.3}


# ----------------------------------------------------------------------------
# merge and StyleSpec
# ----------------------------------------------------------------------------


def merge_layers(layers: Iterable[Layer | None], min_confidence: float = 0.6) -> Delivery:
    """rule -> cue -> llm (the given order). A layer is applied only when ``confidence >= min_confidence``;
    a later accepted layer overrides the fields it sets, events/notes/cues accumulate. No accepted layer
    -> the generic neutral delivery (``layer == "neutral"``)."""
    d = Delivery()
    seen: dict[str, Any] = {}
    accepted = False
    events: list[str] = []
    notes: list[str] = []
    cues: list[str] = []
    laughter: dict[str, Any] = {}
    for lyr in layers:
        if lyr is None:
            continue
        ok = round(lyr.confidence, 3) >= round(min_confidence, 3)
        seen[lyr.name] = {**lyr.summary(), "accepted": ok}
        if not ok:
            continue
        accepted = True
        if lyr.emotion is not None:
            d.emotion = lyr.emotion
        if lyr.intensity is not None:
            d.intensity = lyr.intensity
        if lyr.tempo is not None:
            d.tempo = lyr.tempo
        if lyr.loudness is not None:
            d.loudness = lyr.loudness
        if lyr.arc is not None:
            d.arc = lyr.arc
        events += [e for e in lyr.events if e not in events]
        notes += [n for n in lyr.notes if n not in notes]
        cues += [c for c in lyr.cues if c not in cues]
        if lyr.laughter:
            laughter = {**lyr.laughter, **laughter}  # an earlier layer wins: the rule pass read the actual text, the LLM only guessed a kind
        d.layer = lyr.name
        d.confidence = round(lyr.confidence, 3)
    d.layers = seen
    if not accepted:
        return d
    d.events = events
    d.notes = ", ".join(notes)
    d.cues = cues
    d.laughter = laughter if "laugh" in events else {}
    return d


def to_style(d: Delivery, notes: str = "") -> StyleSpec:
    """Delivery -> :class:`StyleSpec` (events renamed to the StyleSpec vocabulary)."""
    txt = ", ".join(x for x in (d.notes, notes.strip()) if x)
    return StyleSpec(
        emotion=d.emotion, intensity=round(float(d.intensity), 3), tempo=d.tempo, loudness=d.loudness,
        events=[STYLE_EVENT.get(e, e) for e in d.events], voice_quality=[], arc=d.arc, notes=txt,
        confidence=round(float(d.confidence), 3), emotion_probs={d.emotion: 1.0},
        laughter=dict(d.laughter) if d.laughter and "laugh" in d.events else None,
    )
