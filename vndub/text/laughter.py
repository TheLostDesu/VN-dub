"""Laughter in dialogue text: where it is, what kind it is, and how to respell it for a TTS.

Written laughter is onomatopoeia a text-to-speech model reads letter by letter unless it is spelled the way
the model saw laughter in its training data (or replaced by the model's own laughter token). This module is
the shared core for the text (``delivery`` events), synth (respelling / tags) and score (CER reference)
sides:

* :func:`find_laughs` returns every laugh span of a line with its ``kind`` and ``lang``; the three scripts
  (kana, Cyrillic, Latin) do not overlap, so ``lang="auto"`` simply runs all three.
* :func:`is_laughter_only` tells whether a line is nothing but laughter (plus punctuation), i.e. an
  interjection whose transcript cannot be checked word by word.
* :func:`strip_laughter` removes the laughs (CER reference, "what else does the line say").
* :func:`spell_laughter` rewrites each laugh through a renderer (a plain respelling table, or a backend's
  inline event tag); :func:`table_renderer` builds such a renderer from a ``{kind: {short, long}}`` table.

Kinds (:data:`KINDS`): ``hearty`` (ha-ha, ахаха, あはは), ``chuckle`` (heh, хе-хе, へへ), ``giggle`` (hihi,
хи-хи, ふふ, くすくす), ``evil`` (mwahaha, муахаха, くくく), ``nervous`` (ehehe, えへへ, a laugh trailing into
"..."), ``snort`` (pfft, пфф, ぷっ). Stage directions in brackets (``*laughs*``, ``(giggles)``, ``(смеётся)``,
``(笑)``) are laughs too (``stage=True``): they describe the delivery but must not be read aloud.

The second half of the module (below :func:`table_renderer`) is what the pipeline stages add on top: kind notes
and :func:`summarize` (``text_emotion`` / ``StyleSpec.laughter``), the spelling table ``configs/laughter.yaml``
(:func:`load_table`) and :func:`apply_laughter`, which synth uses to turn the laughs of a TTS text into the
spelling / inline token the chosen backend voices best (modes ``respell`` | ``inline`` | ``tag`` | ``keep``).
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from vndub.text import ja
from vndub.util import repo_root

KINDS = ("hearty", "chuckle", "giggle", "evil", "nervous", "snort")

# ----------------------------------------------------------------------------
# patterns
# ----------------------------------------------------------------------------

_LAT_B = r"(?<![A-Za-z])"
_LAT_E = r"(?![A-Za-z]|[\-~][A-Za-z])"
_LAT_SEP = r"(?:[\s\-~]*)"
# "haha", "ha ha ha", "ha-ha", "hehe", "heheh", "hihi", "hohoho", with an optional lead-in syllable
_LAT_LEAD = r"(?:bwa|mwa|mua|gya|gwa|nya|wa|ga|ba|ya|a|e|u)"
_LAT_UNITS = re.compile(
    # the lead-in may be hyphenated from the laugh ("Mwa-ha-ha-ha", "A-ha-ha") and then belongs to it
    _LAT_B + "(?:(?P<lead>" + _LAT_LEAD + r")[\-~]?)?(?P<units>h[aeio](?:" + _LAT_SEP + r"h[aeio])+)h?" + _LAT_E,
    re.I,
)
_LAT_SINGLES = re.compile(
    _LAT_B
    + r"(?:(?P<heh>heh(?:" + _LAT_SEP + r"heh)*)|(?P<fufu>u?fu(?:" + _LAT_SEP + r"fu)+)|(?P<kuku>k[ue](?:" + _LAT_SEP + r"k[ue]){2,})"
    r"|(?P<teehee>(?:tee|hee)(?:" + _LAT_SEP + r"hee)+)|(?P<pfft>pf{1,3}t{0,2}|pfh)|(?P<lol>lol|lmao|rofl))" + _LAT_E,
    re.I,
)
_LAT_STAGE = re.compile(
    r"[\(\*\[]\s*(?P<word>laughs?|laughing|laughter|lol|giggles?|giggling|chuckles?|chuckling|snickers?|snickering|"
    r"cackles?|cackling|snorts?|snorting|guffaws?|titters?|tittering)\b[^)*\]\n]{0,24}[\)\*\]]",
    re.I,
)

_RU_B = r"(?<![а-яёА-ЯЁ])"
_RU_E = r"(?![а-яёА-ЯЁ]|[\-~][а-яёА-ЯЁ])"
_RU_SEP = r"(?:[\s\-~]*)"
# Lead-ins, mirroring the Latin side: bva/mua = evil; э/е ("Эхехехе", "Эхе-хе", the sheepish "ehehe") = nervous; и ("Ихихи") and
# у ("Ухаха") by the vowel of the units; "ня" ("Ня-хаха") by the units. "о" is not a lead: "Ох-хо" is a sigh.
_RU_LEAD = r"(?:бва|буа|муа|бу|му|ня|а|ы|э|е|и|у)"
_RU_UNITS = re.compile(
    # the lead-in may be hyphenated from the laugh ("Муа-ха-ха-ха", "А-ха-ха") and then belongs to it
    _RU_B + "(?:(?P<lead>" + _RU_LEAD + r")[\-~]?)?(?P<units>х[аеиоыэ](?:" + _RU_SEP + r"х[аеиоыэ])+)х?" + _RU_E,
    re.I,
)
# "Ухухуху": у + at least two "ху" (hearty); the units of _RU_UNITS have no "у" vowel
_RU_UKH = re.compile(_RU_B + r"у[\-~]?(?P<units>ху(?:" + _RU_SEP + r"ху)+)х?" + _RU_E, re.I)
# Singles: "хех"; "эхэ"/"эхе" (a lead and ONE unit; "эхо" is an echo, so only е/э after х); "гы-гы"; "бугага"; "пфф";
# "Фу-фу-фу"/"Уфуфу" (giggle; a bare "фу" is disgust, so three units, or two behind the "у"); "Кукуку" (evil; "ку-ку" is a cuckoo,
# so three units).
_RU_SINGLES = re.compile(
    _RU_B
    + r"(?:(?P<heh>хех)|(?P<eh>э[\-~]?х[еэ])|(?P<fufu>у(?:" + _RU_SEP + r"?фу){2,}|фу(?:" + _RU_SEP + r"фу){2,})|(?P<kuku>к[уе](?:" + _RU_SEP + r"к[уе]){2,})|(?P<gy>гы(?:" + _RU_SEP + r"гы)+)|(?P<buga>бу[\-~]?га(?:" + _RU_SEP + r"га)+\w*)|(?P<hah>а?хах)"
    r"|(?P<pff>пф{1,4}(?:[\-~]ф+)*|пхах?|фыр))" + _RU_E,
    re.I,
)
_RU_STAGE = re.compile(
    r"[\(\*\[]\s*(?P<word>смеётся|смеется|смех|смешок|со смехом|хихика\w*|усмеха\w*|усмешк\w*|хохо\w*|фырка\w*|"
    r"ржёт|ржет|прыска\w*|гогоч\w*|хмыка\w*|посмеива\w*)[^)*\]\n]{0,24}[\)\*\]]",
    re.I,
)

_JA_WC = "ぁ-んァ-ヶ一-龥々A-Za-z0-9"
_JA_B = rf"(?<![{_JA_WC}ー])"
_JA_E = rf"(?![{_JA_WC}])"
_JA_TAIL = "[っッー〜~ぁぃぅぇぉ]*"
# (regex body, kind); matched on hiragana-normalized text, longest alternatives first
_JA_RULES: list[tuple[str, str]] = [
    ("ぎゃはは+|あっはっは+|はっはっは+|ふははは+|ふはは|わはは+|がはは+|げらげら|けらけら|ほほほ+|おほほ+", "hearty"),
    ("(?:あ|わ|が|う|え|い|に|ひ|お)は{2,}|は{3,}|は{2}[っッ]", "hearty"),
    ("ぐふふ+|ふひひ+|く{3,}|くっく(?:っく)*|けけけ+", "evil"),
    ("えへへ+|えへ[っッ]|あはは[…\\.]|てへへ*", "nervous"),
    ("(?:に|う|あ|ぬ|ぐ)?へ{2,}", "chuckle"),
    ("(?:う|え|に|ふ|ぬ|ぐ|お)?ふ{2,}|くすくす|くすっ|くすり|ししし+|にしし+|うしし+|(?:に|い|う|あ|ふ)?ひ{2,}", "giggle"),
    ("ぷっ|ぷぷ+|ぷくく+|ぶふっ", "snort"),
]
_JA_RE = re.compile(_JA_B + "(?:" + "|".join(f"(?P<k{i}>{body})" for i, (body, _) in enumerate(_JA_RULES)) + ")" + _JA_TAIL + _JA_E)
_JA_STAGE = re.compile(r"[（(]\s*(?P<word>笑い|笑|苦笑|くすっ|くすくす|微笑|爆笑|高笑い)\s*[)）]|笑\)")
_JA_STAGE_KIND = {"笑い": "hearty", "笑": "hearty", "苦笑": "nervous", "くすっ": "giggle", "くすくす": "giggle", "微笑": "giggle", "爆笑": "hearty", "高笑い": "hearty"}

_ELLIPSIS_AFTER = re.compile(r"^[\s\"'”»」』)]*(?:\.{2,}|…)")

_LAT_STAGE_KIND = {"giggl": "giggle", "titter": "giggle", "chuckl": "chuckle", "snicker": "chuckle", "cackl": "evil", "snort": "snort", "laugh": "hearty", "lol": "hearty", "guffaw": "hearty"}
_RU_STAGE_KIND = {"хихик": "giggle", "усмех": "chuckle", "усмеш": "chuckle", "хмык": "chuckle", "посмеив": "chuckle", "фырк": "snort", "прыск": "snort", "гогоч": "hearty", "рж": "hearty", "хохо": "hearty", "сме": "hearty", "со сме": "hearty"}


@dataclass(frozen=True)
class Laugh:
    """One laugh in a line: ``text[start:end]`` is the written laugh (or the bracketed stage direction)."""

    start: int
    end: int
    text: str
    kind: str
    lang: str  # ja | ru | en
    units: int  # syllables of laughter ("ha ha ha" = 3); 0 for a stage direction
    stage: bool = False

    @property
    def long(self) -> bool:
        return self.units >= 3


def _units_latin(s: str) -> int:
    n = len(re.findall(r"h[aeio]|fu|k[ue]|hee|tee", s, re.I))
    if re.match(r"^[aeu](?=[\-~]?h)", s, re.I):
        n += 1  # "a-ha-ha": the lead-in vowel is a syllable of the laugh
    return n or 1


def _units_ru(s: str) -> int:
    n = len(re.findall(r"х[аеиоыэу]|гы|га|фу|к[уе]", s, re.I))
    if re.match(r"^[аыэеиу](?=[\-~]?х)", s, re.I):
        n += 1
    return n or 1


def _units_ja(s: str) -> int:
    return len(re.findall(r"[はふへひくしほぷ]", s)) or 1


def _kind_latin(m: re.Match[str]) -> str:
    lead = (m.group("lead") or "").lower()
    units = m.group("units").lower()
    if lead in ("bwa", "mwa", "mua", "gya", "gwa"):
        return "evil"
    vowel = re.search(r"h([aeio])", units)
    v = vowel.group(1) if vowel else "a"
    if lead == "e" and v == "e":
        return "nervous"
    if v == "e":
        return "chuckle"
    if v == "i":
        return "giggle"
    return "hearty"


def _kind_ru(m: re.Match[str]) -> str:
    lead = (m.group("lead") or "").lower()
    units = m.group("units").lower()
    if lead in ("бва", "буа", "муа", "му"):
        return "evil"
    vowel = re.search(r"х([аеиоыэ])", units)
    v = vowel.group(1) if vowel else "а"
    if v in ("е", "э"):
        return "nervous" if lead in ("э", "е") else "chuckle"
    if v == "и":
        return "giggle"
    if v == "ы":
        return "snort"
    return "hearty"


def _stage_kind(word: str, table: Mapping[str, str], default: str = "hearty") -> str:
    w = word.lower()
    for key in sorted(table, key=len, reverse=True):
        if w.startswith(key) or key in w:
            return table[key]
    return default


def _nervous_if_trailing(text: str, laugh: Laugh) -> Laugh:
    """A hearty laugh or chuckle that trails into an ellipsis is a nervous one."""
    if laugh.kind in ("hearty", "chuckle") and not laugh.stage and _ELLIPSIS_AFTER.match(text[laugh.end:]):
        return Laugh(laugh.start, laugh.end, laugh.text, "nervous", laugh.lang, laugh.units, laugh.stage)
    return laugh


def _find_latin(text: str) -> list[Laugh]:
    out: list[Laugh] = []
    for m in _LAT_STAGE.finditer(text):
        out.append(Laugh(m.start(), m.end(), m.group(0), _stage_kind(m.group("word"), _LAT_STAGE_KIND), "en", 0, True))
    for m in _LAT_UNITS.finditer(text):
        out.append(_nervous_if_trailing(text, Laugh(m.start(), m.end(), m.group(0), _kind_latin(m), "en", _units_latin(m.group(0)))))
    for m in _LAT_SINGLES.finditer(text):
        kind = "chuckle" if m.group("heh") else "giggle" if (m.group("fufu") or m.group("teehee")) else "evil" if m.group("kuku") else "snort" if m.group("pfft") else "hearty"
        out.append(_nervous_if_trailing(text, Laugh(m.start(), m.end(), m.group(0), kind, "en", _units_latin(m.group(0)))))
    return out


def _find_ru(text: str) -> list[Laugh]:
    out: list[Laugh] = []
    for m in _RU_STAGE.finditer(text):
        out.append(Laugh(m.start(), m.end(), m.group(0), _stage_kind(m.group("word"), _RU_STAGE_KIND), "ru", 0, True))
    for m in _RU_UNITS.finditer(text):
        if (m.group("lead") or "").lower() in ("э", "е") and re.search(r"х[аиоы]", m.group("units"), re.I):
            continue  # "Эхо-хо" is not "ehehe": an э/е lead takes only е/э units
        out.append(_nervous_if_trailing(text, Laugh(m.start(), m.end(), m.group(0), _kind_ru(m), "ru", _units_ru(m.group(0)))))
    for m in _RU_UKH.finditer(text):
        out.append(_nervous_if_trailing(text, Laugh(m.start(), m.end(), m.group(0), "hearty", "ru", _units_ru(m.group(0)))))
    for m in _RU_SINGLES.finditer(text):
        kind = (
            "chuckle" if (m.group("heh") or m.group("gy")) else "nervous" if m.group("eh") else "giggle" if m.group("fufu")
            else "evil" if m.group("kuku") else "snort" if m.group("pff") else "hearty"
        )
        out.append(_nervous_if_trailing(text, Laugh(m.start(), m.end(), m.group(0), kind, "ru", _units_ru(m.group(0)))))
    return out


def _find_ja(text: str) -> list[Laugh]:
    out: list[Laugh] = []
    for m in _JA_STAGE.finditer(text):
        word = m.group("word") or "笑"
        out.append(Laugh(m.start(), m.end(), m.group(0), _JA_STAGE_KIND.get(word, "hearty"), "ja", 0, True))
    hira = ja.kata_to_hira(text)  # one code point per code point: spans map 1:1
    for m in _JA_RE.finditer(hira):
        kind = next(_JA_RULES[i][1] for i in range(len(_JA_RULES)) if m.group(f"k{i}"))
        out.append(Laugh(m.start(), m.end(), text[m.start() : m.end()], kind, "ja", _units_ja(m.group(0))))
    return out


def _dedupe(laughs: Iterable[Laugh]) -> list[Laugh]:
    """Sorted by position; a span inside an earlier (longer) one is dropped."""
    out: list[Laugh] = []
    for l in sorted(laughs, key=lambda x: (x.start, -(x.end - x.start))):
        if out and l.start < out[-1].end:
            continue
        out.append(l)
    return out


def find_laughs(text: str, lang: str = "auto") -> list[Laugh]:
    """Every laugh of ``text`` in reading order. ``lang``: ``ja`` | ``ru`` | ``en`` (any Latin script) | ``auto`` (all)."""
    t = text or ""
    if not t:
        return []
    found: list[Laugh] = []
    if lang in ("auto", "ja"):
        found += _find_ja(t)
    if lang in ("auto", "ru"):
        found += _find_ru(t)
    if lang in ("auto", "en") or (lang not in ("ja", "ru", "auto")):
        found += _find_latin(t)
    return _dedupe(found)


_PLUS_MARK = re.compile(r"\+(?=[аеёиоуыэюяАЕЁИОУЫЭЮЯ])|́")


def unstress_laughs(text: str) -> str:
    """``text`` without the stress marks (ruaccent ``+`` before the vowel, U+0301 after it) inside its laughs; every other mark stays.

    A stresser marks a laugh like any word ("Х+а-х+а-х+а!"), and the patterns of :func:`find_laughs` no longer see it then.
    """
    t = text or ""
    marks = [m.start() for m in _PLUS_MARK.finditer(t)]
    if not marks:
        return t
    laughs = find_laughs(_PLUS_MARK.sub("", t))
    # the i marks before position p are not in the text the laughs were found in
    drop = {p for i, p in enumerate(marks) if any(l.start <= p - i < l.end or (t[p] == "́" and p - i == l.end) for l in laughs)}
    return "".join(ch for i, ch in enumerate(t) if i not in drop)


def has_laughter(text: str, lang: str = "auto") -> bool:
    return bool(find_laughs(text, lang))


def dominant_kind(laughs: Iterable[Laugh]) -> str | None:
    """The kind of the longest spoken laugh (stage directions count only when nothing is spoken)."""
    ls = list(laughs)
    if not ls:
        return None
    spoken = [l for l in ls if not l.stage]
    pool = spoken or ls
    return max(pool, key=lambda l: (l.units, -l.start)).kind


def laughter_kind(text: str, lang: str = "auto") -> str | None:
    return dominant_kind(find_laughs(text, lang))


_LETTER_CATS = ("L", "N")


def _has_letters(s: str) -> bool:
    return any(unicodedata.category(ch)[0] in _LETTER_CATS for ch in s)


def strip_laughter(text: str, lang: str = "auto", laughs: Iterable[Laugh] | None = None) -> str:
    """``text`` without its laughs; punctuation left dangling at the start is dropped, whitespace collapsed.

    ``"Haha, no way!"`` -> ``"no way!"``; ``"Хе-хе."`` -> ``""``.
    """
    t = text or ""
    ls = list(laughs) if laughs is not None else find_laughs(t, lang)
    for l in sorted(ls, key=lambda x: -x.start):
        end = l.end
        if not _has_letters(t[: l.start]):
            # the laugh opened the line: its own sentence mark goes with it ("Ха-ха! Ты серьёзно?" -> "Ты серьёзно?")
            m = re.match(r"[\s!?.…~〜！？。]+", t[end:])
            if m:
                end += m.end()
        t = t[: l.start] + " " + t[end:]
    t = re.sub(r"\s+", " ", t).strip()
    # punctuation that belonged to the laugh: leading commas/dashes, doubled sentence marks
    t = re.sub(r"^[\s,;:\-–—~〜、]+", "", t)
    t = re.sub(r"\s+([,.!?;:…、。！？])", r"\1", t)
    t = re.sub(r"([,、])\s*(?=[,.!?;:…、。！？])", "", t)
    return t.strip()


def is_laughter_only(text: str, lang: str = "auto") -> bool:
    """True when the line says nothing but laughter (stage directions and punctuation allowed)."""
    ls = find_laughs(text, lang)
    if not any(not l.stage for l in ls) and not any(l.stage for l in ls):
        return False
    rest = strip_laughter(text, lang, ls)
    return not _has_letters(rest)


Renderer = Callable[[Laugh], "str | None"]


def spell_laughter(text: str, render: Renderer, lang: str = "auto", laughs: Iterable[Laugh] | None = None) -> str:
    """Replace every laugh by ``render(laugh)``; ``None`` keeps the original spelling.

    A stage direction rendered as ``""`` is removed together with the whitespace around it. The rest of
    the line (punctuation after the laugh, the words) is untouched, so ``"Haha, no way!"`` with a renderer
    returning ``"Hahaha"`` becomes ``"Hahaha, no way!"``.
    """
    t = text or ""
    ls = list(laughs) if laughs is not None else find_laughs(t, lang)
    for l in sorted(ls, key=lambda x: -x.start):
        rep = render(l)
        if rep is None:
            continue
        before, after = t[: l.start], t[l.end :]
        if rep == "":
            before = before.rstrip()
            after = after.lstrip()
            sep = " " if before and after and not before.endswith(("(", "「", "«", '"')) and not after.startswith((",", ".", "!", "?", "…", ")", "」", "»", '"')) else ""
            t = before + sep + after
        else:
            # keep the capitalisation of a laugh that opened the sentence
            if l.text[:1].isupper() and rep[:1].islower():
                rep = rep[0].upper() + rep[1:]
            # the spelling's own mark yields to the line's punctuation: "Ха-ха!" + ", ну" -> "Ха-ха, ну"
            if re.search(r"[!?.…~〜！？。]$", rep) and re.match(r"\s*[!?.,…~〜！？。、]", after):
                rep = re.sub(r"[!?.…~〜！？。]+$", "", rep)
            t = before + rep + after
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def table_renderer(table: Mapping[str, Mapping[str, str] | str], stage: str | None = "") -> Renderer:
    """Renderer from a ``{kind: {"short": .., "long": ..} | "spelling"}`` table.

    Lookup: the laugh's kind, then ``"any"``, then ``"hearty"``; inside a kind ``long`` (3+ syllables) or
    ``short``, falling back to the other. ``stage`` is what a bracketed stage direction becomes (``""`` =
    removed, ``None`` = left as written). Unknown kinds with no fallback keep their spelling.
    """

    def pick(entry: Mapping[str, str] | str, long: bool) -> str | None:
        if isinstance(entry, str):
            return entry
        if long:
            return entry.get("long") or entry.get("short") or entry.get("any")
        return entry.get("short") or entry.get("long") or entry.get("any")

    def render(l: Laugh) -> str | None:
        if l.stage:
            return stage
        for key in (l.kind, "any", "hearty"):
            if key in table:
                return pick(table[key], l.long)
        return None

    return render


# ----------------------------------------------------------------------------
# kind notes and the per-line summary (text_emotion layer, StyleSpec.laughter)
# ----------------------------------------------------------------------------

#: delivery note per kind (``Delivery.notes``); the style renderers drop these again where they print the kind's own phrase
KIND_NOTE = {
    "hearty": "a hearty laugh", "chuckle": "chuckling", "giggle": "giggling",
    "nervous": "a nervous laugh", "evil": "a sinister laugh", "snort": "a snort",
}
LAUGH_NOTES = frozenset(KIND_NOTE.values())


def summarize(text: str, lang: str = "auto", laughs: Iterable[Laugh] | None = None) -> dict[str, Any] | None:
    """What the laughs of ``text`` amount to: ``{"kind", "units", "n", "stage", "only"}`` or ``None`` without laughs.

    ``kind`` / ``units`` are those of the longest spoken laugh (:func:`dominant_kind`); ``stage`` = every laugh is a
    bracketed stage direction; ``only`` = the line says nothing but laughter (:func:`is_laughter_only`).
    """
    ls = list(laughs) if laughs is not None else find_laughs(text, lang)
    if not ls:
        return None
    spoken = [l for l in ls if not l.stage]
    best = max(spoken or ls, key=lambda l: (l.units, -l.start))
    return {
        "kind": best.kind, "units": best.units, "n": len(ls), "stage": not spoken,
        "only": not _has_letters(strip_laughter(text, lang, ls)),
    }


# ----------------------------------------------------------------------------
# the spelling table (configs/laughter.yaml)
# ----------------------------------------------------------------------------

MODES = ("inline", "tag", "respell", "keep")
STAGE_MODES = ("remove", "respell")
_BACKEND_KEYS = ("mode", "token", "tokens", "spelling", "only", "stage")
_LANG_KEY = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z]{2,4})?$|^default$")


def default_table_path() -> Path:
    return repo_root() / "configs" / "laughter.yaml"


def _check_entry(where: str, entry: Any) -> Any:
    if isinstance(entry, str):
        if not entry.strip():
            raise ValueError(f"{where}: empty spelling (use a non-empty string)")
        return entry
    if isinstance(entry, Mapping):
        unknown = sorted(set(entry) - {"short", "long", "any"})
        if unknown:
            raise ValueError(f"{where}: unknown field(s) {unknown} (allowed: short, long)")
        out = {k: v for k, v in entry.items() if v is not None}
        if not out or any(not isinstance(v, str) or not v.strip() for v in out.values()):
            raise ValueError(f"{where}: give 'short' and/or 'long' as non-empty strings")
        return out
    raise ValueError(f"{where}: expected a string or {{short, long}}, got {type(entry).__name__}")


def _check_lang_block(where: str, block: Any) -> dict[str, Any]:
    if not isinstance(block, Mapping):
        raise ValueError(f"{where}: expected a mapping {{kind: spelling}}, got {type(block).__name__}")
    out: dict[str, Any] = {}
    for kind, entry in block.items():
        if kind not in KINDS and kind != "any":
            raise ValueError(f"{where}: unknown laugh kind {kind!r} (allowed: {', '.join(KINDS)}, any)")
        out[str(kind)] = _check_entry(f"{where}.{kind}", entry)
    return out


def _check_backend(where: str, sec: Any) -> dict[str, Any]:
    if not isinstance(sec, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(sec).__name__}")
    out: dict[str, Any] = {}
    for key, val in sec.items():
        k = str(key)
        if k == "mode":
            if val not in MODES:
                raise ValueError(f"{where}.mode: must be one of {', '.join(MODES)}, got {val!r}")
        elif k == "only":
            if val not in ("inline", "tag", "respell"):
                raise ValueError(f"{where}.only: must be inline, tag or respell, got {val!r}")
        elif k == "stage":
            if val not in STAGE_MODES:
                raise ValueError(f"{where}.stage: must be one of {', '.join(STAGE_MODES)}, got {val!r}")
        elif k in ("token", "spelling"):
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{where}.{k}: expected a non-empty string")
        elif k == "tokens":
            if not isinstance(val, Mapping) or any((kk not in KINDS and kk != "any") or not isinstance(vv, str) or not vv.strip() for kk, vv in val.items()):
                raise ValueError(f"{where}.tokens: expected {{kind: token}} with kinds from {', '.join(KINDS)}, any")
            val = dict(val)
        elif _LANG_KEY.match(k) and k not in _BACKEND_KEYS:
            val = _check_lang_block(f"{where}.{k}", val)
        else:
            raise ValueError(f"{where}: unknown key {k!r} (allowed: {', '.join(_BACKEND_KEYS)}, language codes)")
        out[k] = val
    return out


def validate_table(data: Any) -> dict[str, Any]:
    """Validate a laughter table (shape of ``configs/laughter.yaml``); ``ValueError`` names the offending entry.

    ``{stage?: remove|respell, <lang>: {<kind>: "spelling" | {short, long}}, backends?: {<name>: {mode?, token?,
    tokens?, spelling?, only?, stage?, <lang>: {<kind>: ...}}}}``
    """
    if not isinstance(data, Mapping):
        raise ValueError("laughter table must be a mapping")
    out: dict[str, Any] = {}
    for key, val in data.items():
        k = str(key)
        if k == "stage":
            if val not in STAGE_MODES:
                raise ValueError(f"stage: must be one of {', '.join(STAGE_MODES)}, got {val!r}")
            out[k] = val
        elif k == "backends":
            if not isinstance(val, Mapping):
                raise ValueError("backends: expected a mapping {backend: section}")
            out[k] = {str(name): _check_backend(f"backends.{name}", sec) for name, sec in val.items()}
        elif _LANG_KEY.match(k):
            out[k] = _check_lang_block(k, val)
        else:
            raise ValueError(f"{k}: not a language code, 'stage' or 'backends'")
    return out


def load_table(path: str | Path) -> dict[str, Any]:
    """Read and validate ``configs/laughter.yaml``. ``FileNotFoundError`` / ``ValueError`` (the message names the file)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"laughter table not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{p.name}: not valid YAML: {e}") from e
    try:
        return validate_table(data)
    except ValueError as e:
        raise ValueError(f"{p.name}: {e}") from e


def spelling_table(table: Mapping[str, Any], lang: str, backend: str = "") -> dict[str, Any] | None:
    """``{kind: spelling}`` for ``lang``: the language block (else ``default``) with the backend's own block over it."""
    short = lang.split("-")[0]
    base = table.get(lang) or table.get(short) or table.get("default")
    sec = (table.get("backends") or {}).get(backend) or {}
    over = sec.get(lang) or sec.get(short) or {}
    out = {**(base or {}), **over}
    return out or None


# ----------------------------------------------------------------------------
# applying it to a TTS text (synth)
# ----------------------------------------------------------------------------


def parse_decision(decision: str) -> tuple[str, str]:
    """``"inline:[laughter]"`` -> ``("inline", "[laughter]")``; ``"respell"`` -> ``("respell", "")``."""
    mode, _, token = (decision or "respell").partition(":")
    return (mode if mode in MODES else "respell"), token


def laughter_renderer(
    decision: str, table: Mapping[str, Any], lang: str, backend: str = "", kind_tokens: Mapping[str, str] | None = None, only: bool = False,
    has_spoken: bool = False, stage_spell: bool = False,
) -> tuple[Renderer, str]:
    """Renderer for :func:`spell_laughter` and the decision it implements (``respell`` | ``keep`` | ``inline:<tok>`` | ``tag:<tok>``).

    ``decision`` is what the backend capabilities allow (``style_prompt.laughter_decision``); the backend section of the
    table may override it: ``mode``, ``token`` / ``tokens`` (per kind), ``spelling`` (language of the spelled laugh next to a
    tag), ``only`` (the mode for a line that is nothing but laughter), ``stage``. Modes:

    * ``respell`` - the laugh becomes the table spelling for ``lang``;
    * ``inline``  - the backend sound-event token replaces the written laugh;
    * ``tag``     - the token precedes the spelled laugh (``[laughing] Ха-ха``, ``<|sfx:laughter|>Haha``);
    * ``keep``    - spoken laughs are left as written.

    A stage direction (``*laughs*``) is removed from the spoken text unless the mode puts a token or a spelling in its place
    (``inline`` / ``tag`` always do; ``respell`` when ``stage: respell``); next to a spoken laugh of the same line it is
    always removed (``has_spoken``), the spoken one already carries the laugh. ``stage_spell`` speaks a stage direction as a
    laugh of its kind whatever the mode (the fallback for a line that would otherwise be empty).
    """
    sec = (table.get("backends") or {}).get(backend) or {}
    mode, token = parse_decision(decision)
    mode = sec.get("mode") or mode
    token = sec.get("token") or token
    if only and sec.get("only"):
        mode = sec["only"]
    tokens = {**(kind_tokens or {}), **(sec.get("tokens") or {})}
    stage_mode = sec.get("stage") or table.get("stage") or "remove"
    tbl = spelling_table(table, sec.get("spelling") or lang, backend)
    spell = table_renderer(tbl, stage=None) if tbl else None

    def spelled(l: Laugh) -> str | None:
        if spell is None:
            return None
        return spell(dataclasses.replace(l, stage=False) if l.stage else l)

    def token_of(l: Laugh) -> str:
        return tokens.get(l.kind) or token or tokens.get("any") or ""

    if mode in ("inline", "tag") and not any([token, *tokens.values()]):
        mode = "respell"  # no token known for this backend: spell it
    label = mode
    if mode in ("inline", "tag"):
        label = f"{mode}:{token or tokens.get('hearty') or tokens.get('any') or next(iter(tokens.values()))}"

    def render(l: Laugh) -> str | None:
        if l.stage and has_spoken:
            return ""
        if l.stage and stage_spell:
            return spelled(l) or ""
        if mode == "keep":
            return "" if l.stage else None
        if mode == "respell":
            if l.stage:
                return (spelled(l) or "") if stage_mode == "respell" else ""
            return spelled(l)
        tk = token_of(l)
        if mode == "inline":
            return tk
        sp = spelled(l)
        # a <|category:value|> tag is written right against the onomatopoeia it is paired with (Higgs model card)
        return tk + (("" if tk.endswith("|>") else " ") + sp if sp else "")

    return render, label


def apply_laughter(
    text: str, lang: str, decision: str, table: Mapping[str, Any], backend: str = "", kind_tokens: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """The TTS text with its laughs spelled for the backend; ``(text, None)`` when the text has none.

    The second value is what the synth sidecar records: ``{"kind", "n", "mode", "only", "stage"}`` (``kind`` of the longest
    spoken laugh, ``n`` laughs found, ``mode`` = the decision applied).
    """
    laughs = find_laughs(text)
    if not laughs:
        return text, None
    info = summarize(text, laughs=laughs) or {}
    render, label = laughter_renderer(decision, table, lang, backend, kind_tokens, only=bool(info.get("only")), has_spoken=any(not l.stage for l in laughs))
    out = spell_laughter(text, render, laughs=laughs)
    if not _has_letters(out):
        # the line was a stage direction only ("*laughs*") and it was dropped: speak it as a laugh rather than send nothing
        render, _ = laughter_renderer(decision, table, lang, backend, kind_tokens, only=True, stage_spell=True)
        out = spell_laughter(text, render, laughs=laughs)
        if not _has_letters(out):
            out, label = text, "keep"
    if label.startswith(("inline", "tag")):
        sec_tokens = ((table.get("backends") or {}).get(backend) or {}).get("tokens") or {}
        for tk in {t for t in [label.partition(":")[2], *(kind_tokens or {}).values(), *sec_tokens.values()] if t}:
            # a comma after an event token is not part of the sentence: "[laughter], ну" -> "[laughter] ну"
            out = re.sub(re.escape(tk) + r"\s*[,;:]\s*", tk + " ", out)
    return out, {"kind": info.get("kind"), "n": info.get("n", len(laughs)), "mode": label, "only": bool(info.get("only")), "stage": bool(info.get("stage"))}
