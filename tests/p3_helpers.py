"""Helpers for the casting / route P / tuning tests: add an unvoiced speaker to a project built by genproj."""

from __future__ import annotations

from typing import Any

from vndub.manifest.io import Character, CharacterStats, LineRecord, TranslationRecord, load_characters, save_characters, write_json, write_jsonl
from vndub.stages import genutil as gu
from vndub.text.style_prompt import StyleSpec

HERO_RU = ["Доброе утро, как дела у тебя сегодня?", "Пойдём вместе, мы не должны опоздать!", "Я совсем не хочу идти туда сегодня."]


def add_unvoiced(ctx: Any, speaker: str = "hero", n: int = 3, gender: str = "m", text_passport_first: bool = True) -> dict[str, Any]:
    """Append to the project: ``n`` unvoiced lines of ``speaker`` (each with a dialogue id), one narration line,
    one ``voice_sustain`` continuation of the speaker and one menu record; translations; the character.

    The first speaker line gets a text passport (``source: "text"``, angry / fast); the others have none.
    Returns ``{"hero": [line_id], "narrator": id, "sustain": id, "menu": id}``.
    """
    lines = ctx.lines()
    order = max(l.order for l in lines)
    new: list[LineRecord] = []
    trs: list[TranslationRecord] = []

    def add(i: int, sid: str, ja: str, ru: str, flags: list[str] | None = None) -> str:
        nonlocal order
        order += 1
        lid = f"s01_{900 + i:06d}"
        new.append(
            LineRecord(
                line_id=lid, scene="s01", order=order, speaker_raw=speaker if sid == speaker else "", speaker_id=sid, text_src=ja,
                has_voice=False, flags=flags or [], dialogue_id=f"start_{i:08x}",
            )
        )
        trs.append(TranslationRecord(line_id=lid, text_tgt=ru, text_tgt_short=ru, text_tts=ru, text_plain=ru, source="test"))
        return lid

    hero = [add(i, speaker, f"おはよう{i}。", HERO_RU[i % len(HERO_RU)]) for i in range(n)]
    narr = add(n, "narrator", "ナレーション。", "Рассказчик говорит что-то совсем длинное.")
    sus = add(n + 1, speaker, "続き。", "И это продолжение его фразы.", ["voice_sustain"])
    menu = add(n + 2, "narrator", "", "Выбор.", ["menu"])
    write_jsonl(ctx.paths.lines, [*lines, *new])
    old = gu.load_translations(ctx)
    write_jsonl(ctx.paths.translation(ctx.cfg.project.tgt_lang), [*old.values(), *trs])
    chars = load_characters(ctx.paths.characters)
    chars[speaker] = Character(names={"src": speaker, "tgt": "Герой"}, gender=gender, age_hint="teen", description="уставший школьник", stats=CharacterStats())
    save_characters(ctx.paths.characters, chars)
    if text_passport_first:
        style = StyleSpec(emotion="angry", intensity=0.8, tempo="fast", loudness="loud")
        write_json(ctx.paths.passport(hero[0]), {"line_id": hero[0], "source": "text", "style": style.to_dict(), "delivery": {"emotion": "angry", "layer": "rule", "confidence": 0.9}})
    return {"hero": hero, "narrator": narr, "sustain": sus, "menu": menu}
