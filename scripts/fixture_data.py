"""Single source of truth for the synthetic Ren'Py fixture (tests/fixtures/renpy_mini).

``scripts/make_fixtures.py`` turns this table into ``game/script.rpy``, the
Japanese voice files ``game/voice/*.ogg`` and ``tests/fixtures/mock_dictionary.yaml``
(the MockTranslator dictionary). Golden parser output in tests is derived from
the same table, so editing it means regenerating fixtures and goldens.

The game is MIXED on purpose: the three heroines/friends are voiced, the protagonist `mc`
(speaker id `hero`, name 蓮) has no voice at all, and unvoiced narration lines around his
dialogue carry delivery cues in Japanese (…と叫んだ, 小さな声でささやいた, 笑いながら言った,
泣きながら言った, 怒鳴った) -- plus lines with no cue at all. The 20 voiced lines and their audio
are exactly the original fixture (the generator never re-renders existing voice files).

Row kinds:
  ("say", char, ja, ru, voice, delivery)     voiced (voice != None) or unvoiced line (voice None, delivery None)
  ("narr", ja, ru)                           narrator line (never voiced)
  ("sustain", char, ja, ru)                  `voice sustain` + say: continues the previous voice file
  ("menu", [(choice_ja, choice_ru, [rows...]), ...])
  ("label", name)

``delivery`` drives the SAPI rendering of the Japanese voice: (pitch, rate, volume)
offsets on top of the character's base voice.
"""

from __future__ import annotations

CHARACTERS = {
    # var: (display name, voice_tag, gender, ru name, SAPI base pitch, base rate)
    "s": ("さくら", "sakura", "f", "Сакура", "+15%", "+5%"),
    "h": ("陽翔", "haruto", "m", "Харуто", "-45%", "-5%"),
    "y": ("ゆい", "yui", "f", "Юи", "+35%", "-5%"),
    # unvoiced protagonist (the player character): no audio is rendered for him
    "mc": ("蓮", "hero", "m", "Рэн", "+0%", "+0%"),
}

HAPPY = ("+10%", "+10%", "loud")
SAD = ("-10%", "-20%", "soft")
ANGRY = ("-5%", "+15%", "x-loud")
SURPRISED = ("+20%", "+0%", "loud")
CALM = ("+0%", "+0%", "medium")

SCRIPT = [
    ("label", "start"),
    ("narr", "放課後の教室。窓の外は夕焼けだった。", "Класс после уроков. За окном горел закат."),
    ("say", "s", "ねえ、陽翔くん。ちょっと待って！", "Эй, Харуто! Подожди немного!", "voice/sak_0001.ogg", HAPPY),
    ("say", "h", "どうしたの、さくら？", "Что случилось, Сакура?", "voice/har_0001.ogg", CALM),
    ("say", "s", "今日はね、すごくいいことがあったの！", "Сегодня случилось кое-что очень хорошее!", "voice/sak_0002.ogg", HAPPY),
    ("say", "h", "へえ、何があったんだ？", "Ого, и что же?", "voice/har_0002.ogg", CALM),
    ("say", "s", "テストで百点を取ったんだよ！", "Я получила 100 баллов на тесте!", "voice/sak_0003.ogg", HAPPY),
    ("say", "y", "うそ……本当に？", "Не может быть... Правда?", "voice/yui_0001.ogg", SURPRISED),
    ("say", "s", "本当だよ！ゆいちゃんも見る？", "Правда! Юи, хочешь посмотреть?", "voice/sak_0004.ogg", HAPPY),
    ("narr", "ゆいは少し寂しそうに笑った。", "Юи немного грустно улыбнулась."),
    ("say", "y", "わたしは……今回もだめだった。", "А у меня... опять ничего не вышло.", "voice/yui_0002.ogg", SAD),
    ("say", "h", "次はきっと大丈夫だよ。", "В следующий раз всё обязательно получится.", "voice/har_0003.ogg", CALM),
    # --- the unvoiced protagonist steps in; the narration around his lines holds the delivery cues
    ("say", "mc", "……ゆい、大丈夫か？", "...Юи, ты в порядке?", None, None),
    ("narr", "蓮はゆいにだけ聞こえるよう、小さな声でささやいた。", "Рэн прошептал так, чтобы слышала только Юи."),
    ("narr", "さくらが胸を張るのを見て、蓮は笑いながら言った。", "Увидев, как Сакура гордо выпрямилась, Рэн сказал со смехом."),
    ("say", "mc", "はいはい、百点はすごいよ。さくらは偉い偉い。", "Ладно-ладно, сто баллов — это здорово. Молодец, Сакура.", None, None),
    ("say", "mc", "でも、ゆいだって毎日遅くまで頑張ってたんだぞ。", "Но Юи тоже каждый день допоздна старалась.", None, None),
    ("narr", "蓮は悔しさをこらえきれず、泣きながら言った。", "Не в силах сдержать досаду, Рэн сказал сквозь слёзы."),
    (
        "menu",
        [
            (
                "励ます",
                "Подбодрить",
                [
                    ("say", "h", "一緒に勉強しようか？", "Может, позанимаемся вместе?", "voice/har_0004.ogg", CALM),
                    ("say", "y", "うん！ありがとう！", "Угу! Спасибо!", "voice/yui_0003.ogg", HAPPY),
                ],
            ),
            (
                "黙っている",
                "Промолчать",
                [
                    ("say", "h", "うーん……", "Хм...", "voice/har_0005.ogg", CALM),
                    ("say", "y", "ふん、もういいよ。", "Хмф, ну и ладно.", "voice/yui_0004.ogg", ANGRY),
                ],
            ),
        ],
    ),
    ("label", "after_menu"),
    ("say", "s", "ねえ、三人で帰ろう！", "Слушайте, пойдёмте домой втроём!", "voice/sak_0005.ogg", HAPPY),
    ("say", "h", "ああ、そうしよう。", "Ага, давай.", "voice/har_0006.ogg", CALM),
    ("sustain", "h", "雨が降りそうだしな。", "Похоже, скоро пойдёт дождь."),
    ("say", "y", "あはは、さくらちゃんったら！", "Ахаха, ну ты даёшь, Сакура!", "voice/yui_0005.ogg", HAPPY),
    ("say", "s", "えへへ。", "Хе-хе.", "voice/sak_0006.ogg", HAPPY),
    ("narr", "三人は並んで校門を出た。", "Втроём они вышли за школьные ворота."),
    ("say", "h", "また明日な。", "До завтра.", "voice/har_0007.ogg", CALM),
    ("say", "y", "また明日ね！", "До завтра!", "voice/yui_0006.ogg", HAPPY),
    ("say", "s", "じゃあね！", "Пока!", "voice/sak_0007.ogg", HAPPY),
    ("narr", "校門の前で、三人は手を振って別れた。", "У школьных ворот трое помахали друг другу и разошлись."),
    # --- epilogue: the hero line "みんな、帰ったのか。" has no cue within three narration lines either way
    ("label", "epilogue"),
    ("narr", "三人の背中が、夕焼けの中へ遠ざかっていく。", "Три силуэта удалялись в закатном свете."),
    ("say", "mc", "みんな、帰ったのか。", "Все уже ушли?", None, None),
    ("narr", "教室には蓮ひとりだけが残った。", "В классе остался один Рэн."),
    ("narr", "黒板の上の時計が、静かに時を刻んでいる。", "Часы над доской тихо отсчитывали время."),
    ("narr", "机の上には、さくらの忘れ物のノートが置かれていた。", "На парте лежала тетрадь, которую забыла Сакура."),
    ("say", "mc", "おい、さくら、ノートを忘れてるぞ！", "Эй, Сакура, ты забыла тетрадь!", None, None),
    ("narr", "蓮は廊下に飛び出して、待ってくれと叫んだ。", "Рэн выбежал в коридор и крикнул, чтобы она подождала."),
    ("narr", "返事がないので、蓮は腹立ちまぎれに怒鳴った。", "Ответа не было, и Рэн в сердцах рявкнул."),
    ("say", "mc", "もういい、明日渡すからな！", "Ладно, завтра отдам!", None, None),
]


def iter_rows(rows=SCRIPT):
    """Flatten menus (choices first, then their bodies) in script order."""
    for row in rows:
        if row[0] == "menu":
            for choice_ja, choice_ru, body in row[1]:
                yield ("choice", choice_ja, choice_ru)
                yield from iter_rows(body)
        else:
            yield row


def dictionary() -> dict[str, str]:
    """Japanese -> Russian for every translatable string (MockTranslator)."""
    out: dict[str, str] = {}
    for row in iter_rows():
        if row[0] == "say":
            out[row[2]] = row[3]
        elif row[0] in ("narr", "choice"):
            out[row[1]] = row[2]
        elif row[0] == "sustain":
            out[row[2]] = row[3]
    return out
