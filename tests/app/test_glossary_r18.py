"""Round 18, contract C9 / C3: automatic glossary terms in the app's glossary, deletions remembered, the user's gender marked."""

from __future__ import annotations

import yaml

from vndub.manifest.io import Character, load_characters, save_characters
from vndub.stages import translate
from vndub_app import appdirs, session, translation


def _chars_path(h):
    return session.work_paths(h).characters


def _handle(tmp_path):
    h = appdirs.ProjectHandle("p", tmp_path / "p")
    (h.root / "work").mkdir(parents=True)
    _chars_path(h).parent.mkdir(parents=True, exist_ok=True)
    save_characters(_chars_path(h), {"mira": Character(names={"src": "Mira"}, gender="f", gender_source="guess")})
    h.glossary_file.write_text(yaml.safe_dump({"terms": [{"src": "Tokyo", "tgt": "Токио"}]}, allow_unicode=True), encoding="utf-8")
    translate.write_auto_glossary(h.work, [{"src": "Gensokyo", "tgt": "Генсокё", "notes": "auto"}, {"src": "Tokyo", "tgt": "Токё", "notes": "auto"},
                                           {"src": "Darling", "tgt": "Дорогой", "notes": "auto"}], [])
    return h


def test_get_glossary_lists_auto_terms_after_the_users_and_marks_a_guessed_gender(tmp_path):
    h = _handle(tmp_path)
    g = translation.get_glossary(h)
    assert g["terms"] == [
        {"src": "Tokyo", "tgt": "Токио", "note": ""},  # the user's term wins over the automatic one of the same source
        {"src": "Gensokyo", "tgt": "Генсокё", "note": "", "auto": True},
        {"src": "Darling", "tgt": "Дорогой", "note": "", "auto": True},
    ]
    assert g["characters"] == [{"speaker_id": "mira", "src": "Mira", "tgt": "", "gender": "f", "gender_source": "guess"}]


def test_a_deleted_auto_term_is_remembered_and_an_edited_one_becomes_the_users(tmp_path, monkeypatch):
    h = _handle(tmp_path)
    written: dict = {}
    monkeypatch.setattr(translation, "write_glossary", lambda h, terms=None: written.setdefault("terms", terms))
    monkeypatch.setattr(translation, "get_glossary", lambda h: {})
    translation.put_glossary(h, {"terms": [
        {"src": "Tokyo", "tgt": "Токио"}, {"src": "Gensokyo", "tgt": "Гэнсокё", "auto": False},  # edited -> the user's
    ]})
    assert written["terms"] == [{"src": "Tokyo", "tgt": "Токио"}, {"src": "Gensokyo", "tgt": "Гэнсокё"}]
    auto, deleted = translate.read_auto_glossary(h.work)
    assert "Darling" in deleted and [a["src"] for a in auto] == ["Gensokyo", "Tokyo"]
    # the names pass does not add it back
    gloss = {"terms": []}
    assert [t["src"] for t in translate.merge_names(gloss, {"Darling": "Дорогой", "Rumia": "Румия", "Mansion": "особняк"}, deleted)] == ["Rumia"]


def test_a_gender_the_user_changes_is_marked_as_the_users(tmp_path, monkeypatch):
    h = _handle(tmp_path)
    monkeypatch.setattr(translation, "write_glossary", lambda h, terms=None: None)
    monkeypatch.setattr(translation, "get_glossary", lambda h: {})
    from vndub_app import casting

    def edit(h):  # the character sheet in the project's work folder, as casting.edit_characters does
        import contextlib

        @contextlib.contextmanager
        def cm():
            chars = load_characters(_chars_path(h))
            yield chars
            save_characters(_chars_path(h), chars)

        return cm()

    monkeypatch.setattr(casting, "edit_characters", edit)
    translation.put_glossary(h, {"characters": [{"speaker_id": "mira", "gender": "m"}]})
    c = load_characters(_chars_path(h))["mira"]
    assert c.gender == "m" and c.model_extra["gender_source"] == "user"
