"""Round 17 of the translation stage: the player character in the prompt, the names pass, post-checks, the context ladder,
narration as text-only lines and menu choices (`choices_tgt`)."""

from __future__ import annotations

import json

import pytest
import yaml

from analysis_helpers import build_project
from tests.test_translate import _Cuda, _records
from vndub.backends import registry
from vndub.backends.base import BackendSpec
from vndub.config import ProjectYaml, save_project_yaml
from vndub.gpu import ModelRegistry
from vndub.manifest.io import LineRecord, read_jsonl
from vndub.stages import translate
from vndub.stages.common import run_stage

# ----------------------------------------------------------------------------
# prompt
# ----------------------------------------------------------------------------


def _prompt(gloss, tgt="ru", batch=None):
    batch = batch or [{"id": "a", "text": "Welcome back, darling.", "speaker_id": "sakuya"}]
    return translate.system_prompt(tgt, batch, ["sakuya"], gloss, {}, 8, False, "en")


def test_prompt_player_rule_by_gender():
    m = _prompt({"player": {"name": "Anon", "gender": "m"}})
    assert "(Anon) is male" in m and "ты вернулся, дорогой" in m
    f = _prompt({"player": {"gender": "f"}})
    assert "is female" in f and "ты вернулась, дорогая" in f
    for g in ({}, {"player": {"gender": "unknown"}}, {"player": {"gender": "x"}}):
        u = _prompt(g)
        assert "is unknown" in u and "do not reveal it" in u and "use the masculine" in u  # the documented default
    assert "ё" not in _prompt({}, "en") and "Хр-р-р" not in _prompt({}, "en")  # Russian examples only for a Russian target
    assert "use the masculine." in _prompt({}, "en")


def test_prompt_craft_rules():
    p = _prompt({})
    for needle in ("Write the letter ё", "never е for ё", "Ха-ха, Хе-хе, Хи-хи, Эхехе, Ахаха", "Чмок", "Хр-р-р", "С возвращением", "omit nothing",
                   "exactly as the Characters and Terms lists say", "never mix scripts"):
        assert needle in p, needle
    assert "player_choice" not in p and "previous_answer" not in p  # only when the batch has such items
    b = _prompt({}, batch=[{"id": "m#c0", "text": "Go", "speaker_id": "narrator", "kind": "player_choice"}, {"id": "b", "text": "x", "speaker_id": "a", "previous_answer": "y", "fix": "z"}])
    assert "player_choice" in b and "previous_answer" in b


def test_seed_glossary_adds_the_player_and_keeps_an_existing_one(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path, sets={"translate.glossary": str(tmp_path / "g.yaml")})
    build_project(ctx, [dict(line_id="l1", speaker_id="a", text="Hi", seconds=0.5)])
    ctx.extra["seed_glossary"] = True
    translate.run(ctx)
    g = yaml.safe_load((tmp_path / "g.yaml").read_text(encoding="utf-8"))
    assert g["player"] == {"name": "", "gender": "unknown", "notes": ""}
    g["player"]["gender"] = "m"
    (tmp_path / "g.yaml").write_text(yaml.safe_dump(g), encoding="utf-8")
    translate.run(ctx)
    assert yaml.safe_load((tmp_path / "g.yaml").read_text(encoding="utf-8"))["player"]["gender"] == "m"


# ----------------------------------------------------------------------------
# names: candidates, merge, post-checks
# ----------------------------------------------------------------------------

SCRIPT = [
    "Welcome back, darling.",
    "Have you ever been to Gensokyo?",
    "Maybe we can visit Patchouli. Patchouli loves books.",
    "Is Reimu's shrine nice? Maybe tomorrow.",
    "Oh, I'm so sleepy. Hehe.",
]


def test_name_candidates():
    got = translate.collect_name_candidates(SCRIPT, known=["Sakuya Izayoi"])
    assert got == ["Patchouli", "Gensokyo", "Reimu"]  # most frequent first; Welcome/Maybe/Oh are sentence starts or stop words
    assert translate.collect_name_candidates(SCRIPT, known=["Patchouli", "Gensokyo", "Reimu"]) == []
    assert translate.collect_name_candidates(["Sakuya said so.", "Where is Sakuya?"], known=["Sakuya"]) == []
    assert translate.collect_name_candidates(["今日は晴れです。", "ハロー"]) == []  # no letter case, no candidates
    assert translate.collect_name_candidates(["Maybe so.", "I think maybe not."]) == []  # also written in lower case


def test_merge_names_never_overwrites_a_term():
    gloss = {"terms": [{"src": "Reimu", "tgt": "Рейму"}]}
    added = translate.merge_names(gloss, {"reimu": "Реиму", "Gensokyo": "Генсокё", "Same": "Same"})
    assert added == [{"src": "Gensokyo", "tgt": "Генсокё", "notes": "auto"}]
    assert gloss["terms"][0] == {"src": "Reimu", "tgt": "Рейму"} and len(gloss["terms"]) == 2
    assert translate.parse_names('{"names": [{"src": "A", "tgt": "Б"}, {"src": "", "tgt": "x"}, 5]}') == {"A": "Б"}
    assert translate.parse_names('```json\n{"names": {"A": "Б"}}\n```') == {"A": "Б"}
    assert translate.parse_names("garbage") == {}


def test_check_translation():
    chk = translate.check_translation
    assert chk("Watatsuki is here", "Ватаzuki здесь", "ru", "en") == "mixed_script"
    assert chk("Hello you", "Привет there", "ru", "en") == "latin_run"
    assert chk("Call the iPhone", "Позвони на iPhone", "ru", "en") == ""  # a Latin word that the source has
    assert chk("Hello there my friend", "Hello there my friend", "ru", "en") == "untranslated"
    assert chk("Sakuya", "Sakuya", "ru", "en") == ""  # a lone word may be a name
    assert chk("Muah!", "Чмок!", "ru", "en") == ""
    assert chk("Hello there", "Hello there", "en", "en") == ""  # same language is not "untranslated"
    terms = [{"src": "Gensokyo", "tgt": "Генсокё"}]
    assert chk("Go to Gensokyo", "Иди в Генсокё", "ru", "en", terms) == ""


def test_glossary_missing_tolerates_declension_and_yo():
    terms = [{"src": "Patchouli", "tgt": "Пачули"}, {"src": "Gensokyo", "tgt": "Генсокё"}, {"src": "Reimu", "tgt": "Рейму"}]
    assert translate.glossary_missing("See Patchouli", "Иди к Пачули", terms) == []
    assert translate.glossary_missing("In Gensokyo", "В Генсоке", terms) == ["Gensokyo"]
    assert translate.glossary_missing("In Gensokyo", "В Генсокё", terms) == []
    assert translate.glossary_missing("Reimu's shrine", "Храм Рейму", terms) == []
    assert translate.glossary_missing("Nothing here", "Ничего", terms) == []


# ----------------------------------------------------------------------------
# stage with a fake LLM: names pass (merge + cache), post-checks
# ----------------------------------------------------------------------------


def _en_rows():
    rows = [dict(line_id=f"e{i:02d}", speaker_id="sakuya", speaker_raw="Sakuya", text=t, seconds=0.5) for i, t in enumerate(SCRIPT)]
    return rows


class _Script:
    """A fake model that tells names, translations, summaries apart. ``tr`` maps a source text to its (first, later) answers."""

    def __init__(self, names=None, tr=None):
        self.names, self.tr = names if names is not None else {"Gensokyo": "Генсокё", "Patchouli": "Пачули", "Reimu": "Рейму"}, tr or {}
        self.kinds: list[str] = []
        self.asked: list[list] = []
        self.system: list[str] = []

    def __call__(self, backend, kind, messages):
        head = messages[0]["content"]
        if head.startswith("Summarize"):
            self.kinds.append("summary")
            return json.dumps({"summary": "s"})
        if head.startswith("You are a localization editor"):
            self.kinds.append("names")
            self.asked.append(json.loads(messages[-1]["content"].split("Names:\n")[1]))
            return json.dumps({"names": [{"src": k, "tgt": v} for k, v in self.names.items()]}, ensure_ascii=False)
        self.kinds.append("translate")
        self.system.append(head)
        batch = json.loads(messages[-1]["content"].split("\n", 1)[1])
        rows = []
        for b in batch:
            ans = self.tr.get(b["text"], f"перевод {b['id']}")
            if isinstance(ans, tuple):
                ans = ans[1] if "previous_answer" in b else ans[0]
            rows.append({"id": b["id"], "ru": ans, "ru_short": "", "notes": ""})
        return json.dumps({"lines": rows}, ensure_ascii=False)


class _Backend:
    def __init__(self, spec, script, options=None):
        self.spec, self.script, self.options, self.loaded = spec, script, options, False

    def load(self):
        self.loaded = True
        return {"device": "cuda:0"}

    def unload(self):
        self.loaded = False

    def peak_vram(self):
        return None

    def chat(self, messages, temperature=0.3, **kw):
        return self.script(self, "", messages)


def _registry(ctx, tmp_path, script, cuda=None, budget=7.5, margin=0.5):
    made: list[_Backend] = []

    def factory(spec, options, device):
        b = _Backend(spec, script, options)
        made.append(b)
        return b

    ctx.models = ModelRegistry(
        budget_gb=budget, safety_margin_gb=margin, check_usable=False, factory=factory, cuda=cuda or _Cuda(), models_dir=tmp_path / "models",
        log_dir=ctx.paths.logs, timings_path=ctx.paths.work / "gpu_events.jsonl", vram_retry_s=0.0,
    )
    return made


def _en_ctx(make_ctx, tmp_path, monkeypatch, script, rows=None, project=None, **sets):
    monkeypatch.setattr(registry, "_SPECS", {**registry.all_specs(), "llm_one": BackendSpec(name="llm_one", role="llm", env="fake", adapter="llm_one", license="MIT", commercial_ok=True, vram_estimate_gb=2.0, heavy=True)})
    base = {"translate.llm.backend": "llm_one", "translate.llm.fallbacks": [], "translate.stress.tool": "none", "translate.glossary": str(tmp_path / "glossary.yaml"),
            "project.src_lang": "en", "translate.include_narration": "off"}
    ctx = make_ctx(tmp_path, sets={**base, **sets})
    build_project(ctx, rows or _en_rows())
    if project:
        save_project_yaml(ctx.paths.project_yaml, ProjectYaml(game_dir=str(tmp_path), src_lang="en", **project))
    made = _registry(ctx, tmp_path, script)
    return ctx, made


def test_names_pass_merges_into_terms_asks_once_and_is_cached(make_ctx, tmp_path, monkeypatch):
    script = _Script()
    ctx, made = _en_ctx(make_ctx, tmp_path, monkeypatch, script)
    (tmp_path / "glossary.yaml").write_text(yaml.safe_dump({"terms": [{"src": "Reimu", "tgt": "Рэйму"}]}, allow_unicode=True), encoding="utf-8")
    summary = run_stage("translate", ctx)
    assert script.kinds.count("names") == 1
    assert script.asked == [["Patchouli", "Gensokyo"]]  # Reimu is the user's term: not asked again
    # the user's spelling wins, the auto ones are added
    sys_prompt = " ".join(script.system)
    assert "Gensokyo -> Генсокё (auto)" in sys_prompt and "Reimu -> Рэйму" in sys_prompt and "Рейму" not in sys_prompt
    assert summary["names_auto"] == 2
    auto = yaml.safe_load((ctx.paths.work / "glossary_auto.yaml").read_text(encoding="utf-8"))
    assert {t["src"] for t in auto["terms"]} == {"Gensokyo", "Patchouli"}
    assert "Gensokyo" in _records(ctx)["e01"].glossary_hits
    ctx.models.close()

    # a second run: the answer comes from the cache, the glossary file on disk is the user's, nothing is asked, no model is loaded
    ctx3 = ctx
    ctx3.models = ModelRegistry(
        budget_gb=7.5, safety_margin_gb=0.5, check_usable=False, factory=lambda *a: pytest.fail("a model was loaded for a fully cached run"),
        cuda=_Cuda(), models_dir=tmp_path / "models", log_dir=ctx.paths.logs, timings_path=ctx.paths.work / "gpu_events2.jsonl", vram_retry_s=0.0,
    )
    s2 = run_stage("translate", ctx3)
    assert s2["names_auto"] == 2


def test_names_pass_off_and_a_model_without_answer(make_ctx, tmp_path, monkeypatch):
    script = _Script()
    ctx, _ = _en_ctx(make_ctx, tmp_path, monkeypatch, script, **{"translate.names_pass": False})
    run_stage("translate", ctx)
    assert "names" not in script.kinds
    script = _Script(names={})
    ctx, _ = _en_ctx(make_ctx, tmp_path / "b", monkeypatch, script)
    s = run_stage("translate", ctx)
    assert script.kinds.count("names") == 1 and s["names_auto"] == 0 and s["failed"] == 0


def test_a_mixed_script_answer_is_asked_again_once_and_kept_when_it_still_fails(make_ctx, tmp_path, monkeypatch):
    tr = {
        "Welcome back, darling.": ("С возвращением, дорогой.", "С возвращением, дорогой."),
        "Have you ever been to Gensokyo?": ("Ты бывал в Gensokyoе?", "Ты бывал в Генсокё?"),
        "Oh, I'm so sleepy. Hehe.": ("Ох, я такая Ватаzuki. Хихи.", "Ох, я такая Ватаzuki. Хихи."),
    }
    script = _Script(tr=tr)
    ctx, _ = _en_ctx(make_ctx, tmp_path, monkeypatch, script)
    s = run_stage("translate", ctx)
    recs = _records(ctx)
    assert recs["e01"].text_tgt == "Ты бывал в Генсокё?" and "check_failed" not in recs["e01"].flags
    assert recs["e04"].text_tgt.endswith("Хихи.") and "check_failed" in recs["e04"].flags  # still wrong: kept, flagged
    assert s["check_failed"] == 1 and s["failed"] == 0
    assert any("were translated wrongly" in m for m in script.system)  # the redo batch carries the rule
    ctx.models.close()


def test_glossary_misses_are_counted(make_ctx, tmp_path, monkeypatch):
    tr = {"Have you ever been to Gensokyo?": "Ты бывал в Гэнсоку?"}
    script = _Script(tr=tr)
    ctx, _ = _en_ctx(make_ctx, tmp_path, monkeypatch, script)
    s = run_stage("translate", ctx)
    assert s["glossary_misses"] == 3  # e01 (Gensokyo), e02 (Patchouli), e03 (Reimu): the others say "перевод <id>" without their names
    ctx.models.close()


# ----------------------------------------------------------------------------
# context ladder
# ----------------------------------------------------------------------------


def _ladder_ctx(make_ctx, tmp_path, budget, margin=0.0, **sets):
    ctx = make_ctx(tmp_path, sets={"translate.llm.backend": "gemma4_12b", "translate.llm.fallbacks": ["qwen35_9b"], "translate.stress.tool": "none",
                                   "translate.glossary": str(tmp_path / "g.yaml"), "project.src_lang": "en", "translate.names_pass": False,
                                   "translate.include_narration": "off", **sets})
    build_project(ctx, _en_rows())
    made = _registry(ctx, tmp_path, _Script(), budget=budget, margin=margin)
    return ctx, made


def test_ladder_runs_gemma_at_8192_when_16k_does_not_fit(make_ctx, tmp_path):
    ctx, made = _ladder_ctx(make_ctx, tmp_path, budget=7.5, **{"translate.batch_lines": 4, "translate.context_window_lines": 6})
    name, spec, used = translate._llm_first(ctx)
    assert (name, used) == ("gemma4_12b", 8192)
    s = run_stage("translate", ctx)
    assert s["models_used"] == {"gemma4_12b": 5}
    assert made[0].options["ctx"] == 8192 and made[0].spec.name == "gemma4_12b"
    assert made[0].options["n_gpu_layers"] == -1  # never a partial offload
    ctx.models.close()


def test_ladder_scales_the_batches_with_the_context(make_ctx, tmp_path):
    ctx, made = _ladder_ctx(make_ctx, tmp_path, budget=7.5, **{"translate.batch_lines": 4})
    sizes: list[int] = []
    inner = ctx.models.factory

    def spy(spec, options, device):
        b = inner(spec, options, device)
        orig = b.script

        def counting(backend, kind, messages):
            if not messages[0]["content"].startswith(("Summarize", "You are a localization editor")):
                sizes.append(len(json.loads(messages[-1]["content"].split("\n", 1)[1])))
            return orig(backend, kind, messages)

        b.script = counting
        return b

    ctx.models.factory = spy
    run_stage("translate", ctx)
    assert sizes == [2, 2, 1]  # 5 lines, batch_lines 4 x (8192 / 16384) = 2
    ctx.models.close()


def test_ladder_falls_to_qwen_when_not_even_4096_fits(make_ctx, tmp_path):
    ctx, made = _ladder_ctx(make_ctx, tmp_path, budget=7.0)
    name, spec, used = translate._llm_first(ctx)
    assert (name, used) == ("qwen35_9b", 16384)
    plan = translate._llm_plan(ctx)
    reasons: list[str] = []
    assert [n for n, _ in translate._llm_candidates(ctx, plan, reasons)] == ["qwen35_9b"]
    assert "also at ctx 4096" in reasons[0] and plan.ctx_used == {}
    s = run_stage("translate", ctx)
    assert s["models_used"] == {"qwen35_9b": 5} and made[0].options["ctx"] == 16384
    ctx.models.close()


def test_ladder_gemma_at_4096_when_8192_is_a_hair_too_big(make_ctx, tmp_path):
    ctx, _ = _ladder_ctx(make_ctx, tmp_path, budget=7.35)
    assert translate._llm_first(ctx)[::2] == ("gemma4_12b", 4096)
    assert translate._ctx_scale(4096, 16384) == 0.25 and translate._ctx_scale(16384, 16384) == 1.0


def test_ladder_is_not_tried_for_a_missing_gpu_or_a_model_without_vram_model(make_ctx, tmp_path, monkeypatch):
    spec = BackendSpec(name="llm_plain", role="llm", env="fake", adapter="x", license="MIT", commercial_ok=True, vram_estimate_gb=9.0, heavy=True)
    assert translate._ctx_ladder(spec, 16384) == []
    assert translate._ctx_ladder(registry.get("gemma4_12b"), 16384) == [8192, 4096]
    assert translate._ctx_ladder(registry.get("hy_mt15_7b"), 8192) == [4096]


# ----------------------------------------------------------------------------
# narration and menus
# ----------------------------------------------------------------------------


def _narr_rows():
    return [
        dict(line_id="n00", speaker_id="sakuya", speaker_raw="Sakuya", text="Welcome home.", seconds=0.5),
        dict(line_id="n01", speaker_id="narrator", text="The wind was cold.", voiced=False),
        dict(line_id="n02", speaker_id="narrator", text="", voiced=False, flags=["menu"], extra={"choices": ["Stay here", "Leave"]}),
        dict(line_id="n03", speaker_id="narrator", text="Pick one.", voiced=False, flags=["menu_caption"]),
        dict(line_id="n04", speaker_id="sakuya", speaker_raw="Sakuya", text="Goodbye.", seconds=0.5),
    ]


@pytest.mark.parametrize(
    "mode,engine,text_enabled,expect_narr,expect_choices",
    [
        ("auto", "renpy", None, True, True),
        ("auto", "renpy", False, False, True),  # engine renpy: choices are game text anyway
        ("auto", "kag", None, False, False),
        ("on", "kag", None, True, True),
        ("off", "renpy", None, False, True),
        ("off", "kag", None, False, False),
    ],
)
def test_narration_and_choices_selection(make_ctx, tmp_path, monkeypatch, mode, engine, text_enabled, expect_narr, expect_choices):
    sets = {"translate.include_narration": mode}
    if text_enabled is not None:
        sets["pack.renpy"] = {"mode": "loose", "text": {"enabled": text_enabled}}
    ctx, _ = _en_ctx(make_ctx, tmp_path, monkeypatch, _Script(), rows=_narr_rows(), project={"engine": engine}, **sets)
    got = {l.line_id for l in translate.lines_to_translate(ctx)}
    assert {"n00", "n04"} <= got
    assert ("n01" in got) is expect_narr
    assert ("n02" in got) is expect_choices
    assert ("n03" in got) is (expect_narr or expect_choices)  # a menu caption is game text too
    ctx.models.close()


def test_narration_is_translated_but_never_reaches_synth(make_ctx, tmp_path, monkeypatch):
    from tests.genproj import build
    from tests.p3_helpers import add_unvoiced

    ctx = make_ctx(tmp_path / "game", sets={"synth.voice_unvoiced": True, "translate.include_narration": "on", "translate.names_pass": False})
    build(ctx, n_per=2)
    ids = add_unvoiced(ctx)
    # translations of these lines come from the run below, not from the helper
    ctx.paths.translation("ru").unlink()
    run_stage("translate", ctx)
    recs = {r.line_id: r for r in read_jsonl(ctx.paths.translation("ru"), translate.TranslationRecord)}
    assert ids["narrator"] in recs and recs[ids["narrator"]].text_tgt  # translated ...
    assert ids["menu"] not in recs or recs[ids["menu"]].text_tgt == ""
    dub = {l.line_id for l in ctx.dub_lines()}
    assert ids["narrator"] not in dub and ids["menu"] not in dub  # ... and not a dub line
    lines = {l.line_id: l for l in ctx.lines()}
    assert translate.text_only(lines[ids["narrator"]]) and not translate.dubbable(lines[ids["narrator"]])
    assert translate.dubbable(lines[ids["hero"][0]])
    run_stage("voicebank", ctx)
    run_stage("synth", ctx)
    for lid in (ids["narrator"], ids["menu"]):
        assert not ctx.paths.synth_dir(lid).exists(), lid  # no candidate for narration
    assert all(ctx.paths.synth_dir(l).exists() for l in ids["hero"])


def test_menu_choices_are_translated_into_choices_tgt(make_ctx, tmp_path, monkeypatch):
    tr = {"Stay here": "Остаться", "Leave": "Уйти", "The wind was cold.": "Ветер был холодным.", "Pick one.": "Выбери."}
    script = _Script(tr=tr)
    ctx, _ = _en_ctx(make_ctx, tmp_path, monkeypatch, script, rows=_narr_rows(), project={"engine": "renpy"}, **{"translate.include_narration": "auto", "translate.names_pass": False})
    s = run_stage("translate", ctx)
    recs = _records(ctx)
    menu = recs["n02"]
    assert menu.choices_tgt == ["Остаться", "Уйти"] and menu.text_tgt == "" and "translate_failed" not in menu.flags
    assert recs["n01"].text_tgt == "Ветер был холодным." and recs["n03"].text_tgt == "Выбери." and recs["n03"].choices_tgt is None
    assert "#c" not in " ".join(recs)  # the choice items never become records of their own
    assert s["choices_translated"] == 1 and s["text_only"] == 3
    assert any("player_choice" in m for m in script.system)
    ctx.models.close()


def test_choices_follow_the_player_gender(make_ctx, tmp_path, monkeypatch):
    script = _Script()
    (tmp_path / "glossary.yaml").write_text(yaml.safe_dump({"player": {"gender": "f"}}), encoding="utf-8")
    ctx, _ = _en_ctx(make_ctx, tmp_path, monkeypatch, script, rows=_narr_rows(), project={"engine": "renpy"}, **{"translate.names_pass": False})
    run_stage("translate", ctx)
    assert any("is female" in m for m in script.system)
    ctx.models.close()


def test_an_import_without_narration_does_not_flag_it_missing(make_ctx, tmp_path, monkeypatch):
    imp = tmp_path / "imp.jsonl"
    imp.write_text("\n".join(json.dumps({"line_id": k, "text": v}, ensure_ascii=False) for k, v in {"n00": "Добро пожаловать.", "n04": "Пока."}.items()), encoding="utf-8")
    ctx, _ = _en_ctx(make_ctx, tmp_path, monkeypatch, _Script(), rows=_narr_rows(), project={"engine": "renpy"}, **{"translate.import": str(imp)})
    s = run_stage("translate", ctx)
    assert s["missing"] == 0 and set(_records(ctx)) == {"n00", "n04"}


def test_old_translation_files_read_without_choices_tgt():
    rec = translate.TranslationRecord.model_validate({"line_id": "x", "text_tgt": "y"})
    assert rec.choices_tgt is None
    assert LineRecord  # the schema import above stays used
