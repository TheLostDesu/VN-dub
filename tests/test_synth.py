"""synth: route planning, candidate counts, interjection originals, sidecars, VC, regen, cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.genproj import build
from tests.p3_helpers import watch_models
from vndub.gpu import VramBudgetExceeded
from vndub.manifest.io import SelectionRecord, load_characters, read_json, save_characters, write_jsonl
from vndub.stages import genutil as gu
from vndub.stages import synth, voicebank
from vndub.stages.common import StageError, run_stage


@pytest.fixture
def proj(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx)
    return ctx


def _cands(ctx, line_id):
    d = ctx.paths.synth_dir(line_id)
    return sorted(p for p in d.glob("*.wav") if not p.name.endswith(".prevc.wav"))


def test_route_a_counts_and_sidecars(proj):
    ctx = proj
    s = synth.run(ctx)
    assert s["failed"] == 0
    assert "skipped_routes" in s and any(k.endswith(":B") for k in s["skipped_routes"])  # no performer yet
    lines = ctx.voiced_lines()
    normal = [l for l in lines if l.text_src != "えへへ。"]
    interj = [l for l in lines if l.text_src == "えへへ。"][0]
    for l in normal:
        c = _cands(ctx, l.line_id)
        n = ctx.cfg.synth.n_candidates
        # short lines (<= 2 syllables / < 1 s) and high arousal get n_candidates_short
        assert len(c) in (n, ctx.cfg.synth.n_candidates_short)
        side = read_json(c[0].with_suffix(".json"))
        assert side["backend"] == "mock_tts" and side["route"] == "A"
        assert side["text_used"] and side["style_prompt"] and side["ref_paths"]
        assert side["style"]["emotion"]
    ci = _cands(ctx, interj.line_id)
    assert any(p.name.startswith("orig_orig_") for p in ci)
    assert len([p for p in ci if p.name.startswith("mock_tts")]) == ctx.cfg.synth.n_candidates_short
    side = read_json(next(p for p in ci if p.name.startswith("orig_")).with_suffix(".json"))
    assert side["route"] == "orig" and side["text_variant"] == "orig"


def test_cache_second_run(proj):
    s1 = synth.run(proj)
    s2 = synth.run(proj)
    assert s2["cached"] == s1["candidates"] and s2.get("generated", 0) == 0


def test_route_b_with_voicebank_and_vc(proj):
    ctx = proj
    vb = voicebank.run(ctx)
    assert vb["performers"] == 2 and vb["rvc"] == 2
    s = synth.run(ctx)
    assert s["failed"] == 0
    line = ctx.voiced_lines()[0]
    b = [p for p in _cands(ctx, line.line_id) if "_B_" in p.name]
    assert b, "route B candidates expected"
    side = read_json(b[0].with_suffix(".json"))
    assert side["pre_vc_wav"] and Path(ctx.paths.abs(side["pre_vc_wav"])).exists()
    assert side["params"]["vc"]["backend"] == "mock_vc"
    assert "f0_transpose" in side["params"]["vc"]


def test_character_route_override(proj):
    ctx = proj
    voicebank.run(ctx)
    chars = load_characters(ctx.paths.characters)
    chars["haruto"].route = "B"
    save_characters(ctx.paths.characters, chars)
    synth.run(ctx)
    for l in ctx.voiced_lines():
        names = [p.name for p in _cands(ctx, l.line_id) if not p.name.startswith("orig")]
        if l.speaker_id == "haruto":
            assert names and all("_B_" in n for n in names)
        else:
            assert any("_A_" in n for n in names)


def test_regen_new_seeds_keep_old(proj):
    ctx = proj
    synth.run(ctx)
    line = ctx.voiced_lines()[0]
    before = {p.name: read_json(p.with_suffix(".json"))["seed"] for p in _cands(ctx, line.line_id)}
    write_jsonl(ctx.paths.selection, [SelectionRecord(line_id=line.line_id, status="regen", round=0)])
    ctx.extra["regen"] = True
    s = synth.run(ctx)
    assert s["lines"] == 1 and s["round"] == 1
    after = {p.name: read_json(p.with_suffix(".json")) for p in _cands(ctx, line.line_id)}
    new = {n: d for n, d in after.items() if n not in before}
    assert new and all(d["round"] == 1 and d["k"] >= 100 for d in new.values())
    assert not ({d["seed"] for d in new.values()} & set(before.values()))
    assert set(before) <= set(after)  # earlier candidates stay available


def test_set_text_override_bumps_round(proj):
    from vndub.review.app import write_override

    ctx = proj
    synth.run(ctx)
    line = ctx.voiced_lines()[0]
    write_override(ctx.paths, line.line_id, "set_text", {"text": "Совсем другой текст."})
    synth.run(ctx)
    sides = [read_json(p.with_suffix(".json")) for p in _cands(ctx, line.line_id)]
    over = [s for s in sides if s["text_variant"] == "override"]
    assert over and all(s["text_used"] == "Совсем другой текст." and s["round"] == 1 for s in over)


def test_missing_translation_skipped(proj):
    ctx = proj
    trs = (ctx.paths.translation("ru")).read_text(encoding="utf-8").splitlines()
    (ctx.paths.translation("ru")).write_text("\n".join(trs[1:]) + "\n", encoding="utf-8")
    s = synth.run(ctx)
    assert s["no_translation"] == 1


def test_stage_runner_writes_status(proj):
    run_stage("synth", proj)
    st = read_json(proj.paths.stage_status)
    assert st["synth"]["finished"] and st["synth"]["summary"]["candidates"] > 0


# ---- round 8: an unavailable backend is loud, nothing is deleted, resume does not redo the TTS ----------------------------


def _all_sidecars(ctx):
    return {p.name: read_json(p) for line in ctx.voiced_lines() for p in sorted(ctx.paths.synth_dir(line.line_id).glob("*.json"))}


def _cands_of(ctx, marker=""):
    return [p for line in ctx.voiced_lines() for p in _cands(ctx, line.line_id) if marker in p.name]


def test_unavailable_backend_raises_and_keeps_existing_candidates(proj):
    ctx = proj
    synth.run(ctx)
    wavs = {p: p.stat().st_mtime_ns for p in _cands_of(ctx)}
    sidecars = _all_sidecars(ctx)
    ctx.cfg.synth.seed_base += 1  # every candidate is stale now, so every job wants the backend
    counts = watch_models(ctx, unavailable={"mock_tts": VramBudgetExceeded("mock_tts", "needs 6.5 GB but 2.1 GB are free (an orphan process holds the rest)")})
    with pytest.raises(StageError) as ei:
        synth.run(ctx)
    assert "mock_tts" in str(ei.value) and "orphan process" in str(ei.value)
    # not a single wav or sidecar was touched: no error sidecar, no unlinked candidate
    assert {p: p.stat().st_mtime_ns for p in wavs} == wavs
    assert all(p.exists() for p in wavs)
    assert _all_sidecars(ctx) == sidecars and not any(d.get("error") for d in sidecars.values())
    # the registry knows, once per route, and the event log has it
    got = [(u.stage, u.group) for u in ctx.models.unavailable]
    assert ("synth", "route A backend mock_tts") in got
    events = (ctx.paths.work / "gpu_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"unavailable"' in ln and "route A backend mock_tts" in ln for ln in events)
    # once the backend is back, the same run regenerates them
    counts.restore()
    s = synth.run(ctx)
    assert s["failed"] == 0 and s["generated"] > 0


def test_unavailable_backend_is_reported_after_the_cache_is_saved(proj):
    ctx = proj
    synth.run(ctx)
    ctx.cfg.synth.seed_base += 1
    watch_models(ctx, unavailable={"mock_tts": VramBudgetExceeded("mock_tts", "no room")})
    with pytest.raises(StageError, match="synth stopped"):
        synth.run(ctx)
    assert (ctx.paths.cache / "synth.json").exists()


def test_item_error_still_writes_sidecar_and_drops_the_stale_wav(proj, monkeypatch):
    ctx = proj
    synth.run(ctx)
    line = ctx.voiced_lines()[0]
    ctx.cfg.synth.seed_base += 1
    real = gu.run_batched
    hit: list[str] = []

    def flaky(backend, op, items, *a, **k):
        res = real(backend, op, items, *a, **k)
        if op == "synthesize" and not hit:
            for r in res:
                if r["id"].startswith(f"{line.line_id}|"):
                    r["error"] = "ValueError: this one text is broken"
                    hit.append(r["id"])
                    break
        return res

    monkeypatch.setattr(gu, "run_batched", flaky)
    s = synth.run(ctx)  # a single item failing is not a stage failure
    assert s["failed"] == 1 and any("broken" in k for k in s["errors"])
    broken = [p for p in ctx.paths.synth_dir(line.line_id).glob("*.json") if (read_json(p) or {}).get("error")]
    assert len(broken) == 1 and not broken[0].with_suffix(".wav").exists()


def test_out_of_memory_item_keeps_its_wav_and_sidecar(proj, monkeypatch):
    ctx = proj
    synth.run(ctx)
    line = ctx.voiced_lines()[0]
    before = {p.name: read_json(p.with_suffix(".json")) for p in _cands(ctx, line.line_id)}
    ctx.cfg.synth.seed_base += 1
    real = gu.run_batched
    hit: list[str] = []

    def oom_once(backend, op, items, *a, **k):
        res = real(backend, op, items, *a, **k)
        if op == "synthesize" and not hit:
            for r in res:
                if r["id"].startswith(f"{line.line_id}|"):
                    r["error"] = "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
                    hit.append(r["id"])
                    break
        return res

    monkeypatch.setattr(gu, "run_batched", oom_once)
    s = synth.run(ctx)  # a few items out of memory: loud, but the backend works and the rest is regenerated
    assert s["failed"] == 1 and hit
    wav = next(p for p in _cands(ctx, line.line_id) if p.stem == hit[0].split("|", 1)[1].replace("|", "_"))
    assert wav.exists()  # the wav of the earlier run is still there
    assert not (read_json(wav.with_suffix(".json")) or {}).get("error")
    assert (read_json(wav.with_suffix(".json")) or {}) == before[wav.name]  # and so is its sidecar
    assert any("out of memory" in u.reason for u in ctx.models.unavailable if u.stage == "synth")


def test_every_item_out_of_memory_stops_the_stage_and_keeps_everything(proj, monkeypatch):
    ctx = proj
    synth.run(ctx)
    wavs = _cands_of(ctx)
    sidecars = _all_sidecars(ctx)
    ctx.cfg.synth.seed_base += 1
    real = gu.run_batched

    def always_oom(backend, op, items, *a, **k):
        res = real(backend, op, items, *a, **k)
        if op == "synthesize":
            for r in res:
                r["error"] = "RuntimeError: CUDA error: out of memory"
        return res

    monkeypatch.setattr(gu, "run_batched", always_oom)
    with pytest.raises(StageError, match="out of memory"):
        synth.run(ctx)
    assert all(p.exists() for p in wavs) and _all_sidecars(ctx) == sidecars


def test_route_b_tts_is_not_redone_after_a_kill_during_vc(proj):
    ctx = proj
    voicebank.run(ctx)

    def kill_on_vc(op, items):
        if op == "convert":
            raise KeyboardInterrupt("killed during voice conversion")

    counts = watch_models(ctx, on_call=kill_on_vc)
    with pytest.raises(KeyboardInterrupt):
        synth.run(ctx)
    ctx.save_caches()  # what the periodic checkpoint has written by then
    assert counts["synthesize"] > 0
    prevc = sorted(ctx.paths.work.rglob("*_B_*.prevc.wav"))
    assert prevc and not _cands_of(ctx, "_B_")
    counts.restore()
    again = watch_models(ctx)
    events: list[tuple[int, int]] = []
    ctx.extra["progress_cb"] = lambda stage, done, total, item: events.append((done, total))
    s = synth.run(ctx)
    n_b = len(prevc)
    assert again["synthesize"] == 0, "the TTS of route B was redone"  # route A was cached, route B resumes at the VC
    n_a = len([p for p in _cands_of(ctx) if "_B_" not in p.name and not p.name.startswith("orig")])
    assert events[0] == (n_a + n_b, n_a + 2 * n_b)  # the cached candidates and the kept TTS audio are done from the start
    assert events[-1] == (n_a + 2 * n_b, n_a + 2 * n_b) and {t for _, t in events} == {n_a + 2 * n_b}
    assert again["convert"] == n_b and s["tts_reused"] == n_b and s["failed"] == 0
    assert len(_cands_of(ctx, "_B_")) == n_b


def test_route_b_tts_is_redone_when_its_request_changed(proj):
    ctx = proj
    voicebank.run(ctx)

    def kill_on_vc(op, items):
        if op == "convert":
            raise KeyboardInterrupt()

    counts = watch_models(ctx, on_call=kill_on_vc)
    with pytest.raises(KeyboardInterrupt):
        synth.run(ctx)
    ctx.save_caches()
    counts.restore()
    ctx.cfg.synth.seed_base += 1  # a different request: the kept pre-VC audio does not describe it
    again = watch_models(ctx)
    s = synth.run(ctx)
    assert s["tts_reused"] == 0 and again["synthesize"] > 0 and s["failed"] == 0


def test_project_folders_are_not_part_of_the_candidate_key(proj, monkeypatch):
    ctx = proj
    with monkeypatch.context() as m:  # the cache as the previous version built it: folders inside the key
        m.setattr(gu, "stage_cache", lambda c, stage, version, *sections: c.cache(stage, version, *sections))
        s0 = synth.run(ctx)
    assert s0["generated"] > 0
    ctx.save_caches()
    s1 = synth.run(ctx)  # the upgrade: every old entry is still fresh (and migrated)
    assert s1["generated"] == 0 and s1["cached"] == s0["candidates"]
    ctx.save_caches()
    ctx.cfg.project.models_dir = "D:/somewhere/else/models"
    ctx.cfg.project.game_dir = "E:/moved/game"
    ctx.cfg.project.out_dir = "out2"
    s2 = synth.run(ctx)
    assert s2["generated"] == 0 and s2["cached"] == s0["candidates"]
    # the folders are out of the hash, the language of the project is still in it
    h = gu.stable_section_hash(ctx.cfg, "synth", "project")
    ctx.cfg.project.work_dir = "elsewhere"
    assert gu.stable_section_hash(ctx.cfg, "synth", "project") == h
    ctx.cfg.project.tgt_lang = "en"
    assert gu.stable_section_hash(ctx.cfg, "synth", "project") != h


def test_synth_progress_counts_every_unit_and_never_shrinks(proj):
    ctx = proj
    voicebank.run(ctx)
    events: list[tuple[int, int]] = []
    ctx.extra["progress_cb"] = lambda stage, done, total, item: events.append((done, total))
    synth.run(ctx)
    total = events[0][1]
    n_jobs = len([p for p in _cands_of(ctx) if not p.name.startswith("orig")])
    n_b = len(_cands_of(ctx, "_B_"))
    assert total == n_jobs + n_b  # a route-B candidate that is generated is two units (TTS, VC)
    assert events[0][0] == 0 and events[-1] == (total, total)
    assert {t for _, t in events} == {total} and [d for d, _ in events] == sorted(d for d, _ in events)
    events.clear()
    ctx.save_caches()
    synth.run(ctx)  # everything cached: n / n at once (a cached route-B candidate is one unit, its final wav)
    assert events[0] == (n_jobs, n_jobs) and set(events) == {(n_jobs, n_jobs)}


class _Cancelled(Exception):
    pass


def test_a_cancel_after_a_chunk_keeps_every_take_of_that_chunk(proj):
    """The app's cancel is raised by the progress report: it must come after the chunk's takes are on disk and in the cache."""
    ctx = proj

    def cancel_on_the_first_unit(stage, done, total, item):
        if item is not None:  # (the opening report has no item)
            raise _Cancelled()

    ctx.extra["progress_cb"] = cancel_on_the_first_unit
    with pytest.raises(_Cancelled):
        run_stage("synth", ctx)  # saves the caches on the way out, like a cancelled app job
    wavs = [p for p in _cands_of(ctx) if not p.name.startswith("orig")]
    assert len(wavs) > 2, "the project must have more takes than the one that was reported before the cancel"
    assert all(p.with_suffix(".json").exists() for p in wavs)
    del ctx.extra["progress_cb"]
    s = synth.run(ctx)  # the resume finds them all: nothing is generated again
    assert s["generated"] == 0 and s["cached"] == len(wavs)


def test_synth_progress_class_units():
    seen: list[tuple[int, int]] = []

    class C:
        def progress(self, done, total, item=None):
            seen.append((done, total))

    p = synth._Progress(C(), 10, 4)
    assert seen == [(4, 10)]
    p("l1")
    assert seen[-1] == (5, 10)
    p.finish()
    assert seen[-1] == (10, 10)
    seen.clear()
    synth._Progress(C(), 0, 0)
    assert seen == []


# ---- laughter: the text a backend gets (spelling / inline token / tag), the style, the sidecar ---------------------------


def _say(ctx, line_id: str, text: str, plain: str | None = None) -> None:
    """Replace the translation of one line (text_tgt = text_tts; ``plain`` = the CER reference, default the same)."""
    trs = gu.load_translations(ctx)
    r = trs[line_id]
    r.text_tgt = r.text_tgt_short = r.text_tts = text
    r.text_plain = text if plain is None else plain
    write_jsonl(ctx.paths.translation(ctx.cfg.project.tgt_lang), list(trs.values()))


def _sides(ctx, line_id: str) -> list[dict]:
    return [read_json(p.with_suffix(".json")) for p in _cands(ctx, line_id) if not p.name.startswith("orig")]


def _with_caps(monkeypatch, **caps):
    """The mock TTS with the capabilities of another backend (tag syntax, instruct kind)."""
    from vndub.backends import registry

    for k, v in caps.items():
        monkeypatch.setitem(registry.get("mock_tts").caps, k, v)


def test_laughter_is_respelled_for_a_plain_backend(proj):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "Ха-ха-ха, ну ты даёшь!")
    s = synth.run(ctx)
    assert s["failed"] == 0 and s["laughter"]["lines"] >= 1 and s["laughter"]["modes"]["respell"] >= 1
    sides = _sides(ctx, line.line_id)
    assert sides
    for side in sides:
        assert side["text_used"] == "Ахахаха, ну ты даёшь!"  # 3 syllables: the long spelling of configs/laughter.yaml
        assert side["text_plain"] == "Ха-ха-ха, ну ты даёшь!"  # the CER reference is never respelled
        assert side["laughter"] == {"kind": "hearty", "n": 1, "mode": "respell", "only": False, "stage": False}
        assert side["style"]["laughter"]["kind"] == "hearty" and "laughter" in side["style"]["events"]
        assert any(p in side["style_prompt"] for p in ("warm, easy laugh", "laughing heartily", "bursting into hearty laughter"))
        assert "light laugh" not in side["style_prompt"]
    # a line without laughter is untouched: no laughter field, no laughter key in the stored style
    other = ctx.voiced_lines()[1]
    for side in _sides(ctx, other.line_id):
        assert "laughter" not in side and "laughter" not in side["style"] and "laugh" not in side["style_prompt"]


def test_stage_direction_is_removed_from_the_spoken_text(proj):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "*laughs* Нет уж, не выйдет.")
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "Нет уж, не выйдет."
        assert side["text_plain"] == "*laughs* Нет уж, не выйдет."
        assert side["laughter"]["stage"] is True and side["laughter"]["kind"] == "hearty"
    # a line that is only a stage direction is spoken as a laugh, never sent empty
    _say(ctx, line.line_id, "(смеётся)")
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "Ахаха!" and side["laughter"]["only"] is True  # the short hearty spelling of configs/laughter.yaml


def test_laughter_only_line_gets_an_instruction_about_the_laugh(proj):
    ctx = proj
    interj = [l for l in ctx.voiced_lines() if l.text_src == "えへへ。"][0]  # translated "Хе-хе."
    synth.run(ctx)
    sides = _sides(ctx, interj.line_id)
    assert sides
    for side in sides:
        assert side["text_used"] == "Хехе." and side["laughter"]["kind"] == "chuckle" and side["laughter"]["only"] is True  # the short chuckle spelling of configs/laughter.yaml
        assert side["style_prompt"].startswith("Only laugh, no words: make a low, quiet chuckle")
        assert side["style"]["laughter"]["only"] is True


def test_inline_token_for_a_cosy_inline_backend(proj, monkeypatch):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "Ха-ха-ха, ну ты даёшь!")
    _with_caps(monkeypatch, instruct="cosy", tags="cosy_inline")
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "[laughter] ну ты даёшь!"  # the CosyVoice sound event replaces the written laugh
        assert side["laughter"]["mode"] == "inline:[laughter]"
        assert side["text_plain"] == "Ха-ха-ха, ну ты даёшь!"
    # a line that is only a laugh keeps a spelling next to the token (configs/laughter.yaml cosyvoice3 `only: tag` is for that backend name;
    # here the mock backend has no section, so the token alone replaces the laugh)
    interj = [l for l in ctx.voiced_lines() if l.text_src == "えへへ。"][0]
    for side in _sides(ctx, interj.line_id):
        assert side["text_used"] == "[laughter]." and side["laughter"]["mode"] == "inline:[laughter]"


def test_tag_in_front_of_the_spelling_for_fish_and_higgs(proj, monkeypatch):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "Хе-хе, посмотрим.")
    _with_caps(monkeypatch, instruct="none", tags="fish")
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        assert side["text_used"].endswith("[chuckling] Хехе, посмотрим.")
        assert side["text_used"].count("[chuckling]") == 1 and "[laughing]" not in side["text_used"]  # the prefix leaves its own laugh tag out
        assert side["laughter"]["mode"] == "tag:[laughing]"
    _with_caps(monkeypatch, instruct="none", tags="higgs")
    ctx.cfg.synth.seed_base += 1
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        # the tag is written right against the onomatopoeia (Higgs model card); the mock backend has no `higgs_tts3` section in
        # configs/laughter.yaml, so the spelling is the Russian one (that section spells it in English, see test_text_laughter)
        assert "<|sfx:laughter|>Хехе, посмотрим." in side["text_used"]
        assert side["text_used"].count("<|sfx:laughter|>") == 1


def test_laughter_can_be_switched_off_and_a_user_text_is_left_alone(proj):
    from vndub.review.app import write_override

    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "Ха-ха-ха, ну ты даёшь! *laughs*")
    ctx.cfg.synth.laughter.enabled = False
    s = synth.run(ctx)
    assert "laughter" not in s
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "Ха-ха-ха, ну ты даёшь! *laughs*" and "laughter" not in side
    ctx.cfg.synth.laughter.enabled = True
    # round 18: a review text gets translate's preparation, so its laugh is written for the backend like a translated one ...
    write_override(ctx.paths, line.line_id, "set_text", {"text": "Ха-ха-ха, вот так."})
    synth.run(ctx)
    over = [s for s in _sides(ctx, line.line_id) if s["text_variant"] == "override"]
    assert over and all("laughter" in s and "вот так" in s["text_used"] for s in over)
    # ... unless the user marked the stress himself: then the text is spoken as typed (numbers spelled out)
    write_override(ctx.paths, line.line_id, "set_text", {"text": "Ха-ха-ха, вот т+ак."})
    synth.run(ctx)
    over = [s for s in _sides(ctx, line.line_id) if s["text_variant"] == "override" and s["round"] == 2]  # (round 1 says the same words: kept)
    assert over and all("laughter" not in s and "Ха-ха-ха" in s["text_used"] and "вот т" in s["text_used"] for s in over)


def test_missing_and_broken_laughter_table(proj, tmp_path):
    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "Ха-ха-ха, ну ты даёшь!")
    ctx.cfg.synth.laughter.table = str(tmp_path / "nowhere.yaml")
    synth.run(ctx)  # a missing table is a warning: the text goes out as translated
    assert all(s["text_used"] == "Ха-ха-ха, ну ты даёшь!" for s in _sides(ctx, line.line_id))
    bad = tmp_path / "bad.yaml"
    bad.write_text("ru:\n  cackle: Ха\n", encoding="utf-8")
    ctx.cfg.synth.laughter.table = str(bad)
    with pytest.raises(StageError, match="laughter table is invalid"):
        synth.run(ctx)


def test_table_edit_changes_the_spelling_and_only_regenerates_laughing_lines(proj, tmp_path):
    import yaml

    from vndub.text import laughter as lg

    ctx = proj
    line = ctx.voiced_lines()[0]
    _say(ctx, line.line_id, "Ха-ха-ха, ну ты даёшь!")
    s1 = synth.run(ctx)
    ctx.save_caches()
    table = yaml.safe_load(lg.default_table_path().read_text(encoding="utf-8"))
    table["ru"]["hearty"] = {"short": "Хаха!", "long": "Хахахаха!"}
    mine = tmp_path / "my_laughter.yaml"
    mine.write_text(yaml.safe_dump(table, allow_unicode=True), encoding="utf-8")
    ctx.cfg.synth.laughter.table = str(mine)
    s2 = synth.run(ctx)
    assert all(s["text_used"] == "Хахахаха, ну ты даёшь!" for s in _sides(ctx, line.line_id))
    n_laugh = len(_sides(ctx, line.line_id))
    assert s2["generated"] == n_laugh and s2["cached"] == s1["candidates"] - n_laugh  # the other lines keep their candidates


def test_synth_laughter_setting_is_not_part_of_the_cache_hash(proj):
    ctx = proj
    s1 = synth.run(ctx)
    ctx.save_caches()
    base = synth._synth_hash(ctx.cfg)
    ctx.cfg.synth.laughter.table = str(ctx.cfg.resolve("configs/laughter.yaml"))  # another path to the same table
    ctx.cfg.synth.laughter.enabled = False
    assert synth._synth_hash(ctx.cfg) == base  # adding / editing the key must not regenerate a project without laughter
    assert gu.stable_section_hash(ctx.cfg, "synth", "project") != base  # (the generic hash would)
    s2 = synth.run(ctx)
    laugh = [l for l in ctx.voiced_lines() if l.text_src == "えへへ。"][0]  # the only line with a laugh in the test project
    n = len(_sides(ctx, laugh.line_id))
    assert s2["generated"] == n and s2["cached"] == s1["candidates"] - n
    ctx.cfg.synth.seed_base += 1
    assert synth._synth_hash(ctx.cfg) != base


def test_haha_no_way_for_the_qwen_clone(proj, monkeypatch):
    """End to end for the user's setup: Qwen3-TTS Base clones the performer (caps instruct=qwen, no tags)."""
    ctx = proj
    line = ctx.voiced_lines()[0]
    _with_caps(monkeypatch, instruct="qwen", tags="none", knobs="none")
    _say(ctx, line.line_id, "Ха-ха-ха, ни за что!")  # "Haha, no way!" as the translator writes it
    synth.run(ctx)
    for side in _sides(ctx, line.line_id):
        assert side["text_used"] == "Ахахаха, ни за что!"
        assert side["params"]["instruct"] == side["style_prompt"] and "laugh" in side["style_prompt"]
        assert side["laughter"] == {"kind": "hearty", "n": 1, "mode": "respell", "only": False, "stage": False}
