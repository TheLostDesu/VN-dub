"""voicebank: ref selection, performer choice (mock design), RVC meta + f0_transpose, speaker calibration."""

from __future__ import annotations

import math

import pytest

from tests.genproj import build
from vndub.manifest.io import Character, CharacterStats, LineRecord, RefClip, load_characters, read_json, read_yaml
from vndub.stages import voicebank
from vndub.stages.voicebank import performer_description, pitch_word, select_refs


def _line(i: int, text: str = "テストです。") -> LineRecord:
    return LineRecord(line_id=f"l{i}", scene="s", order=i, speaker_id="a", text_src=text, has_voice=True, src_wav=f"audio/src/{i}.wav")


def _pp(snr: float, dur: float, emo: str, bgm: bool = False) -> dict:
    return {"duration_s": dur, "snr_db": snr, "bgm": bgm, "style": {"emotion": emo}, "emotion": {"adv": {"arousal": 0.5, "valence": 0.5}}}


def test_select_refs_filters_and_covers_emotions():
    lines = [_line(i) for i in range(10)]
    pps = {
        "l0": _pp(35, 4, "happy"), "l1": _pp(34, 5, "happy"), "l2": _pp(33, 6, "happy"),
        "l3": _pp(30, 4, "sad"), "l4": _pp(29, 4, "sad"),
        "l5": _pp(40, 4, "neutral"),
        "l6": _pp(10, 4, "angry"),            # too noisy
        "l7": _pp(40, 1.0, "angry"),          # too short
        "l8": _pp(40, 4, "angry", bgm=True),  # BGM
        "l9": _pp(28, 9, "surprised"),
    }
    chosen, relaxed = select_refs(lines, pps, min_snr_db=25, min_sec=3, max_sec=10, per_emotion=2, max_total=10, min_refs=5)
    ids = {c["line"].line_id for c in chosen}
    assert not relaxed
    assert ids.isdisjoint({"l6", "l7", "l8"})
    emos = [c["emotion"] for c in chosen]
    assert emos.count("happy") == 2 and set(emos) == {"happy", "sad", "neutral", "surprised"}
    assert all(c["usage"] in ("timbre", "emotion", "both") for c in chosen)
    # strict filter leaves too few -> relaxed
    chosen, relaxed = select_refs(lines[:3], {k: pps[k] for k in ("l0", "l1", "l2")}, min_snr_db=50, min_sec=3, max_sec=10, per_emotion=2, max_total=10, min_refs=2)
    assert relaxed and len(chosen) >= 2


def test_pitch_and_description():
    assert pitch_word(300, "f") == "high pitch" and pitch_word(180, "f") == "low pitch" and pitch_word(130, "m") == "medium pitch"
    c = Character(gender="f", age_hint="teen", stats=CharacterStats(f0_median_hz=290, emotion_hist={"happy": 0.7, "neutral": 0.3}))
    d = performer_description(c, "ru")
    assert "young female" in d and "high pitch" in d and "native Russian" in d and "cheerful" in d


@pytest.fixture
def vb(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=3)
    summary = voicebank.run(ctx)
    return ctx, summary


def test_voicebank_outputs(vb):
    ctx, s = vb
    assert s["characters"] == 2 and s["performers"] == 2 and s["rvc"] == 2
    chars = load_characters(ctx.paths.characters)
    for sid in ("sakura", "haruto"):
        refs = [RefClip.model_validate(r) for r in read_json(ctx.paths.vb_refs_json(sid))]
        assert 1 <= len(refs) <= ctx.cfg.voicebank.refs.max_total
        assert all(ctx.paths.abs(r.path).exists() for r in refs)
        assert chars[sid].refs == [r.path for r in refs]
        perf = read_yaml(ctx.paths.vb_performer(sid))
        assert perf["refs"] and all(ctx.paths.abs(r["path"]).exists() for r in perf["refs"])
        assert perf["backend"] == "mock_tts"  # sapi_tts -> mock in tests
        assert perf["f0_median_hz"] and perf["f0_delta_st"] is not None
        meta = read_json(ctx.paths.vb_rvc(sid) / "meta.json")
        expect = round(12 * math.log2(chars[sid].stats.f0_median_hz / perf["f0_median_hz"]))
        assert meta["f0_transpose"] == expect
        assert chars[sid].rvc["model"] == meta["model"]
    cal = read_json(ctx.paths.spk_calibration)
    assert cal["n_same"] > 0 and cal["n_diff"] > 0 and "threshold" in cal and 0 <= cal["eer"] <= 1


def test_performer_emotion_variants(vb):
    ctx, _ = vb
    perf = read_yaml(ctx.paths.vb_performer("sakura"))
    emos = {r["emotion"] for r in perf["refs"]}
    assert "neutral" in emos  # the chosen design's test sentences


def test_voicebank_cached_rerun(vb):
    ctx, _ = vb
    events = len(ctx.models.events)
    s2 = voicebank.run(ctx)
    assert s2["performers"] == 2 and s2["rvc"] == 2
    loads = [e for e in ctx.models.events[events:] if e["event"] == "load" and e["model"] in ("mock_tts", "mock_vc")]
    assert not loads  # designs and RVC were cached


# ---- round 8 ------------------------------------------------------------------------------------------------------------------


def test_voicebank_cache_ignores_the_project_folders(make_ctx, tmp_path, monkeypatch):
    from vndub.stages import genutil as gu

    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=3)
    with monkeypatch.context() as m:  # the cache as the previous version built it: the folders were part of every key
        m.setattr(gu, "stage_cache", lambda c, stage, version, *sections: c.cache(stage, version, *sections))
        voicebank.run(ctx)
    ctx.save_caches()
    events = len(ctx.models.events)
    ctx.cfg.project.models_dir = "D:/elsewhere/models"  # the install folder moved
    ctx.cfg.project.game_dir = "E:/moved/game"
    s = voicebank.run(ctx)
    assert s["performers"] == 2 and s["rvc"] == 2
    loads = [e for e in ctx.models.events[events:] if e["event"] == "load" and e["model"] in ("mock_tts", "mock_vc")]
    assert not loads  # neither the designs nor the RVC training were redone for a moved folder


def test_voicebank_reports_a_missing_embedder(make_ctx, tmp_path):
    from tests.p3_helpers import watch_models
    from vndub.gpu import VramBudgetExceeded

    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=3)
    watch_models(ctx, unavailable={"ecapa": VramBudgetExceeded("ecapa", "an orphan process holds the card")})
    s = voicebank.run(ctx)
    assert "unavailable" in str(s["calibration"])
    got = {(u.stage, u.group): u.reason for u in ctx.models.unavailable}
    assert "orphan process" in got[("voicebank", "speaker embedder")]


# ---- release 1.0 review fixes -------------------------------------------------------------------------------------------------


class _HashEmbedder:
    """A fake embedder whose vector is a fingerprint of the file's bytes: a cache that reuses a stale npy shows up as a wrong vector."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, op, items, **kw):
        import numpy as np

        assert op == "embed"
        out = []
        for it in items:
            self.calls.append(it["id"])
            np.save(it["out"], fingerprint(it["wav"]))
            out.append({"id": it["id"], "path": it["out"]})
        return out


def fingerprint(path):
    import hashlib
    from pathlib import Path

    import numpy as np

    return np.frombuffer(hashlib.sha256(Path(path).read_bytes()).digest()[:8], dtype=np.uint8).astype(np.float32)


def _chosen(line):
    return [{"line": line, "emotion": "neutral", "arousal": None, "valence": None, "snr": 20.0, "duration": 1.0, "usage": "both"}]


def test_new_refs_under_an_old_name_are_not_scored_with_the_old_embedding(vb):
    import os
    import time

    import numpy as np

    from vndub.stages import genutil as gu

    ctx, _ = vb
    sid = "sakura"
    ls = [l for l in ctx.voiced_lines(apply_only=False) if l.speaker_id == sid]
    # the second source clip is OLDER than the embedding made in between: shutil.copy2 kept that mtime on the copy
    os.utime(ctx.paths.abs(ls[1].src_wav), (time.time() - 7200, time.time() - 7200))
    fake = _HashEmbedder()

    def embed():
        refs = gu.load_refs(ctx, sid)
        jobs = voicebank._ref_clip_jobs(ctx, sid, refs, [], {}, "fake")
        return jobs, gu.embed_paths(ctx, fake, jobs)

    voicebank._write_refs(ctx, sid, _chosen(ls[0]))
    jobs, vecs = embed()
    assert len(jobs) == 1 and np.array_equal(vecs[jobs[0][0]], fingerprint(jobs[0][1]))
    first_calls = len(fake.calls)
    voicebank._write_refs(ctx, sid, _chosen(ls[1]))  # a different clip, the same name neutral_01.wav
    jobs2, vecs2 = embed()
    assert jobs2[0][0] == jobs[0][0] and len(fake.calls) == first_calls + 1  # recomputed, not reused
    assert np.array_equal(vecs2[jobs2[0][0]], fingerprint(jobs2[0][1])) and not np.array_equal(vecs[jobs[0][0]], vecs2[jobs2[0][0]])
    # the F0 of the old refs goes too (it is keyed by the same stem)
    stale = ctx.paths.vb(sid) / "feat" / "neutral_01.f0.npy"
    stale.parent.mkdir(parents=True, exist_ok=True)
    np.save(stale, np.zeros(3))
    voicebank._write_refs(ctx, sid, _chosen(ls[0]))
    assert not stale.exists()


def test_refs_cache_key_follows_the_passport_fields_select_refs_reads(vb):
    from vndub.manifest.io import write_json
    from vndub.stages import genutil as gu

    ctx, _ = vb
    sid = "sakura"
    ref = gu.load_refs(ctx, sid)[0]
    pp = read_json(ctx.paths.passport(ref.line_id))
    assert (pp["emotion"].get("adv") or {}).get("arousal") != 0.123
    # analyze ran again (the ADV model is available now): the new arousal and emotion must reach refs.json
    pp["emotion"]["adv"] = {"arousal": 0.123, "valence": 0.321, "dominance": 0.5}
    pp["style"] = {**(pp.get("style") or {}), "emotion": "angry"}
    write_json(ctx.paths.passport(ref.line_id), pp)
    voicebank.run(ctx)
    now = {r.line_id: r for r in gu.load_refs(ctx, sid)}
    assert now[ref.line_id].arousal == 0.123 and now[ref.line_id].valence == 0.321 and now[ref.line_id].emotion == "angry"


def test_refs_without_any_passport_are_chosen_by_duration(make_ctx, tmp_path):
    import shutil

    from vndub.stages import genutil as gu

    ctx = make_ctx(tmp_path / "game")
    build(ctx, n_per=3)
    shutil.rmtree(ctx.paths.passport("x").parent)  # analyze never ran
    s = voicebank.run(ctx)
    assert s["refs"] > 0 and s["characters"] == 2  # "refs use durations only" used to leave every character without refs
    refs = gu.load_refs(ctx, "sakura")
    assert refs and all(r.snr is None and r.arousal is None and r.duration > 0 for r in refs)


def test_run_keeps_casting_edits_made_while_it_trains(vb, monkeypatch):
    from vndub.manifest.io import save_characters

    ctx, _ = vb
    ctx.force = True  # train again

    def slow_training(ctx_, cache, sid, clips, chars, char_f0, holder, reasons, progress=None):
        edited = load_characters(ctx.paths.characters)  # the user works in the casting screen meanwhile
        edited[sid].description = f"edited while {sid} trains"
        edited[sid].names = {**edited[sid].names, "tgt": "New name"}
        edited[sid].performer = {**edited[sid].performer, "tuning": {"pitch_st": 2.0}}
        save_characters(ctx.paths.characters, edited)
        chars[sid].rvc = {"model": f"voicebank/{sid}/rvc/trained.pth", "f0_transpose": 0}
        return "done"

    monkeypatch.setattr(voicebank, "_rvc_train_character", slow_training)
    voicebank.run(ctx)
    after = load_characters(ctx.paths.characters)
    for sid in ("sakura", "haruto"):
        assert after[sid].description == f"edited while {sid} trains" and after[sid].names["tgt"] == "New name"
        assert after[sid].performer["tuning"] == {"pitch_st": 2.0}  # the stage's own performer record kept the rest
        assert after[sid].performer["design_id"] and after[sid].rvc["model"].endswith("trained.pth")
        assert after[sid].refs  # what the run itself produced is saved too


def test_only_a_speaker_with_unvoiced_lines_does_not_run_the_voiced_ones(make_ctx, tmp_path):
    from tests.p3_helpers import add_unvoiced

    ctx = make_ctx(tmp_path / "game", sets={"synth.voice_unvoiced": True})
    build(ctx, n_per=3)
    add_unvoiced(ctx)
    ctx.only = {"hero"}
    s = voicebank.run(ctx)
    assert s["mode_b"] == ["hero"] and s["characters"] == 0 and s["refs"] == 0
    assert not ctx.paths.vb_refs_json("sakura").exists() and not ctx.paths.vb_refs_json("haruto").exists()
    assert ctx.paths.vb_performer("hero").exists()


def _without_design_backend(monkeypatch):
    from vndub.stages import genutil as gu

    real = gu.acquire_first

    def no_design_backend(ctx_, names, options=None, *, group=None, stage=None):
        if group == "performer design backend":
            return None, None, "the design backend is not installed"
        return real(ctx_, names, options, group=group, stage=stage)

    monkeypatch.setattr(gu, "acquire_first", no_design_backend)


def test_no_design_backend_leaves_variants_undecided(vb, monkeypatch):
    from vndub.manifest.io import write_yaml
    from vndub.stages import designs as dz

    ctx, _ = vb
    sid = "sakura"
    did = next(d["design_id"] for d in voicebank.list_designs(ctx, sid) if d["kind"] == "voice_design")
    voicebank.choose_design(ctx, sid, did, "B")
    # the chosen design has no emotion variants yet and the performer says so
    ds = dz.load_designs(ctx, sid)
    for d in ds:
        d.pop("variants", None)
    dz.save_designs(ctx, sid, ds)
    perf = read_yaml(ctx.paths.vb_performer(sid))
    perf["variants_done"] = False
    write_yaml(ctx.paths.vb_performer(sid), perf)

    with monkeypatch.context() as m:
        _without_design_backend(m)
        voicebank.run(ctx)
    assert "variants" not in dz.get_design(ctx, sid, did)  # used to be stored as [] and never tried again
    assert read_yaml(ctx.paths.vb_performer(sid))["variants_done"] is False
    voicebank.run(ctx)  # the backend is back: the variants are made now
    assert "variants" in dz.get_design(ctx, sid, did)
    assert read_yaml(ctx.paths.vb_performer(sid))["variants_done"] is True


def test_designs_json_does_not_keep_designs_whose_files_were_deleted(vb, monkeypatch):
    from vndub.stages import designs as dz

    ctx, _ = vb
    sid = "sakura"
    before = dz.load_designs(ctx, sid)
    assert before and all(d["origin"] == "stage" for d in before)
    ctx.paths.vb_performer(sid).unlink()  # the performer is lost: the stage designs are made again
    _without_design_backend(monkeypatch)
    voicebank.run(ctx)
    assert dz.load_designs(ctx, sid) == []  # their wavs were deleted above; the list used to keep them
    assert not any((ctx.paths.vb(sid) / "designs" / d["design_id"]).exists() for d in before)


def test_relaxed_refs_never_admit_music_or_extreme_lengths_and_say_why():
    lines = [_line(i) for i in range(6)]
    pps = {
        "l0": _pp(35, 4, "neutral"), "l1": _pp(33, 5, "happy"),
        "l2": _pp(40, 6, "sad", bgm=True),   # music under the voice
        "l3": _pp(30, 0.5, "angry"),         # a half-second "eh?"
        "l4": _pp(28, 30.0, "surprised"),    # a monologue
        "l5": _pp(9, 4, "fearful"),          # noisy but usable when few clips pass
    }
    chosen, relaxed = select_refs(lines, pps, min_snr_db=25, min_sec=3, max_sec=10, per_emotion=2, max_total=10, min_refs=5)
    ids = {c["line"].line_id for c in chosen}
    assert relaxed and ids == {"l0", "l1", "l5"}
    by = {c["line"].line_id: c for c in chosen}
    assert by["l0"]["relaxed_reasons"] == [] and any("snr" in r for r in by["l5"]["relaxed_reasons"])


def test_a_clip_without_a_measured_noise_floor_is_unknown_not_bad():
    lines = [_line(i) for i in range(5)]
    pps = {f"l{i}": {**_pp(30, 4, "neutral"), "snr_db": None} for i in range(5)}
    chosen, relaxed = select_refs(lines, pps, min_snr_db=25, min_sec=3, max_sec=10, per_emotion=5, max_total=10, min_refs=5)
    assert not relaxed and len(chosen) == 5
