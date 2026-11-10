"""Round 18: the voicebank cache key, one design for an uncast character without audio."""

from __future__ import annotations

import time

from tests.genproj import build
from tests.p3_helpers import add_unvoiced
from vndub.stages import designs as dz
from vndub.stages import genutil as gu
from vndub.stages import voicebank
from vndub.stages.common import run_stage

BASE = {"synth.voice_unvoiced": True, "voicebank.rvc.enabled": False}


def _stamps(ctx):
    return {s: [d["created"] for d in dz.load_designs(ctx, s)] for s in ("hero", "sakura")}


def test_a_score_tweak_does_not_rerender_but_a_new_embedder_does(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game", sets=BASE)
    build(ctx, n_per=3)
    add_unvoiced(ctx)
    run_stage("voicebank", ctx)
    a = _stamps(ctx)
    time.sleep(1.1)
    ctx2 = make_ctx(tmp_path / "game", sets={**BASE, "score.regen_cer_max": 0.3, "score.gates.cer_max": 0.2})
    run_stage("voicebank", ctx2)
    assert _stamps(ctx2) == a  # nothing of the voicebank reads those
    assert voicebank.voicebank_hash(ctx.cfg) == voicebank.voicebank_hash(ctx2.cfg)
    ctx3 = make_ctx(tmp_path / "game", sets={**BASE, "score.speaker": "other_embedder"})
    assert voicebank.voicebank_hash(ctx3.cfg) != voicebank.voicebank_hash(ctx.cfg)
    ctx4 = make_ctx(tmp_path / "game", sets={**BASE, "analyze.f0": "torchcrepe"})
    assert voicebank.voicebank_hash(ctx4.cfg) != voicebank.voicebank_hash(ctx.cfg)


def test_an_entry_stored_under_the_old_key_is_honoured(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game", sets=BASE)
    old = gu.stage_cache(ctx, voicebank.NAME, voicebank.VERSION, "voicebank", "score", "project")  # what 1.1.0 wrote with
    old.put("rvc:x", old.key("rvc", 1))
    old.save()
    new = voicebank._cache(ctx)
    assert new.fresh("rvc:x", new.key("rvc", 1))
    ctx2 = make_ctx(tmp_path / "game", sets={**BASE, "score.speaker": "other_embedder"})
    assert not voicebank._cache(ctx2).fresh("rvc:x", voicebank._cache(ctx2).key("rvc", 1))


def test_an_uncast_character_without_audio_gets_one_design(make_ctx, tmp_path):
    ctx = make_ctx(tmp_path / "game", sets={**BASE, "voicebank.performer.n_designs": 4})
    build(ctx, n_per=3)
    add_unvoiced(ctx)
    run_stage("voicebank", ctx)
    assert len(dz.load_designs(ctx, "hero")) == 1  # unranked without audio: the stage would pick the first anyway
    assert len(dz.load_designs(ctx, "sakura")) >= 2  # a character with audio is ranked: the full batch stays
    more = voicebank.propose_designs(ctx, "hero", n=2)  # "generate more" continues the indexes
    assert [d["design_id"] for d in more] and len(dz.load_designs(ctx, "hero")) == 3
