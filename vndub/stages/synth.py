"""synth: N candidates per voiced line x enabled backend x route.

Plan -> execute, grouped by model (one acquire per backend, all lines under it):

1. TTS for routes A and B per backend (route B writes ``<cand>.prevc.wav``),
2. route C per (backend, LoRA adapter),
3. voice conversion of route-B pre-VC wavs (RVC when the character has a
   trained model, else the zero-shot fallback; no VC available -> the
   candidate fails with that reason, it is never passed through unconverted),
4. the original audio as an ``orig`` candidate for interjections.

Route P (performer voice only, no VC) serves speaker lines without original audio
(``synth.voice_unvoiced``, characters without voice): the same performer TTS as route B, written
straight to the candidate path. Their style comes from the text passport (``source: "text"``, stage
``text_emotion``) or is neutral when there is none.

Per-character tuning (``characters.yaml performer.tuning``) and the per-line ``tuning`` override go through
``text/style_prompt.apply_tuning`` / ``tuning_knobs`` / ``tuning_plan``; the tuning is part of every
candidate's cache key, so changing it regenerates only that character's (or that line's) candidates, and
candidates of an older tuning that are no longer planned are removed.

Every candidate gets ``synth/<line_id>/<backend>_<route>_<k>.wav`` and a sidecar
``.json`` (:class:`SynthSidecar`); an item that fails on its own gets a sidecar with ``error`` and
never stops the stage. Regen rounds number candidates ``k = round*100 + i`` so
earlier candidates stay available to the scorer.

A backend that cannot be acquired (``BackendUnavailable`` incl. ``VramBudgetExceeded``: an orphan process holds the
VRAM, a thin margin, the env or the weights are gone) is not an item failure: the registry is told
(``record_unavailable("synth", "route R backend B", reason)``), no wav and no sidecar of its jobs is touched (a
candidate generated earlier stays), the other backends still run, and the stage ends with a ``StageError`` naming the
backend and the reason, so the job is not reported as done with lines that have no candidates. An adapter error that
says the GPU ran out of memory is treated the same way (``Job.keep``), whether it hit every item or a few.

Resume: a route-B candidate stores ``<cid>:tts`` in the stage cache when its pre-VC wav is complete; a rerun whose TTS
request is unchanged skips the TTS and converts the kept ``.prevc.wav``.

Non-speech sounds (``synth.sounds``, ``configs/sounds.yaml``, ``vndub/text/sounds.py``): "ZZZ...", "Muah!", "Achoo!", ``*yawns*``
are letters to a TTS. After the laughs, every sound of the text is written the way :func:`vndub.text.sounds.render_for_backend` says
for the backend (the target language's stand-in spelling, a backend event tag + the spelling, stage directions removed); a stand-in
never gets stress marks (the pass runs before the acute conversion, marks inside a sound are dropped first). The kind reaches the
style (``StyleSpec.sound`` + the kind as an event) and the sidecar records ``sound: {kind, n, mode, only, stage}``. A line that is
nothing but a sound follows ``synth.sounds.only``: ``spell`` (the stand-in, the default), ``sample`` (a recorded clip from
``synth.sounds.samples`` becomes the one candidate ``sample_P_<k>.wav``, no model; a kind without a clip is spelled, warned once) or
``silent`` (no candidates; pack lists the line as ``sound_silent``). The record's ``sound`` (written by translate) is read with
``getattr`` and the regexes are the fallback, so a project translated before the table ("З-З-З...", "Му-ах!") is handled too. Like
the laughter spelling, the spelled text is part of every candidate key and ``synth.sounds`` is not part of the section hash.

Laughter (``synth.laughter``, ``configs/laughter.yaml``, ``vndub/text/laughter.py``): a written laugh in the text a backend
gets ("Ха-ха-ха", "Haha", "あはは", a stage direction like ``*laughs*``) is read letter by letter by an LLM-based TTS, so
after the text variant is picked every laugh is written the way that backend voices as a laugh: the table spelling for the
target language (``respell``, the default, and the only lever of a cloner such as Qwen3-TTS Base, which ignores ``instruct``),
the backend own sound-event token (``inline``, CosyVoice ``[laughter]``) or a tag in front of the spelled laugh (``tag``, Fish
``[laughing]``, Higgs ``<|sfx:laughter|>``). Stage directions never reach the TTS as words. The laugh kind and strength also
reach the style (``StyleSpec.laughter``: specific instruct phrase, a laughter-only line is an instruction about the laugh) and
the sidecar records ``laughter: {kind, n, mode, only, stage}``. ``text_used`` is the text that was sent; ``text_plain`` (the CER
reference) is never respelled. A ``set_text`` override is sent as the user wrote it. The setting is not part of the cache hash
(the spelled text is part of every candidate key), so adding it re-generates only lines that contain a laugh.

Russian stress: a backend whose ``stress_format`` is ``plus`` or ``acute`` (caps; ``synth.backends.<name>.stress_format`` overrides)
gets the ``text_tts_plus`` variant, ``acute`` turned into U+0301 marks after the laughs are written. A project translated while no
such backend was enabled has no variant: :func:`_backfill_stress` makes it for this run from translate's stress cache.
"""

from __future__ import annotations

import re
import shutil
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vndub.backends.base import BackendSpec, BackendUnavailable
from vndub.manifest.io import LineRecord, load_overrides, write_json
from vndub.stages import genutil as gu
from vndub.stages.common import StageError
from vndub.text import laughter as lg
from vndub.text import sounds as snd
from vndub.text import ru
from vndub.text.style_prompt import (
    CALM_BELOW,
    StyleSpec,
    apply_tuning,
    build_style,
    laughter_decision,
    laughter_tokens,
    normalize_tuning,
    render,
    to_emotion_ref,
    tuning_key,
    tuning_knobs,
    tuning_plan,
)
from vndub.util import hash_obj

NAME = "synth"
VERSION = "1"
CONFIG_SECTIONS = ("synth", "project")
STRESSED = ("plus", "acute")  # the stress formats that speak the text_tts_plus variant


@dataclass
class Job:
    line_id: str
    speaker_id: str
    backend: str  # resolved backend name
    route: str
    k: int
    round: int
    seed: int
    text: str  # text sent to the backend (prefix + text)
    text_plain: str  # CER reference
    text_variant: str
    style: StyleSpec
    rendered: dict[str, Any]
    item: dict[str, Any]
    out: Path
    vc: dict[str, Any] | None = None  # route B: VC plan
    tuning: dict[str, Any] | None = None  # normalized tuning, None when default
    plan: dict[str, Any] | None = None  # tuning_plan(): what master / score still need to know
    options: dict[str, Any] = field(default_factory=dict)
    laughter: dict[str, Any] | None = None  # {kind, n, mode, only, stage}: how the laughs of the text were written for this backend
    sound: dict[str, Any] | None = None  # {kind, n, mode, only, stage}: how the non-speech sounds of the text were handled
    error: str = ""
    keep: bool = False  # the failure is about the machine (backend unavailable, out of memory): leave wav and sidecar alone
    elapsed: float = 0.0
    peak_vram: float | None = None
    cache_key: str = ""
    tts_key: str = ""  # route B: key of the TTS request alone (the VC plan is not part of it)

    @property
    def cid(self) -> str:
        return f"{self.line_id}|{self.backend}|{self.route}|{self.k}"

    @property
    def prevc(self) -> Path:
        return self.out.with_name(self.out.stem + ".prevc.wav")


def _is_short(text_tgt: str, src_dur: float) -> bool:
    return gu.count_syllables(text_tgt) <= 2 or (0 < src_dur < 1.0)


def _latest_override(overrides: dict[str, list[Any]], line_id: str, actions: tuple[str, ...]) -> Any | None:
    for o in reversed(overrides.get(line_id, [])):
        if o.action in actions:
            return o
    return None


def _prev_dur_ratio(ctx: Any, line_id: str) -> float | None:
    vals = [s.metrics.get("dur_ratio") for s in gu.load_scores(ctx, line_id) if s.route != "orig"]
    vals = [float(v) for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def run(ctx: Any) -> dict[str, Any]:
    cfg = ctx.cfg
    sc = cfg.synth
    tgt = cfg.project.tgt_lang
    lines = ctx.dub_lines()
    if not lines:
        raise StageError("no voiced lines to synthesize (run parse/extract/link first)")
    trs = gu.load_translations(ctx)
    if not trs:
        raise StageError(f"{ctx.paths.translation(tgt)} not found: run `vndub translate` first")
    overrides = load_overrides(ctx.paths.overrides)
    regen = bool(ctx.extra.get("regen"))
    if regen:
        sel = gu.load_selection(ctx)
        todo = {lid for lid, s in sel.items() if s.status == "regen"}
        if not todo:
            return {"lines": 0, "note": "no lines with status regen"}
        lines = [l for l in lines if l.line_id in todo]

    def line_round(line_id: str) -> int:
        """0 for a first pass; previous round + 1 in regen mode or when a review
        override (regen / set_text / set_ref) is newer than the line's candidates."""
        sides = [s for _, s in gu.candidate_files(ctx, line_id) if s is not None]
        prev = max([s.round for s in sides], default=-1)
        if regen:
            return max(1, prev + 1)
        newest = max([str((s.model_extra or {}).get("created", "")) for s in sides], default="")
        ov = _latest_override(overrides, line_id, ("regen", "set_text", "set_ref"))
        if ov is not None and sides and ov.ts > newest:
            return prev + 1
        return 0

    chars = ctx.characters()
    summary: dict[str, Any] = {"lines": len(lines), "candidates": 0, "cached": 0, "failed": 0}
    rounds: list[int] = []
    unavailable: dict[str, str] = {}
    skipped_routes: dict[str, str] = {}
    no_text: list[str] = []

    # ---- resolve backends (dedupe on the backend actually used) --------------
    backends: list[tuple[str, BackendSpec, dict[str, Any]]] = []
    requested: dict[str, str] = {}  # resolved name -> configured name (acquire records substitutions)
    seen: set[str] = set()
    for name in cfg.enabled_backends():
        try:
            spec, sub = ctx.models.resolve(name)
        except BackendUnavailable as e:
            unavailable[name] = e.reason
            ctx.log.warning("synth backend %s unavailable: %s", name, e)
            gu.record_unavailable(ctx, NAME, f"backend {name}", e.reason)
            continue
        except KeyError as e:
            unavailable[name] = str(e)
            gu.record_unavailable(ctx, NAME, f"backend {name}", str(e))
            continue
        if spec.name in seen:
            ctx.log.info("backend %s resolves to %s which is already scheduled; skipped", name, spec.name)
            continue
        seen.add(spec.name)
        backends.append((spec.name, spec, gu.backend_options(cfg, name)))
        requested[spec.name] = name
    if not backends:
        raise StageError(f"no usable TTS backend: {unavailable or 'none enabled in synth.backends'}")
    if tgt == "ru" and any(_stress_format(s, o) in STRESSED for _, s, o in backends):
        _backfill_stress(ctx, trs, lines)

    # ---- per-speaker voicebank info -------------------------------------------
    spk_info: dict[str, dict[str, Any]] = {}

    def info(sid: str) -> dict[str, Any]:
        if sid not in spk_info:
            refs = gu.ref_dicts(ctx, gu.load_refs(ctx, sid))
            perf = gu.load_performer(ctx, sid)
            prefs = []
            if perf:
                for r in perf.get("refs", []):
                    p = ctx.paths.abs(r["path"])
                    if p.exists():
                        prefs.append({**r, "path": str(p)})
            c = chars.get(sid)
            route_override = c.route if c is not None else None
            spk_info[sid] = {
                "tuning": normalize_tuning(((c.performer or {}).get("tuning")) if c is not None else None),
                "refs": refs,
                "performer": perf,
                "performer_refs": prefs,
                "rvc": gu.load_rvc_meta(ctx, sid),
                "lora": gu.load_lora_meta(ctx, sid),
                "stats": gu.char_stats(ctx, chars, sid),
                "gender": c.gender if c is not None else "unknown",
                "route": route_override,
            }
        return spk_info[sid]

    enabled_routes = [r for r, on in sc.routes.items() if on]
    cache = gu.stage_cache(ctx, NAME, VERSION, "synth", "project")  # the project folders are not part of a candidate's key
    # nor is synth.laughter (or synth.sounds, same reasoning): its effect (the spelled text, the laugh phrase of the style) is part of every candidate's key already,
    # and a key that moved with the section would regenerate every candidate of a project that has no laughter at all
    cache.rekey(_synth_hash(cfg))
    laughter_table = _laughter_table(ctx) if sc.laughter.enabled else None
    laughter_lines: set[str] = set()
    laughter_modes: Counter[str] = Counter()
    sounds_table = _sounds_table(ctx) if sc.sounds.enabled else None
    sound_lines: set[str] = set()
    sound_modes: Counter[str] = Counter()
    silent_sound_lines: list[str] = []
    sample_warned: set[str] = set()
    jobs: list[Job] = []
    orig_jobs: list[tuple[LineRecord, int]] = []
    interjection_lex = _interjections(ctx)

    for line in lines:
        tr = trs.get(line.line_id)
        if not _usable(tr):
            # translate_failed keeps the *source* text in text_tgt: never synthesize it
            no_text.append(line.line_id)
            continue
        sid = line.speaker_id
        si = info(sid)
        rnd = line_round(line.line_id)
        rounds.append(rnd)
        passport = gu.load_passport(ctx, line.line_id) or {}
        src_dur = gu.src_duration(ctx, line, passport)
        set_text = _latest_override(overrides, line.line_id, ("set_text",))
        set_ref = _latest_override(overrides, line.line_id, ("set_ref",))
        regen_ov = _latest_override(overrides, line.line_id, ("regen",))
        tuning_ov = _latest_override(overrides, line.line_id, ("tuning",))
        notes = tr.notes
        base_spec = _base_style(passport, si["stats"], notes, cfg)
        # the line's own tuning replaces the character's (it is a full tuning, not a delta)
        tuning = normalize_tuning(tuning_ov.payload) if tuning_ov is not None else si["tuning"]
        tkey = tuning_key(tuning)
        spec = apply_tuning(base_spec, tuning)
        style_override = ""
        for ov in (regen_ov, set_text):
            if ov is not None and ov.payload.get("style_prompt"):
                style_override = str(ov.payload["style_prompt"])

        # text variant
        interj = "interjection" in tr.flags or _is_interjection(line.text_src, interjection_lex, sc.interjection_max_syllables)
        variant = "interjection" if interj else "full"
        prev_ratio = _prev_dur_ratio(ctx, line.line_id) if rnd > 0 else None
        use_short = prev_ratio is not None and prev_ratio > sc.use_short_text_if_ratio_above and bool(tr.text_tts_short)
        if use_short:
            variant = "short"
        # voice sustain: the head line's voice file also covers its continuation lines
        cont_ids = (line.model_extra or {}).get("sustain_lines") or []
        cont = [trs[c] for c in cont_ids if _usable(trs.get(c))]
        if len(cont) < len(cont_ids):
            ctx.log.warning("%s: %d sustain continuation line(s) without a usable translation", line.line_id, len(cont_ids) - len(cont))

        user_text = set_text is not None and bool(set_text.payload.get("text"))
        lsound = None if (sounds_table is None or user_text) else line_sound(tr, tr.text_tts or tr.text_tgt, tgt)
        sound_only = bool(lsound and lsound["only"] and not cont)
        policy = sc.sounds.only if sound_only else "spell"
        if policy == "silent":
            silent_sound_lines.append(line.line_id)
            _purge_stale(ctx, line.line_id, tkey, set(), drop_all=True)
            continue
        sample_clip: Path | None = None
        if policy == "sample":
            sample_clip = _sample_clip(ctx, sc.sounds.samples, lsound["kind"], sample_warned)

        def texts(plus: bool) -> tuple[str, str]:
            if user_text:
                t = str(set_text.payload["text"])
                return t, gu.strip_stress(t)
            return join_texts([tr, *cont], use_short, plus)

        high = spec.intensity >= sc.high_arousal
        n = sc.n_candidates_short if (interj or _is_short(tr.text_tgt, src_dur) or high) else sc.n_candidates
        if si["route"]:
            routes = [si["route"]]
        elif not line.has_voice and not si["refs"]:
            routes = ["P"]  # no original voice to clone or convert to: the performer speaks as is
        else:
            routes = enabled_routes
        planned: set[Path] = set()
        if sample_clip is not None:
            sj = _sample_job(ctx, cache, line, sid, rnd, lsound, sample_clip, sc.sounds.samples[lsound["kind"]], tr, tgt, spec)
            planned.add(sj.out)
            jobs.append(sj)
            sound_lines.add(line.line_id)
            sound_modes[sj.sound["mode"]] += 1
            if interj and sc.keep_original_for_interjections and line.src_wav and not list(ctx.paths.synth_dir(line.line_id).glob("orig_orig_*.wav")):
                orig_jobs.append((line, rnd))
            _purge_stale(ctx, line.line_id, tkey, planned, drop_all=True)
            continue
        if interj and sc.keep_original_for_interjections and line.src_wav and not list(ctx.paths.synth_dir(line.line_id).glob("orig_orig_*.wav")):
            orig_jobs.append((line, rnd))

        for route in routes:
            if route in ("B", "P") and not si["performer_refs"]:
                skipped_routes[f"{sid}:{route}"] = "no performer (run voicebank, or choose a design)"
                continue
            if route == "A" and not (si["refs"] or line.src_wav):
                skipped_routes[f"{sid}:A"] = "no original audio to clone"
                continue
            if route == "C" and not si["lora"]:
                skipped_routes[f"{sid}:C"] = "no LoRA adapter (voicebank.route_c)"
                continue
            route_backends = backends
            if route == "C":
                lname = si["lora"].get("backend")
                route_backends = [(b, s, o) for b, s, o in backends if b == lname] or []
                if not route_backends:
                    try:
                        lspec, _ = ctx.models.resolve(lname)
                        route_backends = [(lspec.name, lspec, gu.backend_options(cfg, lname))]
                        requested.setdefault(lspec.name, lname)
                    except (BackendUnavailable, KeyError) as e:
                        skipped_routes[f"{sid}:C"] = f"LoRA backend {lname} unavailable: {e}"
                        gu.record_unavailable(ctx, NAME, f"route C backend {lname}", str(e))
                        continue
            for bname, bspec, bopts in route_backends:
                caps = bspec.caps or {}
                stress = _stress_format(bspec, bopts)
                text, plain = texts(stress in STRESSED)
                spec_b, laugh = spec, None
                if laughter_table is not None and not user_text:
                    text, spec_b, laugh = _laughter_for(lg.unstress_laughs(text), tgt, bname, caps, laughter_table, spec, tuning)
                sound = None
                if sounds_table is not None and not user_text:
                    text, plain, spec_b, sound = _sounds_for(_unstress_sounds(text), plain, tgt, bname, sounds_table, spec_b, lsound)
                if stress == "acute" and tgt == "ru" and not user_text:
                    text = ru.to_acute_format(text)
                rendered = render(spec_b, caps, bopts, cross_lingual=(route == "A"), laughter_inline=bool(laugh) and laugh["mode"].startswith(("inline", "tag")),
                                  sound_inline=bool(sound) and sound["mode"].startswith(("inline", "tag")))
                if sound:
                    sound_lines.add(line.line_id)
                    sound_modes[sound["mode"]] += 1
                if laugh:
                    rendered["laughter"] = laugh["mode"]
                    laughter_lines.add(line.line_id)
                    laughter_modes[laugh["mode"]] += 1
                plan = tuning_plan(tuning, caps, route) if tkey else None
                if style_override:
                    rendered["instruct"] = style_override if caps.get("instruct", "none") not in ("none",) else rendered["instruct"]
                    rendered["style_prompt"] = style_override
                text_used = (rendered["text_prefix"] + " " + text).strip() if rendered["text_prefix"] else text
                knobs = dict(rendered["knobs"] or {})
                if tkey:
                    knobs.update(tuning_knobs(tuning, caps, knobs))
                if bspec.env is None:  # local backends (mock / sapi) also take generic hints
                    gen = {"tempo": spec.tempo, "loudness": spec.loudness, "intensity": spec.intensity, "arc": spec.arc, "emotion": spec.emotion}
                    for k2, v2 in gen.items():
                        knobs.setdefault(k2, v2)
                timbre_ref = timbre_text = emotion_ref = None
                ref_paths: list[str] = []
                if route == "A":
                    pool = si["refs"]
                    if set_ref is not None and set_ref.payload.get("ref"):
                        timbre_ref = str(ctx.paths.abs(set_ref.payload["ref"]))
                    elif pool:
                        timbre_ref = to_emotion_ref(spec, pool, usage="timbre")
                    else:
                        timbre_ref = str(ctx.paths.abs(line.src_wav)) if line.src_wav else None
                    timbre_text = next((r.get("text") for r in pool if r["path"] == timbre_ref), None) or (line.text_src if timbre_ref and line.src_wav and timbre_ref == str(ctx.paths.abs(line.src_wav)) else None)
                    emotion_ref = to_emotion_ref(spec, pool) if pool else timbre_ref
                    if bspec.env is None:
                        knobs.setdefault("f0_hz", si["stats"].get("f0_median_hz"))
                        knobs.setdefault("gender", si["gender"])
                elif route in ("B", "P"):
                    prefs = si["performer_refs"]
                    timbre_ref = to_emotion_ref(spec, prefs) if prefs else None
                    timbre_text = next((r.get("text") for r in prefs if r["path"] == timbre_ref), None)
                    emotion_ref = timbre_ref
                    if bspec.env is None:
                        knobs.setdefault("f0_hz", (si["performer"] or {}).get("f0_median_hz"))
                        knobs.setdefault("gender", si["gender"])
                ref_paths = [p for p in dict.fromkeys([timbre_ref, emotion_ref]) if p]
                options = dict(bopts)
                if route == "C":
                    options["lora"] = str(ctx.paths.abs(si["lora"]["adapter"]))
                target_duration = round(src_dur * (plan["target_duration_scale"] if plan else 1.0), 3) if caps.get("duration") and src_dur > 0 else None
                vc0 = _vc_plan(ctx, si) if route == "B" else None
                if vc0 is not None and plan and plan["vc_f0_transpose"] and vc0.get("backend"):
                    vc0["f0_transpose"] = int(vc0.get("f0_transpose", 0)) + int(plan["vc_f0_transpose"])
                for i in range(n):
                    k = rnd * 100 + i
                    seed = sc.seed_base + gu.stable_seed(line.line_id, bname, route, k, rnd) % 1_000_000
                    out = ctx.paths.candidate(line.line_id, bname, route, k)
                    item = {
                        "id": f"{line.line_id}|{bname}|{route}|{k}",
                        "text": text_used,
                        "lang": tgt,
                        "timbre_ref": timbre_ref,
                        "timbre_ref_text": timbre_text,
                        "emotion_ref": emotion_ref,
                        "instruct": rendered["instruct"],
                        "knobs": knobs,
                        "seed": seed,
                        "target_duration": target_duration,
                        "out": str(out.with_name(out.stem + ".prevc.wav") if route == "B" else out),
                    }
                    vc = dict(vc0) if vc0 is not None else None
                    job = Job(
                        line_id=line.line_id, speaker_id=sid, backend=bname, route=route, k=k, round=rnd, seed=seed,
                        text=text_used, text_plain=plain, text_variant=variant if set_text is None else "override",
                        style=spec_b, rendered=rendered, item=item, out=out, vc=vc, options=options, tuning=tkey, plan=plan, laughter=laugh, sound=sound,
                    )
                    parts: list[Any] = [
                        text_used, spec_b.to_dict(), {k2: v2 for k2, v2 in rendered.items() if k2 != "laughter"},
                        [Path(p).name for p in ref_paths], options, seed, tgt, item["target_duration"], vc,
                    ]
                    if tkey:  # default tuning keeps the pre-tuning key
                        parts.append({"tuning": tkey, "plan": plan})
                    job.cache_key = cache.key(*parts, files=ref_paths)
                    if route == "B":
                        job.tts_key = cache.key("tts", {k2: v2 for k2, v2 in item.items() if k2 not in ("id", "out")}, bname, options, files=ref_paths)
                    planned.add(out)
                    jobs.append(job)
        _purge_stale(ctx, line.line_id, tkey, planned)

    # ---- execute ------------------------------------------------------------
    todo = [j for j in jobs if not cache.fresh(j.cid, j.cache_key, [j.out])]
    samples_todo = [j for j in todo if j.backend == "sample"]
    todo_ids = {j.cid for j in todo}
    summary["cached"] = len(jobs) - len(todo)
    # route B: the pre-VC audio of an interrupted run is complete (its ``:tts`` entry is written when the file was) and
    # still describes the same TTS request: the rerun goes straight to voice conversion
    tts_done = {j.cid for j in todo if j.route == "B" and cache.fresh(f"{j.cid}:tts", j.tts_key, [j.prevc])}
    summary["tts_reused"] = len(tts_done)
    # every unit of the stage, cached ones included (they count as done from the start; the total never shrinks to the
    # remaining work): a candidate is one unit, a route-B candidate that still has to be generated two (TTS, then VC)
    progress = _Progress(
        ctx, len(jobs) + sum(1 for j in jobs if j.route == "B" and j.cid in todo_ids),
        (len(jobs) - len(todo)) + len(tts_done),  # cached candidates, and the TTS unit of the kept pre-VC audio
    )
    blocked: dict[str, str] = {}  # "backend (routes)" -> why the stage cannot finish: raised once everything else is done

    def block(label: str, js: list[Job], reason: str) -> None:
        """A backend of a planned route cannot be used (VRAM taken, env or weights gone): loud, and nothing is deleted."""
        routes = sorted({j.route for j in js})
        for route in routes:
            gu.record_unavailable(ctx, NAME, f"route {route} backend {label}", reason)
        for j in js:
            if not j.error:
                j.error, j.keep = f"backend unavailable: {reason}", True
        blocked[f"{label} (route {', '.join(routes)})"] = reason
        unavailable[label] = reason
        ctx.log.error("synth: backend %s (route %s) cannot be used: %s", label, ", ".join(routes), reason)

    for j in samples_todo:  # recorded clips: no model, no VRAM
        _render_sample(j)
        _finish(ctx, cache, j)
        progress(j.line_id)
    groups: dict[tuple[str, str], list[Job]] = {}
    for j in todo:
        if j.backend == "sample":
            continue
        if j.cid not in tts_done:
            groups.setdefault((j.backend, j.options.get("lora", "")), []).append(j)
            if j.route == "B":
                cache.drop(f"{j.cid}:tts")  # the pre-VC wav is about to be rewritten: an old entry must not vouch for a half-written file
    for (bname, lora), js in groups.items():
        try:
            backend = ctx.models.acquire(requested.get(bname, bname), js[0].options or None)
        except BackendUnavailable as e:
            block(bname, js, e.reason or str(e))
            continue
        vram_failed: list[Job] = []
        for i in range(0, len(js), 16):
            part = js[i : i + 16]
            t0 = time.perf_counter()
            try:
                res = gu.run_batched(backend, "synthesize", [j.item for j in part], chunk=16)
            except BackendUnavailable as e:  # the adapter lost its backend mid-run
                block(bname, js[i:], e.reason or str(e))
                break
            el = (time.perf_counter() - t0) / max(1, len(part))
            peak = None
            try:
                peak = backend.peak_vram()
            except Exception:
                pass
            by = {r.get("id"): r for r in res}
            for j in part:
                r = by.get(j.item["id"], {})
                target = Path(j.item["out"])
                if r.get("error"):
                    j.error = str(r["error"])
                    if gu.is_vram_error(j.error):  # the machine, not the item: keep whatever is on disk
                        j.keep = True
                        vram_failed.append(j)
                elif not target.exists():
                    j.error = "backend returned no audio"
                j.elapsed, j.peak_vram = el, peak
                if j.route == "B":
                    if not j.error:
                        cache.put(f"{j.cid}:tts", j.tts_key)  # the pre-VC wav is complete: a kill during VC does not redo it
                else:
                    _finish(ctx, cache, j)
            # progress last: it is where the app's cancel is raised, so every take of the chunk is on disk and in the cache by then
            for j in part:
                progress(j.line_id)
        if vram_failed:
            reason = f"out of memory on {len(vram_failed)} of {len(js)} item(s): {vram_failed[0].error[:160]}"
            if len(vram_failed) == len(js):
                block(bname, vram_failed, reason)
            else:  # a long line that does not fit: loud, but the backend works
                for route in sorted({j.route for j in vram_failed}):
                    gu.record_unavailable(ctx, NAME, f"route {route} backend {bname}", reason)

    # ---- VC for route B -------------------------------------------------------
    vc_groups: dict[str, list[Job]] = {}
    for j in todo:
        if j.route != "B":
            continue
        if j.error:  # TTS failed or the backend was unavailable (``keep``: nothing is touched)
            _finish(ctx, cache, j)
            continue
        if not j.vc or not j.vc.get("backend"):
            j.error = (j.vc or {}).get("reason") or "no voice conversion available"
            j.keep = j.out.exists()  # an earlier converted candidate is better than none
            gu.record_unavailable(ctx, NAME, "route B voice conversion", j.error)
            _finish(ctx, cache, j)
            continue
        vc_groups.setdefault(j.vc.get("request") or j.vc["backend"], []).append(j)
    for vname, js in vc_groups.items():
        try:
            vb = ctx.models.acquire(vname)
        except BackendUnavailable as e:
            block(vname, js, e.reason or str(e))
            continue
        vram_failed = []
        for i in range(0, len(js), 16):  # chunks of 16: progress between them
            part = js[i : i + 16]
            items = []
            for j in part:
                v = j.vc
                items.append({
                    "id": j.cid, "src": str(j.prevc), "out": str(j.out), "target_ref": v.get("target_ref"),
                    "model": v.get("model"), "index": v.get("index"), "f0_transpose": v.get("f0_transpose", 0), "params": v.get("params", {}),
                })
            t0 = time.perf_counter()
            try:
                res = {r.get("id"): r for r in gu.run_batched(vb, "convert", items, chunk=16)}
            except BackendUnavailable as e:
                block(vname, js[i:], e.reason or str(e))
                break
            el = (time.perf_counter() - t0) / max(1, len(items))
            for j in part:
                r = res.get(j.cid, {})
                if r.get("error"):
                    j.error = f"vc: {r['error']}"
                    if gu.is_vram_error(j.error):
                        j.keep = True
                        vram_failed.append(j)
                elif not j.out.exists():
                    j.error = "vc returned no audio"
                j.elapsed += el
                _finish(ctx, cache, j)
            for j in part:  # (after the whole chunk is recorded: the cancel check lives in progress)
                progress(j.line_id)
        if vram_failed:
            reason = f"out of memory on {len(vram_failed)} of {len(js)} item(s): {vram_failed[0].error[:160]}"
            if len(vram_failed) == len(js):
                block(vname, vram_failed, reason)
            else:
                gu.record_unavailable(ctx, NAME, f"route B backend {vname}", reason)

    # ---- original audio candidates (interjections) ------------------------------
    for line, r in orig_jobs:
        out = ctx.paths.candidate(line.line_id, "orig", "orig", r * 100)
        if not out.exists():
            src = ctx.paths.abs(line.src_wav)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            _write_sidecar(ctx, out, {
                "line_id": line.line_id, "backend": "orig", "route": "orig", "k": r * 100, "seed": None,
                "text_used": line.text_src, "text_variant": "orig", "text_plain": "", "round": r,
                "ref_paths": [ctx.paths.rel(src)], "style_prompt": "", "created": gu.ts_now(),
            })
        summary.setdefault("orig_candidates", 0)
        summary["orig_candidates"] += 1

    failed = [j for j in todo if j.error]
    summary["candidates"] = len(jobs) - len(failed)
    summary["generated"] = len(todo) - len(failed)
    summary["failed"] = len(failed)
    if failed:
        errs: dict[str, int] = {}
        for j in failed:
            key = f"{j.backend}/{j.route}: {j.error[:80]}"
            errs[key] = errs.get(key, 0) + 1
        summary["errors"] = errs
    cache.save()
    if blocked:
        # Everything that could be generated is on disk and in the cache; the rest waits for the backend. The job must not end
        # `done` with lines that have no candidates (it used to, and the old wavs of those lines were deleted).
        why = "; ".join(f"{k}: {v[:200]}" for k, v in blocked.items())
        raise StageError(
            f"synth stopped: {why}. The candidates that were already generated are kept; free the GPU memory / fix the backend and run synth again "
            f"({len(failed)} candidate(s) are still missing)"
        )
    progress.finish()
    summary["backends"] = [b for b, _, _ in backends]
    summary["round"] = max(rounds, default=0)
    if laughter_lines:
        summary["laughter"] = {"lines": len(laughter_lines), "modes": dict(laughter_modes.most_common())}
    if sound_lines or silent_sound_lines:
        summary["sounds"] = {"lines": len(sound_lines), "modes": dict(sound_modes.most_common()), "silent": len(silent_sound_lines)}
        if silent_sound_lines:
            ctx.log.info("synth: %d line(s) are only a sound and are left silent (synth.sounds.only: silent)", len(silent_sound_lines))
    if unavailable:
        summary["unavailable"] = unavailable
    if skipped_routes:
        summary["skipped_routes"] = skipped_routes
    if no_text:
        summary["no_translation"] = len(no_text)
        ctx.log.warning("%d line(s) without a usable translation were skipped: %s", len(no_text), no_text[:20])
    return summary


class _Progress:
    """``ctx.progress`` over the units of the stage: a candidate is one unit, a route-B candidate that is generated in this run
    two (TTS, then VC). Units that are already done (cached candidates, TTS audio kept from an interrupted run) are part of
    ``total`` and count as done from the start: a fully cached run reports n / n at once, a resumed one never reports only
    the remaining work."""

    def __init__(self, ctx: Any, total: int, done: int = 0) -> None:
        self.ctx, self.total, self.done = ctx, max(0, int(total)), min(max(0, int(done)), max(0, int(total)))
        if self.total:
            ctx.progress(self.done, self.total, None)

    def __call__(self, item: str | None = None) -> None:
        self.done = min(self.total, self.done + 1)
        self.ctx.progress(self.done, self.total, item)

    def finish(self) -> None:
        """Units that were skipped (a failed TTS never reaches its VC unit) are done too."""
        if self.total and self.done < self.total:
            self.done = self.total
            self.ctx.progress(self.done, self.total, None)


def _synth_hash(cfg: Any) -> str:
    """``gu.stable_section_hash(cfg, "synth", "project")`` without ``synth.laughter`` and ``synth.sounds`` (see the module docstring)."""
    dump = cfg.model_dump(mode="json", by_alias=True)
    synth_d = dump.get("synth")
    if isinstance(synth_d, dict):
        synth_d = {k: v for k, v in synth_d.items() if k not in ("laughter", "sounds")}
    project_d = dump.get("project")
    if isinstance(project_d, dict):
        project_d = {k: v for k, v in project_d.items() if k not in gu.PROJECT_PATH_KEYS}
    return hash_obj({"synth": synth_d, "project": project_d})


def _laughter_table(ctx: Any) -> dict[str, Any] | None:
    """``configs/laughter.yaml`` (``synth.laughter.table``); ``None`` (laughs are left as written) when the file is missing."""
    path = ctx.cfg.resolve(ctx.cfg.synth.laughter.table)
    try:
        return lg.load_table(path)
    except FileNotFoundError:
        ctx.log.warning("laughter table %s not found: laughs are sent to the TTS as written", path)
        return None
    except ValueError as e:
        raise StageError(f"laughter table is invalid: {e}") from e


def _laughter_for(
    text: str, tgt: str, backend: str, caps: dict[str, Any], table: dict[str, Any], spec: StyleSpec, tuning: dict[str, Any] | None,
) -> tuple[str, StyleSpec, dict[str, Any] | None]:
    """The TTS text with its laughs written for ``backend`` and the StyleSpec that knows the laugh.

    Returns ``(text, spec, record)``; ``record`` = ``{kind, n, mode, only, stage}`` for the sidecar (``None``: no laugh in the
    text, ``text`` and ``spec`` are returned as they came). The laugh kind and length of the text set ``spec.laughter`` (the
    kind-specific instruct phrase, a laughter-only line) and add the ``laughter`` event when the passport had none, unless the
    tuning is "always calm".
    """
    new_text, rec = lg.apply_laughter(text, tgt, laughter_decision(caps), table, backend, laughter_tokens(caps))
    if rec is None:
        return text, spec, None
    info = lg.summarize(text) or {}
    spec_b = StyleSpec.from_dict(spec.to_dict())
    if normalize_tuning(tuning)["expressiveness"] > CALM_BELOW:
        spec_b.laughter = {"kind": info.get("kind"), "units": info.get("units", 0), "stage": bool(info.get("stage")), "only": bool(info.get("only"))}
        if "laughter" not in spec_b.events:
            spec_b.events = [*spec_b.events, "laughter"]
    return new_text, spec_b, rec


def _sounds_table(ctx: Any) -> dict[str, Any] | None:
    """``configs/sounds.yaml`` (``synth.sounds.table``); ``None`` (sounds are left as written) when the file is missing."""
    path = ctx.cfg.resolve(ctx.cfg.synth.sounds.table)
    try:
        return snd.load_table(path)
    except FileNotFoundError:
        ctx.log.warning("sounds table %s not found: non-speech sounds are sent to the TTS as written", path)
        return None
    except ValueError as e:
        raise StageError(f"sounds table is invalid: {e}") from e


def line_sound(tr: Any, text: str, lang: str = "auto") -> dict[str, Any] | None:
    """The non-speech sound of a line as ``{kind, n, only, stage}`` (``None`` = the line has none).

    The regexes read the text (a project translated before the sounds table still carries "З-З-З..." / "Му-ах!", which they
    find); when they find nothing but the translation record's ``sound`` (written by translate from the LLM's answer) says the
    line is only a sound, that names the kind. Shared by synth, score and pack, so all three agree on which lines are sound-only.
    """
    found = snd.find_sounds(text or "", lang)
    if found:
        s = snd.summarize(text, lang, found) or {}
        return {"kind": s.get("kind"), "n": s.get("n", len(found)), "only": bool(s.get("only")), "stage": bool(s.get("stage"))}
    hint = getattr(tr, "sound", None)
    if isinstance(hint, dict) and hint.get("only") and hint.get("kind") in snd.KINDS:
        return {"kind": hint["kind"], "n": 1, "only": True, "stage": bool(hint.get("stage"))}
    return None


def silent_sound_line(cfg: Any, tr: Any, line: Any = None) -> bool:
    """Does synth leave this line without candidates on purpose (``synth.sounds.only: silent`` and the line is only a sound)?

    Score and pack ask the same question synth answered: such a line is neither a failure nor a review item.
    """
    so = cfg.synth.sounds
    if not so.enabled or so.only != "silent" or tr is None or not _usable(tr):
        return False
    if line is not None and (line.model_extra or {}).get("sustain_lines"):
        return False
    s = line_sound(tr, tr.text_tts or tr.text_tgt)
    return bool(s and s["only"])


def _unstress_sounds(text: str) -> str:
    """``text`` without the stress marks inside its sounds (the stresser marks "Хр-р-р" like a word and the patterns miss it then)."""
    t = text or ""
    marks = [m.start() for m in lg._PLUS_MARK.finditer(t)]
    if not marks:
        return t
    found = snd.find_sounds(lg._PLUS_MARK.sub("", t))
    drop = {p for i, p in enumerate(marks) if any(s.start <= p - i < s.end or (t[p] == "́" and p - i == s.end) for s in found)}
    return "".join(ch for i, ch in enumerate(t) if i not in drop)


# one written sound per kind (the pattern of vndub.text.sounds finds exactly this one): rendering it gives the stand-in once
_PROBE = {
    "snore": "zzz", "kiss": "muah", "yawn": "yawn", "gasp": "gasp", "sneeze": "achoo", "cough": "ahem", "hum": "mmm", "hmph": "hmph",
    "shiver": "brrr", "eat": "nom nom",
}


def _sounds_for(
    text: str, plain: str, tgt: str, backend: str, table: dict[str, Any], spec: StyleSpec, line: dict[str, Any] | None,
) -> tuple[str, str, StyleSpec, dict[str, Any] | None]:
    """The TTS text with its non-speech sounds written for ``backend``, the CER reference to match and the StyleSpec that knows them.

    Returns ``(text, plain, spec, record)``; ``record`` = ``{kind, n, mode, only, stage}`` (``None``: no sound, everything comes
    back as it went in). A line that is only a sound (``line``: regexes, or the translation record's ``sound`` when they find
    nothing) becomes the target language's spelling of its kind as a whole.
    """
    if line and line.get("only") and line.get("kind") in _PROBE:
        # the line is only a sound: its text is the stand-in, built from ONE sound of that kind so that a spelling the patterns
        # match twice ("Хр-р-р... хр-р-р...") is not rendered again on top of itself
        new_text, rec = snd.render_for_backend(_PROBE[line["kind"]], tgt, backend, table)
        if rec is None:
            return text, plain, spec, None
        rec = {**rec, "kind": line["kind"], "only": True, "stage": bool(line.get("stage"))}
        new_plain = snd.render_for_backend(_PROBE[line["kind"]], tgt, "", table)[0]
    else:
        found = snd.find_sounds(text, "auto")
        if not found:
            return text, plain, spec, None
        new_text, rec = snd.render_for_backend(text, tgt, backend, table, found)
        new_plain = snd.render_for_backend(plain, tgt, "", table)[0] if snd.has_sound(plain) else plain
    if rec is None:
        return text, plain, spec, None
    spec_b = StyleSpec.from_dict(spec.to_dict())
    spec_b.sound = {"kind": rec["kind"], "n": rec.get("n", 1), "only": bool(rec.get("only")), "stage": bool(rec.get("stage"))}
    if rec["kind"] not in spec_b.events:
        spec_b.events = [*spec_b.events, rec["kind"]]
    return new_text, new_plain, spec_b, rec


def _sample_clip(ctx: Any, samples: dict[str, str], kind: str, warned: set[str]) -> Path | None:
    """The recorded clip for ``kind`` (``synth.sounds.samples``), or ``None`` (the line is spelled instead; warned once per kind)."""
    raw = samples.get(kind)
    if not raw:
        reason = f"no clip configured for {kind!r} in synth.sounds.samples"
    else:
        p = ctx.cfg.resolve(raw)
        if Path(p).is_file():
            return Path(p)
        reason = f"clip {p} not found"
    if kind not in warned:
        warned.add(kind)
        ctx.log.warning("synth.sounds.only is sample, but %s: lines that are only a %s are spelled instead", reason, kind)
    return None


def _sample_job(
    ctx: Any, cache: Any, line: LineRecord, sid: str, rnd: int, lsound: dict[str, Any], clip: Path, configured: str, tr: Any, tgt: str,
    spec: StyleSpec,
) -> Job:
    """The single candidate of a sound-only line whose sound is a recorded clip (backend ``sample``, route ``P``): no model runs."""
    k = rnd * 100
    out = ctx.paths.candidate(line.line_id, "sample", "P", k)
    spelled = (tr.text_tts or tr.text_tgt or "").strip()
    rec = {"kind": lsound["kind"], "n": lsound.get("n", 1), "mode": f"sample:{configured}", "only": True, "stage": bool(lsound.get("stage"))}
    spec_s = StyleSpec.from_dict(spec.to_dict())
    spec_s.sound = {k2: rec[k2] for k2 in ("kind", "n", "only", "stage")}
    item = {"id": f"{line.line_id}|sample|P|{k}", "text": spelled, "lang": tgt, "out": str(out), "clip": str(clip)}
    job = Job(
        line_id=line.line_id, speaker_id=sid, backend="sample", route="P", k=k, round=rnd, seed=0, text=spelled, text_plain="",
        text_variant="sound", style=spec_s, rendered={"instruct": None, "text_prefix": "", "knobs": {}, "style_prompt": f"recorded sample: {configured}"},
        item=item, out=out, sound=rec,
    )
    job.cache_key = cache.key("sample", lsound["kind"], clip.name, tgt, files=[str(clip)])
    return job


def _render_sample(j: Job) -> None:
    """The clip of ``j.item["clip"]`` as the candidate wav: mono, at the clip's own sample rate (master resamples)."""
    from vndub.audio import features

    t0 = time.perf_counter()
    try:
        y, sr = features.load(j.item["clip"], mono=True)
        if not len(y):
            raise ValueError("the clip is empty")
        features.save(j.out, y, sr)
    except Exception as e:
        j.error = f"sample clip {j.item['clip']}: {e}"
    j.elapsed = time.perf_counter() - t0


def _base_style(passport: dict[str, Any], stats: dict[str, Any], notes: str, cfg: Any) -> StyleSpec:
    """The line's StyleSpec before tuning: built from an audio passport; taken as is from a text passport
    (``source: "text"``, written by ``text_emotion``); neutral when there is no passport."""
    if passport.get("source") == "text":
        st = passport.get("style")
        spec = StyleSpec.from_dict(st) if isinstance(st, dict) and st else StyleSpec()
        joined = "; ".join(x for x in (spec.notes, notes) if x)
        spec.notes = joined
        return spec
    return build_style(passport, stats, notes, cfg.analyze.e2v_weight, cfg.analyze.anime_weight, cfg.analyze.caption)


def _purge_stale(ctx: Any, line_id: str, tkey: dict[str, Any] | None, planned: set[Path], drop_all: bool = False) -> None:
    """Delete candidates rendered under another tuning that this run does not regenerate in place
    (e.g. extra regen-round candidates): they would compete with the tuned ones in score.

    A recorded sample (``sample_*``) that this run does not plan is stale too (the line is spoken again), and with ``drop_all``
    (a sound line that is left silent, or replaced by a sample) so is every synthesized take."""
    for wav, side in gu.candidate_files(ctx, line_id):
        if side is None or side.route == "orig" or wav in planned:
            continue
        old = ((side.params or {}).get("tuning") or {}).get("values")
        if old == tkey and not drop_all and side.backend != "sample":
            continue
        for p in (wav, wav.with_suffix(".json"), wav.with_name(wav.stem + ".prevc.wav")):
            p.unlink(missing_ok=True)


def _usable(tr: Any) -> bool:
    """A translation synth may speak: present, not failed/missing, non-empty text_tts."""
    return tr is not None and not ({"translate_failed", "translate_missing"} & set(tr.flags)) and bool((tr.text_tts or "").strip())


def _stress_format(spec: BackendSpec, options: dict[str, Any]) -> str:
    return str(options.get("stress_format") or (spec.caps or {}).get("stress_format") or "plain")


def _backfill_stress(ctx: Any, trs: dict[str, Any], lines: list[LineRecord]) -> None:
    """The ``+`` variants translate writes when a backend needs them, for the lines translated while none did.

    They stay in memory: the translation file is translate's output. translate stressed these texts for its plain variant too, so
    its cache normally has them all and no model loads. Without a stresser the lines keep their plain text, with a warning.
    """
    from vndub.stages.translate import make_stresser

    ids = dict.fromkeys(i for l in lines for i in (l.line_id, *((l.model_extra or {}).get("sustain_lines") or [])))
    recs = [trs[i] for i in ids if _usable(trs.get(i)) and "text_tts_plus" not in (trs[i].model_extra or {})]
    if not recs:
        return
    stresser, _, reason = make_stresser(ctx, [ru.normalize_for_tts(x) for r in recs for x in (r.text_tgt, r.text_tgt_short) if x])
    if stresser is None:
        ctx.log.warning("%d line(s) stay without stress marks: %s", len(recs), reason)
        return
    yo = ctx.cfg.translate.stress.yo
    for r in recs:
        r.text_tts_plus = ru.prepare_tts_text(r.text_tgt, stresser, "plus", yo)[0]
        r.text_tts_short_plus = ru.prepare_tts_text(r.text_tgt_short or r.text_tgt, stresser, "plus", yo)[0]


def join_texts(recs: list[Any], short: bool, plus: bool) -> tuple[str, str]:
    """(TTS text, CER reference) for a head line + its sustain continuations, in line order.

    short -> text_tts_short / text_plain_short; plus -> the '+' stress variants when present.
    """
    parts, plains = [], []
    for rec in recs:
        ex = rec.model_extra or {}
        if short:
            body = (ex.get("text_tts_short_plus") if plus else None) or rec.text_tts_short or rec.text_tts
            plain = ex.get("text_plain_short") or gu.strip_stress(rec.text_tgt_short or body)
        else:
            body = (ex.get("text_tts_plus") if plus else None) or rec.text_tts
            plain = rec.text_plain or gu.strip_stress(rec.text_tgt)
        parts.append(body.strip())
        plains.append(plain.strip())
    return " ".join(p for p in parts if p), " ".join(p for p in plains if p)


def _vc_plan(ctx: Any, si: dict[str, Any]) -> dict[str, Any]:
    vc = ctx.cfg.synth.vc
    perf_f0 = (si["performer"] or {}).get("f0_median_hz")
    char_f0 = si["stats"].get("f0_median_hz")
    f0_t = int(round(12 * np.log2(char_f0 / perf_f0))) if char_f0 and perf_f0 else 0
    reasons = []
    if si["rvc"] and vc.primary:
        try:
            spec, _ = ctx.models.resolve(vc.primary)
            meta = si["rvc"]
            return {
                "backend": spec.name, "request": vc.primary, "model": str(ctx.paths.abs(meta["model"])),
                "index": str(ctx.paths.abs(meta["index"])) if meta.get("index") else None,
                "f0_transpose": int(meta.get("f0_transpose", f0_t)), "params": dict(vc.rvc), "target_ref": None,
            }
        except (BackendUnavailable, KeyError) as e:
            reasons.append(f"{vc.primary}: {e}")
    elif not si["rvc"]:
        reasons.append("no RVC model for this character")
    if vc.fallback:
        try:
            spec, _ = ctx.models.resolve(vc.fallback)
            timbre = [r for r in si["refs"] if r.get("usage") in ("timbre", "both")] or si["refs"]
            target = max(timbre, key=lambda r: float(r.get("snr") or 0))["path"] if timbre else None
            if target is None:
                reasons.append("no character reference clip for zero-shot VC")
            else:
                return {"backend": spec.name, "request": vc.fallback, "target_ref": target, "f0_transpose": f0_t, "params": dict(vc.seedvc), "model": None, "index": None}
        except (BackendUnavailable, KeyError) as e:
            reasons.append(f"{vc.fallback}: {e}")
    return {"backend": None, "reason": "no voice conversion available (" + "; ".join(reasons) + ")"}


def _interjections(ctx: Any) -> dict[str, str]:
    try:
        from vndub.text.ja import load_interjections

        return load_interjections(ctx.cfg.resolve(ctx.cfg.translate.interjections))
    except Exception:
        return {}


_JA_PUNCT = re.compile(r"[\s、。，．・！？!?…‥ー〜~「」『』（）()\[\]【】]")


def _is_interjection(text: str, lex: dict[str, str], max_morae: int) -> bool:
    """fork C's text.ja.is_interjection when present, else lexicon / length heuristic."""
    try:
        from vndub.text.ja import is_interjection

        return bool(is_interjection(text, lex, max_morae))
    except Exception:
        pass
    norm = _JA_PUNCT.sub("", unicodedata.normalize("NFKC", text or ""))
    return bool(norm) and (norm in lex or len(norm) <= max_morae)


def _write_sidecar(ctx: Any, wav: Path, data: dict[str, Any]) -> None:
    data = dict(data)
    data["wav"] = ctx.paths.rel(wav)
    write_json(wav.with_suffix(".json"), data)


def _finish(ctx: Any, cache: Any, j: Job) -> None:
    if j.error and j.keep:
        # the machine failed, not the item: the wav and the sidecar of an earlier run stay as they are (and the cache entry
        # is not updated, so the next run tries again)
        ctx.log.warning("%s kept as is: %s", j.cid, j.error[:200])
        return
    data = {
        "line_id": j.line_id, "backend": j.backend, "route": j.route, "k": j.k, "seed": j.seed,
        "params": {
            "options": j.options, "knobs": j.item.get("knobs"), "instruct": j.item.get("instruct"), "target_duration": j.item.get("target_duration"), "vc": j.vc,
            **({"tuning": {"values": j.tuning, "plan": j.plan}} if j.tuning else {}),
        },
        "style_prompt": j.rendered.get("style_prompt", ""), "style": j.style.to_dict(),
        "ref_paths": [ctx.paths.rel(p) for p in [j.item.get("timbre_ref"), j.item.get("emotion_ref")] if p],
        "text_used": j.text, "text_plain": j.text_plain, "text_variant": j.text_variant,
        "elapsed_s": round(j.elapsed, 3), "peak_vram_gb": j.peak_vram, "round": j.round, "created": gu.ts_now(),
    }
    if j.route == "B" and j.prevc.exists():
        data["pre_vc_wav"] = ctx.paths.rel(j.prevc)
    if j.laughter:
        data["laughter"] = j.laughter
    if j.sound:
        data["sound"] = j.sound
    if j.error:
        # this run tried to generate this very item and it failed on its own: a wav of an earlier request would be stale
        data["error"] = j.error
        if j.out.exists():
            j.out.unlink()
        write_json(j.out.with_suffix(".json"), {**data, "wav": ""})
        return
    _write_sidecar(ctx, j.out, data)
    cache.put(j.cid, j.cache_key)
