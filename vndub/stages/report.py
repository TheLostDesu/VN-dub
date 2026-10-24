"""report: self-contained HTML + Markdown overview of a project's state.

Every section tolerates missing inputs (a stage not run yet) and says so
instead of failing. Output: ``work/report/index.html`` and ``work/report/report.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from vndub.manifest.io import (
    AudioIndexRecord,
    CandidateScore,
    FinalIndexRecord,
    LinkRecord,
    SelectionRecord,
    iter_jsonl,
    load_characters,
    load_lines,
    read_json,
    read_jsonl,
)
from vndub.stages import genutil as gu
from vndub.util import now_iso

NAME = "report"
VERSION = "1"
CONFIG_SECTIONS = ("report",)

#: sections `collect` goes through (progress units besides the per-file ones)
SECTIONS = 7

TEMPLATES = Path(__file__).resolve().parent.parent / "report" / "templates"
METRICS = ["composite", "utmos", "pq", "spk_sim", "emo_sim", "adv_delta", "cer", "dur_ratio", "prosody", "f0_corr", "energy_corr", "pause_match"]


def _stats(vals: list[float]) -> dict[str, Any]:
    v = [float(x) for x in vals if x is not None]
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": len(v), "min": round(float(a.min()), 3), "median": round(float(np.median(a)), 3), "max": round(float(a.max()), 3), "mean": round(float(a.mean()), 3)}


def _rel(report_dir: Path, target: Path) -> str:
    try:
        return Path(os.path.relpath(target, report_dir)).as_posix()
    except ValueError:  # different drive on Windows
        return target.as_uri()


def collect(ctx: Any) -> dict[str, Any]:
    p = ctx.paths
    rd = p.report_dir
    data: dict[str, Any] = {"generated": now_iso(), "work": str(p.work), "out": str(p.out), "profile": ctx.cfg.profile}
    proj = ctx.project
    data["project"] = proj.model_dump(mode="json") if proj else None

    # progress: one tick per section and per score / passport file read (nothing is written until render)
    score_files = sorted(p.scores_dir.glob("*.jsonl")) if p.scores_dir.exists() else []
    passport_files = list(p.analysis.glob("*.json"))[:5000] if p.analysis.exists() else []
    total = SECTIONS + len(score_files) + len(passport_files)
    done = 0

    def tick(item: str | None = None, section: bool = True) -> None:
        nonlocal done
        done += 1
        if section or done % 25 == 0 or done >= total:  # per-file ticks are throttled
            ctx.progress(min(done, total), total, item)

    # lines / overview (source text with mode C recoveries and user edits applied)
    lines = ctx.lines() if p.lines.exists() else []
    trs = gu.load_translations(ctx)
    sel = {s.line_id: s for s in read_jsonl(p.selection, SelectionRecord)} if p.selection.exists() else {}
    finals = read_jsonl(p.final_index, FinalIndexRecord) if p.final_index.exists() else []
    n_pass = len(list(p.analysis.glob("*.json"))) if p.analysis.exists() else 0
    n_cand = sum(1 for _ in p.synth.rglob("*.wav") if not _.name.endswith(".prevc.wav")) if p.synth.exists() else 0
    manifest_patch = None
    if proj:
        for cand in p.out.glob("*/manifest_patch.json"):
            manifest_patch = read_json(cand)
    data["overview"] = {
        "lines": len(lines),
        "voiced": sum(1 for l in lines if l.has_voice),
        "speakers": len({l.speaker_id for l in lines if l.has_voice}),
        "translated": len(trs),
        "passports": n_pass,
        "candidates": n_cand,
        "selected": sum(1 for s in sel.values() if s.winner),
        "finals": len(finals),
        "packed": len(manifest_patch) if isinstance(manifest_patch, list) else (len(manifest_patch.get("files", [])) if isinstance(manifest_patch, dict) else None),
    }
    tick("overview")

    # mapping coverage
    if p.link.exists():
        links = read_jsonl(p.link, LinkRecord)
        cnt: dict[str, int] = {}
        for r in links:
            cnt[r.status] = cnt.get(r.status, 0) + 1
        data["link"] = {
            "counts": cnt,
            "total": len(links),
            "ok_pct": round(100.0 * cnt.get("ok", 0) / max(1, len(links)), 1),
            "relinked": sum(1 for r in links if r.relinked_from),
            "match": _stats([r.match_score for r in links]),
            "consistency": _stats([r.spk_consistency for r in links if r.spk_consistency is not None]),
        }
    else:
        data["link"] = None
    tick("link")

    # characters
    chars = load_characters(p.characters) if p.characters.exists() else {}
    rows = []
    for sid, c in chars.items():
        hist = c.stats.emotion_hist or {}
        tot = sum(hist.values()) or 1.0
        rows.append(
            {
                "id": sid,
                "name": c.names.get("src", sid),
                "tgt": c.names.get("tgt", ""),
                "gender": c.gender,
                "n_lines": c.stats.n_lines,
                "total_min": round((c.stats.total_sec or 0.0) / 60.0, 2),
                "f0": c.stats.f0_median_hz,
                "lufs": c.stats.lufs_median,
                "refs": len(c.refs),
                "performer": (c.performer or {}).get("backend") or "",
                "rvc": "yes" if (c.rvc or {}).get("model") else "",
                "lora": "yes" if (c.lora or {}).get("adapter") else "",
                "route": c.route or "",
                "hist": [{"label": k, "pct": round(100.0 * v / tot, 1)} for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:8]],
            }
        )
    data["characters"] = rows
    tick("characters")

    # scores per backend/route
    allscores: list[CandidateScore] = []
    for f in score_files:
        allscores += read_jsonl(f, CandidateScore)
        tick(f.stem, section=False)
    groups: dict[str, list[CandidateScore]] = {}
    for c in allscores:
        groups.setdefault(f"{c.backend}/{c.route}", []).append(c)
    score_rows = []
    for g, cs in sorted(groups.items()):
        gate_names = sorted({k for c in cs for k in c.gates})
        score_rows.append(
            {
                "group": g,
                "n": len(cs),
                "pass_pct": round(100.0 * sum(1 for c in cs if c.passed) / max(1, len(cs)), 1),
                "gates": {k: round(100.0 * sum(1 for c in cs if c.gates.get(k)) / max(1, sum(1 for c in cs if k in c.gates)), 1) for k in gate_names},
                "metrics": {m: _stats([c.composite if m == "composite" else c.metrics.get(m) for c in cs]) for m in METRICS},
                "wins": sum(1 for s in sel.values() if s.winner and f"{s.backend}/{s.route}" == g),
            }
        )
    data["scores"] = score_rows
    data["metrics"] = METRICS
    tick("scores")

    # route comparison
    routes = []
    for r in ("A", "B", "C", "P", "orig"):
        cs = [c for c in allscores if c.route == r]
        if not cs:
            continue
        routes.append(
            {
                "route": r,
                "n": len(cs),
                "composite": _stats([c.composite for c in cs]).get("mean"),
                "utmos": _stats([c.metrics.get("utmos") for c in cs]).get("mean"),
                "spk_sim": _stats([c.metrics.get("spk_sim") for c in cs]).get("mean"),
                "emo_sim": _stats([c.metrics.get("emo_sim") for c in cs]).get("mean"),
                "cer": _stats([c.metrics.get("cer") for c in cs]).get("mean"),
                "dur_ratio": _stats([c.metrics.get("dur_ratio") for c in cs]).get("mean"),
                "pass_pct": round(100.0 * sum(1 for c in cs if c.passed) / len(cs), 1),
                "wins": sum(1 for s in sel.values() if s.route == r),
            }
        )
    data["routes"] = routes if ctx.cfg.report.compare_routes else []
    tick("routes")

    # selection status + review queue
    st: dict[str, int] = {}
    for s in sel.values():
        st[s.status] = st.get(s.status, 0) + 1
    data["selection"] = st if sel else None
    queue = []
    by_line = {l.line_id: l for l in lines}
    if p.review_queue.exists():
        for r in iter_jsonl(p.review_queue):
            queue.append({"line_id": r.get("line_id"), "source": "link", "status": r.get("status", "review"), "reason": "; ".join(r.get("reasons", []) or []) or r.get("reason", "")})
    for s in sel.values():
        if s.status in ("review", "regen"):
            queue.append({"line_id": s.line_id, "source": "score", "status": s.status, "reason": s.reason})
    # what the patch holds: how many lines are provisional / left out, and which review lines ended up where
    in_final = {f.line_id: f for f in finals if p.abs(f.encoded).exists()}
    data["coverage"] = None
    if p.final_index.exists():
        from vndub.stages.pack import patch_coverage

        data["coverage"] = patch_coverage(ctx, list(in_final.values()), sel)
        data["coverage"]["left_out_lines"] = data["coverage"]["left_out_lines"][:100]
        data["coverage"]["left_out_reasons_text"] = ", ".join(f"{k} {v}" for k, v in sorted(data["coverage"]["left_out_reasons"].items()))
    for q in queue:
        l = by_line.get(q["line_id"])
        q["speaker"] = l.speaker_id if l else ""
        q["text"] = (l.text_src if l else "")[:60]
        q["patch"] = ""  # where a review line of the quality step ended up: installed as provisional, or left out of the patch
        if q["source"] == "score" and p.final_index.exists():
            q["patch"] = "provisional" if q["line_id"] in in_final else "left out"
    data["queue"] = queue[:300]
    data["queue_total"] = len(queue)
    tick("queue")

    # substitutions, timings, VRAM
    events = list(iter_jsonl(p.work / "gpu_events.jsonl")) if (p.work / "gpu_events.jsonl").exists() else []
    subs: dict[tuple[str, str], int] = {}
    for e in events:
        if e.get("event") == "load" and e.get("substituted_for"):
            subs[(e["substituted_for"], e["model"])] = subs.get((e["substituted_for"], e["model"]), 0) + 1
    missing: dict[str, int] = {}
    passport_backends: dict[str, int] = {}
    for f in passport_files:
        d = read_json(f) or {}
        for part, reason in (d.get("missing") or {}).items():
            key = f"{part}: {str(reason)[:80]}"
            missing[key] = missing.get(key, 0) + 1
        for part, b in (d.get("backends") or {}).items():
            passport_backends[f"{part}={b}"] = passport_backends.get(f"{part}={b}", 0) + 1
        tick(f.stem, section=False)
    data["substitutions"] = [{"requested": a, "used": b, "loads": n} for (a, b), n in sorted(subs.items())]
    data["analysis_missing"] = sorted(missing.items(), key=lambda kv: -kv[1])
    data["analysis_backends"] = sorted(passport_backends.items(), key=lambda kv: -kv[1])
    unavailable: dict[tuple[str, str, str], int] = {}  # a stage that went on without a part (VRAM, env, weights)
    for e in events:
        if e.get("event") == "unavailable":
            key = (str(e.get("stage", "")), str(e.get("group", "")), str(e.get("reason", ""))[:120])
            unavailable[key] = unavailable.get(key, 0) + 1
    data["unavailable"] = [{"stage": s, "group": g, "reason": r, "count": n} for (s, g, r), n in sorted(unavailable.items())]
    vram: dict[str, dict[str, Any]] = {}
    for e in events:
        if e.get("event") == "unavailable":  # no model behind it: not a row of the VRAM table
            continue
        m = vram.setdefault(e.get("model", "?"), {"loads": 0, "load_s": 0.0, "peak_vram_gb": None, "role": e.get("role", ""), "estimate": e.get("vram_estimate_gb"), "device": ""})
        if e.get("event") == "load":
            m["loads"] += 1
            m["load_s"] = round(m["load_s"] + float(e.get("load_s") or 0), 2)
            m["device"] = e.get("device", m["device"])
        elif e.get("event") == "unload" and e.get("peak_vram_gb") is not None:
            m["peak_vram_gb"] = max(float(e["peak_vram_gb"]), m["peak_vram_gb"] or 0.0)
    data["vram"] = [{"model": k, **v} for k, v in sorted(vram.items())]
    status = read_json(p.stage_status) or {}
    data["stages"] = [
        {"stage": k, "elapsed_s": v.get("elapsed_s"), "finished": v.get("finished"), "error": v.get("error"), "summary": v.get("summary")}
        for k, v in status.items()
    ]

    # audio samples
    samples = []
    audio_idx = {a.id: a for a in read_jsonl(p.audio_index, AudioIndexRecord)} if p.audio_index.exists() else {}
    for s in list(sel.values())[:12]:
        l = by_line.get(s.line_id)
        if l is None:
            continue
        orig = p.abs(l.src_wav) if l.src_wav else None
        fin = p.final_wav(s.line_id)
        win = p.abs(s.winner) if s.winner else None
        tr = trs.get(s.line_id)
        samples.append(
            {
                "line_id": s.line_id,
                "speaker": l.speaker_id,
                "text_src": l.text_src,
                "text_tgt": tr.text_tgt if tr else "",
                "status": s.status,
                "composite": s.scores.get("composite"),
                "backend": f"{s.backend}/{s.route}",
                "orig": _rel(rd, orig) if orig and orig.exists() else None,
                "winner": _rel(rd, fin if fin.exists() else win) if (fin.exists() or (win and win.exists())) else None,
            }
        )
    data["samples"] = samples
    data["n_audio"] = len(audio_idx)
    tick("samples")
    return data


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html", "j2"]), trim_blocks=True, lstrip_blocks=True)

    def fmt(v: Any, nd: int = 3) -> str:
        if v is None:
            return "–"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    env.filters["fmt"] = fmt
    return env


def render(data: dict[str, Any]) -> tuple[str, str]:
    env = _env()
    html = env.get_template("index.html.j2").render(**data)
    md_env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    md_env.filters["fmt"] = env.filters["fmt"]
    md = md_env.get_template("report.md.j2").render(**data)
    return html, md


def run(ctx: Any) -> dict[str, Any]:
    data = collect(ctx)
    html, md = render(data)
    rd = ctx.paths.report_dir
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "index.html").write_text(html, encoding="utf-8")
    if ctx.cfg.report.markdown:
        (rd / "report.md").write_text(md, encoding="utf-8")
    out = {
        "html": str(rd / "index.html"),
        "lines": data["overview"]["lines"],
        "finals": data["overview"]["finals"],
        "review_queue": data["queue_total"],
        "substitutions": len(data["substitutions"]),
    }
    cov = data.get("coverage")
    if cov:  # the patch's gaps are in the summary, not only in the document
        out["provisional"] = cov["provisional"]
        out["left_out"] = cov["left_out"]
    return out
