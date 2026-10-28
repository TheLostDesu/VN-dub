"""Review UI (Gradio): listen to the original vs candidates, override the automatic choice.

Everything the UI decides is appended to ``work/overrides.jsonl`` via
:func:`write_override`; ``vndub synth --only ...`` then ``vndub score --only ...``
applies regen / set_text / set_ref, and ``score`` honours accept / reject /
keep_original immediately.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from vndub.manifest.io import CandidateScore, OverrideRecord, append_jsonl, read_json, read_jsonl
from vndub.paths import WorkPaths

ACTIONS = ("accept", "reject", "regen", "set_text", "set_ref", "keep_original", "set_src_text", "tuning")


def write_override(paths: WorkPaths, line_id: str, action: str, payload: dict[str, Any] | None = None) -> OverrideRecord:
    if action not in ACTIONS:
        raise ValueError(f"unknown override action {action!r}; one of {ACTIONS}")
    from vndub.stages.genutil import ts_now

    rec = OverrideRecord(line_id=line_id, action=action, payload=payload or {}, ts=ts_now())
    append_jsonl(paths.overrides, rec)
    return rec


# ----------------------------------------------------------------------------
# data access (plain functions, testable without gradio)
# ----------------------------------------------------------------------------


def load_rows(ctx: Any) -> list[dict[str, Any]]:
    from vndub.stages import genutil as gu

    p = ctx.paths
    # the lines synth / score work on: the link rebindings applied (a relinked line shows its new voice) and, with
    # synth.voice_unvoiced, the route-P lines that have no original audio (the raw has_voice hid them)
    lines = ctx.dub_lines(apply_only=False) if p.lines.exists() else []
    sel = gu.load_selection(ctx)
    trs = gu.load_translations(ctx)
    rows = []
    for l in lines:
        s = sel.get(l.line_id)
        tr = trs.get(l.line_id)
        rows.append(
            {
                "line_id": l.line_id,
                "speaker": l.speaker_id,
                "scene": l.scene,
                "status": s.status if s else "-",
                "composite": (s.scores.get("composite") if s else None),
                "winner": (f"{s.backend}/{s.route}" if s and s.winner else ""),
                "text_src": l.text_src,
                "text_tgt": tr.text_tgt if tr else "",
                "reason": s.reason if s else "",
            }
        )
    return rows


def filter_rows(rows: list[dict[str, Any]], speaker: str = "all", scene: str = "all", status: str = "all", lo: float = 0.0, hi: float = 1.0) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if speaker not in ("", "all") and r["speaker"] != speaker:
            continue
        if scene not in ("", "all") and r["scene"] != scene:
            continue
        if status not in ("", "all") and r["status"] != status:
            continue
        c = r["composite"]
        if c is not None and not (lo <= float(c) <= hi):
            continue
        out.append(r)
    return out


def candidates(ctx: Any, line_id: str) -> list[dict[str, Any]]:
    p = ctx.paths
    scores = {c.wav: c for c in read_jsonl(p.scores(line_id), CandidateScore)} if p.scores(line_id).exists() else {}
    out = []
    d = p.synth_dir(line_id)
    for wav in sorted(d.glob("*.wav")) if d.exists() else []:
        if wav.name.endswith(".prevc.wav"):
            continue
        rel = p.rel(wav)
        side = read_json(wav.with_suffix(".json")) or {}
        sc = scores.get(rel)
        label = f"{wav.stem}"
        if sc is not None and sc.composite is not None:
            label += f"  composite={sc.composite:.3f}{'  PASS' if sc.passed else ''}"
        out.append({"label": label, "wav": rel, "abs": str(wav), "side": side, "score": sc})
    return out


def batch_accept(ctx: Any, threshold: float) -> int:
    """Accept the current winner of every auto/review line whose composite >= threshold."""
    from vndub.stages import genutil as gu

    n = 0
    for lid, s in gu.load_selection(ctx).items():
        if s.status == "override" or not s.winner:
            continue
        if (s.scores.get("composite") or 0.0) >= threshold:
            write_override(ctx.paths, lid, "accept", {"candidate": s.winner, "batch": True})
            n += 1
    return n


def _quote(arg: Any) -> str:
    """Double quotes when needed: valid in cmd.exe, PowerShell and POSIX shells (a project folder with a space broke the line)."""
    s = str(arg)
    # (a backslash path is quoted too: bash would eat the backslashes outside quotes)
    return s if re.fullmatch(r"[A-Za-z0-9_./:,=+@%-]+", s) else '"' + s.replace('"', '\\"') + '"'


def score_command(ctx: Any, line_ids: list[str]) -> str:
    return f"uv run vndub score --project {_quote(ctx.project_dir)} --only {_quote(','.join(line_ids))}"


def rerun_command(ctx: Any, line_ids: list[str]) -> str:
    """The synth + score commands for ``line_ids``, one per line: ``&&`` is not a statement separator of cmd.exe or
    Windows PowerShell 5, so the two are run one after the other."""
    proj, ids = _quote(ctx.project_dir), _quote(",".join(line_ids))
    return f"uv run vndub synth --project {proj} --only {ids}\nuv run vndub score --project {proj} --only {ids}"


# ----------------------------------------------------------------------------
# gradio
# ----------------------------------------------------------------------------


def build_app(ctx: Any):
    import gradio as gr

    from vndub.stages import genutil as gu

    rows_all = load_rows(ctx)
    cols = ["line_id", "speaker", "scene", "status", "composite", "winner", "text_src", "text_tgt", "reason"]
    speakers = ["all", *sorted({r["speaker"] for r in rows_all})]
    scenes = ["all", *sorted({r["scene"] for r in rows_all})]
    statuses = ["all", "auto", "regen", "review", "override", "-"]
    touched: list[str] = []

    def table(speaker: str, scene: str, status: str, lo: float, hi: float) -> list[list[Any]]:
        rows = filter_rows(load_rows(ctx), speaker, scene, status, lo, hi)
        return [[r[c] if r[c] is not None else "" for c in cols] for r in rows]

    def show(line_id: str):
        line_id = (line_id or "").strip()
        lines = {l.line_id: l for l in ctx.dub_lines(apply_only=False)} if ctx.paths.lines.exists() else {}
        l = lines.get(line_id)
        if l is None:
            return ("unknown line", None, gr.update(choices=[], value=None), gr.update(choices=[], value=None), None, None, "", "", gr.update(choices=[], value=None))
        cands = candidates(ctx, line_id)
        labels = [c["label"] for c in cands]
        sel = gu.load_selection(ctx).get(line_id)
        win = next((c for c in cands if sel and c["wav"] == sel.winner), cands[0] if cands else None)
        trs = gu.load_translations(ctx)
        tr = trs.get(line_id)
        refs = gu.load_refs(ctx, l.speaker_id)
        info = f"**{line_id}** · {l.speaker_id} · {l.scene}\n\n{l.text_src}\n\n{tr.text_tgt if tr else ''}\n\nstatus: {sel.status if sel else '-'} — {sel.reason if sel else ''}"
        orig = str(ctx.paths.abs(l.src_wav)) if l.src_wav else None
        a = win["label"] if win else None
        b = labels[1] if len(labels) > 1 else a
        return (
            info,
            orig,
            gr.update(choices=labels, value=a),
            gr.update(choices=labels, value=b),
            win["abs"] if win else None,
            next((c["abs"] for c in cands if c["label"] == b), None),
            (win["side"].get("style_prompt", "") if win else ""),
            (tr.text_tts or tr.text_tgt) if tr else "",
            gr.update(choices=[r.path for r in refs], value=None),
        )

    def pick(line_id: str, label: str):
        for c in candidates(ctx, line_id):
            if c["label"] == label:
                sc = c["score"]
                return c["abs"], (sc.metrics if sc else {})
        return None, {}

    def act(action: str, line_id: str, label: str, style: str, text: str, ref: str) -> str:
        line_id = (line_id or "").strip()
        if not line_id:
            return "select a line first"
        cand = next((c for c in candidates(ctx, line_id) if c["label"] == label), None)
        payload: dict[str, Any] = {}
        if action in ("accept", "reject"):
            if cand is None:
                return "select a candidate"
            payload = {"candidate": cand["wav"]}
        elif action == "regen":
            payload = {"style_prompt": style, "text": text}
            if text:
                write_override(ctx.paths, line_id, "set_text", {"text": text, "style_prompt": style})
        elif action == "set_ref":
            if not ref:
                return "pick a reference clip"
            payload = {"ref": ref}
        write_override(ctx.paths, line_id, action, payload)
        if line_id not in touched:
            touched.append(line_id)
        msg = f"{action} recorded for {line_id}."
        if action in ("regen", "set_ref"):
            msg += f"\n\nRe-run (one command after the other):\n\n```\n{rerun_command(ctx, touched)}\n```"
        else:
            msg += f"\n\nApply:\n\n```\n{score_command(ctx, touched)}\n```"
        return msg

    def do_batch(th: float) -> str:
        n = batch_accept(ctx, float(th))
        return f"accepted {n} line(s) with composite >= {th}"

    with gr.Blocks(title="vndub review") as app:
        gr.Markdown("# vndub review")
        with gr.Row():
            f_spk = gr.Dropdown(speakers, value="all", label="character")
            f_scene = gr.Dropdown(scenes, value="all", label="scene")
            f_status = gr.Dropdown(statuses, value="all", label="status")
            f_lo = gr.Slider(0, 1, value=0, step=0.01, label="composite >=")
            f_hi = gr.Slider(0, 1, value=1, step=0.01, label="composite <=")
        grid = gr.Dataframe(headers=cols, value=table("all", "all", "all", 0, 1), interactive=False, wrap=True)
        for comp in (f_spk, f_scene, f_status, f_lo, f_hi):
            comp.change(table, [f_spk, f_scene, f_status, f_lo, f_hi], grid)
        with gr.Row():
            line_box = gr.Textbox(label="line_id", scale=2)
            load_btn = gr.Button("Load line", scale=1)
        info = gr.Markdown()
        orig = gr.Audio(label="original", type="filepath", interactive=False)
        with gr.Row():
            with gr.Column():
                cand_a = gr.Dropdown([], label="candidate A")
                audio_a = gr.Audio(label="A", type="filepath", interactive=False)
                metrics_a = gr.JSON(label="A metrics")
            with gr.Column():
                cand_b = gr.Dropdown([], label="candidate B")
                audio_b = gr.Audio(label="B", type="filepath", interactive=False)
                metrics_b = gr.JSON(label="B metrics")
        swap = gr.Button("A <-> B")
        style = gr.Textbox(label="style prompt (used on regen)")
        text = gr.Textbox(label="text for TTS (used on regen / set_text)")
        ref = gr.Dropdown([], label="reference clip (set_ref)")
        with gr.Row():
            b_acc = gr.Button("Accept A", variant="primary")
            b_rej = gr.Button("Reject A")
            b_regen = gr.Button("Regenerate")
            b_ref = gr.Button("Set reference")
            b_orig = gr.Button("Keep original")
        with gr.Row():
            th = gr.Number(value=ctx.cfg.review.batch_accept_threshold, label="batch accept threshold")
            b_batch = gr.Button("Batch accept >= threshold")
        status = gr.Markdown()

        load_btn.click(show, [line_box], [info, orig, cand_a, cand_b, audio_a, audio_b, style, text, ref])
        cand_a.change(pick, [line_box, cand_a], [audio_a, metrics_a])
        cand_b.change(pick, [line_box, cand_b], [audio_b, metrics_b])
        swap.click(lambda a, b: (gr.update(value=b), gr.update(value=a)), [cand_a, cand_b], [cand_a, cand_b])
        b_acc.click(lambda *a: act("accept", *a), [line_box, cand_a, style, text, ref], status)
        b_rej.click(lambda *a: act("reject", *a), [line_box, cand_a, style, text, ref], status)
        b_regen.click(lambda *a: act("regen", *a), [line_box, cand_a, style, text, ref], status)
        b_ref.click(lambda *a: act("set_ref", *a), [line_box, cand_a, style, text, ref], status)
        b_orig.click(lambda *a: act("keep_original", *a), [line_box, cand_a, style, text, ref], status)
        b_batch.click(do_batch, [th], status)
    return app


def launch(ctx: Any, host: str = "127.0.0.1", port: int = 7861, open_browser: bool = False) -> None:
    app = build_app(ctx)
    app.launch(server_name=host, server_port=port, inbrowser=open_browser, allowed_paths=[str(ctx.paths.work), str(Path(ctx.project_dir))], share=False)
