"""Generate the synthetic test fixtures under tests/fixtures/.

    uv run python scripts/make_fixtures.py            # all fixtures
    uv run python scripts/make_fixtures.py renpy kag  # selected ones
    uv run python scripts/make_fixtures.py --synthetic  # never use a TTS voice

renpy_mini   game/script.rpy built from scripts/fixture_data.py (3 voiced characters, an
             UNVOICED protagonist `hero` with narrator lines around his dialogue that carry
             delivery cues, a menu, voice statements, one `voice sustain`, an epilogue scene)
             and Japanese voices game/voice/*.ogg. Existing voice files are kept as they are
             (script.rpy, mock_dictionary.yaml and FIXTURE.md are regenerated); pass
             --rerender-voices (or --synthetic) to render the audio again. Voice source, in
             order of preference:
               1. Windows SAPI ja-JP voice (Microsoft Haruka) with per-character pitch/rate
                  and per-line delivery (vndub.backends.tts.sapi),
               2. synthetic speech-like signals with a distinct pitch per character
                  (any OS, or --synthetic).
             (Qwen3-TTS Japanese would be preferred when its weights are present; this
             build machine has no GPU, so it is not wired in.)
             Encoded as OGG Vorbis mono 22.05 kHz ~48 kbps; total < 2 MB, committed.
kag_mini     macro.ks ([nm] macro with a voice parameter via [playse storage=%s]), a
             scenario with 【名前】 lines and direct [playse], tiny voices, and
             xp3/sample.xp3 written by the built-in XP3 writer (reader tests).
nscripter_mini  nscript.dat (XOR 0x84 of a plain script with dwave lines) + arc.nsa
             (built-in NSA writer) holding the voices, plus one loose override file.
siglus_mini  koe_master.csv + KOE(#########).ogg files in per-character folders.
bgi_mini     bgi_json/*.json (bgi-extractor-like records) + voice files.
tyrano_mini  data/scenario/*.ks with [voconfig]/[vostart] and #name lines + data/sound.
mock_dictionary.yaml  MockTranslator dictionary (ja -> ru) for the renpy_mini lines.

Regenerating the script changes line ids and counts, so refresh the golden parser outputs afterwards:
    VNDUB_UPDATE_GOLDEN=1 uv run pytest tests/test_engine_goldens.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fixture_data as FD  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
SR = 22050


# ----------------------------------------------------------------------------
# audio helpers
# ----------------------------------------------------------------------------


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg not found on PATH")
    return exe


def encode_ogg(wav: Path, ogg: Path, sr: int = SR, kbps: int = 48) -> None:
    ogg.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", str(sr), "-c:a", "libvorbis", "-b:a", f"{kbps}k", "-map_metadata", "-1", "-fflags", "+bitexact", str(ogg)],
        check=True,
    )


def synth_speechlike(text: str, f0: float, seed: int, sr: int = SR, rate: float = 1.0) -> np.ndarray:
    """Harmonic bursts, one per kana-ish character, with an F0 contour."""
    rng = np.random.default_rng(seed)
    n_syl = max(2, sum(1 for c in text if ord(c) > 0x3000 and c not in "、。！？…「」"))
    dur = 0.15 + n_syl * 0.13 / rate
    n = int(dur * sr)
    t = np.arange(n) / sr
    contour = f0 * 2 ** ((1.5 * np.sin(2 * np.pi * 0.8 * t + rng.uniform(0, 3))) / 12)
    ph = 2 * np.pi * np.cumsum(contour) / sr
    y = sum((0.5 / k) * np.sin(k * ph) for k in range(1, 9))
    env = np.zeros(n)
    for i in range(n_syl):
        a = 0.08 + (dur - 0.16) * i / n_syl
        b = a + (dur - 0.16) / n_syl * 0.8
        m = (t >= a) & (t < b)
        env[m] = np.sin(np.pi * (t[m] - a) / (b - a)) ** 0.6
    return (0.3 * env * y / 2 + rng.normal(0, 0.003, n)).astype(np.float32)


def tone(path: Path, f0: float, seconds: float = 0.4, sr: int = 16000) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    y = (0.2 * np.sin(2 * np.pi * f0 * t) * np.hanning(t.size)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path.read_bytes()


def tiny_ogg(path: Path, f0: float, seconds: float = 0.5) -> None:
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "t.wav"
        tone(w, f0, seconds)
        encode_ogg(w, path, sr=16000, kbps=32)


def _pct(*vals: str) -> str:
    total = sum(int(v.strip().rstrip("%")) for v in vals if v)
    return f"{total:+d}%"


# ----------------------------------------------------------------------------
# renpy_mini
# ----------------------------------------------------------------------------


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def renpy_script() -> tuple[str, list[dict]]:
    """script.rpy text and the voice jobs [{file, char, text, delivery}]."""
    out = [
        "# Synthetic fixture generated by scripts/make_fixtures.py -- do not edit by hand.",
        "# 3 voiced characters + an unvoiced protagonist, narrator lines, a menu, voice statements and one `voice sustain`.",
        "",
    ]
    for var, (name, tag, _g, _ru, _p, _r) in FD.CHARACTERS.items():
        out.append(f'define {var} = Character({_q(name)}, color="#c8c8ff", voice_tag={_q(tag)})')
    out += ["", "init python:", "    config.has_voice = True", ""]
    jobs: list[dict] = []
    last_voice: dict | None = None

    def emit(rows: list, indent: str) -> None:
        nonlocal last_voice
        for row in rows:
            kind = row[0]
            if kind == "label":
                if out[-1] != "":
                    out.append("")
                out.append(f"label {row[1]}:")
                if row[1] == "start":
                    out.append(f"{indent}scene black")
                    out.append(f"{indent}$ renpy.music.set_volume(0.8)")
                continue
            if kind == "narr":
                out.append(f"{indent}{_q(row[1])}")
                last_voice = None
            elif kind == "say":
                _, var, ja, _ru, voice, delivery = row
                if voice:
                    out.append(f"{indent}voice {_q(voice)}")
                    job = {"file": voice, "char": var, "text": ja, "delivery": delivery}
                    jobs.append(job)
                    last_voice = job
                else:
                    last_voice = None
                out.append(f"{indent}{var} {_q(ja)}")
            elif kind == "sustain":
                _, var, ja, _ru = row
                out.append(f"{indent}voice sustain")
                out.append(f"{indent}{var} {_q(ja)}")
                if last_voice is not None:
                    last_voice["text"] += ja  # the sustained file speaks both lines
            elif kind == "menu":
                out.append(f"{indent}menu:")
                for cap, _cru, body in row[1]:
                    out.append(f"{indent}    {_q(cap)}:")
                    emit(body, indent + "        ")
                out.append(f"{indent}jump after_menu")
        return None

    emit(FD.SCRIPT, "    ")
    out += ["    with dissolve", "    return", ""]
    text = "\n".join(l for i, l in enumerate(out) if not (l == "" and i > 0 and out[i - 1] == ""))
    return text + "\n", jobs


def _voices_label(root: Path) -> str:
    """The "Voices: ..." value recorded in FIXTURE.md by the run that rendered the audio."""
    md = root / "FIXTURE.md"
    if md.exists():
        for line in md.read_text(encoding="utf-8").splitlines():
            if line.startswith("Voices:"):
                return line[len("Voices:"):].strip().removesuffix(" No game data.").strip().rstrip(".")
    return "existing files"


def make_renpy(synthetic: bool, rerender: bool = False) -> None:
    root = FIX / "renpy_mini"
    game = root / "game"
    voice_dir = game / "voice"
    text, jobs = renpy_script()
    game.mkdir(parents=True, exist_ok=True)
    (game / "script.rpy").write_text(text, encoding="utf-8")
    keep = not rerender and not synthetic and all((game / j["file"]).exists() for j in jobs)
    if keep:
        used = _voices_label(root)
        print(f"renpy_mini: keeping the {len(jobs)} existing voice files ({used}); --rerender-voices renders them again")
    else:
        used = _render_renpy_voices(jobs, game, voice_dir, synthetic)
    _finish_renpy(root, voice_dir, jobs, used)


def _render_renpy_voices(jobs: list[dict], game: Path, voice_dir: Path, synthetic: bool) -> str:
    """Render (and OGG-encode) every voiced line; returns the voice source label."""
    if voice_dir.exists():
        shutil.rmtree(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    used = "synthetic"
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        wavs: dict[str, Path] = {}
        if not synthetic:
            try:
                from vndub.backends.tts.sapi import sapi_available, synthesize_batch

                if sapi_available():
                    batch = []
                    for i, j in enumerate(jobs):
                        _n, _t, _g, _ru, base_pitch, base_rate = FD.CHARACTERS[j["char"]]
                        dp, dr, vol = j["delivery"]
                        batch.append({"id": str(i), "text": j["text"], "lang": "ja", "out": str(tdp / f"{i}.wav"), "pitch": _pct(base_pitch, dp), "rate": _pct(base_rate, dr), "volume": vol})
                    res = synthesize_batch(batch)
                    if all(ok for ok, _ in res.values()):
                        wavs = {j["file"]: tdp / f"{i}.wav" for i, j in enumerate(jobs)}
                        used = "sapi:" + next(iter(res.values()))[1]
                    else:
                        print("SAPI failed for some lines, falling back to synthetic:", [d for ok, d in res.values() if not ok][:2])
            except Exception as e:  # pragma: no cover - environment dependent
                print("SAPI unavailable:", e)
        if not wavs:
            f0s = {"s": 260.0, "h": 120.0, "y": 300.0}
            for i, j in enumerate(jobs):
                y = synth_speechlike(j["text"], f0s[j["char"]], seed=i)
                p = tdp / f"{i}.wav"
                sf.write(str(p), y, SR, subtype="PCM_16")
                wavs[j["file"]] = p
        for j in jobs:
            encode_ogg(wavs[j["file"]], game / j["file"])
    return used


def _finish_renpy(root: Path, voice_dir: Path, jobs: list[dict], used: str) -> None:
    n_say = sum(1 for r in FD.iter_rows() if r[0] == "say")
    n_mc = sum(1 for r in FD.iter_rows() if r[0] == "say" and r[1] == "mc")
    n_narr = sum(1 for r in FD.iter_rows() if r[0] == "narr")
    (root / "FIXTURE.md").write_text(
        "Synthetic Ren'Py fixture generated by `scripts/make_fixtures.py` from `scripts/fixture_data.py`.\n"
        f"Voices: {used}. No game data.\n"
        "\n"
        "Mixed game (voiced heroines + unvoiced protagonist):\n"
        f"- {len(jobs)} voiced lines: sakura, haruto, yui (+ one `voice sustain` continuation). A normal regeneration never re-renders the audio.\n"
        f"- {n_mc} unvoiced lines of the protagonist `hero` (name 蓮, voice_tag \"hero\"): no audio at all.\n"
        f"- {n_narr} narrator lines (unvoiced). Five of them carry Japanese delivery cues next to hero lines\n"
        "  (小さな声でささやいた, 笑いながら言った, 泣きながら言った, ...と叫んだ, 怒鳴った); the others have no cue.\n"
        "  The hero line みんな、帰ったのか。 (scene epilogue) has no cue within three narration lines either way.\n"
        f"- scenes: start (hero lines before the menu), after_menu, epilogue; {n_say} say rows in all.\n",
        encoding="utf-8",
    )
    size = sum(p.stat().st_size for p in voice_dir.glob("*.ogg"))
    print(f"renpy_mini: {len(jobs)} voices ({used}), {size / 1024:.0f} KiB")
    (FIX / "mock_dictionary.yaml").write_text(
        "# MockTranslator dictionary for tests/fixtures/renpy_mini (generated by scripts/make_fixtures.py)\n"
        + yaml.safe_dump(FD.dictionary(), allow_unicode=True, sort_keys=False, width=200),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------------
# kag_mini
# ----------------------------------------------------------------------------

KAG_MACRO = """; macro.ks -- name/voice macros (synthetic fixture)
[macro name=nm]
[ch text="【%t】"][r]
[playse storage=%s buf=1 cond="mp.s!=void"]
[endmacro]

[macro name=voi]
[playse storage=%file buf=1]
[endmacro]
"""

KAG_FIRST = """; first.ks
[call storage="macro.ks"]
[jump storage="s01.ks"]
"""

KAG_S01 = """*start|放課後
[cm]
夕暮れの屋上。風が少し冷たい。[p]

[nm t="寧々" s=nen_0001]「……本当に？」[p]

[nm t="ハルキ" s=har_0001]「ああ、本当だよ。[r]嘘はつかない」[p]

【寧々】[playse storage=nen_0002 buf=1]「じゃあ、信じる」[l][r]
[playse storage=se_wind buf=0]
風の音が響いた。[p]

#ハルキ
[voi file=har_0002]「帰ろうか」[p]

[ns]寧々[nse][playse storage=nen_0003]「うん！」[p]
"""


def make_kag() -> None:
    from vndub.engines.xp3 import write_xp3

    root = FIX / "kag_mini"
    if root.exists():
        shutil.rmtree(root)
    (root / "scenario").mkdir(parents=True)
    (root / "scenario" / "macro.ks").write_text(KAG_MACRO, encoding="utf-8")
    (root / "scenario" / "first.ks").write_text(KAG_FIRST, encoding="utf-8")
    (root / "scenario" / "s01.ks").write_bytes(KAG_S01.encode("cp932"))  # Shift-JIS on purpose (encoding sniffing)
    for name, f0 in [("nen_0001", 300), ("nen_0002", 310), ("nen_0003", 320), ("har_0001", 130), ("har_0002", 140)]:
        tiny_ogg(root / "voice" / f"{name}.ogg", f0)
    tiny_ogg(root / "sound" / "se_wind.ogg", 900, 0.3)
    xp3_files = [
        ("scenario/test.ks", "[nm t=\"寧々\" s=xp_0001]「アーカイブの中」[p]\n".encode("utf-8")),
        ("voice/xp_0001.wav", tone(Path(tempfile.gettempdir()) / "vndub_xp_0001.wav", 330, 0.3)),
    ]
    write_xp3(root / "xp3" / "sample.xp3", xp3_files)
    print("kag_mini: ok")


# ----------------------------------------------------------------------------
# nscripter_mini
# ----------------------------------------------------------------------------

NS_SCRIPT = """;mode800
*define
game
*start
bg black,1
放課後、教室には誰もいなかった。\\
dwave 0,"voice\\mio001.wav"
【澪】「遅かったね」@
dwave 0,"voice\\mio002.wav":wait 10
【澪】「ずっと待ってたんだよ」\\
dwave 1,"voice\\tak001.wav"
拓海「ごめん、委員会が長引いて」\\
彼女は小さく笑った。\\
dwave 0,"voice\\mio003.wav"
【澪】「……許してあげる」\\
end
"""


def make_nscripter() -> None:
    from vndub.engines.nsa import COMP_LZSS, encode_nscript, write_nsa

    root = FIX / "nscripter_mini"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "nscript.dat").write_bytes(encode_nscript(NS_SCRIPT))
    tmp = Path(tempfile.gettempdir())
    files = [
        ("voice/mio001.wav", tone(tmp / "vndub_mio001.wav", 280, 0.3, 8000)),
        ("voice/mio002.wav", tone(tmp / "vndub_mio002.wav", 290, 0.3, 8000)),
        ("voice/tak001.wav", tone(tmp / "vndub_tak001.wav", 130, 0.3, 8000)),
    ]
    write_nsa(root / "arc.nsa", files[:2])
    write_nsa(root / "arc1.nsa", files[2:], compression=COMP_LZSS)
    # a loose file overrides archive members (and supplies mio003)
    tone(root / "voice" / "mio003.wav", 300, 0.3, 8000)
    print("nscripter_mini: ok")


# ----------------------------------------------------------------------------
# siglus_mini / bgi_mini / tyrano_mini
# ----------------------------------------------------------------------------


def make_siglus() -> None:
    root = FIX / "siglus_mini"
    if root.exists():
        shutil.rmtree(root)
    rows = [
        (100001, "古河渚", "あの……", "1.2", "s01:12"),
        (100002, "岡崎朋也", "どうした？", "0.9", "s01:15"),
        (100003, "古河渚", "一緒に、帰りませんか", "2.1", "s01:18"),
        (200001, "岡崎朋也", "また明日な", "1.0", "s02:4"),
    ]
    root.mkdir(parents=True)
    lines = ["koe_no,character,text,duration,callsite"] + [",".join(map(str, r)) for r in rows]
    (root / "koe_master.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for koe, ch, *_ in rows:
        tiny_ogg(root / ch / f"KOE({koe:09d}).ogg", 250 if "渚" in ch else 130, 0.4)
    print("siglus_mini: ok")


def make_bgi() -> None:
    import json

    root = FIX / "bgi_mini"
    if root.exists():
        shutil.rmtree(root)
    data = {
        "messages": [
            {"speaker": "", "text": "朝の駅は静かだった。"},
            {"speaker": "ユキ", "text": "おはよう！", "voice": "yuki_000001"},
            {"name": "ソウタ", "message": "早いな", "voice_id": "sota_000001"},
            {"speaker": "ユキ", "text": "(yuki_000002)今日は特別な日だから"},
        ]
    }
    (root / "bgi_json").mkdir(parents=True)
    (root / "bgi_json" / "01_morning.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, f0 in [("yuki_000001", 300), ("yuki_000002", 310), ("sota_000001", 130)]:
        tiny_ogg(root / "voice" / f"{n}.ogg", f0, 0.4)
    print("bgi_mini: ok")


TYRANO_FIRST = """;first.ks
[chara_new name="akane" storage="chara/akane/normal.png" jname="あかね"]
[chara_new name="yamato" storage="chara/yamato/normal.png" jname="やまと"]
[jump storage="scene1.ks"]
"""

TYRANO_SCENE = """*start
[voconfig sebuf=2 name="akane" vostorage="akane_{number}.ogg" number=1]
[voconfig sebuf=2 name="yamato" vostorage="yamato_{number}.ogg" number=1]
[vostart]
#
放課後の教室。[p]
#akane
ねえ、ちょっといい？[p]
#yamato
なに？[p]
#akane:happy
これ、あげる！[p]
[vostop]
#akane
（独り言は録音されていない）[p]
[vostart]
#yamato
……ありがとう。[p]
"""


def make_tyrano() -> None:
    root = FIX / "tyrano_mini"
    if root.exists():
        shutil.rmtree(root)
    sc = root / "data" / "scenario"
    sc.mkdir(parents=True)
    (sc / "first.ks").write_text(TYRANO_FIRST, encoding="utf-8")
    (sc / "scene1.ks").write_text(TYRANO_SCENE, encoding="utf-8")
    for n, f0 in [("akane_1", 290), ("akane_2", 300), ("yamato_1", 140), ("yamato_2", 150)]:
        tiny_ogg(root / "data" / "sound" / f"{n}.ogg", f0, 0.4)
    print("tyrano_mini: ok")


MAKERS = {"renpy": make_renpy, "kag": make_kag, "nscripter": make_nscripter, "siglus": make_siglus, "bgi": make_bgi, "tyrano": make_tyrano}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", help=f"fixtures to build ({', '.join(MAKERS)}); default all")
    ap.add_argument("--synthetic", action="store_true", help="renpy_mini: synthetic signals instead of a TTS voice (re-renders the audio)")
    ap.add_argument("--rerender-voices", action="store_true", help="renpy_mini: render the voice files again (default: keep the existing ones)")
    args = ap.parse_args()
    bad = [w for w in args.which if w not in MAKERS]
    if bad:
        ap.error(f"unknown fixture(s) {bad}; choose from {list(MAKERS)}")
    for name in args.which or list(MAKERS):
        if name == "renpy":
            make_renpy(args.synthetic, args.rerender_voices)
        else:
            MAKERS[name]()


if __name__ == "__main__":
    main()
