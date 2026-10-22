"""Smoke test of a built VNDub.exe: the whole wizard through the HTTP API of the frozen app.

Starts ``VNDub.exe --browser --profile apptest`` (all-mock pipeline, no browser opened) on a throw-away data
folder and a copy of the ``renpy_mini`` fixture, then walks Step 0 -> 7 (doctor, import, translate,
automatic casting, preview, full run, review apply, patch install with a backup), imports a BGI game made
from scratch (``tests/bgi_archive.py``) with the bundled ``bin\\bgi-extractor``, downloads a tiny GGUF
from Hugging Face through "add a model by link", and starts a backend adapter from ``<data>/envs`` (an
empty venv: the adapter must start from the env interpreter and fail on its first third-party import,
not report "env not created"). Modules the frozen app imports only along the way show up here as job
failures, which ``--selfcheck`` cannot catch.

    uv run python packaging/smoke_exe.py [path\\to\\VNDub.exe] [--no-network]

Exit code 0 = every step passed. Nothing outside the temporary folder is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "renpy_mini"
sys.path.insert(0, str(ROOT / "tests"))
from bgi_archive import write_bgi_game  # noqa: E402  (a BURIKO ARC20 game built from scratch)
TINY_GGUF = ("https://huggingface.co/ggml-org/tiny-llamas", "stories260K.gguf")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Smoke:
    def __init__(self, base: str) -> None:
        self.c = httpx.Client(base_url=base, timeout=60)
        self.results: list[tuple[str, bool, str]] = []

    def step(self, name: str, ok: bool, note: str = "") -> bool:
        self.results.append((name, ok, note))
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {note}" if note else ""), flush=True)
        return ok

    def wait(self, job: dict[str, Any], timeout: float = 900) -> dict[str, Any]:
        t0 = time.monotonic()
        while True:
            j = self.c.get(f"/api/jobs/{job['id']}").json()
            if j["state"] in ("done", "failed", "cancelled", "interrupted"):
                return j
            if time.monotonic() - t0 > timeout:
                return {**j, "state": "timeout"}
            time.sleep(0.25)

    def job(self, name: str, path: str, body: Any = None, query: str = "") -> dict[str, Any]:
        r = self.c.post(path + query, json=body if body is not None else {})
        if r.status_code != 200:
            self.step(name, False, f"HTTP {r.status_code} {r.text[:300]}")
            return {"state": "http_error"}
        j = self.wait(r.json())
        err = j.get("error") or {}
        self.step(name, j["state"] == "done", "" if j["state"] == "done" else f"{j['state']} {err.get('code')} {str(err.get('details'))[:400]}")
        return j

    def idle(self, slug: str) -> None:
        for j in self.c.get("/api/jobs", params={"project": slug, "active": 1}).json():
            self.wait(j)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=str(ROOT / "dist" / "VNDub" / "VNDub.exe"))
    ap.add_argument("--no-network", action="store_true", help="skip the Hugging Face download")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder")
    args = ap.parse_args()
    exe = Path(args.exe).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="vndub-smoke-"))
    data, game = tmp / "data", tmp / "games" / "renpy_mini"
    bgi_game = tmp / "games" / "ゲーム Игра BGI"
    shutil.copytree(FIXTURE, game)
    voice = game / "game" / "voice" / "sak_0001.ogg"
    orig = sha(voice)
    port = free_port()
    env = {**os.environ, "VNDUB_NO_BROWSER": "1", "HF_HOME": str(tmp / "hf_home")}
    env.pop("VNDUB_ENVS_DIR", None)
    proc = subprocess.Popen([str(exe), "--browser", "--port", str(port), "--profile", "apptest", "--data-dir", str(data), "--no-idle-exit"], env=env)
    s = Smoke(f"http://127.0.0.1:{port}")
    t_start = time.monotonic()
    try:
        for _ in range(240):
            try:
                if s.c.get("/api/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        health = s.c.get("/api/health").json()
        s.step("server up (frozen exe)", True, f"{time.monotonic() - t_start:.1f}s, {health}")

        rows = {r["id"]: r for r in s.c.get("/api/doctor").json()["rows"]}
        s.step("step 0 doctor", bool(rows), ", ".join(f"{k}={v.get('status')}" for k, v in rows.items()))
        slug = s.c.post("/api/project/open", json={"game_dir": str(game)}).json()["slug"]
        s.job("step 1 import (detect..analyze, text_emotion)", "/api/project/import")
        s.job("step 2 translate", "/api/project/translate")
        s.job("step 3 automatic casting", "/api/casting/auto")
        s.idle(slug)
        s.job("step 4 preview", "/api/preview/run", {"mode": "first_n", "n": 3})
        s.job("step 5 full run", "/api/run/full", {"shutdown_when_done": False})
        items = s.c.get("/api/review/lines").json()["items"]
        s.step("step 6 review list", bool(items), f"{len(items)} lines")
        lid = next(i["line_id"] for i in items if i["speaker_id"] == "sakura")  # one change, so apply re-renders it
        r = s.c.post(f"/api/review/lines/{lid}/action", json={"action": "regen", "payload": {"text": "Ладно, договорились."}})
        s.step("step 6 review: new text for one line", r.status_code == 200, r.text[:200] if r.status_code != 200 else lid)
        s.job("step 6 review apply", "/api/review/apply")
        plan = s.c.get("/api/install/plan", params={"slug": slug}).json()
        s.step("step 7 install plan", bool(plan.get("auto_possible")), f"engine={plan.get('engine')} files={len(plan.get('files') or [])}")
        s.job("step 7 install (with backup)", "/api/install/apply", {"backup": True}, query=f"?slug={slug}")
        backups = list((data / "backups").rglob("sak_0001.ogg"))
        s.step("patch replaced the game's voice file, original backed up", sha(voice) != orig and bool(backups) and sha(backups[0]) == orig)
        s.job("step 7 patch zip", "/api/install/zip", {}, query=f"?slug={slug}")

        # a BGI game in a folder with Japanese and Cyrillic characters: read by the bgi-extractor next to the exe
        bgi = {t["name"]: t for t in s.c.get("/api/tools").json()}.get("bgi_extractor") or {}
        s.step("bgi-extractor bundled with the exe", bool(bgi.get("present")) and str(bgi.get("path", "")).startswith(str(exe.parent / "bin")), str(bgi.get("path")))
        write_bgi_game(bgi_game, (FIXTURE / "game" / "voice" / "sak_0001.ogg").read_bytes())
        s.c.post("/api/project/open", json={"game_dir": str(bgi_game)})
        s.job("BGI import (bundled bgi-extractor)", "/api/project/import")
        summ = s.c.get("/api/project/summary").json()
        s.step("BGI game: 4 lines, 3 voiced, no tool missing", summ.get("lines_total") == 4 and summ.get("voiced") == 3 and not summ.get("tools_missing"),
               f"engine={summ.get('engine')} lines={summ.get('lines_total')} voiced={summ.get('voiced')} missing={summ.get('tools_missing')}")

        if not args.no_network:
            url, fname = TINY_GGUF
            ins = s.c.post("/api/models/hf/inspect", json={"url": url})
            ok = ins.status_code == 200 and ins.json().get("kind") == "gguf_llm"
            s.step("models: inspect a Hugging Face link", ok, ins.text[:200] if not ok else f"{ins.json()['repo']}, {len(ins.json()['files'])} files")
            if ok:
                i = ins.json()
                s.job("models: download it (huggingface_hub + hf_xet in the exe)", "/api/models/hf/add", {"repo": i["repo"], "revision": i["revision"], "files": [fname], "kind": "gguf_llm"})
                custom = s.c.get("/api/models/custom").json()
                s.step("models: listed as present", any(m.get("present") for m in (custom if isinstance(custom, list) else custom.get("items", []))))

        # a backend adapter from <data>/envs: an empty venv made with the bundled uv (the env ecapa lives in, per the registry)
        from vndub.backends import registry

        env_name = registry.get("ecapa").env
        uv = exe.parent / "_internal" / "bin" / "uv.exe"
        venv = data / "envs" / env_name
        r = subprocess.run([str(uv), "venv", str(venv), "--python", "3.11"], capture_output=True, text=True)
        models = data / "models"
        (models / "spkrec-ecapa-voxceleb").mkdir(parents=True, exist_ok=True)
        (models / "spkrec-ecapa-voxceleb" / ".vndub_complete").write_text("smoke\n", encoding="utf-8")
        j = s.job(f"adapter: start ecapa from <data>/envs/{env_name} (empty venv)", "/api/models/check", {"names": ["ecapa"]})
        row = ((j.get("result") or {}).get("rows") or [{}])[0]
        details = str(row.get("details") or "")
        log = models / f"adapter_{env_name}.log"
        log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        started = "not created" not in details and (str(venv).lower() in log_text.lower() or "ModuleNotFoundError" in log_text or "No module named" in details)
        s.step("adapter process started with the env interpreter", r.returncode == 0 and started, f"code={row.get('code')} details={details[:240]!r} log={log_text.strip().splitlines()[-1][:200] if log_text.strip() else '-'}")
    finally:
        try:
            s.c.post("/api/system/quit")
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    failed = [n for n, ok, _ in s.results if not ok]
    print(json.dumps({"steps": len(s.results), "failed": failed, "seconds": round(time.monotonic() - t_start, 1)}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
