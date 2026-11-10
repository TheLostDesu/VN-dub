"""Round 18: backend env health (build record, pin drift, import probe), the torch kind, failed extras, disk / memory /
Hugging Face rows of Diagnostics.

0.7.1, 1.0.1 and 1.0.2 fixed envs that broke inside a vendor import after minutes of weight loading; nothing in the app
noticed a stale or broken env before a job reached it. The fake envs below are folders with the files the checks read
(``pyvenv.cfg``, ``*.dist-info`` names, ``vndub_env.json``, ``vndub_probe.json``); no interpreter is started.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
from o_helpers import drop_runner, fresh_runner, make_env, wait_job

from vndub_app import __version__, appdirs, doctor, installer, jobs

REPO = Path(__file__).resolve().parents[2]
FUNASR_PINS = {  # what envs/funasr.txt + funasr+cosyvoice.txt pin, as the dist-info names of a healthy env
    "torch": "2.3.1+cpu", "torchaudio": "2.3.1+cpu", "numpy": "1.26.4", "scipy": "1.13.1", "soundfile": "0.12.1", "librosa": "0.10.2",
    "funasr": "1.2.7", "modelscope": "1.20.0", "huggingface_hub": "0.30.2", "transformers": "4.51.3", "omegaconf": "2.3.0",
    "hydra_core": "1.3.2", "sentencepiece": "0.2.0", "conformer": "0.3.2", "diffusers": "0.29.0", "hyperpyyaml": "1.2.3",
    "ruamel.yaml": "0.18.17", "inflect": "7.3.1", "lightning": "2.2.4", "onnx": "1.16.0", "onnxruntime": "1.18.0",
    "openai_whisper": "20250625", "protobuf": "4.25.8", "pyarrow": "18.1.0", "pydantic": "2.7.0", "pyworld": "0.3.4", "rich": "13.7.1",
    "x_transformers": "2.11.24", "wetext": "0.0.4", "gdown": "5.1.0", "wget": "3.2", "tiktoken": "0.8.0", "matplotlib": "3.7.5",
    "setuptools": "80.10.2",
}


@pytest.fixture
def o_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    data = make_env(monkeypatch, tmp_path)
    fresh_runner()
    monkeypatch.setattr(installer, "ensure_python", lambda version, ctx: Path("python"))
    monkeypatch.setattr(installer, "find_uv", lambda: Path("uv"))
    yield data
    doctor.set_probes(None)
    drop_runner()


def fake_env(name: str, packages: dict[str, str], *, record: dict[str, Any] | None | bool = True, probe: dict[str, Any] | None = None) -> Path:
    env = installer.envs_dir() / name
    sp = env / "Lib" / "site-packages"
    sp.mkdir(parents=True, exist_ok=True)
    py = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_bytes(b"")
    (env / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    for old in sp.glob("*.dist-info"):  # a second call replaces the packages: two versions of one package have no defined winner
        old.rmdir()
    for n, v in packages.items():
        (sp / f"{n}-{v}.dist-info").mkdir(exist_ok=True)
    if record is True:
        record = {"env": name, "cpu": True, "extras_failed": [], "req_sha256": installer.envrecord().req_sha256(appdirs.resource("envs"), name)}
    if record:
        (env / "vndub_env.json").write_text(json.dumps(record), encoding="utf-8")
    if probe is not None:
        (env / "vndub_probe.json").write_text(json.dumps(probe), encoding="utf-8")
    return env


def ok_probe(name: str) -> dict[str, Any]:
    return {"ok": True, "at": "2026-11-10T00:00:00", "app_version": __version__, "req_sha256": installer.envrecord().req_sha256(appdirs.resource("envs"), name), "results": {}}


def codes(st: dict[str, Any]) -> list[str]:
    return [p["code"] for p in st["problems"]]


# ---- the build record helper --------------------------------------------------------------------------------------


def test_requirement_hash_is_stable_and_ignores_line_endings(tmp_path: Path) -> None:
    er = installer.envrecord()
    for f in ("x.txt", "x+extra.txt", "build.constraints.txt"):
        (tmp_path / f).write_bytes(b"a==1\nb==2\n")
    h = er.req_sha256(tmp_path, "x")
    (tmp_path / "x+extra.txt").write_bytes(b"a==1\r\nb==2\r\n")
    assert er.req_sha256(tmp_path, "x") == h
    (tmp_path / "x+extra.txt").write_bytes(b"a==1\nb==3\n")
    assert er.req_sha256(tmp_path, "x") != h
    assert [f.name for f in er.req_files(tmp_path, "x")] == ["x.txt", "x+extra.txt", "build.constraints.txt"]


def test_the_record_says_what_torch_is_installed_not_what_was_asked_for(tmp_path: Path) -> None:
    er = installer.envrecord()
    venv = tmp_path / "e"
    (venv / "Lib" / "site-packages" / "torch-2.8.0+cpu.dist-info").mkdir(parents=True)
    (venv / "src" / "Repo" / ".git").mkdir(parents=True)
    (venv / "src" / "Repo" / ".git" / "HEAD").write_text("074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc\n", encoding="utf-8")
    assert er.torch_kind(venv) == "cpu"
    rec = er.write_record(venv, "main", REPO / "envs", cpu_flag=False, torch="torch==2.8.0", extras_failed=["main+voxcpm.txt"], version="9.9")
    on_disk = json.loads((venv / "vndub_env.json").read_text(encoding="utf-8"))
    assert on_disk == rec and rec["cpu"] is True and rec["torch_kind"] == "cpu" and rec["record_version"] == 2  # -Cpu was not passed
    assert rec["git"] == {"Repo": "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"} and rec["app_version"] == "9.9" and rec["extras_failed"] == ["main+voxcpm.txt"]
    shutil.rmtree(venv / "Lib" / "site-packages" / "torch-2.8.0+cpu.dist-info")
    (venv / "Lib" / "site-packages" / "torch-2.8.0+cu128.dist-info").mkdir()
    assert er.torch_kind(venv) == "cuda"


# ---- env_states / env_problems (contract C4) ------------------------------------------------------------------


def test_a_healthy_env_has_no_problems_and_the_contract_keys(o_env: Path) -> None:
    fake_env("funasr", FUNASR_PINS, probe=ok_probe("funasr"))
    st = installer.env_states()["funasr"]
    assert {"present", "complete", "drift", "req_changed", "extras_failed", "torch_kind", "probe", "packages", "problems"} <= set(st)
    assert st["present"] and st["complete"] and st["drift"] == [] and not st["req_changed"] and st["torch_kind"] == "cpu"
    assert st["problems"] == [] and "ruamel-yaml==0.18.17" in st["packages"]
    assert installer.env_states()["chatterbox"]["problems"][0]["code"] == "env_missing"


def test_the_team_s_pre_0_7_1_funasr_env_is_stale_with_the_three_pins_named(o_env: Path) -> None:
    pkgs = {k: v for k, v in FUNASR_PINS.items() if k not in ("setuptools",)}
    pkgs.update({"ruamel.yaml": "0.19.1", "hyperpyyaml": "1.2.2"})
    fake_env("funasr", pkgs, probe=ok_probe("funasr"))
    st = installer.env_states()["funasr"]
    assert st["drift"] == [
        "funasr+cosyvoice.txt: HyperPyYAML==1.2.3 installed 1.2.2",
        "funasr+cosyvoice.txt: ruamel.yaml==0.18.17 installed 0.19.1",
        "funasr+cosyvoice.txt: setuptools==80.10.2 installed none",
    ]
    assert codes(st) == ["env_stale"] and installer.env_problems(["funasr"]) == []  # a warning: the shims keep it working
    assert installer.env_problems(["funasr"], "warn")[0]["env"] == "funasr"


def test_an_unfinished_build_a_failed_probe_and_a_failed_extra(o_env: Path) -> None:
    fake_env("funasr", FUNASR_PINS, record=None)  # make_envs deletes the record when a build starts
    assert codes(installer.env_states()["funasr"]) == ["env_incomplete"]
    bad = {**ok_probe("funasr"), "ok": False, "results": {"cosyvoice3": {"ok": False, "error": "ModuleNotFoundError: No module named 'pkg_resources'"}, "sensevoice": {"ok": True}}}
    fake_env("funasr", FUNASR_PINS, probe=bad)
    probs = installer.env_problems(["funasr", "unknown_env"])
    assert probs == [{"level": "error", "code": "env_broken", "detail": "cosyvoice3: ModuleNotFoundError: No module named 'pkg_resources'", "env": "funasr"}]
    rec = {"env": "funasr", "extras_failed": ["funasr+cosyvoice.txt"], "req_sha256": "x"}
    fake_env("funasr", {k: v for k, v in FUNASR_PINS.items() if k in ("torch", "numpy", "funasr")}, record=rec, probe=ok_probe("funasr"))
    st = installer.env_states()["funasr"]
    assert all("cosyvoice" not in d for d in st["drift"])  # the failed extra's pins are its own problem, not drift
    assert "env_extras_failed" in codes(st) and st["extras_failed"] == ["funasr+cosyvoice.txt"]


def test_a_probe_of_an_older_version_or_none_asks_for_a_check(o_env: Path) -> None:
    fake_env("funasr", FUNASR_PINS)
    assert codes(installer.env_states()["funasr"]) == ["env_unchecked"]
    fake_env("funasr", FUNASR_PINS, probe={**ok_probe("funasr"), "app_version": "1.0.0"})
    assert codes(installer.env_states()["funasr"]) == ["env_unchecked"]


def test_env_states_never_raises_and_is_fast_on_the_real_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    real = REPO / ".venvs"
    if not (real / "main" / "pyvenv.cfg").exists():
        pytest.skip("no built envs on this machine")
    monkeypatch.setenv("VNDUB_ENVS_DIR", str(real))
    t0 = time.perf_counter()
    states = installer.env_states()
    assert time.perf_counter() - t0 < 1.0
    assert states["main"]["present"] and states["main"]["drift"] == [] and states["funasr"]["drift"] == []
    monkeypatch.setattr(installer, "env_state", lambda *a, **k: 1 / 0)
    assert all(s["problems"][0]["code"] == "env_state_failed" for s in installer.env_states().values())


# ---- install_envs: record, extras, probe, disk ------------------------------------------------------------------


def make_envs_stand_in(tmp_path: Path, *, extras_failed: str = "", fail_line: str = "") -> Any:
    script = tmp_path / "fake_make_envs_r18.py"
    script.write_text(
        "import sys, os, json\nfrom pathlib import Path\n"
        "envs_dir, extras, fail, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]\n"
        "for n in names:\n"
        "    print(f'== env {n} (CPU)', flush=True)\n"
        "    e = envs_dir / n\n"
        "    (e / 'vndub_env.json').unlink(missing_ok=True)\n"
        "    py = e / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'); py.parent.mkdir(parents=True, exist_ok=True); py.write_bytes(b'')\n"
        "    (e / 'pyvenv.cfg').write_text('home = x')\n"
        "    if fail:\n"
        "        print(fail, flush=True); sys.exit(2)\n"
        "    (e / 'vndub_env.json').write_text(json.dumps({'env': n, 'cpu': True, 'extras_failed': [x for x in extras.split(',') if x]}))\n",
        encoding="utf-8",
    )
    return lambda envs, torch, path, python=None: ([sys.executable, str(script), str(path), extras_failed, fail_line, *envs], {})


def run_install(names: list[str]) -> dict[str, Any]:
    job = jobs.runner().submit("envs_install", lambda ctx: installer.install_envs(names, "cpu", ctx), gpu=False)
    return wait_job(job.id)


def test_a_failed_extra_finishes_the_install_with_a_warning_naming_the_backends(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "build_make_envs_command", make_envs_stand_in(tmp_path, extras_failed="funasr+cosyvoice.txt"))
    monkeypatch.setattr(installer, "_check_env_space", lambda names, path: None)
    probed: list[str] = []
    monkeypatch.setattr(installer, "probe_env", lambda n, ctx=None, timeout=0: (probed.append(n), {"ok": True, "s": 1.0, "results": {}})[1])
    done = run_install(["funasr"])
    assert done["state"] == "done", done.get("error")
    assert done["result"]["extras_failed"] == [{"env": "funasr", "extra": "funasr+cosyvoice.txt", "backends": ["cosyvoice3"]}]
    assert done["result"]["files"][0]["warnings"] == ["funasr+cosyvoice.txt"]
    assert any(w["kind"] == "unavailable" and "cosyvoice3" in w["group"] for w in done["result"]["warnings"])
    assert probed == ["funasr"]


def test_a_failed_import_probe_fails_the_row_with_env_broken(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "build_make_envs_command", make_envs_stand_in(tmp_path))
    monkeypatch.setattr(installer, "_check_env_space", lambda names, path: None)
    monkeypatch.setattr(installer, "probe_env", lambda n, ctx=None, timeout=0: {"ok": False, "s": 1.0, "results": {"chatterbox_ml": {"ok": False, "error": "ModuleNotFoundError: No module named 'pkg_resources'"}}})
    done = run_install(["chatterbox"])
    assert done["state"] == "failed" and done["error"]["code"] == "env_broken" and done["error"]["params"]["env"] == "chatterbox"
    assert "pkg_resources" in done["error"]["params"]["detail"] and done["result"]["files"][0]["state"] == "failed"


def test_an_interrupted_build_is_not_counted_as_built(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "build_make_envs_command", make_envs_stand_in(tmp_path, fail_line="error: build failed for pyworld"))
    monkeypatch.setattr(installer, "_check_env_space", lambda names, path: None)
    done = run_install(["funasr"])
    assert done["state"] == "failed" and done["result"]["envs"] == []  # pyvenv.cfg exists, the record does not
    assert codes(installer.env_states()["funasr"])[0] == "env_incomplete"


def test_not_enough_space_stops_before_the_build_and_a_full_disk_is_disk_full(o_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vndub_app import downloads

    started: list[int] = []
    build = make_envs_stand_in(tmp_path, fail_line="error: failed to write to file: There is not enough space on the disk. (os error 112)")
    monkeypatch.setattr(installer, "build_make_envs_command", lambda *a, **k: (started.append(1), build(*a, **k))[1])
    monkeypatch.setattr(downloads, "free_space_gb", lambda p: 3.0)
    done = run_install(["main"])
    assert done["error"]["code"] == "disk_full" and done["error"]["params"]["need_gb"] == 9.0 and done["error"]["params"]["free_gb"] == 3.0 and started == []
    monkeypatch.setattr(downloads, "free_space_gb", lambda p: 500.0)
    done = run_install(["main"])
    assert started == [1] and done["error"]["code"] == "disk_full" and "os error 112" in done["error"]["details"]


# ---- Diagnostics ------------------------------------------------------------------------------------------------------


def gpu8() -> doctor.GpuInfo:
    return doctor.GpuInfo("NVIDIA GeForce RTX 4060", 8.0, "572.16", "12.8")


def envs_probe(states: dict[str, Any]) -> dict[str, Any]:
    return {"needed": sorted(states), "present": sorted(states), "states": states}


def st(problems: list[dict[str, str]] | None = None, torch_kind: str | None = "cuda") -> dict[str, Any]:
    return {"present": True, "complete": True, "torch_kind": torch_kind, "problems": problems or []}


def test_envs_row_from_the_worst_finding_with_its_fix() -> None:
    broken = {"level": "error", "code": "env_broken", "detail": "cosyvoice3: No module named 'pkg_resources'"}
    r = doctor.row_envs(envs_probe({"funasr": st([broken]), "main": st()}), False, gpu8())
    assert r["status"] == "red" and r["params"]["reason"] == "broken" and r["params"]["envs"] == "funasr" and "pkg_resources" in r["params"]["detail"]
    assert r["fix"]["action"] == "rebuild_envs"
    r = doctor.row_envs(envs_probe({"main": st(torch_kind="cpu"), "funasr": st()}), False, gpu8())
    assert r["status"] == "yellow" and r["params"]["reason"] == "cpu_torch" and r["params"]["envs"] == "main"
    assert doctor.row_envs(envs_probe({"main": st(torch_kind="cpu")}), False, None)["status"] == "green"  # no NVIDIA card: CPU torch is right
    r = doctor.row_envs(envs_probe({"main": st([{"level": "warn", "code": "env_unchecked", "detail": "x"}])}), False, gpu8())
    assert r["params"]["reason"] == "unchecked" and r["fix"]["action"] == "check_envs"
    assert doctor.row_envs(envs_probe({"main": st()}), False, gpu8())["status"] == "green"


def test_rebuild_envs_submits_exactly_the_envs_with_problems(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stale = {"level": "warn", "code": "env_stale", "detail": "x"}
    doctor.set_probes(doctor.Probes(gpu=gpu8, envs=lambda: envs_probe({"funasr": st([stale]), "main": st(torch_kind="cpu"), "rvc": st()})))
    seen: list[tuple[list[str], str]] = []
    monkeypatch.setattr(installer, "install_envs", lambda names, torch, ctx: (seen.append((names, torch)), {"envs": names})[1])
    job = doctor.apply_fix("rebuild_envs")
    assert wait_job(job["id"])["state"] == "done" and seen == [(["main", "funasr"], "auto")]
    checked: list[list[str]] = []
    monkeypatch.setattr(installer, "check_envs", lambda names, ctx: (checked.append(names), {"checked": names})[1])
    job = doctor.apply_fix("check_envs")
    assert wait_job(job["id"])["state"] == "done" and checked == [["funasr", "main", "rvc"]] and job["kind"] == "envs_check"


def test_disk_row_judges_each_drive_of_data_models_and_envs() -> None:
    free = {"C:": 50.0, "D:": 4.0}
    drives = doctor.disk_by_drive({"data": Path("C:/VNDub"), "models": Path("D:/models"), "envs": Path("C:/VNDub/envs")},
                                  {"default_missing_gb": 30.0}, {"needed": ["main"], "present": []}, lambda p: free[str(p)[:2].upper()])
    assert sorted((d["drive"], d["need_gb"], d["free_gb"]) for d in drives) == [("C:", 19.0, 50.0), ("D:", 30.0, 4.0)]
    p = doctor.Probes(
        gpu=gpu8, disk_free_gb=lambda p: free[str(p)[:2].upper()], internet=lambda: True, tools=lambda: {"ffmpeg": {"present": True, "required": True}},
        vcredist=lambda: doctor.VcRuntime(True, "14.44", "registry"), vcredist_bundled=lambda: True, git=lambda: {"path": "git"},
        envs=lambda: {"needed": ["main"], "present": []}, models=lambda: {"default_missing_gb": 30.0}, data_dir=lambda: Path("C:/VNDub"),
        profile=lambda: "default", processes=lambda: [], huggingface=lambda: {"ok": True, "host": "huggingface.co"},
        memory=lambda: {"ram_gb": 32.0, "commit_gb": 40.0}, folders=lambda: {"models": Path("D:/models"), "envs": Path("C:/VNDub/envs")}, platform="win32",
    )
    disk = {r["id"]: r for r in doctor.run_doctor(p)["rows"]}["disk"]
    assert disk["status"] == "red" and disk["params"]["drive"] == "D:" and disk["params"]["need_gb"] == 30.0 and "C: 50.0 / 19.0" in disk["params"]["drives"]


def test_memory_row_and_its_thresholds() -> None:
    assert doctor.row_memory("win32", {"ram_gb": 32.0, "commit_gb": 40.0})["status"] == "green"
    r = doctor.row_memory("win32", {"ram_gb": 15.9, "commit_gb": 40.0})
    assert r["status"] == "yellow" and r["params"]["reason"] == "low_ram"
    assert doctor.row_memory("win32", {"ram_gb": 16.0, "commit_gb": 20.0})["params"]["reason"] == "small_page_file"
    assert doctor.row_memory("linux", None)["status"] == "green"
    if sys.platform.startswith("win"):
        m = doctor.probe_memory()
        assert m is not None and m["ram_gb"] > 1 and m["commit_gb"] >= m["ram_gb"]


def test_huggingface_has_its_own_row_and_follows_a_mirror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert doctor.hf_endpoint_host() == "huggingface.co"
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert doctor.hf_endpoint_host() == "hf-mirror.com"
    tried: list[tuple[str, int]] = []

    def connect(addr: tuple[str, int], timeout: float) -> Any:
        tried.append(addr)
        raise OSError("blocked")

    monkeypatch.setattr(doctor.socket, "create_connection", connect)
    assert doctor.probe_huggingface() == {"ok": False, "host": "hf-mirror.com"} and tried == [("hf-mirror.com", 443)]
    r = doctor.row_huggingface({"ok": False, "host": "huggingface.co"})
    assert r["status"] == "yellow" and r["params"]["host"] == "huggingface.co"
    assert doctor.row_huggingface({"ok": True, "host": "hf-mirror.com"})["params"]["reason"] == "mirror"


def test_the_release_gate_fails_on_drift_and_on_a_failed_probe(o_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env("funasr", FUNASR_PINS)
    lines: list[str] = []
    verdict = {"ok": True, "s": 2.0, "results": {}}
    monkeypatch.setattr(installer, "probe_env", lambda n, ctx=None, timeout=0: verdict)
    assert installer.release_gate(lines.append) == 0 and any(l.startswith("funasr: ok") for l in lines)
    verdict = {"ok": False, "s": 2.0, "results": {"cosyvoice3": {"ok": False, "error": "ModuleNotFoundError: pkg_resources"}}}
    lines.clear()
    assert installer.release_gate(lines.append) == 1 and "import check: cosyvoice3: ModuleNotFoundError: pkg_resources" in lines[-1]
    fake_env("funasr", {**FUNASR_PINS, "ruamel.yaml": "0.19.1"})
    lines.clear()
    assert installer.release_gate(lines.append) == 1 and "ruamel.yaml==0.18.17 installed 0.19.1" in lines[-1]


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell scripts")
def test_the_powershell_scripts_parse() -> None:
    for script in (REPO / "scripts" / "make_envs.ps1", REPO / "packaging" / "build_app.ps1"):
        cmd = f"$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile('{script}', [ref]$null, [ref]$e); if ($e) {{ $e | % {{ $_.Message }}; exit 1 }}"
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (script.name, r.stdout, r.stderr)
    text = (REPO / "packaging" / "build_app.ps1").read_text(encoding="utf-8")
    assert "[string]$SkipEnvProbe" in text and "installer.release_gate()" in text


# ---- make_envs.ps1 with a stand-in uv --------------------------------------------------------------------------


FAKE_UV = r'''
import json, os, subprocess, sys
from pathlib import Path
log = Path(os.environ["FAKE_UV_LOG"])
with log.open("a", encoding="utf-8") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
args = sys.argv[1:]
if args[:1] == ["venv"]:
    subprocess.run([os.environ["FAKE_UV_PY"], "-m", "venv", "--without-pip", args[1]], check=True)
elif args[:2] == ["pip", "install"] and "--index-url" in args:
    py = Path(args[args.index("--python") + 1]); sp = py.parent.parent / "Lib" / "site-packages"
    kind = "cpu" if args[args.index("--index-url") + 1].endswith("/cpu") else "cu128"
    for p in [a for a in args if "==" in a]:
        n, v = p.split("==")
        have = list(sp.glob(f"{n}-*.dist-info"))
        if have and "--reinstall-package" not in args:
            continue  # uv: torch==2.8.0 is met by 2.8.0+cpu
        for h in have:
            h.rmdir()
        (sp / f"{n}-{v}+{kind}.dist-info").mkdir()
elif os.environ.get("FAKE_UV_FAIL") and args[:2] == ["pip", "install"] and "-r" in args:
    sys.exit(1)
'''


@pytest.mark.skipif(os.name != "nt", reason="make_envs.ps1 is the Windows script")
def test_make_envs_reinstalls_cpu_torch_for_cuda_and_writes_the_record(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "fake_uv.py").write_text(FAKE_UV, encoding="utf-8")
    (bin_dir / "uv.cmd").write_text(f'@"{sys.executable}" "{bin_dir / "fake_uv.py"}" %*\r\n', encoding="utf-8")
    envs = tmp_path / "envs"
    log = tmp_path / "uv.log"
    base = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "FAKE_UV_LOG": str(log), "FAKE_UV_PY": sys.executable, "VNDUB_APP_VERSION": "7.7.7"}
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(REPO / "scripts" / "make_envs.ps1"), "-Env", "f5", "-EnvsDir", str(envs), "-SkipExtras"]

    r = subprocess.run([*cmd, "-Cpu"], env=base, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((envs / "f5" / "vndub_env.json").read_text(encoding="utf-8"))
    assert rec["torch_kind"] == "cpu" and rec["cpu"] is True and rec["app_version"] == "7.7.7"
    assert rec["req_sha256"] == installer.envrecord().req_sha256(REPO / "envs", "f5")

    log.write_text("", encoding="utf-8")
    r = subprocess.run(cmd, env=base, capture_output=True, text=True, timeout=300)  # CUDA asked for over the CPU build
    assert r.returncode == 0, r.stdout + r.stderr
    torch_call = next(json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if "--index-url" in l)
    assert torch_call[torch_call.index("--reinstall-package") + 1] == "torch" and "torchaudio" in torch_call
    rec = json.loads((envs / "f5" / "vndub_env.json").read_text(encoding="utf-8"))
    assert rec["torch_kind"] == "cuda" and rec["cpu"] is False

    r = subprocess.run(cmd, env={**base, "FAKE_UV_FAIL": "1"}, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and not (envs / "f5" / "vndub_env.json").exists()  # the failed rebuild left no record
